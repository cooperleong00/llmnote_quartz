---
description: 通过 Clip-Higher、Dynamic Sampling 等四项技术改进 GRPO，解决大规模 LLM RL 中的 entropy collapse 和训练不稳定问题
type: method
aliases:
  - Decoupled Clip and Dynamic sAmpling Policy Optimization
  - 解耦裁剪与动态采样策略优化
prerequisites:
  - "[[GRPO]]"
  - "[[PPO]]"
tags:
  - post-training
  - reinforcement-learning
  - reasoning
created: 2025-03-17
updated: 2026-02-01T01:02
---

# DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)

DAPO 是 [[GRPO]] 的改进版本，专门针对**大规模 long-CoT 推理场景**设计。作为 [[RLHF]] 的一种高效实现，它通过四项关键技术解决了 naive GRPO 在实践中遇到的 entropy collapse、梯度消失和训练不稳定问题，在 AIME 2024 上达到 50 分，超越 DeepSeek-R1-Zero-Qwen-32B (47 分)。

> [!paper] 论文出处
> ByteDance Seed & Tsinghua AIR, "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", 2025
> - 完全开源：算法、代码（基于 verl）、数据集（DAPO-Math-17K）
> - 项目主页：https://dapo-sia.github.io/

---

## 动机

> [!intuition] 为什么需要 DAPO？

在尝试复现 DeepSeek-R1 的 RL 训练时，naive [[GRPO]] 只能在 AIME 上达到 30 分，远低于 DeepSeek 的 47 分。分析发现三个核心问题：

1. **[[Entropy Collapse]]**：策略的 entropy 快速下降，采样多样性丧失，探索能力受限
2. **梯度消失**：随着训练进行，越来越多 prompt 的所有输出都正确（accuracy=1），导致 advantage=0，有效梯度减少
3. **Reward 噪声**：超长样本被截断后的惩罚引入噪声，干扰训练

这些问题在 long-CoT 场景下尤为严重，因为输出长度可达数万 token。

---

## 核心机制

### DAPO 目标函数

> [!math] 数学形式

$$
\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot | q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min \left( r_{i,t} \hat{A}_{i,t}, \text{clip}(r_{i,t}, 1 - \varepsilon_{\text{low}}, 1 + \varepsilon_{\text{high}}) \hat{A}_{i,t} \right) \right]
$$

约束条件：$0 < |\{o_i \mid \text{is\_equivalent}(a, o_i)\}| < G$

其中：
- $r_{i,t} = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}$ 是概率比
- $\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_i\})}{\text{std}(\{R_i\})}$ 是 group relative advantage
- $\varepsilon_{\text{low}}, \varepsilon_{\text{high}}$ 是**解耦的** clip 范围

**与 GRPO 的关键区别**：
1. 解耦的 clip 范围（Clip-Higher）
2. Token-level loss（而非 sample-level）
3. Dynamic Sampling 约束
4. 移除 KL penalty（long-CoT 场景不需要）

---

## 四项关键技术

### 1. Clip-Higher：解决 Entropy Collapse

> [!intuition] 问题：为什么 [[PPO]]/GRPO 会 entropy collapse？

标准 clip 机制使用对称的 $\varepsilon$（如 0.2），即 $r_t \in [0.8, 1.2]$。

考虑 $A_t > 0$（好 token，应增加概率）的情况：
- 高概率 token（$\pi_{old} = 0.9$）：上界是 $0.9 \times 1.2 = 1.08$，几乎不受限
- 低概率 token（$\pi_{old} = 0.01$）：上界是 $0.01 \times 1.2 = 0.012$，增幅极小

**结果**：低概率的"探索" token 很难被强化，策略逐渐收敛到高概率 token，entropy 下降。

> [!definition] Clip-Higher 策略

**解耦上下 clip 范围**：
- $\varepsilon_{\text{low}} = 0.2$（保持不变，防止概率被压到 0）
- $\varepsilon_{\text{high}} = 0.28$（放宽上界，给低概率 token 更多增长空间）

$$
\text{clip}(r_{i,t}, 1 - \varepsilon_{\text{low}}, 1 + \varepsilon_{\text{high}})
$$

**效果**：entropy 保持健康增长，采样多样性得以维持。

---

### 2. Dynamic Sampling：解决梯度消失

> [!intuition] 问题：为什么梯度会消失？

在 GRPO 中，如果一个 prompt 的所有 $G$ 个输出都正确（或都错误），则：
- 所有 reward 相同 → $\text{std}(\{R_i\}) = 0$ → advantage 无法计算或为 0
- 零 advantage = 零梯度

随着训练进行，模型变强，accuracy=1 的 prompt 越来越多（实验中超过 40%），有效梯度信号持续减少。

> [!definition] Dynamic Sampling 策略

**过滤并重采样**：只保留 $0 < \text{accuracy} < 1$ 的 prompt。

约束条件：$0 < |\{o_i \mid \text{is\_equivalent}(a, o_i)\}| < G$

**实现**：
1. 采样一批 prompt
2. 过滤掉 accuracy=0 或 1 的
3. 继续采样直到 batch 填满

**效果**：每个 batch 都有 100% 有效梯度，训练效率反而提升（更少的 step 达到相同性能）。

---

### 3. Token-Level Loss：解决长序列问题

> [!intuition] 问题：Sample-level loss 有什么问题？

GRPO 使用 sample-level loss：先对每个样本内的 token loss 取平均，再对样本取平均。

$$
\text{GRPO: } \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} L_{i,t}
$$

问题：
- 长样本中的每个 token 权重被稀释
- 高质量长推理的 pattern 难以学习
- 低质量长样本（重复、乱码）的惩罚不够

> [!definition] Token-Level Loss

**按 token 总数归一化**：

$$
\text{DAPO: } \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} L_{i,t}
$$

**效果**：
- 长序列对梯度贡献更大
- 无论出现在长还是短样本中，同一 pattern 的强化/惩罚力度相同
- 训练更稳定，长度增长更健康

---

### 4. Overlong Reward Shaping：减少 Reward 噪声

> [!intuition] 问题：截断样本的 reward 怎么处理？

Long-CoT 场景设置最大长度（如 16K token），超长样本被截断。

默认做法：给截断样本惩罚 reward。但这会引入噪声——一个正确的推理过程可能仅因为太长而被惩罚。

> [!definition] Overlong Reward Shaping

**两阶段策略**：

1. **Overlong Filtering**：直接 mask 截断样本的 loss（不参与训练）
2. **Soft Overlong Punishment**：渐进式惩罚

$$
R_{\text{length}}(y) = \begin{cases}
0, & |y| \le L_{\max} - L_{\text{cache}} \\
\frac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < |y| \le L_{\max} \\
-1, & |y| > L_{\max}
\end{cases}
$$

**参数设置**：$L_{\max} = 16384$，$L_{\text{cache}} = 4096$

**效果**：训练更稳定，模型学会控制输出长度。

---

## 实验结果

### Ablation Study

| 配置 | AIME24 avg@32 |
|------|---------------|
| DeepSeek-R1-Zero-Qwen-32B | 47 |
| Naive GRPO | 30 |
| + Overlong Filtering | 36 (+6) |
| + Clip-Higher | 38 (+2) |
| + Soft Overlong Punishment | 41 (+3) |
| + Token-level Loss | 42 (+1) |
| + Dynamic Sampling (**DAPO**) | **50** (+8) |

> [!intuition] 关键发现
> - 每项技术都有贡献，Dynamic Sampling 提升最大
> - Token-level loss 虽然分数提升小，但显著改善训练稳定性
> - 最终超越 DeepSeek-R1-Zero，且只用 50% 的训练 step

### 训练超参数

| 参数 | 值 |
|------|-----|
| Base model | Qwen2.5-32B |
| Learning rate | $1 \times 10^{-6}$ |
| Prompt batch size | 512 |
| Group size $G$ | 16 |
| $\varepsilon_{\text{low}}$ | 0.2 |
| $\varepsilon_{\text{high}}$ | 0.28 |
| Max length | 20480 (16384 + 4096 cache) |

---

## 训练动态监控

> [!example] 关键监控指标

1. **Response Length**：应该逐渐增长（更复杂的推理），但不是单调的
2. **Reward**：稳定上升，但训练集 reward 与验证集 accuracy 相关性不高（过拟合风险）
3. **Entropy**：应保持缓慢上升趋势，过低表示探索不足，过高表示乱码/重复
4. **Generation Probability**：与 entropy 相反

> [!warning] 涌现现象
> 训练过程中观察到推理模式的涌现：早期模型几乎不会 self-reflection，但随着训练进行，逐渐出现检查和回溯行为。

---

## 局限性

> [!warning] DAPO 的局限

1. **仅验证于数学任务**：论文只在 AIME 等数学 benchmark 上验证，其他领域效果未知
2. **Rule-based Reward**：依赖可验证的答案，不适用于开放式任务（但避免了 [[Reward Hacking]] 问题）
3. **计算开销**：Dynamic Sampling 需要过采样，增加推理成本
4. **超参数敏感**：$\varepsilon_{\text{high}}$ 的选择需要调优

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: DAPO 相比 GRPO 的核心改进是什么？**
> A: 四项技术：(1) Clip-Higher 解耦上下 clip 范围避免 entropy collapse；(2) Dynamic Sampling 过滤零梯度样本；(3) Token-level loss 解决长序列权重问题；(4) Overlong Reward Shaping 减少截断噪声。
>
> **Q2: 为什么标准 clip 会导致 entropy collapse？**
> A: 对称 clip 对低概率 token 的增幅限制太严格。例如 $\pi_{old}=0.01$ 的 token 最多只能增到 0.012，而 $\pi_{old}=0.9$ 的 token 几乎不受限。这导致探索 token 难以被强化。
>
> **Q3: Dynamic Sampling 为什么能提升效率？**
> A: 过滤 accuracy=0/1 的样本后，每个 batch 都有 100% 有效梯度。虽然需要更多采样，但训练 step 减少，总体收敛更快。
>
> **Q4: DAPO 为什么移除了 KL penalty？**
> A: Long-CoT 推理场景下，模型分布需要显著偏离初始模型才能学会复杂推理，KL 约束反而是障碍。
>
> **Q5: Token-level vs Sample-level loss 的区别？**
> A: Sample-level 先对样本内 token 平均再对样本平均，长样本中每个 token 权重被稀释。Token-level 直接对所有 token 平均，长样本贡献更大，pattern 强化更一致。

---

## 参考资料

- [[250314476v2|DAPO Paper (2025)]] — 原始论文
- [[GRPO]] — 前置方法
- [verl](https://github.com/volcengine/verl) — DAPO 的实现框架
- [DAPO-Math-17K](https://dapo-sia.github.io/) — 开源数据集
