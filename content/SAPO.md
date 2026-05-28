---
description: 用平滑的温度控制门替代硬裁剪的 RL 算法，通过非对称温度设计实现序列级连贯性和 token 级自适应性
type: method
aliases:
  - SAPO
  - Soft Adaptive Policy Optimization
  - 软自适应策略优化
prerequisites:
  - "[[GRPO]]"
  - "[[GSPO]]"
  - "[[PPO]]"
tags:
  - post-training
  - reinforcement-learning
  - rlhf
  - optimization
created: 2026-01-27
updated: 2026-01-31T22:42
---

# SAPO (Soft Adaptive Policy Optimization)

软自适应策略优化（SAPO, Soft Adaptive Policy Optimization）是一种**平滑且自适应**的 RL 算法，通过用温度控制的软门函数替代 [[GRPO]] 和 [[GSPO]] 的硬裁剪，在保持训练稳定性的同时提升样本效率。SAPO 的核心创新是**非对称温度设计**，对正负 token 使用不同的衰减速率，显著改善了大规模 LLM（尤其是 MoE 模型）的 RL 训练稳定性。

> [!paper] 论文出处
> Gao et al., "Soft Adaptive Policy Optimization", Qwen Team, 2025
> - 在 Qwen3-VL 系列上验证有效性
> - 相比 GSPO 和 GRPO 提升训练稳定性和 Pass@1 性能
> - 支持文本和多模态任务

---

## 动机

> [!intuition] 为什么需要 SAPO？

[[GRPO]] 和 [[GSPO]] 使用**硬裁剪（hard clipping）**来约束策略更新，但这带来了根本性的权衡困境：

1. **GRPO 的问题**：Token 级硬裁剪是"全有或全无"的
   - 在裁剪范围内：梯度完全保留
   - 在裁剪范围外：梯度直接归零
   - 结果：要么样本利用率低（裁剪太紧），要么训练不稳定（裁剪太松）

2. **GSPO 的问题**：序列级硬裁剪过于粗粒度
   - 如果序列中有**少数几个** off-policy token，整个序列的梯度都被抑制
   - 丢失了序列中**大量 near-on-policy token** 的学习信号
   - 样本效率低

3. **MoE 模型的挑战**：Token 级重要性比率方差更大
   - Expert routing 的异质性导致 token 间差异放大
   - 长响应进一步加剧方差
   - 硬裁剪在高方差场景下更容易导致训练崩溃

> [!warning] 硬裁剪的根本问题
> 硬裁剪是**不连续**的，在裁剪边界处梯度突变，导致：
> - 优化噪声大
> - 难以平衡稳定性和样本效率
> - 在 MoE 等高方差场景下容易崩溃

---

## 核心机制

### 软门函数（Soft Gate）

> [!definition] SAPO 的核心思想
> 用**平滑的 sigmoid 形状函数**替代硬裁剪，实现连续的信任域（continuous trust region）。

**SAPO 目标函数**：

$$
\mathcal{J}_{SAPO}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} f_{i,t}(r_{i,t}(\theta)) \hat{A}_{i,t} \right]
$$

其中软门函数：

$$
f_{i,t}(x) = \sigma(\tau_{i,t}(x - 1)) \cdot \frac{4}{\tau_{i,t}}, \quad \tau_{i,t} = \begin{cases} \tau_{pos}, & \text{if } \hat{A}_{i,t} > 0 \\ \tau_{neg}, & \text{otherwise} \end{cases}
$$

- $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 sigmoid 函数
- $r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}|q, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|q, y_{i,<t})}$ 是重要性比率
- $\tau_{pos}, \tau_{neg}$ 是正负 token 的温度参数

### 梯度权重

对目标函数求导，得到加权的 log-policy 梯度：

$$
\nabla_\theta \mathcal{J}_{SAPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} w_{i,t}(\theta) \, r_{i,t}(\theta) \, \nabla_\theta \log \pi_\theta(y_{i,t}|q, y_{i,<t}) \hat{A}_{i,t} \right]
$$

其中梯度权重：

$$
w_{i,t}(\theta) = 4 p_{i,t}(\theta) (1 - p_{i,t}(\theta)), \quad p_{i,t}(\theta) = \sigma(\tau_{i,t}(r_{i,t}(\theta) - 1))
$$

> [!intuition] 软门的特性
>
> 1. **在 on-policy 点峰值**：$r_{i,t} = 1$ 时，$w_{i,t} = 1$（与无裁剪目标一致）
> 2. **平滑衰减**：随着 $r_{i,t}$ 偏离 1，权重平滑下降（近似指数衰减）
> 3. **避免梯度消失**：即使 off-policy，仍保留部分梯度信号
> 4. **连续信任域**：没有硬边界，优化更稳定

### 非对称温度设计

> [!definition] 为什么正负 token 需要不同温度？

**关键洞察**：负 token 的梯度更容易引起不稳定。

通过分析梯度在 logits 上的传播：

$$
\frac{\partial \log \pi_\theta(y_{i,t}|q, y_{i,<t}) \hat{A}_{i,t}}{\partial z_v} = \begin{cases}
(1 - \pi_\theta(y_{i,t}|q, y_{i,<t})) \cdot \hat{A}_{i,t} & \text{if } v = y_{i,t} \text{ (采样 token)} \\
-\pi_\theta(v|q, y_{i,<t}) \cdot \hat{A}_{i,t} & \text{otherwise (未采样 token)}
\end{cases}
$$

**正 advantage**（$\hat{A} > 0$）：
- 增加采样 token 的 logit
- 减少所有未采样 token 的 logit
- 影响集中，稳定

**负 advantage**（$\hat{A} < 0$）：
- 减少采样 token 的 logit
- **增加所有未采样 token 的 logit**
- 在大词表（几十万 token）中，梯度扩散到大量无关 token
- 引入高方差，容易不稳定

> [!math] 非对称温度的作用
>
> 设置 $\tau_{neg} > \tau_{pos}$（论文中 $\tau_{pos} = 1.0, \tau_{neg} = 1.05$）：
> - 更大的 $\tau$ → 更快的衰减
> - 负 token 的梯度在 off-policy 时衰减更快
> - 抑制高方差的负梯度，提升稳定性

---

## 与 GRPO/GSPO 的关系

### 统一视角：门函数

三种算法可以用统一的框架表示：

$$
\mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} f_{i,t}(r_{i,t}(\theta)) \hat{A}_{i,t} \right]
$$

区别在于门函数 $f_{i,t}$ 的选择：

| 算法 | 门函数 $f_{i,t}$ | 特点 |
|------|------------------|------|
| **GRPO** | $\min(r_{i,t}, 1+\varepsilon)$ 或 $\max(r_{i,t}, 1-\varepsilon)$ | Token 级硬裁剪 |
| **GSPO** | $\min(s_{i,t}, 1+\varepsilon)$ 或 $\max(s_{i,t}, 1-\varepsilon)$ | 序列级硬裁剪 |
| **SAPO** | $\sigma(\tau_{i,t}(r_{i,t} - 1)) \cdot \frac{4}{\tau_{i,t}}$ | Token 级软门 |

### SAPO vs GRPO

> [!comparison] 平滑 vs 硬裁剪

**GRPO 的梯度权重**（硬裁剪）：

$$
f_{i,t}^{GRPO'}(r_{i,t}) = \begin{cases}
1, & \text{if } 1-\varepsilon \le r_{i,t} \le 1+\varepsilon \\
0, & \text{otherwise}
\end{cases}
$$

**SAPO 的梯度权重**（软门）：

$$
f_{i,t}^{SAPO'}(r_{i,t}) = \text{sech}^2\left(\frac{\tau_{i,t}}{2}(r_{i,t} - 1)\right)
$$

**优势**：
- GRPO：二元信任域（全有或全无）→ 样本利用率低或训练不稳定
- SAPO：连续信任域（平滑衰减）→ 保留 moderate off-policy 的学习信号

### SAPO vs GSPO

> [!comparison] Token 级自适应 vs 序列级粗粒度

**GSPO 的问题**：
- 序列中有少数 off-policy token → 整个序列被抑制
- 丢失大量 near-on-policy token 的信号

**SAPO 的解决方案**：
- 在温和假设下（小步长、低序列内方差），SAPO 等价于序列级软门
- 当假设不成立时（异质 token、outlier），SAPO 退化为 token 级门函数
- **选择性衰减**：只降低 offending token 的权重，保留 near-on-policy token

> [!math] 序列级等价性
>
> 在假设 (A1) 小步长和 (A2) 低序列内方差下，SAPO 的平均 token 门近似为序列级门：
>
> $$
> \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} f_{i,t}^{SAPO'}(r_{i,t}) \approx g_{\tau_i}(\log s_i(\theta)) = \text{sech}^2\left(\frac{\tau_i}{2} \log s_i(\theta)\right)
> $$
>
> 其中 $s_i(\theta) = \left(\frac{\pi_\theta(y_i|q)}{\pi_{\theta_{old}}(y_i|q)}\right)^{1/|y_i|}$ 是长度归一化的序列级比率。

**实证验证**：
- 论文在 Qwen3-30B-A3B (MoE) 和 Qwen3-4B (dense) 上验证假设
- $r_{i,t}$ 集中在 1 附近，$\text{Var}_i(\theta)$ 通常 < 0.02
- MoE 模型方差稍大（expert routing 异质性），但假设仍大致成立

---

## 实验结果

### 数学推理任务

在 Qwen3-30B-A3B-Base 上对比 SAPO、GSPO、GRPO-R2：

| 方法 | 训练稳定性 | 最终性能 | 特点 |
|------|------------|----------|------|
| GSPO | 早期崩溃 | 低 | 需要 routing replay |
| GRPO-R2 | 早期崩溃 | 低 | 需要 routing replay |
| **SAPO** | 持续稳定 | **最高** | 不需要 routing replay |

**Benchmark 结果**（Pass@1）：
- AIME25、HMMT25、BeyondAIME 上均优于 GSPO 和 GRPO-R2
- 训练过程更长时间保持稳定
- 最终达到更高准确率

### 温度消融实验

对比三种温度设置：

| 设置 | $\tau_{pos}$ | $\tau_{neg}$ | 训练稳定性 |
|------|--------------|--------------|------------|
| 非对称（推荐） | 1.0 | 1.05 | **最稳定** |
| 对称 | 1.0 | 1.0 | 中等 |
| 反向非对称 | 1.0 | 0.95 | **最不稳定** |

> [!warning] 关键发现
> $\tau_{neg} < \tau_{pos}$ 导致显著不稳定，验证了负 token 梯度是主要不稳定来源。

### Qwen3-VL 大规模训练

在 Qwen3-VL 系列（多种规模、MoE 和 dense 架构）上验证：

**任务**：
- 文本任务：数学、编程、逻辑推理
- 多模态任务：视觉理解

**结果**：
- SAPO 在所有任务和模型规模上持续优于 GSPO 和 GRPO-R2
- 支持多任务学习（固定任务采样比例）
- 不需要 routing replay 等额外工程技巧

---

## 优势总结

> [!intuition] SAPO 的核心优势

1. **序列级连贯性 + Token 级自适应性**
   - 正常情况：等价于序列级优化（如 GSPO）
   - 异常情况：退化为 token 级门函数，选择性处理 outlier

2. **平滑信任域**
   - 避免硬裁剪的不连续性
   - 保留 moderate off-policy 的学习信号
   - 优化更稳定

3. **非对称温度**
   - 针对性抑制高方差负梯度
   - 显著提升训练稳定性
   - 尤其适合 MoE 模型

4. **工程简化**
   - 不需要 routing replay
   - 不需要复杂的稳定性技巧
   - 更少的超参数调优

---

## 局限性

> [!warning] SAPO 的局限

1. **仍需 Reward Model**：与 GRPO/GSPO 一样，需要训练 reward model
2. **采样开销**：每个问题需要采样多个输出（G 个）
3. **温度调优**：虽然论文给出了推荐值（$\tau_{pos}=1.0, \tau_{neg}=1.05$），但不同任务可能需要调整
4. **理论保证有限**：序列级等价性依赖假设 (A1) 和 (A2)，极端情况下可能不成立
5. **计算开销**：相比硬裁剪，sigmoid 计算略有额外开销（但可忽略）

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: SAPO 相比 GRPO/GSPO 的核心改进是什么？**
> A: 用平滑的温度控制软门替代硬裁剪。硬裁剪是"全有或全无"的，SAPO 的软门在 on-policy 点峰值为 1，随着偏离平滑衰减，保留 moderate off-policy 的学习信号，提升样本效率和稳定性。
>
> **Q2: 为什么正负 token 需要不同温度？**
> A: 负 advantage 的梯度会扩散到大量未采样 token（大词表中几十万个），引入高方差和不稳定性。设置 $\tau_{neg} > \tau_{pos}$ 让负梯度衰减更快，抑制不稳定来源。
>
> **Q3: SAPO 如何同时实现序列级连贯性和 token 级自适应性？**
> A: 在温和假设下（小步长、低序列内方差），SAPO 的平均 token 门近似为序列级门，类似 GSPO。当假设不成立时（异质 token、outlier），SAPO 退化为 token 级门，选择性降低 offending token 权重，保留 near-on-policy token 的信号。
>
> **Q4: SAPO 的软门函数是什么形状？**
> A: $f_{i,t}(r_{i,t}) = \sigma(\tau_{i,t}(r_{i,t} - 1)) \cdot \frac{4}{\tau_{i,t}}$，梯度权重是 $w_{i,t} = \text{sech}^2(\frac{\tau_{i,t}}{2}(r_{i,t} - 1))$，在 $r_{i,t}=1$ 处峰值为 1，随偏离平滑衰减。
>
> **Q5: SAPO 在 MoE 模型上的优势是什么？**
> A: MoE 模型的 expert routing 导致 token 级重要性比率方差更大，硬裁剪容易导致训练崩溃。SAPO 的软门和非对称温度设计能更好地处理高方差场景，不需要 routing replay 等额外技巧。

---

## 相关概念

- [[GRPO]] — SAPO 改进的基础，token 级硬裁剪
- [[GSPO]] — SAPO 改进的基础，序列级硬裁剪
- [[PPO]] — 所有方法的理论基础
- [[Dr. GRPO]] — 另一个 GRPO 改进方向，修正优化偏差
- [[RLHF]] — SAPO 是 RLHF 的一种实现
- [[Reward Model]] — SAPO 仍需要 reward model

---

## 参考资料

- [[251120347v2|Soft Adaptive Policy Optimization (2025)]] — SAPO 的原始论文
- [[GSPO]] — 序列级优化的前身
- [[GRPO]] — Token 级优化的前身
