---
type: paper
description: 提出 RL 训练的理论框架，证明 token-level 目标是 sequence-level 奖励的一阶近似，需最小化训练-推理差异和策略陈旧度
aliases:
  - Stabilizing RL with LLMs
  - Qwen RL Formulation
prerequisites:
  - "[[Policy Gradient]]"
  - "[[Importance Sampling]]"
  - "[[Training-Inference Mismatch]]"
  - "[[Mixture of Experts]]"
tags:
  - post-training
  - reinforcement-learning
  - stability
  - theory
  - moe
created: 2026-02-09
updated: 2026-02-09T15:07
---

# Stabilizing Reinforcement Learning with LLMs (2025)

这篇论文由 Qwen 团队提出，为 LLM 强化学习训练建立了理论框架，解释了**为什么 token-level 目标可以优化 sequence-level 奖励**，并给出了稳定训练的充要条件。核心贡献是证明 token-level 目标是 sequence-level 目标的**一阶近似**，其有效性依赖于同时最小化**训练-推理差异**和**策略陈旧度**。

> [!paper] 论文信息
> - **标题**: Stabilizing Reinforcement Learning with LLMs: Formulation and Practices
> - **作者**: An Yang, Jingren Zhou, Junyang Lin, Chujie Zheng 等（Qwen Team, Alibaba）
> - **arXiv**: 2512.01374v3
> - **年份**: 2025
> - **Clipping**: [[Clippings/Paper/251201374v3/251201374v3]]

---

## 动机

> [!intuition] 为什么需要这个理论框架？

LLM 强化学习训练中存在一个基本问题：我们想优化的是**序列级奖励**（sequence-level reward），但实际优化的是**token 级目标**（token-level objective）。

**核心困惑**：
- **目标**：最大化 $\mathbb{E}_{y \sim \pi_\theta}[R(x, y)]$（整个序列的奖励）
- **实践**：优化 $\sum_{t=1}^{|y|} \log \pi_\theta(y_t|x, y_{<t}) \cdot R(x, y)$（每个 token 的对数概率加权）

**为什么这样做有效？** 现有算法（[[PPO]]、[[GRPO]]、[[DPO]]）都在用 token-level 目标，但缺乏理论解释。

**实践中的不稳定性**：
- 训练崩溃（collapse）
- 梯度范数剧烈波动
- 在 [[Mixture of Experts|MoE]] 模型和 long CoT 场景下问题尤为严重

这篇论文回答了两个关键问题：
1. **理论**：token-level 目标在什么条件下是 sequence-level 目标的有效近似？
2. **实践**：如何设计算法来满足这些条件，从而稳定训练？

---

## 核心理论：一阶近似

### Sequence-level 目标（真实目标）

我们想优化的是期望序列级奖励：

$$
\mathcal{J}^{\text{seq}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} [R(x, y)]
$$

**问题**：直接优化很困难，因为：
1. 需要从当前策略 $\pi_\theta$ 采样（on-policy），样本效率低
2. 梯度估计方差大

### Token-level 目标（实际优化）

实践中使用的 surrogate 目标：

$$
\mathcal{J}^{\text{token}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \mu_{\theta_{\text{old}}}(\cdot|x)} \left[ \sum_{t=1}^{|y|} \frac{\pi_\theta(y_t|x, y_{< t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{< t})} R(x, y) \right]
$$

其中：
- $\mu_{\theta_{\text{old}}}$：推理引擎中的 rollout 策略（生成样本）
- $\pi_\theta$：训练引擎中的目标策略（计算梯度）
- $\frac{\pi_\theta(y_t|x, y_{< t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{< t})}$：token 级[[Importance Sampling|重要性权重]]

### 为什么是一阶近似？

> [!math] 推导

序列级重要性比率可以分解为：

$$
\frac{\pi_\theta(y|x)}{\mu_{\theta_{\text{old}}}(y|x)} = \prod_{t=1}^{|y|} \frac{\pi_\theta(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})} = \prod_{t=1}^{|y|} (1 + \delta_t)
$$

其中 $\delta_t = \frac{\pi_\theta(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})} - 1$ 是小量。

**泰勒展开**（忽略二阶及以上项）：

$$
\prod_{t=1}^{|y|} (1 + \delta_t) \approx 1 + \sum_{t=1}^{|y|} \delta_t
$$

因此：

$$
\nabla_\theta \mathcal{J}^{\text{seq}}(\theta) \approx \nabla_\theta \mathcal{J}^{\text{token}}(\theta)
$$

> [!intuition] 直觉理解
>
> 当每个 token 的概率比 $\frac{\pi_\theta}{\mu_{\theta_{\text{old}}}}$ 都接近 1 时，序列级的乘积 $\prod (1+\delta_t)$ 可以近似为求和 $1 + \sum \delta_t$。
>
> 这就像复利计算：当利率很小时，$(1+r_1)(1+r_2) \approx 1 + r_1 + r_2$。

---

## 两个关键条件

> [!warning] 一阶近似何时成立？

论文将 token 级重要性权重分解为两部分：

$$
\frac{\pi_\theta(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})}}_{\text{training-inference discrepancy}} \times \underbrace{\frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t})}}_{\text{policy staleness}}
$$

其中：
- $\mu_{\theta_{\text{old}}}$：推理引擎的策略
- $\pi_{\theta_{\text{old}}}$：训练引擎加载相同参数的策略
- $\pi_\theta$：更新后的策略

### 条件 1：最小化训练-推理差异

**来源**：推理引擎和训练引擎对同一参数计算的概率不同。

**原因**：
- 数值精度差异（FP8 vs BF16）
- 并行策略差异
- 算子实现差异
- [[Mixture of Experts|MoE]] 的动态路由敏感性

**解决方法**：
- [[Importance Sampling|重要性采样]]校正（如 TIS）
- [[Rollout Routing Replay]]（针对 MoE）
- [[IcePop]]（masking 不可靠 token）

详见 [[Training-Inference Mismatch]]。

### 条件 2：控制策略陈旧度

**来源**：off-policy 训练中，rollout 策略 $\pi_{\theta_{\text{old}}}$ 和当前策略 $\pi_\theta$ 的差异。

**解决方法**：
- Clipping（如 [[PPO]] 的 clip 机制）
- 限制更新步数（mini-batch size）
- KL 惩罚项

> [!intuition] 为什么需要同时满足两个条件？
>
> 想象你在用旧地图（rollout 策略）导航，但地图本身就画错了（训练-推理差异）。即使你小心翼翼地跟着地图走（控制策略陈旧度），最终还是会迷路。
>
> 反之，即使地图准确，但你走得太远偏离了地图范围（策略陈旧度过大），也会出问题。
>
> **两个条件缺一不可**。

---

## MoE 的特殊挑战

### 路由不一致放大差异

在 [[Mixture of Experts|MoE]] 模型中，token 级重要性权重进一步分解：

$$
\frac{\pi_\theta(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})} = \underbrace{\text{routing discrepancy}}_{\text{expert selection}} \times \underbrace{\text{weight discrepancy}}_{\text{gating weights}}
$$

**问题**：
- 路由器对输入极其敏感，微小差异可能导致完全不同的专家选择
- 不同专家的输出差异远大于数值精度差异
- 约 10% 的路由器选择不同专家 → 94% 的 token 至少在一层中选择了不同专家

### Routing Replay 的解决方案

[[Rollout Routing Replay]] (R3) 的核心思想：

**记录推理时的路由决策，训练时重放**：

1. **推理阶段**：记录每层每个 token 的 Top-K 专家掩码 $\mathbf{I}_{\text{infer}}$
2. **训练阶段**：
   - 使用 $\mathbf{I}_{\text{infer}}$ 确定专家选择（对齐推理）
   - 但用训练引擎的 logits 计算门控权重（保留梯度流）

$$
g_{\text{replay},i} = \frac{I_{\text{infer},i} \exp(s_{\text{train},i})}{\sum_{j=1}^{M} I_{\text{infer},j} \exp(s_{\text{train},i})}
$$

**效果**：
- 训练-推理 KL 散度降低约 50%
- 极端 token 比例减少 10 倍
- 防止训练崩溃

> [!warning] R3 的权衡
> R3 通过重放路由引入了**偏差**（bias）——训练的策略不再是原始的 $\pi_\theta$，而是一个修改后的策略 $\pi_\theta^{R3}$。
>
> 但这种偏差是值得的，因为它使一阶近似更可能成立，从而稳定训练。

---

## 实验验证

### MiniRL：最小化基线算法

论文设计了 MiniRL 算法，尽可能忠实于理论推导：

$$
\mathcal{J}_{\text{MiniRL}}(\theta) = \mathbb{E} \left[ \sum_{t=1}^{|y|} M_t \cdot \text{sg}\left[\frac{\pi_\theta(y_t)}{\mu_{\theta_{\text{old}}}(y_t)}\right] \widehat{A}(x, y) \log \pi_\theta(y_t) \right]
$$

其中：
- $M_t$：masking 函数（类似 PPO clip）
- $\text{sg}[\cdot]$：stop gradient（重要性权重不回传梯度）
- $\widehat{A}(x, y)$：advantage 估计

### On-policy 实验（验证理论）

**设置**：global batch size = mini-batch size（无 off-policy）

**对比三个变体**：
1. **MiniRL**：完整的训练-推理 IS 校正
2. **MiniRL + length norm**：添加长度归一化（破坏一阶近似）
3. **MiniRL w/o train-infer IS**：移除训练-推理 IS 校正（破坏一阶近似）

**结果**：
- MiniRL：稳定训练，最佳性能
- Length norm：稳定但性能次优（一阶近似被破坏）
- w/o IS：**快速崩溃**，熵急剧下降（一阶近似完全失效）

> [!intuition] 关键洞察
> 只有保持一阶近似有效性的设计才能稳定训练。任何破坏一阶近似的修改（如移除 IS 校正）都会导致崩溃。

### Off-policy 实验（验证实践方法）

**设置**：global batch size = 2 × mini-batch size

**关键发现**：
1. **Routing Replay 和 clipping 都是必需的**：
   - 移除任一都会导致训练崩溃
   - R3 比 R2 更有效（同时解决训练-推理差异和策略陈旧度）

2. **Off-policiness 影响方法选择**：
   - 小 off-policiness：R2 足够且更有效
   - 大 off-policiness：需要 R3

3. **R3 与 TIS 的关系**：
   - R3 已大幅降低差异，TIS 的额外收益有限
   - 在某些情况下，TIS + R3 甚至不如单独 R3

### 大规模实验

**规模**：
- 模型：Qwen3-30B-A3B（MoE）
- 任务：数学推理（AIME24, AIME25, AMC23, MATH500）
- 计算：数十万 GPU 小时（5-6 GPU 小时/梯度步）

**结果**：
- GRPO：60 步崩溃
- GRPO + TIS：105 步崩溃
- GRPO + R3：不崩溃，性能 71.83
- GSPO + R3：性能 69.00（最佳组合）

---

## 实践建议

> [!example] 稳定 RL 训练的 Recipe

基于论文的理论和实验，以下是实践建议：

### 1. 必须做的（核心）

- **训练-推理 IS 校正**：不要移除 $\frac{\pi_{\theta_{\text{old}}}}{\mu_{\theta_{\text{old}}}}$ 项
- **Clipping 或 masking**：控制策略陈旧度（如 PPO clip）
- **避免 length normalization**：会破坏一阶近似

### 2. MoE 模型特定

- **优先使用 R3**：从根本上解决路由不一致
- **小 off-policiness 可用 R2**：更简单，效果也好
- **R3 + TIS 收益有限**：R3 已经很有效

### 3. 超参数建议

- **Clipping 阈值**：$\epsilon_{\text{high}} = 0.27$, $\epsilon_{\text{low}} = 0.2$
- **TIS 截断阈值**：5（如果使用）
- **Mini-batch size**：控制 off-policiness

### 4. 监控指标

- **训练-推理 KL 散度**：应保持在较低水平
- **梯度范数**：剧烈波动是不稳定的信号
- **熵**：急剧下降表明可能崩溃
- **极端 token 比例**：$\tau > 2$ 的 token 应尽量少

---

## 局限性

> [!warning] 论文的边界

1. **理论假设**：
   - 一阶近似忽略了高阶项，在策略差异较大时可能不准确
   - 没有给出"多小的差异"才算"足够小"的定量标准

2. **实验范围**：
   - 主要在数学推理任务上验证
   - 其他任务（如对话、代码生成）的泛化性未知

3. **MoE 特定**：
   - R3 只适用于 MoE 模型
   - Dense 模型不需要 Routing Replay

4. **计算成本**：
   - 大规模实验需要数十万 GPU 小时
   - 小团队难以复现

5. **未解决的问题**：
   - Reward hacking
   - 长序列的 credit assignment
   - 其他不稳定性来源

---

## 相关概念

**理论基础**：
- [[Policy Gradient]] — RL 的基本框架
- [[Importance Sampling]] — Off-policy 学习的核心技术
- [[Training-Inference Mismatch]] — 训练-推理差异的详细分析

**相关方法**：
- [[PPO]] — 使用 clipping 控制策略陈旧度
- [[GRPO]] — Token 级 RL 算法
- [[GSPO]] — 序列级重要性采样
- [[Rollout Routing Replay]] — 针对 MoE 的解决方案
- [[IcePop]] — 通过 masking 解决训练-推理差异

**模型架构**：
- [[Mixture of Experts]] — 路由不一致的根源

---

## 延伸阅读

**原始论文与深入材料**：
- [[Clippings/Paper/251201374v3/251201374v3]] — 论文完整内容
- [[Policy Gradient, Sequence, and Token— Part II Learner-Sampler Mismatch]] — Learner-Sampler Mismatch 的理论分析

**相关工作**：
- [[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training]] — TIS 方法
- [[Clippings/Paper/251011370v1/251011370v1]] — R3 的原始论文

