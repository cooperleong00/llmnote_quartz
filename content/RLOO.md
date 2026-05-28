---
type: method
description: 用 leave-one-out 均值作为 baseline 的 REINFORCE 变体，无需 critic 即可实现低方差策略梯度估计，在 RLHF 中性能超越 PPO
aliases:
  - REINFORCE Leave-One-Out
  - RLOO 算法
prerequisites:
  - "[[REINFORCE]]"
  - "[[Reward Model]]"
  - "[[RLHF]]"
tags:
  - post-training
  - reinforcement-learning
  - rlhf
  - variance-reduction
created: 2026-02-26
updated: 2026-02-26
---

# RLOO (REINFORCE Leave-One-Out)

RLOO 是 [[REINFORCE]] 的一种多样本变体：对每个 prompt 采样 $k$ 个回答，用**其余 $k-1$ 个回答的平均 reward** 作为当前回答的 baseline。这个简单的 leave-one-out 技巧提供了无偏、低方差的梯度估计，完全不需要学习 value function（critic）。在 [[RLHF]] 场景中，RLOO 以更简单的实现超越了 [[PPO]]。

## 动机：PPO 在 RLHF 中是否过度设计？

[[PPO]] 是 RLHF 的标准算法，但它继承了通用 RL 的全套复杂机制——[[GAE]]、critic network、[[Importance Sampling|importance sampling]] ratio、clip。这些机制是为解决通用 RL 的难题设计的。但 RLHF 有三个特殊性质，使得这些机制变得不必要：

> [!intuition] RLHF 不是典型的 RL 问题
>
> 1. **预训练提供了极强的初始化**：LLM 在每个 token 位置的概率质量高度集中，策略梯度的方差天然很低。PPO 通过 GAE 做 bias-variance tradeoff 是多余的——实验表明 $\lambda=1$（无 bias、最大方差，即退化为 Vanilla PG）反而效果最好。
>
> 2. **策略变化缓慢，clip 几乎不触发**：RLHF 中相邻迭代的策略差异很小，PPO 的 clip 机制在 <5% 的 token 上触发。去掉 clip 和 IS ratio（退化为 Vanilla PG）效果相当甚至更好。
>
> 3. **Reward 只在 `<EOS>` 给出，环境是确定性的**：下一个 state = 当前 state + 新 token，没有随机转移。整个生成过程等价于一个 **bandit 问题**——把整段回答视为一个 action（如 REINFORCE 所做的）就够了，不需要建模 partial completion。

既然 PPO 的复杂性不必要，那能不能用更简单的方法？RLOO 就是答案：回归 REINFORCE 的本质，只用一个聪明的 baseline 来降低方差。

## 核心机制

### Leave-One-Out Baseline

标准 [[REINFORCE]] 的问题是高方差。常见的解决方案是减去一个 baseline $b$：

$$\nabla_\theta J(\theta) \approx [R(y, x) - b] \nabla_\theta \log \pi_\theta(y|x)$$

Baseline 的选择至关重要。一个简单方案是用 reward 的移动平均（moving average baseline），但它是全局的、跨 prompt 的，对特定 prompt 的 reward 分布不敏感。

RLOO 的核心洞察：**用同一个 prompt 的其他样本的平均 reward 作为 baseline**。

> [!definition] RLOO 梯度估计器
> 对每个 prompt $x$，从当前策略 $\pi_\theta$ 采样 $k$ 个 i.i.d. 回答 $y_{(1)}, \ldots, y_{(k)}$：
>
> $$\nabla_\theta J \approx \frac{1}{k} \sum_{i=1}^{k} \left[R(y_{(i)}, x) - \underbrace{\frac{1}{k-1} \sum_{j \neq i} R(y_{(j)}, x)}_{\text{leave-one-out baseline}}\right] \nabla_\theta \log \pi_\theta(y_{(i)}|x)$$
>
> 其中 $R(y, x)$ 是 [[Reward Model]] 给出的 reward（通常包含 [[KL Divergence|KL]] penalty）。

> [!intuition] 为什么 leave-one-out 有效？
> 想象你对同一个问题生成了 4 个回答，reward 分别是 [0.8, 0.3, 0.6, 0.5]。
>
> - 对第 1 个回答（0.8），baseline = (0.3+0.6+0.5)/3 = 0.47，advantage = +0.33 → **强正向更新**
> - 对第 2 个回答（0.3），baseline = (0.8+0.6+0.5)/3 = 0.63，advantage = -0.33 → **负向更新**
>
> 这个 baseline 是**per-prompt、per-step** 的——它精确反映了"对这个特定问题，当前策略的平均水平"，比全局移动平均有效得多。而且它是**无偏的**：因为其他 $k-1$ 个样本独立于当前样本，它们的均值是 $\mathbb{E}[R]$ 的无偏估计。

### 两重方差降低

RLOO 同时从两个维度降低方差：

1. **Baseline 效果**：leave-one-out 均值是一个高质量的、per-prompt 的 baseline，减去它后 advantage 的方差大幅降低
2. **多样本平均**：对 $k$ 个样本的梯度取平均，方差再降低 $k$ 倍（多样本 Monte Carlo 估计的标准效果）

## 与其他方法的对比

> [!comparison] RLOO vs PPO vs GRPO

| 方面 | [[PPO]] | [[GRPO]] | RLOO |
|------|---------|----------|------|
| **Advantage 来源** | [[GAE]]（需要 critic） | 组内 reward 归一化 | Leave-one-out 均值 |
| **额外模型** | Critic（与 policy 同规模） | 无 | 无 |
| **IS ratio + clip** | 有 | 有 | 无 |
| **KL 正则化** | Reward 中加 KL penalty | Loss 中加 KL term | Reward 中加 KL penalty |
| **内存占用** | 高（~2x policy） | 低 | 低 |
| **归一化** | 无 | 有（mean + std） | 无 |

RLOO 和 [[GRPO]] 的哲学非常接近——都是"去掉 critic，用同组样本的 reward 做 baseline"。关键区别在于：
- GRPO 保留了 PPO 风格的 IS ratio 和 clip 机制
- RLOO 是纯粹的 on-policy REINFORCE，完全不需要 IS ratio 和 clip
- GRPO 对 advantage 做了 z-score 归一化（减均值除标准差），RLOO 只减均值

> [!comparison] RLOO vs RAFT
> RAFT（Reward rAnked FineTuning）也采样 $k$ 个回答，但只用 reward 最高的那个做 SFT。RLOO 利用**全部** $k$ 个样本构建梯度估计。实验表明 RLOO $k=2$ 就能匹配或超越 RAFT $k=4$，样本效率显著更高。

> [!comparison] RLOO vs DPO
> [[DPO]] 是 offline 方法，直接在静态偏好数据上优化。RLOO 是 online RL 方法，持续从当前策略采样新数据。Online 采样使 RLOO 能探索策略改进后的新区域，避免 DPO 的分布偏移问题。实验中 RLOO 在 win-rate 上显著优于 DPO。

## 实验结果

> [!paper] Ahmadian et al., 2024
> *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*
> arXiv: 2402.14740

在 TL;DR 摘要和 Anthropic-HH 对话任务上：

- **RLOO $k=4$ 取得最高 win-rate**，在 TL;DR 上超越 PPO 10-32 个百分点
- **Vanilla PG 一致优于 PPO**——验证了"PPO 的复杂性在 RLHF 中不必要"的论点
- **RLOO 对噪声和 KL penalty 系数更鲁棒**，而 RAFT 对这些超参数敏感
- **RLOO $k=2$ 匹配 RAFT $k=4$**，样本利用率更高

## 局限性

> [!warning] 边界条件
> 1. **采样成本**：$k$ 个样本意味着 $k$ 倍的生成开销。当 $k$ 较大时，训练吞吐量下降明显。这是用计算换方差降低的 tradeoff
> 2. **Bandit 假设的适用范围**：RLOO 将整个生成视为单一 action，这在 reward 只在 `<EOS>` 给出时成立。如果需要 process reward（逐步反馈），RLOO 的 bandit 框架不直接适用
> 3. **Baseline 质量依赖 $k$**：$k$ 太小时 leave-one-out 均值的估计不稳定（$k=2$ 时 baseline 就是另一个样本的 reward）
> 4. **论文的实验规模有限**：主要在 1B-7B 模型上验证，更大规模下的表现尚待确认

## 面试要点

> [!interview] 面试视角
>
> **Q: RLOO 和 GRPO 有什么区别？**
> A: 核心思想相似——都用同组样本的 reward 做 baseline 来替代 critic。区别在于 GRPO 保留了 PPO 的 IS ratio 和 clip 机制，并对 advantage 做 z-score 归一化；RLOO 是纯 on-policy REINFORCE，不需要 IS ratio 和 clip。RLOO 更简单，GRPO 更接近 PPO 的框架。
>
> **Q: 为什么 PPO 在 RLHF 中是过度设计的？**
> A: 三个原因：(1) 预训练 LLM 的策略梯度方差天然低，GAE 的 bias-variance tradeoff 不必要；(2) 策略变化慢，clip 几乎不触发；(3) reward 只在 EOS 给出且环境确定性，问题退化为 bandit，不需要建模 partial completion。
>
> **Q: RLOO 的 leave-one-out baseline 为什么是无偏的？**
> A: 因为 $k$ 个样本是 i.i.d. 的。对于样本 $y_{(i)}$，其余 $k-1$ 个样本的均值是 $\mathbb{E}[R]$ 的无偏估计，且与 $y_{(i)}$ 独立。减去任何与当前样本独立的常数都不改变梯度的期望。

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/240214740v2/240214740v2|Ahmadian et al., 2024]] — 完整实验和分析
- Kool et al., 2019 — RLOO 估计器的原始提出（组合优化场景）

**后续发展**：
- [[GRPO]] — 类似思路但保留 PPO 框架的变体，被 DeepSeek 广泛使用
- [[CISPO]] — 另一种简化 PPO 的 REINFORCE 风格方法，clip IS weight 而非 token update
