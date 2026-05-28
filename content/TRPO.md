---
description: 通过 KL 散度约束限制策略更新幅度的 RL 算法，是 PPO 的前身
type: method
prerequisites:
  - "[[Policy Gradient|策略梯度]]"
  - "[[KL Divergence]]"
tags:
  - reinforcement-learning
  - optimization
created: 2025-01-25
updated: 2026-02-01T01:15
---

# TRPO (Trust Region Policy Optimization)

信任域策略优化（TRPO, Trust Region Policy Optimization）是一种带约束的[[Policy Gradient|策略梯度]]算法，通过限制策略更新的幅度来保证单调改进。它是 [[PPO]] 的前身。

> [!paper] 论文出处
> Schulman et al., "Trust Region Policy Optimization", ICML 2015

---

## 核心思想

> [!intuition] 直觉理解
> 策略梯度的问题：步长难以选择。太大可能导致策略崩溃，太小学习太慢。
>
> TRPO 的解决方案：**在"信任域"内优化**——每次更新保证新策略不会偏离旧策略太远。

---

## 数学形式

### 优化目标

最大化代理目标（surrogate objective）：

$$
L(\theta) = \mathbb{E}_{s,a \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{old}}(s, a)\right]
$$

### KL 约束

同时满足 [[KL Divergence|KL 散度]]约束：

$$
\mathbb{E}_s\left[D_{KL}(\pi_{\theta_{old}}(\cdot|s) \| \pi_\theta(\cdot|s))\right] \leq \delta
$$

### 完整优化问题

> [!definition] TRPO 优化问题
> $$
> \begin{aligned}
> &\max_\theta \quad \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s, a)\right] \\
> &\text{s.t.} \quad \mathbb{E}\left[D_{KL}(\pi_{\theta_{old}} \| \pi_\theta)\right] \leq \delta
> \end{aligned}
> $$

---

## 理论保证

### 单调改进定理

TRPO 的理论基础是**策略改进下界**：

$$
J(\pi_{new}) \geq L_{\pi_{old}}(\pi_{new}) - C \cdot D_{KL}^{max}(\pi_{old}, \pi_{new})
$$

其中：
- $J(\pi_{new})$ 是新策略的真实期望回报
- $L_{\pi_{old}}(\pi_{new})$ 是**代理目标（surrogate objective）**，用旧策略采样的数据来评估新策略：
  $$L_{\pi_{old}}(\pi_{new}) = \mathbb{E}_{s,a \sim \pi_{old}}\left[\frac{\pi_{new}(a|s)}{\pi_{old}(a|s)} A^{\pi_{old}}(s, a)\right]$$
  下标 $\pi_{old}$ 表示数据来源（旧策略采样），参数 $\pi_{new}$ 表示被评估的策略
- $C = \frac{4\epsilon\gamma}{(1-\gamma)^2}$ 是与 discount factor $\gamma$ 相关的常数（$\epsilon$ 是优势函数的上界）
- $D_{KL}^{max}$ 是状态空间上的最大 KL 散度

> [!intuition] 含义
> 只要 KL 散度足够小，新策略的性能就有保证。这就是"信任域"的来源。

---

## 求解方法

### 1. 二阶近似

将 KL 约束用 Fisher 信息矩阵近似：

$$
D_{KL}(\pi_{\theta_{old}} \| \pi_\theta) \approx \frac{1}{2}(\theta - \theta_{old})^T F (\theta - \theta_{old})
$$

其中 $F$ 是 Fisher 信息矩阵。

### 2. 共轭梯度法

使用共轭梯度法求解约束优化问题，避免直接计算 $F^{-1}$。

### 3. 线搜索

在更新方向上进行线搜索，确保满足 KL 约束和改进条件。

---

## 与 PPO 的对比

| 方面 | TRPO | PPO |
|------|------|-----|
| **约束方式** | 硬约束（KL ≤ δ） | 软约束（clip 或 KL penalty） |
| **求解方法** | 二阶优化（共轭梯度） | 一阶优化（SGD） |
| **实现复杂度** | 高 | 低 |
| **计算开销** | 大（需要 Fisher 矩阵） | 小 |
| **效果** | 理论保证更强 | 实践中效果相当 |

> [!intuition] PPO 是 TRPO 的简化
> PPO 用 clip 机制或 KL penalty 近似 TRPO 的硬约束，换取更简单的实现和更低的计算开销。

---

## 为什么 PPO 更常用？

1. **实现简单**：PPO 只需要一阶梯度，TRPO 需要二阶优化
2. **计算高效**：PPO 不需要计算 Fisher 矩阵
3. **效果相当**：在大多数任务上，PPO 和 TRPO 效果差不多
4. **更易调参**：PPO 的超参数（clip ratio）比 TRPO（δ）更直观

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: TRPO 的核心思想是什么？**
> A: 在信任域内优化策略。通过 KL 散度约束限制每次更新的幅度，保证策略单调改进。
>
> **Q2: TRPO 的优化目标是什么？**
> A: 最大化 importance-weighted advantage，同时满足 KL 约束：$\max E[r(\theta) A]$ s.t. $E[D_{KL}] \leq \delta$。
>
> **Q3: TRPO 和 PPO 的主要区别？**
> A: TRPO 用硬约束 + 二阶优化，PPO 用软约束（clip/penalty）+ 一阶优化。PPO 更简单高效，效果相当。
>
> **Q4: 为什么 PPO 比 TRPO 更常用？**
> A: 实现简单、计算高效、效果相当、更易调参。

---

## 延伸阅读

**后续发展**：
- [[PPO]] — TRPO 的简化版本，用 clip 机制替代硬约束
