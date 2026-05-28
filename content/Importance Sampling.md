---
description: 用一个分布的样本估计另一个分布的期望，是 off-policy RL 和 PPO 的理论基础
type: concept
aliases:
  - 重要性采样
  - IS
prerequisites:
  - "[[Probability Theory]]"
  - "[[Expected Value]]"
tags:
  - reinforcement-learning
  - statistics
  - sampling
created: 2026-01-28
updated: 2026-01-31T22:42
---

# Importance Sampling

重要性采样（Importance Sampling）是一种用**一个分布的样本**来估计**另一个分布的期望**的技术。在强化学习中，它使得 off-policy 学习成为可能，是 [[PPO]]、[[GRPO]] 等算法的理论基础。

---

## 动机

> [!intuition] 为什么需要重要性采样？

**问题场景**：我们想计算 $\mathbb{E}_{x \sim p}[f(x)]$，但：
1. **无法从 $p$ 采样**：目标分布 $p$ 难以采样，但另一个分布 $q$ 容易采样
2. **已有旧样本**：在 RL 中，我们有旧策略 $\pi_{old}$ 生成的数据，但想用它来更新新策略 $\pi$

**核心思想**：通过**重新加权**，用 $q$ 的样本来估计 $p$ 的期望。

---

## 数学定义

> [!definition] 重要性采样公式

$$
\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]
$$

其中 $w(x) = \frac{p(x)}{q(x)}$ 称为**重要性权重**（importance weight）。

> [!math] 推导

$$
\begin{aligned}
\mathbb{E}_{x \sim p}[f(x)] &= \int p(x) f(x) dx \\
&= \int q(x) \frac{p(x)}{q(x)} f(x) dx \\
&= \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]
\end{aligned}
$$

**前提条件**：$q(x) > 0$ 对所有 $p(x) > 0$ 的 $x$ 成立（support 覆盖）。

---

## 蒙特卡洛估计

在实践中，用有限样本估计期望：

$$
\mathbb{E}_{x \sim p}[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} \frac{p(x_i)}{q(x_i)} f(x_i), \quad x_i \sim q
$$

> [!warning] 方差问题
> 重要性采样是**无偏**的，但**方差可能很大**：
> - 当 $p(x) \gg q(x)$ 时，权重 $w(x)$ 很大
> - 少数样本主导估计，导致高方差
> - 需要足够多的样本才能得到稳定估计

---

## 在强化学习中的应用

### On-Policy vs Off-Policy

> [!comparison] 两种学习范式

| 方面 | On-Policy | Off-Policy |
|------|-----------|------------|
| **数据来源** | 当前策略 $\pi_\theta$ | 旧策略 $\pi_{old}$ 或其他策略 |
| **样本效率** | 低（数据用完即弃） | 高（数据可复用） |
| **是否需要 IS** | 不需要 | 需要 |
| **代表算法** | REINFORCE, A2C | PPO, Off-policy AC |

### 策略梯度中的重要性采样

[[Policy Gradient|策略梯度]]的目标是最大化期望回报：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$

如果我们有旧策略 $\pi_{old}$ 的样本，可以用重要性采样：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{old}}\left[\frac{\pi_\theta(\tau)}{\pi_{old}(\tau)} R(\tau)\right]
$$

对于轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots)$，重要性权重为：

$$
\frac{\pi_\theta(\tau)}{\pi_{old}(\tau)} = \prod_{t=0}^{T} \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

> [!warning] 轨迹级 IS 的问题
> 权重是各时间步的**乘积**，随序列长度指数增长，方差极大。这是 off-policy RL 的核心挑战。

### PPO 中的重要性采样

[[PPO]] 使用 token 级（或 action 级）的重要性比率：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

PPO 的目标函数：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

> [!intuition] PPO 的设计思想
> - 用 $r_t(\theta)$ 做重要性校正，允许复用旧数据
> - 用 clip 限制 $r_t$ 的范围，防止权重过大导致的不稳定
> - 这是一种**截断重要性采样**（Truncated Importance Sampling）

---

## 截断重要性采样

> [!definition] Truncated Importance Sampling

为了控制方差，将重要性权重截断到一定范围：

$$
w_{clip}(x) = \text{clip}\left(\frac{p(x)}{q(x)}, w_{min}, w_{max}\right)
$$

**权衡**：
- 截断引入**偏差**（bias）
- 但显著降低**方差**（variance）
- 在实践中，这种 bias-variance 权衡通常是值得的

---

## Token 级 vs 序列级重要性采样

> [!comparison] GRPO vs GSPO 的核心区别

[[GRPO]] 和 [[GSPO]] 的主要区别在于重要性采样的粒度：

| 方面 | GRPO (Token 级) | GSPO (序列级) |
|------|-----------------|---------------|
| **重要性权重** | $w_t = \frac{\pi_\theta(y_t\|x, y_{<t})}{\pi_{old}(y_t\|x, y_{<t})}$ | $s = \left(\frac{\pi_\theta(y\|x)}{\pi_{old}(y\|x)}\right)^{1/\|y\|}$ |
| **理论基础** | 每个 token 独立校正 | 序列整体校正 |
| **问题** | 单样本无法有效校正 | 与序列级 reward 对齐 |

> [!warning] GRPO 的理论缺陷
> 重要性采样要求从行为分布采样**多个样本**来校正分布差异。GRPO 在每个 token 位置只有**单个样本**，无法执行有效的分布校正，引入高方差噪声。
>
> [[GSPO]] 通过将重要性采样提升到序列级解决了这一问题。

---

## Learner-Sampler Mismatch

> [!definition] Learner-Sampler Mismatch
> 在现代 RL 训练中，**采样器**（sampler，用于生成 rollout）和**学习器**（learner，用于计算梯度）可能是不同的系统。即使使用相同的模型参数，两者计算的概率也可能不同。

这种 mismatch 与传统 off-policy RL 不同：
- **传统 off-policy**：参数差异导致分布不同
- **Learner-Sampler Mismatch**：**系统差异**导致分布不同（数值精度、并行策略、算子实现等）

### 重要性采样能否修复？

| 粒度 | 能否通过 IS 修复 | 原因 |
|------|------------------|------|
| **序列级** | ✅ 可以 | 状态分布与策略无关，IS 可以完全校正 |
| **Token 级** | ❌ 不完全 | 状态访问频率 $d^{\pi}$ 和 advantage $A^{\pi}$ 都依赖策略，IS 无法校正 |

> [!intuition] 为什么 Token 级无法完全修复？
> Token 级目标中有两个额外的差异：
> 1. **状态访问频率**：$d^{\pi_{learner}}$ vs $d^{\pi_{sampler}}$
> 2. **Advantage 估计**：$A^{\pi_{learner}}$ vs $A^{\pi_{sampler}}$
>
> 这些差异无法通过简单的重要性比率校正。

### 实践中的处理

尽管理论上不完美，实践中有几种有效方法：

1. **Naive Recompute**：忽略 mismatch，用 learner 重新计算概率（当 mismatch 小时有效）
2. **TIS（Truncated IS）**：用截断的重要性比率校正，保持 trust region 在 learner 上
3. **Masking**：直接丢弃 mismatch 过大的样本（[[IcePop]]）

> [!note] 理论支持
> 虽然 IS 不能完全消除 mismatch，但通过三角不等式可以证明，adapted surrogate 仍然是策略改进的下界，因此仍是合理的学习目标。

详见 [[Training-Inference Mismatch]]。

---

## 局限性

> [!warning] 重要性采样的边界条件

1. **分布差异过大**：当 $p$ 和 $q$ 差异很大时，需要大量样本才能得到稳定估计
2. **高维空间**：在高维空间中，权重的方差随维度指数增长
3. **单样本失效**：重要性采样本质上是一种**期望校正**，需要多个样本才能生效
4. **Support 覆盖**：$q$ 必须覆盖 $p$ 的 support，否则估计有偏

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: 什么是重要性采样？**
> A: 用一个分布 $q$ 的样本来估计另一个分布 $p$ 的期望。通过重要性权重 $w(x) = p(x)/q(x)$ 重新加权样本。
>
> **Q2: 重要性采样在 RL 中有什么用？**
> A: 使 off-policy 学习成为可能。可以用旧策略生成的数据来更新新策略，提高样本效率。PPO 的概率比 $r_t(\theta)$ 就是重要性权重。
>
> **Q3: 重要性采样的主要问题是什么？**
> A: 方差大。当两个分布差异大时，少数样本的权重很大，主导估计。解决方法包括截断（clipping）、自归一化等。
>
> **Q4: PPO 的 clip 机制和重要性采样有什么关系？**
> A: PPO 的 $r_t(\theta)$ 是重要性权重，clip 是一种截断重要性采样，通过限制权重范围来控制方差，代价是引入一定偏差。
>
> **Q5: GRPO 的 token 级重要性采样有什么问题？**
> A: 重要性采样需要多个样本来校正分布差异，但 GRPO 在每个 token 位置只有单个样本，无法有效校正，引入高方差噪声。GSPO 通过序列级重要性采样解决了这个问题。

---

## 相关概念

- [[Policy Gradient]] — 重要性采样使 off-policy 策略梯度成为可能
- [[PPO]] — 使用截断重要性采样的代表算法
- [[GRPO]] — Token 级重要性采样
- [[GSPO]] — 序列级重要性采样，修正了 GRPO 的理论缺陷
- [[On-Policy vs Off-Policy]] — 两种学习范式的对比
- [[Variance Reduction]] — 降低方差的技术
- [[Training-Inference Mismatch]] — Learner-Sampler Mismatch 的详细分析

---

## 参考资料

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 5.5
- Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- [[GSPO]] 论文对 token 级 vs 序列级重要性采样的分析
- [[Policy Gradient, Sequence, and Token— Part II Learner-Sampler Mismatch|Policy Gradient Part II]] — Learner-Sampler Mismatch 的深入分析
