---
description: 用当前估计更新当前估计的 RL 学习方法，是 Actor-Critic 和 GAE 的理论基础
type: concept
aliases:
  - 时序差分学习
  - Temporal Difference Learning
prerequisites:
  - "[[Value Function]]"
  - "[[Markov Decision Process]]"
tags:
  - reinforcement-learning
  - value-estimation
created: 2026-01-29
updated: 2026-02-01T01:14
---

# TD Learning

时序差分学习（Temporal Difference Learning, TD Learning）是强化学习中最核心的思想之一。它结合了 Monte Carlo 方法的采样思想和动态规划的 bootstrapping 思想，能够在不等待完整轨迹的情况下进行学习。TD Learning 是 [[Actor-Critic]] 和 [[GAE]] 的理论基础。

---

## 核心思想

> [!intuition] 直觉理解
> **Monte Carlo**：等到游戏结束，用实际获得的总回报来更新估计
> **Dynamic Programming**：用已知的环境模型计算期望
> **TD Learning**：不等游戏结束，用**当前估计**来更新**当前估计**（bootstrapping）
>
> 核心洞察：我们可以用 $r + \gamma V(s')$ 作为 [[Value Function|$V(s)$]] 的目标，即使 $V(s')$ 本身也是估计值。

---

## TD(0) 更新规则

> [!definition] TD(0) 更新
> $$
> V(s_t) \leftarrow V(s_t) + \alpha \left[ r_t + \gamma V(s_{t+1}) - V(s_t) \right]
> $$
>
> 其中：
> - $\alpha$：学习率
> - $\gamma$：折扣因子
> - $r_t + \gamma V(s_{t+1})$：**TD target**（目标值）
> - $r_t + \gamma V(s_{t+1}) - V(s_t)$：**TD error**（时序差分误差）

---

## TD Error

> [!definition] TD Error（时序差分误差）
> $$
> \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
> $$

TD error 是 TD Learning 的核心概念，表示**当前估计与更好估计之间的差距**。

> [!intuition] TD Error 的含义
> - $\delta_t > 0$：实际情况比预期好，应该提高 $V(s_t)$
> - $\delta_t < 0$：实际情况比预期差，应该降低 $V(s_t)$
> - $\delta_t = 0$：估计准确，无需更新

TD error 在现代 RL 算法（如 [[PPO]]）中有广泛应用：
- **Critic 训练**：最小化 TD error 的平方
- **Actor 更新**：TD error 可作为优势函数的估计（结合 [[Policy Gradient]]）
- **[[GAE]]**：TD error 的指数加权和

---

## TD vs Monte Carlo

> [!comparison] TD 与 Monte Carlo 对比

| 方面 | TD Learning | Monte Carlo |
|------|-------------|-------------|
| **更新时机** | 每步更新 | 等待完整轨迹 |
| **目标值** | $r + \gamma V(s')$（估计） | $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$（实际） |
| **Bias** | 有偏（依赖 $V$ 的准确性） | 无偏 |
| **Variance** | 低（只用一步奖励） | 高（累积多步随机性） |
| **需要完整轨迹** | 否 | 是 |
| **适用场景** | 连续任务、在线学习 | 回合制任务 |

> [!math] 为什么 TD 有偏？
> TD target $r + \gamma V(s')$ 使用了估计值 $V(s')$，而非真实值 $V^*(s')$。
> 如果 $V(s') \neq V^*(s')$，则 TD target 是有偏的。
>
> 但随着学习进行，$V \to V^*$，偏差会逐渐减小。

> [!math] 为什么 MC 方差高？
> Monte Carlo 使用完整回报 $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots$
> 每一步的随机性都会累积，导致 $G_t$ 的方差随轨迹长度增加。

---

## TD(lambda) 和 Eligibility Traces

TD(0) 只看一步，Monte Carlo 看完整轨迹。**TD(lambda)** 提供了一个连续的谱，通过 $\lambda$ 参数在两者之间权衡。

### n-step TD

> [!definition] n-step TD Return
> $$
> G_t^{(n)} = r_t + \gamma r_{t+1} + \ldots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
> $$

| n 值 | 方法 | 特点 |
|------|------|------|
| n=1 | TD(0) | 高 bias，低 variance |
| n=2,3,... | n-step TD | 中间状态 |
| n=infinity | Monte Carlo | 低 bias，高 variance |

### TD(lambda)

TD(lambda) 不是选择某个固定的 n，而是对所有 n-step returns 做**指数加权平均**：

> [!definition] lambda-return
> $$
> G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
> $$
>
> 其中 $\lambda \in [0, 1]$：
> - $\lambda = 0$：等价于 TD(0)
> - $\lambda = 1$：等价于 Monte Carlo

### Eligibility Traces

直接计算 lambda-return 需要等待未来所有奖励，这违背了 TD 的在线学习优势。**Eligibility traces** 提供了一种等价但可在线计算的方法。

> [!definition] Eligibility Trace
> $$
> e_t(s) = \gamma \lambda e_{t-1}(s) + \mathbf{1}[s_t = s]
> $$
>
> 更新规则：
> $$
> V(s) \leftarrow V(s) + \alpha \delta_t e_t(s), \quad \forall s
> $$

> [!intuition] Eligibility Trace 的直觉
> Eligibility trace 记录了"哪些状态对当前 TD error 负有责任"。
> - 刚访问过的状态：责任大（$e$ 值高）
> - 很久前访问的状态：责任小（$e$ 值因 $\gamma\lambda$ 衰减）

---

## 在 Actor-Critic 中的应用

[[Actor-Critic]] 架构中，Critic 的训练本质上就是 TD Learning：

```
# Critic 更新（最小化 TD error）
delta = r + gamma * V(s') - V(s)  # TD error
loss_critic = delta^2
critic.update(loss_critic)

# Actor 更新（用 TD error 作为优势估计）
loss_actor = -log(pi(a|s)) * delta
actor.update(loss_actor)
```

> [!intuition] 为什么用 TD error 作为优势估计？
> 优势函数 $A(s,a) = Q(s,a) - V(s)$
>
> 而 $Q(s,a) \approx r + \gamma V(s')$（一步展开）
>
> 因此 $A(s,a) \approx r + \gamma V(s') - V(s) = \delta$

---

## 在 GAE 中的应用

[[GAE]]（Generalized Advantage Estimation）是 TD(lambda) 思想在优势函数估计上的应用：

$$
A^{GAE}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD error。

| lambda 值 | GAE 行为 | Bias | Variance |
|-----------|----------|------|----------|
| lambda=0 | $A_t = \delta_t$（TD(0)） | 高 | 低 |
| lambda=1 | Monte Carlo advantage | 低 | 高 |
| lambda=0.95 | 典型值，平衡 bias-variance | 中 | 中 |

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: TD Learning 的核心思想是什么？**
> A: Bootstrapping——用当前估计更新当前估计。TD 用 $r + \gamma V(s')$ 作为目标，而不是等待完整轨迹的实际回报。
>
> **Q2: TD error 是什么？有什么用？**
> A: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，表示当前估计与更好估计的差距。用于 Critic 训练（最小化 $\delta^2$）和 Actor 更新（作为优势估计）。
>
> **Q3: TD 和 Monte Carlo 的主要区别？**
> A: TD 每步更新、有偏但低方差；MC 等完整轨迹、无偏但高方差。TD 适合连续任务和在线学习，MC 适合回合制任务。
>
> **Q4: TD(lambda) 中 lambda 的作用？**
> A: 控制 bias-variance 权衡。lambda=0 是 TD(0)，lambda=1 是 MC。通过 eligibility traces 可以在线计算。
>
> **Q5: GAE 和 TD(lambda) 的关系？**
> A: GAE 是 TD(lambda) 在优势函数估计上的应用。TD(lambda) 估计 $V(s)$，GAE 估计 $A(s,a)$。

---

## 参考资料

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 6 (TD Learning), Chapter 12 (Eligibility Traces)
- Sutton, R. S. (1988). "Learning to Predict by the Methods of Temporal Differences"
