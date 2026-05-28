---
description: 直接优化策略参数的 RL 方法，是 PPO/TRPO 等现代算法的理论基础
type: concept
aliases:
  - 策略梯度
prerequisites:
  - "[[Probability Theory]]"
  - "[[Gradient Descent]]"
tags:
  - reinforcement-learning
  - optimization
created: 2025-01-25
updated: 2026-03-31T22:10
---

# Policy Gradient

策略梯度（Policy Gradient）是一类直接优化策略参数的强化学习方法。与 value-based 方法不同，策略梯度直接学习从状态到动作的映射，是 [[PPO]]、[[TRPO]] 等现代 RL 算法的理论基础。

---

## 核心思想

> [!intuition] 直觉理解
> **Value-based**：先学习"每个状态/动作有多好"，再据此选择动作
> **Policy-based**：直接学习"在每个状态应该怎么做"
>
> 策略梯度的核心问题：**如何计算策略参数的梯度？**

策略 $\pi_\theta(a|s)$ 是一个参数化的概率分布，表示在状态 $s$ 下选择动作 $a$ 的概率。

---

## 优化目标

最大化期望累积奖励：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

其中 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ 是一条轨迹。

---

## 策略梯度定理

> [!definition] 策略梯度定理 (Policy Gradient Theorem)
> $$
> \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]
> $$

### 推导

轨迹概率：
$$
P(\tau|\theta) = \rho_0(s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_\theta(a_t|s_t)
$$

对 $J(\theta)$ 求梯度：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{\tau}[R(\tau)] = \nabla_\theta \sum_\tau P(\tau|\theta) R(\tau) \\
&= \sum_\tau R(\tau) \nabla_\theta P(\tau|\theta) \\
&= \sum_\tau R(\tau) P(\tau|\theta) \frac{\nabla_\theta P(\tau|\theta)}{P(\tau|\theta)} \\
&= \sum_\tau R(\tau) P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) \\
&= \mathbb{E}_{\tau}\left[R(\tau) \nabla_\theta \log P(\tau|\theta)\right]
\end{aligned}
$$

由于 $\log P(\tau|\theta) = \log \rho_0(s_0) + \sum_t \log P(s_{t+1}|s_t,a_t) + \sum_t \log \pi_\theta(a_t|s_t)$

只有最后一项与 $\theta$ 有关，因此：
$$
\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

> [!intuition] 直觉
> $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 指向"增加动作 $a_t$ 概率"的方向。
> 乘以 $R(\tau)$ 后：好轨迹的动作概率增加，坏轨迹的动作概率减少。

---

## [[REINFORCE]] 算法

最简单的策略梯度算法：

```
for each episode:
    采样轨迹 τ = (s_0, a_0, r_0, ..., s_T)
    计算回报 R(τ) = Σ γ^t r_t
    更新: θ ← θ + α Σ_t ∇log π_θ(a_t|s_t) · R(τ)
```

### 问题：高方差

$R(\tau)$ 是整条轨迹的回报，方差很大：
- 同一个好动作，在不同轨迹中可能得到很不同的 $R(\tau)$
- 需要大量样本才能得到稳定的梯度估计

---

## 降低方差的方法

### 1. 使用 Baseline

引入与动作无关的 baseline $b(s)$：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (R(\tau) - b(s_t))\right]
$$

> [!math] 为什么 baseline 不引入偏差？
> $$
> \mathbb{E}_{a \sim \pi}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0
> $$

常用的 baseline：状态价值函数 $V(s)$

### 2. 使用 Advantage

用优势函数 $A(s,a) = Q(s,a) - V(s)$ 替代 $R(\tau)$：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t)\right]
$$

这就是 [[Actor-Critic]] 方法的基础。

### 3. 使用因果性

$t$ 时刻的动作只影响 $t$ 之后的奖励：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}\right]
$$

---

## 与 Value-based 方法的对比

| 方面 | Policy Gradient | Value-based (如 DQN) |
|------|-----------------|---------------------|
| 输出 | 动作概率分布 | 动作价值 |
| 动作空间 | 连续/离散均可 | 通常离散 |
| 随机策略 | 天然支持 | 需要额外处理 |
| 收敛性 | 局部最优 | 可能不收敛 |
| 样本效率 | 较低 | 较高（可用经验回放） |

---

## Token 级 vs 序列级策略梯度

在 LLM 的 RL 训练中，一个关键设计选择是在 **token 级**还是**序列级**应用策略梯度。

> [!intuition] 核心区别
> - **Token 级**：每个 token 是一个 action，prefix 是 state → 类似 MDP
> - **序列级**：整个序列是一个 action，prompt 是 state → 类似 Bandit

### REINFORCE：两者等价

对于基础的 REINFORCE 算法，token 级和序列级在数学上**完全等价**：

$$
\nabla J(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ R(y) \cdot \nabla \log \pi_\theta(y) \right] = \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_{t=1}^T R(y) \cdot \nabla \log \pi_\theta(y_t|y_{<t}) \right]
$$

因为 $\log \pi_\theta(y) = \sum_t \log \pi_\theta(y_t|y_{<t})$，序列级梯度自然分解为 token 级梯度之和。

### Trust Region 方法：两者不同

对于 [[TRPO]]、[[PPO]] 等 trust region 方法，两种粒度的 surrogate objective 形式不同。这些方法使用 [[Importance Sampling]] 来复用旧策略采集的数据：

**Token 级目标**（MDP 视角）：
$$
L(\theta) = \mathbb{E}_{y_{<t}, y_t} \left[ \frac{\pi_\theta(y_t|y_{<t})}{\pi_{\theta_{old}}(y_t|y_{<t})} \cdot A(y_{<t}, y_t) \right]
$$

**序列级目标**（Bandit 视角）：
$$
L(\theta) = \mathbb{E}_{y} \left[ \frac{\pi_\theta(y)}{\pi_{\theta_{old}}(y)} \cdot A(y) \right]
$$

> [!warning] 关键区别
> - **序列级**：状态分布 $d$ 与策略无关（stationary），$L(\theta)$ 与 $J(\theta)$ 在梯度意义上等价
> - **Token 级**：状态分布 $d^{\pi_{\theta_{old}}}$ 依赖旧策略，$L(\theta)$ 是 $J(\theta) - J(\theta_{old})$ 的**下界**，而非等价
>
> 这种差异是**有意设计**的——trust region 方法通过保守更新换取训练稳定性。

### 实践选择

两种粒度各有优劣，没有绝对答案：

| 粒度 | 适用场景 | 代表方法 |
|------|----------|----------|
| Token 级 | 需要细粒度 [[Credit Assignment]] | [[GRPO]]、[[PPO]] |
| 序列级 | 只有序列级 reward | [[GSPO]]、REINFORCE |

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: 策略梯度的核心思想是什么？**
> A: 直接优化策略参数，通过增加高回报轨迹中动作的概率、减少低回报轨迹中动作的概率来改进策略。
>
> **Q2: 策略梯度定理是什么？**
> A: $\nabla J = \mathbb{E}[\sum_t \nabla \log \pi(a_t|s_t) \cdot R(\tau)]$。梯度方向由 $\nabla \log \pi$ 决定，大小由回报 $R$ 加权。
>
> **Q3: REINFORCE 的主要问题是什么？如何解决？**
> A: 高方差。解决方法：引入 baseline（如 $V(s)$）、使用 advantage、利用因果性。
>
> **Q4: 为什么 baseline 不引入偏差？**
> A: 因为 $\mathbb{E}[\nabla \log \pi \cdot b(s)] = 0$，baseline 与动作无关，期望为零。

---

## 相关概念

- [[GAE]] — 优势估计方法，用于降低策略梯度的方差

---

## 参考资料

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 13
- [[Policy Gradient, Sequence, and Token— Part I Basic Concepts|Policy Gradient, Sequence, and Token— Part I]] — Token 级 vs 序列级策略梯度的详细分析
