---
description: 结合策略网络（Actor）和价值网络（Critic）的 RL 架构，是 PPO 的基础
type: concept
prerequisites:
  - "[[Policy Gradient|策略梯度]]"
  - "[[价值函数]]"
tags:
  - reinforcement-learning
created: 2025-01-25
updated: 2026-03-02
---

# Actor-Critic

Actor-Critic 是一类结合了**策略梯度**（Actor）和**价值函数**（Critic）的强化学习方法。它是 [[PPO]]、[[A2C]]、[[SAC]] 等现代 RL 算法的基础架构。

---

## 核心思想

> [!intuition] 直觉理解
> - **Actor**（演员）：学习策略 $\pi_\theta(a|s)$，决定"做什么"
> - **Critic**（评论家）：学习价值函数 $V_\phi(s)$ 或 $Q_\phi(s,a)$，评估"做得怎么样"
>
> Actor 根据 Critic 的反馈来改进策略，Critic 根据 Actor 的行为来更新价值估计。

**为什么需要 Critic？**
- 纯策略梯度（如 REINFORCE）方差太大
- Critic 提供更稳定的价值估计，降低方差

---

## 架构

```
        ┌─────────────────────────────────────┐
        │            Environment              │
        └─────────────┬───────────────────────┘
                      │ state s, reward r
                      ▼
        ┌─────────────────────────────────────┐
        │              Actor                  │
        │         π_θ(a|s) → action           │
        └─────────────┬───────────────────────┘
                      │ action a
                      ▼
        ┌─────────────────────────────────────┐
        │             Critic                  │
        │    V_φ(s) or Q_φ(s,a) → value       │
        └─────────────────────────────────────┘
```

---

## 数学形式

### Actor 更新

使用优势函数 $A(s,a)$ 作为策略梯度的权重：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)\right]
$$

其中优势函数：
$$
A(s, a) = Q(s, a) - V(s)
$$

### Critic 更新

最小化 TD 误差：

$$
L(\phi) = \mathbb{E}\left[(r + \gamma V_\phi(s') - V_\phi(s))^2\right]
$$

或者使用 n-step return / [[GAE]] 作为目标。

---

## 优势函数的估计

### 方法 1：TD Error

$$
A(s_t, a_t) \approx r_t + \gamma V(s_{t+1}) - V(s_t) = \delta_t
$$

- **优点**：低方差
- **缺点**：有偏（依赖 $V$ 的准确性）

### 方法 2：Monte Carlo

$$
A(s_t, a_t) \approx \sum_{k=0}^{T-t} \gamma^k r_{t+k} - V(s_t)
$$

- **优点**：无偏
- **缺点**：高方差

### 方法 3：GAE（推荐）

$$
A^{GAE}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

通过 $\lambda$ 参数平衡 bias 和 variance。详见 [[GAE]]。

---

## 训练流程

```python
for episode in episodes:
    s = env.reset()
    while not done:
        # Actor 选择动作
        a = actor.sample(s)
        s', r, done = env.step(a)

        # 计算 TD error
        delta = r + gamma * critic(s') - critic(s)

        # 更新 Critic
        critic_loss = delta ** 2
        critic.update(critic_loss)

        # 更新 Actor
        actor_loss = -log_prob(a|s) * delta
        actor.update(actor_loss)

        s = s'
```

---

## Actor 和 Critic 的协同关系

> [!intuition] 核心洞察：优势趋于 0 不是坏事
>
> 初看 Critic loss $L(\phi) = \mathbb{E}[(r + \gamma V(s') - V(s))^2]$，会产生困惑：
> - Critic 在让 TD error（即优势）趋于 0
> - 如果所有动作的优势都是 0，怎么区分好坏？
>
> **答案**：优势趋于 0 是**策略趋于最优的自然结果**，而非所有动作无法区分。

### 价值评估 ↔ 策略迭代的循环

Actor-Critic 本质上是两个交替进行的过程：

**1. 策略迭代（Actor 的视角）**

假设当前有策略 $\pi$ 和能准确评估它的 $V_\pi$：
- 计算优势 $A_\pi(s_t, a_t) = Q_\pi(s_t, a_t) - V_\pi(s_t)$
- 如果 $A_\pi(s_t, a_t) > 0$，提升 $\pi(a_t|s_t)$ 的概率
- 重复此过程，得到改进后的策略 $\pi'$

**2. 价值评估（Critic 的视角）**

策略从 $\pi$ 变为 $\pi'$ 后，旧的 $V_\pi$ 不再准确：
- 需要重新估计 $V_{\pi'}$ 来正确评估新策略
- Critic loss 让 $V_\phi(s_t)$ 拟合 $\sum_{a_t} \pi'(a_t|s_t) Q_{\pi'}(s_t, a_t)$
- 这样才能为下一轮策略迭代提供准确的价值估计

### 优势趋于 0 的含义

> [!math] 最优策略时的优势
>
> 回顾价值函数的关系：
> $$
> V_\pi(s_t) = \sum_{a_t \in \mathcal{A}} \pi(a_t|s_t) Q_\pi(s_t, a_t)
> $$
>
> 假设在状态 $s_t$ 下，存在客观最优动作 $a_t^*$。
>
> **当策略趋于最优时**：
> - $\pi(a_t^*|s_t) \to 1$（几乎总是选择最优动作）
> - $V_\pi(s_t) \to Q_\pi(s_t, a_t^*)$（状态价值接近最优动作价值）
> - $A_\pi(s_t, a_t^*) = Q_\pi(s_t, a_t^*) - V_\pi(s_t) \to 0$

**两种解读**：

| 阶段 | 优势的含义 | Actor 的行为 | Critic 的行为 |
|------|-----------|-------------|--------------|
| **策略未优化** | $A > 0$ 的动作明显更好 | 增加好动作的概率，区分动作好坏 | 估计当前策略的价值 |
| **策略趋于最优** | 优势趋于 0 | 已经几乎总是选最优动作，无需再调整 | 准确反映最优策略的价值 $V_{\pi^*}$ |

> [!warning] 常见误解
>
> **误解**：优势趋于 0 → 所有动作无法区分 → 训练失败
>
> **正解**：优势趋于 0 → 策略已经趋于最优 → 训练成功
>
> 在策略未优化时，不同动作的优势差异大；当策略优化后，最优动作的概率已经很高，优势自然变小。

### 单步更新的形式

在实际训练中，通常使用单步更新（而非等待整个 episode 结束）：

**Actor 目标**：
$$
\arg \max_{\pi_\theta} J(\pi_\theta) = \mathbb{E}_t \left[ A_\phi(s_t, a_t) \log \pi_\theta(a_t|s_t) \right]
$$

**Critic 目标**：
$$
\arg \min_{V_\phi} L(V_\phi) = \mathbb{E}_t \left[ (r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2 \right]
$$

这样即使 episode 很长（$T \to \infty$），也能持续更新。

---

## 在 RLHF 中的应用

[[RLHF]] 使用 Actor-Critic 架构：

| 组件          | RLHF 中的对应                              |
| ----------- | -------------------------------------- |
| Actor       | 策略模型 $\pi_\theta$（LLM）                 |
| Critic      | Value head $V_\phi$（通常共享 LLM backbone） |
| Reward      | Reward Model 的输出                       |
| Environment | 文本生成过程                                 |

**特殊之处**：
- Reward 只在序列末尾给出（sparse reward）
- 加入 KL 惩罚防止偏离参考策略
- 使用 [[GAE]] 估计优势

---

## 变体

### A2C (Advantage Actor-Critic)

- **同步更新**：多个 worker 并行采样，同步更新参数
- **优势**：比 A3C 更稳定（避免异步更新的梯度冲突）
- **缺点**：需要等待最慢的 worker

### A3C (Asynchronous Advantage Actor-Critic)

- **异步更新**：多个 worker 独立采样和更新，不需要同步
- **优势**：更快（不等待），探索更多样化
- **缺点**：梯度可能冲突，不如 A2C 稳定

### PPO (Proximal Policy Optimization)

- **核心改进**：加入 clip 约束，限制策略更新幅度
- **优势**：更稳定，样本效率高，是 [[RLHF]] 的标准选择
- **详见**：[[PPO]]

### SAC (Soft Actor-Critic)

- **核心改进**：加入熵正则化 $\alpha \mathcal{H}(\pi)$，鼓励探索
- **适用场景**：连续控制任务（机器人、自动驾驶）
- **特点**：off-policy，样本效率高

### DDPG (Deep Deterministic Policy Gradient)

- **核心特点**：确定性策略（输出具体动作值，而非概率分布）
- **适用场景**：连续动作空间
- **架构**：Actor 输出动作，Critic 估计 $Q(s,a)$

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: Actor-Critic 的核心思想是什么？**
> A: 结合策略梯度和价值函数。Actor 学习策略，Critic 学习价值函数来评估 Actor 的行为，降低策略梯度的方差。
>
> **Q2: Actor 和 Critic 分别学什么？**
> A: Actor 学习策略 $\pi(a|s)$，输出动作概率；Critic 学习价值函数 $V(s)$ 或 $Q(s,a)$，输出状态/动作的价值。
>
> **Q3: 为什么 Actor-Critic 比纯策略梯度好？**
> A: 纯策略梯度用整条轨迹的回报作为权重，方差大。Actor-Critic 用 Critic 估计的优势函数，方差更低。
>
> **Q4: RLHF 中 Actor-Critic 是怎么用的？**
> A: Actor 是 LLM 策略，Critic 是 value head（通常共享 backbone）。用 Reward Model 的输出作为 reward，加入 KL 惩罚。
>
> **Q5: Critic loss 让优势趋于 0，为什么不会导致无法区分动作好坏？**
> A: 优势趋于 0 是策略趋于最优的自然结果。当策略未优化时，不同动作优势差异大；当策略优化后，最优动作概率已经很高（接近 1），此时 $V(s) \approx Q(s, a^*)$，优势自然变小。这不是"无法区分"，而是"已经找到最优"。
>
> **Q6: Actor 和 Critic 是如何协同工作的？**
> A: 两者交替进行"价值评估 ↔ 策略迭代"循环。Actor 根据 Critic 的价值估计改进策略；策略改进后，Critic 重新估计新策略的价值。循环往复，直到收敛到最优策略和最优价值函数。

---

## 延伸阅读

**基础概念**：
- [[Policy Gradient|策略梯度]] — Actor 的理论基础
- [[GAE]] — 优势估计方法，平衡方差与偏差

**主要变体**：
- [[PPO]] — 最常用的 Actor-Critic 算法，RLHF 的标准选择
- [[A2C]] — 同步 Actor-Critic
- [[A3C]] — 异步 Actor-Critic

**应用场景**：
- [[RLHF]] — Actor-Critic 在 LLM 对齐中的应用
