---
type: method
description: 最基础的 policy gradient 算法，用 Monte Carlo 采样估计策略梯度，是理解 PPO/GRPO 等现代 RL 算法的起点
aliases:
  - REINFORCE 算法
  - Williams REINFORCE
  - Monte Carlo Policy Gradient
prerequisites:
  - "[[Policy Gradient]]"
  - "[[Value Function]]"
  - "[[Markov Decision Process]]"
tags:
  - reinforcement-learning
  - optimization
  - policy-gradient
created: 2026-02-24
updated: 2026-02-26T16:49
---

# REINFORCE

REINFORCE（Williams, 1992）是 [[Policy Gradient]] 最直接的实现：采样完整轨迹，用实际回报作为梯度信号更新策略。它的名字是 "REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility" 的缩写，但更重要的是理解它做了什么——把"好动作的概率调高，坏动作的概率调低"这个直觉变成了可计算的算法。

REINFORCE 本身因为高方差问题很少直接使用，但它是理解 [[Actor-Critic]]、[[PPO]]、[[GRPO]] 等现代算法的必经之路。这些算法本质上都在解决 REINFORCE 留下的问题。

---

## 动机

> [!intuition] 为什么需要 REINFORCE？
> 假设你有一个策略 $\pi_\theta$，你想让它在环境中表现更好。[[Policy Gradient|策略梯度定理]]告诉你梯度的数学形式：
> $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$
> 但这个期望怎么算？你不可能遍历所有可能的轨迹。
>
> REINFORCE 的回答很朴素：**采样**。跑几条轨迹，用样本均值近似期望。这就是 Monte Carlo 估计。

---

## 算法推导

### 从策略梯度定理到更新规则

[[Policy Gradient|策略梯度定理]]给出了目标函数 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ 的梯度。REINFORCE 做了两步简化：

**第一步：Monte Carlo 估计期望**

用 $N$ 条采样轨迹的均值替代期望：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \cdot R(\tau^{(i)})$$

**第二步：利用因果性（Causality）**

$t$ 时刻的动作不可能影响 $t$ 之前的奖励。所以 $R(\tau)$ 可以替换为从 $t$ 开始的回报 $G_t$：

$$G_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$$

这给出 REINFORCE 的核心更新规则：

> [!definition] REINFORCE 更新规则
> $$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$
>
> 参数更新：$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$

> [!intuition] 每一项的含义
> - $\nabla_\theta \log \pi_\theta(a_t|s_t)$：**方向**——增加动作 $a_t$ 概率的参数更新方向
> - $G_t$：**幅度**——这个动作之后实际获得了多少回报
>
> 两者相乘：回报高的动作，概率增加得多；回报低的动作，概率增加得少（甚至减少，如果 $G_t < 0$）。

### 完整算法

```
REINFORCE:
    初始化策略参数 θ
    for each episode:
        用 π_θ 采样完整轨迹 τ = (s_0, a_0, r_0, ..., s_T)
        for t = 0, 1, ..., T:
            计算 G_t = Σ_{t'=t}^{T} γ^{t'-t} r_{t'}
        更新 θ ← θ + α Σ_t γ^t ∇log π_θ(a_t|s_t) · G_t
```

注意两个关键特征：
1. **必须等 episode 结束**才能计算 $G_t$（因为需要未来所有奖励）
2. **On-policy**：每次更新后，旧轨迹就不能再用了（分布变了）

---

## 高方差问题

REINFORCE 在理论上是无偏的（unbiased），但方差极大。这是它最核心的缺陷。

> [!intuition] 为什么方差大？
> 想象你在训练一个下棋策略。某一局你赢了（$G_t > 0$），REINFORCE 会增加这局**所有动作**的概率——包括那些无关紧要甚至是坏的动作。
>
> 问题在于 $G_t$ 是从 $t$ 到终局的**累积回报**，它混合了：
> - 动作 $a_t$ 本身的好坏
> - 之后所有动作的好坏
> - 环境的随机性
>
> 这就是 [[Credit Assignment]] 问题：REINFORCE 无法精确区分"这个动作好"还是"后面运气好"。

**方差的具体来源**：

1. **轨迹回报的波动**：同一个好动作，在不同轨迹中可能得到截然不同的 $G_t$
2. **长 horizon 放大效应**：$T$ 越大，$G_t$ 的方差越大（更多随机因素累积）
3. **稀疏奖励**：如果只有终局有奖励，中间步骤的 $G_t$ 完全取决于最终结果

> [!math] 方差的量级
> 对于长度为 $T$ 的轨迹，$G_t$ 的方差大致与 $T$ 成正比。这意味着在 LLM 场景中（生成几百个 token），原始 REINFORCE 的方差会非常大。

---

## Baseline 技巧

降低方差最经典的方法是引入 baseline $b(s)$。

### 核心思想

将更新规则中的 $G_t$ 替换为 $G_t - b(s_t)$：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

> [!intuition] 为什么 baseline 有效？
> 没有 baseline 时，如果所有轨迹的回报都是正的（比如 $G_t \in [5, 10]$），那么**所有动作的概率都会增加**，只是增加的幅度不同。这很浪费——我们真正想知道的是"这个动作比平均水平好多少"。
>
> 减去 baseline 后，$G_t - b(s_t)$ 可以是负的：比平均好的动作概率增加，比平均差的动作概率减少。信号更清晰，方差自然更小。

### 为什么不引入偏差？

> [!math] Baseline 不改变梯度期望的证明
> 对于任何只依赖状态的 $b(s)$：
> $$\mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0$$
>
> 关键条件：$b$ 不能依赖于动作 $a$，只能依赖于状态 $s$。否则上面的推导不成立。

### 最优 Baseline 的选择

理论上，最优 baseline（最小化方差的 $b^*$）是：

$$b^*(s_t) = \frac{\mathbb{E}\left[\|\nabla_\theta \log \pi_\theta\|^2 \cdot G_t \mid s_t\right]}{\mathbb{E}\left[\|\nabla_\theta \log \pi_\theta\|^2 \mid s_t\right]}$$

实践中这个太复杂了。最常用的选择是 [[Value Function|状态价值函数]] $V(s)$，因为：
- $V(s_t) = \mathbb{E}[G_t | s_t]$ 就是 $G_t$ 的条件期望，是自然的"平均水平"
- 虽然不是理论最优，但接近最优且容易估计

### REINFORCE with Baseline

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \underbrace{(G_t - V(s_t))}_{\approx A(s_t, a_t)}\right]$$

$G_t - V(s_t)$ 是 [[Value Function|优势函数]] $A(s_t, a_t)$ 的 Monte Carlo 估计。到这里，REINFORCE with baseline 已经非常接近 [[Actor-Critic]] 了。

---

## 从 REINFORCE 到现代算法

REINFORCE 的每个缺陷都催生了一个改进方向，最终形成了现代 RL 算法的谱系：

```
REINFORCE (1992)
    │
    │ 问题：高方差
    ▼
REINFORCE + Baseline
    │
    │ 问题：V(s) 也需要学习，且 G_t 仍是 MC 估计
    ▼
Actor-Critic（用 learned V(s) 做 bootstrap）
    │
    │ 问题：TD 估计有 bias，如何平衡？
    ▼
GAE（λ 参数平衡 bias-variance）
    │
    │ 问题：策略更新步长不好控制
    ▼
TRPO（KL 约束）→ PPO（clip 近似）
    │
    │ 问题：Critic 太大（LLM 场景）
    ▼
GRPO（用 group reward 做 baseline，去掉 Critic）

另一条路线（回归简单）：

REINFORCE + Baseline
    │
    │ 改进：用多样本互为 baseline，无需 learned V(s)
    ▼
RLOO（Leave-One-Out baseline，去掉 Critic）
```

> [!comparison] 关键演进对比
>
> | 算法 | 梯度信号 | Baseline/Advantage | 数据使用 |
> |------|----------|-------------------|----------|
> | REINFORCE | $G_t$（MC return） | 无 | On-policy |
> | REINFORCE + Baseline | $G_t - V(s_t)$ | Learned $V(s)$ | On-policy |
> | [[Actor-Critic]] | $r_t + \gamma V(s_{t+1}) - V(s_t)$ | TD error | On-policy |
> | [[PPO]] | [[GAE]] | $\lambda$-weighted TD | Near on-policy（clip） |
> | [[GRPO]] | $r_i - \text{mean}(r_{\text{group}})$ | Group mean | Near on-policy（clip） |
> | [[RLOO]] | $R(y_i) - \text{mean}(R(y_{j \neq i}))$ | Leave-one-out mean | On-policy |

### 与 Actor-Critic 的关键区别

REINFORCE with baseline 和 [[Actor-Critic]] 看起来很像，但有本质区别：

- **REINFORCE**：用 $G_t$（实际回报）估计 advantage，$V(s)$ 只是 baseline
- **Actor-Critic**：用 $r_t + \gamma V(s_{t+1})$（TD target）替代 $G_t$，引入 bootstrap

Bootstrap 的代价是引入 bias（因为 $V(s)$ 不完美），但大幅降低方差。这就是 bias-variance tradeoff 在 RL 中的核心体现，[[GAE]] 通过 $\lambda$ 参数提供了精细的控制。

### 在 LLM 中的体现

在 RLHF 场景中，REINFORCE 的思想无处不在：

- **[[PPO]] 的目标函数**本质上是 REINFORCE + [[Importance Sampling|IS 修正]] + clip，其中 $\nabla_\theta \log \pi_\theta$ 的结构直接来自 REINFORCE
- **[[GRPO]]** 回归了 REINFORCE 的精神——不用 Critic，直接用 group reward 的均值作为 baseline
- **[[RLOO]]** 更彻底地回归 REINFORCE——用 leave-one-out 多样本互为 baseline，不需要 Critic、不需要 clipping、不需要 IS ratio
- **[[CISPO]]** 在 REINFORCE 目标上做 IS weight 的 clip，而非 token update 的 clip

理解 REINFORCE 为什么需要 clip：因为 REINFORCE 是 on-policy 的，数据用完即弃。[[PPO]] 想复用旧数据（提高样本效率），就需要 [[Importance Sampling]] 修正分布差异，而 IS ratio 可能爆炸，所以需要 clip 来限制。

---

## 局限性

> [!warning] REINFORCE 的根本限制
> 1. **高方差**：即使加了 baseline，MC 估计的方差仍然比 TD 方法大得多
> 2. **样本效率极低**：on-policy + 必须等 episode 结束 = 需要大量交互
> 3. **[[Credit Assignment]] 困难**：$G_t$ 混合了当前动作和未来所有动作的影响，无法精确归因
> 4. **不适用于连续任务**：必须有明确的 episode 终止（因为需要计算 $G_t$）
> 5. **步长敏感**：没有信任域约束，学习率太大会导致策略崩溃

---

## 面试要点

> [!interview] 面试视角
> **Q: REINFORCE 的核心思想是什么？**
> A: 用 Monte Carlo 采样估计策略梯度。采样完整轨迹，用实际回报 $G_t$ 加权 $\nabla \log \pi$ 来更新策略。好动作的概率增加，坏动作的概率减少。
>
> **Q: REINFORCE 的主要问题是什么？如何解决？**
> A: 高方差。三个层次的解决方案：(1) 加 baseline $b(s)$ 降低方差但不引入 bias；(2) 用 learned $V(s)$ 做 baseline，演变为 Actor-Critic；(3) 用 GAE 的 $\lambda$ 参数精细控制 bias-variance tradeoff。
>
> **Q: REINFORCE with baseline 和 Actor-Critic 有什么区别？**
> A: 关键区别在于是否 bootstrap。REINFORCE 用实际回报 $G_t$ 减去 $V(s)$；Actor-Critic 用 TD target $r + \gamma V(s')$ 替代 $G_t$。Bootstrap 降低方差但引入 bias。
>
> **Q: 为什么 PPO 需要 clip？和 REINFORCE 有什么关系？**
> A: REINFORCE 是 on-policy 的，数据不能复用。PPO 想复用旧策略的数据来提高样本效率，就需要 importance sampling 修正。但 IS ratio 可能很大导致更新不稳定，所以用 clip 限制更新幅度。这就是从 REINFORCE → TRPO（KL 约束）→ PPO（clip 近似）的演进逻辑。

> [!paper] 论文出处
> Williams, R.J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning*, 8(3-4), 229-256.
