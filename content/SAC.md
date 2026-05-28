---
description: 最大熵强化学习算法，同时最大化累积奖励和策略熵，通过自动温度调节平衡探索与利用
type: concept
aliases:
  - Soft Actor-Critic
  - 软演员-评论家
prerequisites:
  - "[[Actor-Critic]]"
  - "[[Entropy]]"
  - "[[Policy Gradient]]"
tags:
  - reinforcement-learning
  - maximum-entropy
created: 2026-01-29
updated: 2026-02-01T01:12
---

# SAC

Soft Actor-Critic（SAC）是一种基于 [[Actor-Critic]] 架构的**最大熵强化学习**（Maximum Entropy RL）算法，核心思想是在最大化累积奖励的同时，也最大化策略的[[Entropy|熵]]。这使得 SAC 天然具有更好的探索能力和鲁棒性，是连续动作空间中最成功的 off-policy 算法之一。

---

## 动机

> [!intuition] 为什么需要最大熵？
> 传统 RL 只追求最大化奖励，容易陷入局部最优。最大熵框架的核心洞察是：
> - **探索**：高熵策略会尝试更多可能性，避免过早收敛
> - **鲁棒性**：当多条路径都能获得相似奖励时，保持对它们的概率分布，而非只选一条
> - **学习信号**：即使在稀疏奖励环境中，熵也提供了持续的学习信号

**与 PPO 的对比**：
- [[PPO]] 用 **clip 机制**限制策略更新幅度，防止更新过大
- SAC 用 **熵正则化**鼓励探索，防止策略过早确定

---

## 核心目标函数

> [!math] 最大熵目标
> SAC 优化的目标是**软价值函数**（soft value function）：
>
> $$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$
>
> 其中：
> - $r(s_t, a_t)$：环境奖励
> - $\mathcal{H}(\pi(\cdot|s_t)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s_t)]$：策略熵
> - $\alpha$：温度参数，控制熵的重要性

展开熵的定义，目标等价于：

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}\left[r(s_t, a_t) - \alpha \log \pi(a_t|s_t)\right]$$

---

## 软贝尔曼方程

SAC 引入了**软 Q 函数**和**软价值函数**：

> [!math] 软价值函数
> $$V(s) = \mathbb{E}_{a \sim \pi}\left[Q(s, a) - \alpha \log \pi(a|s)\right]$$

> [!math] 软 Q 函数
> $$Q(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim p}\left[V(s')\right]$$

合并得到**软贝尔曼方程**：

$$Q(s, a) = r(s, a) + \gamma \mathbb{E}_{s', a'}\left[Q(s', a') - \alpha \log \pi(a'|s')\right]$$

---

## 三个网络的训练

SAC 使用三个网络（实际实现中常用两个 Q 网络 + 一个策略网络）：

### 1. Critic（Q 网络）

最小化软贝尔曼残差：

$$L_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(Q_\phi(s,a) - \hat{Q}(s,a)\right)^2\right]$$

其中目标值：
$$\hat{Q}(s,a) = r + \gamma \left(\min_{i=1,2} Q_{\bar{\phi}_i}(s', a') - \alpha \log \pi_\theta(a'|s')\right)$$

> [!warning] 双 Q 网络
> SAC 使用两个 Q 网络并取最小值，缓解 Q 值过估计问题（类似 TD3）。

### 2. Actor（策略网络）

最大化期望 Q 值减去熵惩罚：

$$L_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta}\left[\alpha \log \pi_\theta(a|s) - Q_\phi(s, a)\right]$$

使用**重参数化技巧**（reparameterization trick）使梯度可以流过采样过程：
$$a = f_\theta(\epsilon; s), \quad \epsilon \sim \mathcal{N}(0, I)$$

---

## 温度参数 $\alpha$ 的自动调节

> [!intuition] 为什么需要自动调节？
> - $\alpha$ 太大：过度探索，忽略奖励
> - $\alpha$ 太小：探索不足，容易陷入局部最优
> - 不同任务、不同训练阶段需要不同的 $\alpha$

SAC 将 $\alpha$ 也作为可学习参数，通过约束优化自动调节：

> [!math] 温度优化目标
> $$L(\alpha) = \mathbb{E}_{a \sim \pi_\theta}\left[-\alpha \left(\log \pi_\theta(a|s) + \bar{\mathcal{H}}\right)\right]$$
>
> 其中 $\bar{\mathcal{H}}$ 是目标熵（通常设为 $-\dim(\mathcal{A})$，即动作空间维度的负值）。

**直觉**：
- 当策略熵 $< \bar{\mathcal{H}}$（探索不足）：$\alpha$ 增大，鼓励更多探索
- 当策略熵 $> \bar{\mathcal{H}}$（探索过度）：$\alpha$ 减小，更关注奖励

---

## 与 PPO 的对比

> [!comparison] SAC vs PPO

| 维度 | SAC | PPO |
|------|-----|-----|
| **策略类型** | Off-policy | On-policy |
| **稳定性机制** | 熵正则化 + 双 Q 网络 | Clip 机制 |
| **样本效率** | 高（可重用历史数据） | 低（只能用当前数据） |
| **探索方式** | 熵最大化（内在） | 需要额外探索策略 |
| **动作空间** | 主要用于连续 | 离散/连续都适用 |
| **超参数敏感性** | 较低（自动调节 $\alpha$） | 中等 |
| **LLM 应用** | 较少 | 主流（RLHF） |

**为什么 LLM 训练主要用 PPO 而非 SAC？**
1. **离散动作空间**：LLM 的 token 选择是离散的，SAC 原生设计用于连续空间
2. **On-policy 偏好**：LLM 训练通常希望策略不要偏离太远，on-policy 更自然
3. **KL 约束**：PPO + KL penalty 的组合已经很成熟

> [!note] 稳定性机制的对比
> SAC 和 [[TRPO]]/PPO 代表了两种不同的稳定 RL 训练的思路：
> - **TRPO/PPO**：通过信任域约束（trust region）限制策略更新幅度
> - **SAC**：通过熵正则化保持策略的随机性和探索能力

---

## 在 LLM 训练中的潜在应用

虽然 SAC 在 LLM 训练中不是主流，但其**最大熵思想**对解决 [[Entropy Collapse]] 问题有重要启发：

> [!intuition] 熵正则化与 Entropy Collapse
> LLM RL 训练中的 [[Entropy Collapse]] 问题本质是策略熵过快下降。SAC 的熵最大化目标天然对抗这一问题：
> - 显式地将熵纳入优化目标
> - 自动调节温度参数维持目标熵水平

**相关方法**：
- [[DAPO]] 的 Clip-Higher 机制：防止概率过快下降
- [[CISPO]] 的 clip importance sampling：保留低概率 token 的梯度
- Entropy-based advantage：在优势函数中加入熵项

这些方法都可以看作是将 SAC 的最大熵思想引入 LLM RL 训练。

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: SAC 的核心思想是什么？**
> A: 最大熵强化学习。在最大化累积奖励的同时，也最大化策略熵。这样做的好处是：(1) 更好的探索，避免局部最优；(2) 更鲁棒，保持对多条可行路径的概率；(3) 即使稀疏奖励也有学习信号。
>
> **Q2: SAC 和 PPO 的主要区别？**
> A: (1) SAC 是 off-policy，PPO 是 on-policy；(2) SAC 用熵正则化鼓励探索，PPO 用 clip 限制更新；(3) SAC 样本效率更高但主要用于连续动作空间，PPO 更通用。
>
> **Q3: SAC 的温度参数 $\alpha$ 是什么？如何自动调节？**
> A: $\alpha$ 控制熵的重要性。通过约束优化自动调节：设定目标熵 $\bar{\mathcal{H}}$，当实际熵低于目标时增大 $\alpha$，反之减小。这样可以自适应地平衡探索与利用。
>
> **Q4: 为什么 LLM 训练主要用 PPO 而非 SAC？**
> A: (1) LLM 是离散动作空间，SAC 原生设计用于连续空间；(2) LLM 训练需要 KL 约束防止偏离太远，on-policy 的 PPO 更自然；(3) PPO + KL penalty 的组合已经很成熟。

---

## 参考资料

- Haarnoja et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
- Haarnoja et al. (2018). "Soft Actor-Critic Algorithms and Applications"（自动温度调节版本）
