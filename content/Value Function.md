---
type: concept
description: 衡量状态或状态-动作对的长期价值，是 RL 中评估和改进策略的核心工具
aliases:
  - 价值函数
  - State Value Function
  - Action Value Function
  - Bellman Equation
  - 贝尔曼方程
  - Advantage Function
  - 优势函数
prerequisites:
  - "[[Markov Decision Process]]"
tags:
  - reinforcement-learning
  - foundations
created: 2026-02-24
updated: 2026-02-24
---

# Value Function

价值函数（Value Function）回答了 RL 中最基本的问题：**从某个状态出发，遵循某个策略，长期来看能获得多少回报？** 它是连接"即时奖励"和"长期目标"的桥梁——有了价值函数，agent 就不需要每次都模拟到终局才能判断当前处境的好坏。

在 [[Markov Decision Process]] 中，我们定义了 Return $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ 作为长期累积奖励。价值函数就是 Return 的**期望值**——对随机性（策略的随机选择、环境的随机转移）取平均后的结果。

## 两种价值函数：V 和 Q

RL 中有两种互补的价值函数，分别从不同粒度评估"好坏"。

### State Value Function $V^\pi(s)$

> [!definition] State Value Function
> 在状态 $s$ 下，遵循策略 $\pi$ 的期望回报：
> $$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s \right]$$

$V^\pi(s)$ 回答的问题是：**"我在状态 $s$，之后一直按策略 $\pi$ 行动，平均能拿多少回报？"**

它只关心"在哪里"，不关心"做什么"——因为"做什么"已经由策略 $\pi$ 决定了。

### Action Value Function $Q^\pi(s, a)$

> [!definition] Action Value Function（Q 函数）
> 在状态 $s$ 下执行动作 $a$，之后遵循策略 $\pi$ 的期望回报：
> $$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$$

$Q^\pi(s, a)$ 比 $V^\pi(s)$ 多了一个维度：它不仅关心"在哪里"，还关心"第一步做什么"。之后的动作仍然由 $\pi$ 决定，但**第一步是指定的**。

> [!intuition] V 和 Q 的关系
> $V$ 是 $Q$ 在策略 $\pi$ 下的期望——把所有可能的第一步动作按策略概率加权平均：
> $$V^\pi(s) = \sum_{a} \pi(a|s) \, Q^\pi(s, a) = \mathbb{E}_{a \sim \pi} \left[ Q^\pi(s, a) \right]$$
>
> 反过来，$Q$ 可以用 $V$ 表示——执行动作 $a$ 后拿到即时奖励，再加上下一个状态的价值：
> $$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \, V^\pi(s')$$
>
> 这两个等式是理解 Bellman Equation 的基础。

## Bellman Equation：价值的递归结构

Bellman Equation 是价值函数最重要的性质：**当前状态的价值可以用下一个状态的价值来表示**。这个递归结构是几乎所有 RL 算法的理论基础。

### 直觉：为什么价值可以递归？

回忆 [[Markov Decision Process#Return：如何衡量长期收益？|Return 的递归性质]]：$G_t = r_t + \gamma G_{t+1}$。对两边取期望，就得到了价值函数的递归关系。

> [!intuition] Bellman Equation 的核心直觉
> **当前状态的价值 = 即时奖励 + 折扣后的下一状态价值**
>
> 这就像评估一份工作的价值：不只看这个月的工资（即时奖励），还要考虑这份工作带来的未来发展（后续状态的价值）。

### Bellman Expectation Equation

对于给定策略 $\pi$，V 和 Q 满足以下递归关系：

> [!math] V 的 Bellman Expectation Equation
> $$V^\pi(s) = \sum_{a} \pi(a|s) \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) \, V^\pi(s') \right]$$
>
> 展开来读：在状态 $s$，按策略 $\pi$ 选择动作 $a$，拿到即时奖励 $R(s,a)$，然后转移到 $s'$，未来的价值是 $\gamma V^\pi(s')$。对所有可能的 $a$ 和 $s'$ 取期望。

> [!math] Q 的 Bellman Expectation Equation
> $$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \sum_{a'} \pi(a'|s') \, Q^\pi(s', a')$$
>
> 执行动作 $a$ 后拿到奖励，转移到 $s'$，然后在 $s'$ 按策略 $\pi$ 选择 $a'$，继续下去。

这两个方程描述的是**评估**——给定策略 $\pi$，它的价值是多少。[[TD Learning]] 正是利用这个递归结构，用 $r + \gamma V(s')$ 作为 $V(s)$ 的更新目标，实现增量式的价值估计。

### Bellman Optimality Equation

如果我们不固定策略，而是问"最优策略下的价值是多少"，就得到了 Bellman Optimality Equation：

> [!math] Bellman Optimality Equation
> $$V^*(s) = \max_{a} \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) \, V^*(s') \right]$$
> $$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \, \max_{a'} Q^*(s', a')$$
>
> 关键区别：Expectation 版本对动作取**期望**（按 $\pi$ 加权），Optimality 版本对动作取 **max**（选最好的）。

> [!intuition] 从 Expectation 到 Optimality
> - Bellman Expectation：**"按照策略 $\pi$ 走，平均能拿多少？"** → 用于**评估**策略
> - Bellman Optimality：**"如果每步都做最优选择，能拿多少？"** → 用于**找到**最优策略
>
> 两者的唯一区别就是 $\sum_a \pi(a|s) [\cdots]$ 变成了 $\max_a [\cdots]$。

### Optimal Policy

有了 $V^*$ 或 $Q^*$，最优策略可以直接导出：

$$\pi^*(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_{a'} Q^*(s, a') \\ 0 & \text{otherwise} \end{cases}$$

也就是说，**最优策略是确定性的**——在每个状态选择 Q 值最大的动作。这是 Q-learning 和 [[DQN]] 等方法的理论基础：只要学到了 $Q^*$，最优策略就自动得到了。

如果只有 $V^*$，则需要知道环境的转移概率才能导出策略：

$$\pi^*(s) = \arg\max_{a} \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) \, V^*(s') \right]$$

这也是为什么 Q 函数在 model-free RL 中更常用——它不需要环境模型就能直接决策。

## Advantage Function：相对价值

在策略优化中，我们通常不直接用 $Q$ 或 $V$，而是用它们的**差值**——Advantage Function。

> [!definition] Advantage Function
> $$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$
>
> 衡量在状态 $s$ 下，选择动作 $a$ 比"按策略 $\pi$ 的平均水平"好多少。

> [!intuition] 为什么要减去 baseline？
> 考虑一个状态 $s$，所有动作的 $Q$ 值都很高（比如 $Q = [100, 102, 98]$）。如果直接用 $Q$ 作为策略梯度的权重，所有动作都会被强化，区分度很低。
>
> 减去 $V(s) = 100$（平均水平）后，$A = [0, +2, -2]$，信号变得清晰：
> - $A > 0$：这个动作**优于**平均，应该增加概率
> - $A < 0$：这个动作**劣于**平均，应该减少概率
> - $A = 0$：和平均水平持平
>
> $V(s)$ 作为 baseline 不改变梯度的期望（因为 $\mathbb{E}_a[A^\pi(s,a)] = 0$），但**显著降低方差**。这就是为什么 [[PPO]] 使用 Advantage 而非 Q 值作为 clip 目标的权重，也是 [[GAE]] 存在的意义——它提供了一种 bias-variance 可调的 Advantage 估计方法。

### Advantage 的性质

- **零均值**：$\mathbb{E}_{a \sim \pi}[A^\pi(s, a)] = 0$（因为 $V = \mathbb{E}[Q]$）
- **正值 = 好动作**：$A > 0$ 意味着该动作优于策略平均
- **与 TD error 的关系**：一步 Advantage 可以用 TD error 近似：$A(s_t, a_t) \approx r_t + \gamma V(s_{t+1}) - V(s_t) = \delta_t$，这是 [[Actor-Critic]] 中 Critic 提供信号的基础

## 在 LLM 对齐中的对应

将价值函数映射到 LLM 的文本生成场景（参见 [[Markov Decision Process#MDP 在 LLM 中的对应]]）：

> [!comparison] Value Function 在 LLM 中的语义
>
> | RL 概念 | LLM 对应 | 直觉 |
> |---------|----------|------|
> | $V(s_t)$ | 给定已生成序列 $y_{<t}$ 的期望回报 | "到目前为止，这个回答大概能拿多少分？" |
> | $Q(s_t, a_t)$ | 选择 token $y_t$ 后的期望回报 | "如果下一个词选 X，最终能拿多少分？" |
> | $A(s_t, a_t)$ | 选择 token $y_t$ 相对于平均水平的优势 | "选这个词比随便选一个好多少？" |

在 [[RLHF]] 的 [[PPO]] 训练中，Critic 网络（通常是在 LLM backbone 上加一个 value head）负责估计 $V(s_t)$，然后通过 [[GAE]] 计算 Advantage 来更新 Actor（策略模型）。

[[GRPO]] 的核心创新正是**去掉了 Critic 网络**：它不再用 $V(s)$ 作为 baseline，而是对同一个 prompt 采样一组输出，用组内 reward 的均值和标准差来归一化，得到 group relative advantage。这省去了与策略模型同等规模的 Critic 网络，大幅降低了内存开销。

## 局限性

> [!warning] 价值函数的实际挑战
> - **高维状态空间**：LLM 的状态空间（所有可能的 token 序列）是天文数字级别的，精确的价值函数不可能用表格表示，必须用函数逼近（神经网络），这引入了逼近误差
> - **稀疏奖励下的估计困难**：在 RLHF 中，[[Reward Model]] 通常只在序列末尾给出奖励，中间 token 没有直接的奖励信号。Critic 需要学会把序列级奖励分配到每个 token——这就是 [[Credit Assignment]] 问题
> - **Deadly Triad**：当函数逼近 + bootstrapping（如 TD Learning）+ off-policy 学习三者结合时，价值函数的训练可能发散。这是 [[DQN]] 需要 target network 和 experience replay 的原因

> [!interview] 面试视角
> **Q: V 和 Q 的区别是什么？什么时候用哪个？**
> A: $V(s)$ 评估状态的好坏，$Q(s,a)$ 评估状态-动作对的好坏。$Q$ 在 model-free 设定下更有用，因为可以直接通过 $\arg\max_a Q(s,a)$ 得到策略，不需要知道环境模型。$V$ 在 Actor-Critic 中常用作 baseline。
>
> **Q: Bellman Expectation 和 Bellman Optimality 的区别？**
> A: Expectation 版本是对给定策略 $\pi$ 的评估（对动作取期望），Optimality 版本是对最优策略的刻画（对动作取 max）。前者用于 policy evaluation，后者用于 policy improvement / Q-learning。
>
> **Q: 为什么 Advantage Function 要减去 V(s)？**
> A: $V(s)$ 作为 baseline 不改变梯度期望（因为 $\mathbb{E}[A] = 0$），但显著降低方差。直觉上，我们关心的是"这个动作比平均好多少"，而非"这个动作的绝对回报是多少"。

## 延伸阅读

**基于 Value Function 的算法**：
- [[DQN]] — 用神经网络逼近 $Q^*$，直接从 Bellman Optimality Equation 导出
- [[REINFORCE]] — 不使用 value function 的策略梯度方法（对比理解 baseline 的作用）
