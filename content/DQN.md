---
type: method
description: 用深度神经网络逼近 Q 函数的 value-based RL 算法，通过 Experience Replay 和 Target Network 两大创新解决了函数逼近下 Q-Learning 的不稳定性
aliases:
  - Deep Q-Network
  - 深度 Q 网络
prerequisites:
  - "[[Value Function]]"
  - "[[Markov Decision Process]]"
  - "[[TD Learning]]"
tags:
  - reinforcement-learning
  - foundations
created: 2026-02-24
updated: 2026-02-24
---

# DQN

DQN（Deep Q-Network, Mnih et al., 2015）是第一个成功将深度学习与强化学习结合的算法。它用神经网络逼近 [[Value Function|Q 函数]]，配合 Experience Replay 和 Target Network 两个关键技巧，解决了传统 Q-Learning 在高维状态空间下无法扩展的问题。DQN 在 Atari 游戏上达到了人类水平，标志着深度强化学习时代的开始。

---

## 从 Q-Learning 到 DQN

### Q-Learning：表格方法的极限

Q-Learning 是一种 [[On-Policy vs Off-Policy|off-policy]] 的 [[TD Learning|TD 学习]]算法，核心思想是：用 [[Value Function#Bellman Optimality Equation|Bellman Optimality Equation]] 迭代更新 Q 值，直到收敛到最优 $Q^*$。

> [!definition] Q-Learning 更新规则
> $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
>
> 其中：
> - $r + \gamma \max_{a'} Q(s', a')$ 是 **TD target**（用 Bellman Optimality Equation 构造）
> - $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ 是 **TD error**
> - $\max_{a'}$ 体现了 off-policy 的本质：不管当前策略怎么选动作，更新时总是朝最优方向走

表格 Q-Learning 为每个 $(s, a)$ 对维护一个 Q 值。当状态空间很小（如网格世界）时这没问题，但面对高维状态（如 Atari 的 $210 \times 160$ 像素画面），状态数量爆炸，表格方法完全不可行。

### 函数逼近：用神经网络替代表格

自然的想法是用参数化函数 $Q_\theta(s, a)$ 逼近 Q 值，其中 $\theta$ 是神经网络参数。这样：
- 不需要枚举所有状态，网络可以泛化到未见过的状态
- 输入是原始状态（如像素），输出是所有动作的 Q 值

但直接把神经网络塞进 Q-Learning 会导致训练极不稳定。原因有二：

1. **样本相关性**：agent 按时间顺序与环境交互，连续的样本高度相关（$s_t$ 和 $s_{t+1}$ 几乎一样），违反了 SGD 对 i.i.d. 数据的假设
2. **Moving target**：TD target $r + \gamma \max_{a'} Q_\theta(s', a')$ 依赖于正在更新的网络 $\theta$，每次更新参数都会改变目标值，导致训练震荡甚至发散

DQN 的两个关键创新正是为了解决这两个问题。

---

## DQN 的两大创新

### 1. Experience Replay

> [!intuition] 核心思想
> 不要用完数据就扔掉——把所有经验存起来，训练时随机抽样。

Experience Replay 维护一个固定大小的缓冲区（replay buffer）$\mathcal{D}$，存储 agent 的交互经验 $(s, a, r, s')$。训练时从 $\mathcal{D}$ 中均匀随机采样 mini-batch，而非使用最新的连续样本。

这解决了两个问题：
- **打破相关性**：随机采样使 mini-batch 中的样本近似 i.i.d.
- **提高 sample efficiency**：每条经验可以被多次使用，而非用完即弃

> [!warning] Replay Buffer 的局限
> - 存储成本：需要维护大量历史数据（DQN 原文用 100 万条）
> - 数据过时：buffer 中的旧数据来自过去的策略，可能与当前策略差距很大
> - 仅适用于 off-policy 算法：on-policy 方法（如 [[PPO]]）不能用旧数据训练

### 2. Target Network

> [!intuition] 核心思想
> 把"出题人"和"答题人"分开——用一个冻结的旧网络计算目标值，避免自己追自己的尾巴。

DQN 维护两个网络：
- **Online network** $Q_\theta$：正常训练，每步更新
- **Target network** $Q_{\theta^-}$：参数冻结，每隔 $C$ 步从 online network 复制一次

TD target 用 target network 计算：

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

损失函数：

> [!math] DQN 损失函数
> $$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]$$
>
> 注意：梯度只对 $\theta$ 求，不对 $\theta^-$ 求（target network 是冻结的）。

这解决了 moving target 问题：在 $C$ 步之内，目标值是固定的，网络可以稳定地朝一个固定目标优化。

---

## DQN 算法流程

> [!example] DQN 伪代码
> ```
> 初始化 replay buffer D（容量 N）
> 初始化 online network Q_θ（随机权重）
> 初始化 target network Q_θ⁻ ← Q_θ
>
> for episode = 1 to M:
>     获取初始状态 s
>     for t = 1 to T:
>         用 ε-greedy 选动作：
>             以概率 ε 随机选动作
>             否则 a = argmax_a Q_θ(s, a)
>         执行 a，观察 r, s'
>         存储 (s, a, r, s') 到 D
>
>         从 D 中随机采样 mini-batch
>         计算 target: y = r + γ max_a' Q_θ⁻(s', a')
>         更新 θ：最小化 (y - Q_θ(s, a))²
>
>         每 C 步：θ⁻ ← θ
> ```

关键超参数：
- $\epsilon$：探索率，通常从 1.0 线性衰减到 0.1
- $C$：target network 更新频率（原文 10,000 步）
- $|\mathcal{D}|$：replay buffer 大小（原文 1,000,000）
- mini-batch size：32

---

## 局限性与改进

DQN 虽然是里程碑式的工作，但存在几个已知问题，催生了一系列改进方法。

### Q-value Overestimation

> [!warning] 核心问题
> $\max_{a'} Q_\theta(s', a')$ 中的 $\max$ 操作会系统性地高估 Q 值。

直觉：假设 Q 值的估计有噪声，$\max$ 操作总是选到噪声最大的那个，导致正向偏差。这不是偶然误差，而是结构性的高估。

**Double DQN**（van Hasselt et al., 2016）的解决方案：将动作选择和动作评估分开。

$$y = r + \gamma Q_{\theta^-}\left(s', \arg\max_{a'} Q_\theta(s', a')\right)$$

- Online network $Q_\theta$ 负责选动作（$\arg\max$）
- Target network $Q_{\theta^-}$ 负责评估该动作的价值

这样即使 online network 选到了被高估的动作，target network 给出的评估也不会同样高估（两个网络的噪声不同）。

### 状态价值与动作优势的纠缠

标准 DQN 直接输出 $Q(s, a)$，但很多状态下不同动作的 Q 值差异很小——真正重要的是"这个状态本身好不好"和"这个动作相对于平均水平好不好"。

**Dueling DQN**（Wang et al., 2016）将网络分成两个分支：

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')$$

- $V(s)$：状态价值（这个状态本身有多好）
- $A(s, a)$：[[Value Function#Advantage Function：相对价值|优势函数]]（这个动作相对于平均水平好多少）
- 减去均值是为了保证可辨识性（否则 $V$ 和 $A$ 的分解不唯一）

### 均匀采样的低效

标准 Experience Replay 均匀随机采样，但不是所有经验都同样有价值——TD error 大的样本包含更多学习信号。

**Prioritized Experience Replay**（Schaul et al., 2016）按 TD error 的大小赋予采样优先级，让网络更多地从"意外"的经验中学习。

---

## 为什么 DQN 不适合 LLM 训练

DQN 是 value-based RL 的经典方法，但在 LLM 对齐/训练中几乎不被使用。核心原因是 action space 的规模：

> [!warning] Action Space 爆炸
> - Atari 游戏：4-18 个离散动作 → DQN 输出层有 18 个节点，轻松搞定
> - LLM 生成：vocabulary size 通常 32K-128K → 每个 token 位置都要对所有 token 计算 Q 值
> - 而且 LLM 的"动作"是序列级的（生成整个回复），组合空间是 $|V|^T$，完全不可枚举

这就是为什么 LLM 训练转向了 [[Policy Gradient|policy-based 方法]]：
- [[PPO]] 直接优化策略 $\pi_\theta(a|s)$，输出动作的概率分布，天然适配大 action space
- 不需要对每个动作计算 Q 值，只需要能采样和计算 log probability

> [!comparison] Value-based vs Policy-based 在 LLM 中的角色
>
> | 方面 | Value-based (DQN) | Policy-based (PPO) |
> |------|-------------------|-------------------|
> | 动作空间 | 离散、小规模 | 连续或大规模离散均可 |
> | 输出 | 每个动作的 Q 值 | 动作的概率分布 |
> | LLM 适用性 | 不适用（vocab 太大） | 直接适用（输出 token 概率） |
> | 探索方式 | $\epsilon$-greedy | 策略本身的随机性 |
> | 样本效率 | 高（experience replay） | 较低（on-policy 数据） |
>
> 不过 value function 并没有在 LLM 训练中消失——它以 Critic 的形式出现在 [[PPO]] 的 Actor-Critic 架构中，用于估计 baseline 和计算 advantage。

---

## 面试要点

> [!interview] 面试视角
> **Q: DQN 相比 Q-Learning 的核心改进是什么？**
> A: 两点——Experience Replay 打破样本相关性，Target Network 稳定训练目标。本质上是解决"用神经网络做函数逼近"带来的不稳定性。
>
> **Q: 为什么 DQN 不能用于 LLM？**
> A: Action space 太大。DQN 需要对每个动作输出 Q 值，LLM 的 vocabulary 有 32K-128K 个 token，而且生成是序列级的，组合空间是指数级的。所以 LLM 用 policy-based 方法（如 PPO）。
>
> **Q: Double DQN 解决了什么问题？**
> A: Q-value overestimation。标准 DQN 的 max 操作会系统性高估 Q 值，Double DQN 把"选动作"和"评估动作"分给两个网络，打破了高估的正反馈循环。

---

## 延伸阅读

**原始论文**：
- Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015

**改进方法**：
- [[Double DQN]] — 解决 Q-value overestimation
- [[Dueling DQN]] — 分离状态价值和动作优势
- [[Prioritized Experience Replay]] — 按 TD error 优先采样

**对比方法**：
- [[Policy Gradient]] — 直接优化策略的另一条路线
- [[PPO]] — policy-based 方法在 LLM 中的实际应用
