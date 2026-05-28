---
type: concept
description: 将 agent 的交互经验存入 buffer 并随机采样训练，打破数据时间相关性并提高样本效率，是 DQN 成功的关键创新之一
aliases:
  - 经验回放
  - Replay Buffer
  - 经验重放
prerequisites:
  - "[[Q-Learning]]"
  - "[[TD Learning]]"
  - "[[On-Policy vs Off-Policy]]"
tags:
  - reinforcement-learning
  - off-policy
  - foundations
created: 2026-02-26
updated: 2026-02-26
---

# Experience Replay

Experience Replay（经验回放）将 agent 与环境交互产生的经验 $(s, a, r, s')$ 存入一个固定大小的缓冲区（replay buffer），训练时从中随机采样 mini-batch 而非直接使用最新的连续样本。这个看似简单的想法解决了深度 RL 中一个根本性问题：**神经网络训练需要 i.i.d. 数据，但 RL 的在线交互天然产生高度相关的序列数据**。它是 [[DQN]] 成功将深度学习引入 RL 的两大关键创新之一。

## 动机：为什么在线数据不能直接用？

> [!intuition] 核心问题
> 想象你在玩一个迷宫游戏。如果你连续 100 步都在同一个走廊里转悠，这 100 个样本几乎描述的是同一件事——"走廊长什么样"。用这样的 batch 训练神经网络，网络会过度拟合到"走廊"，忘掉之前学到的"岔路口该怎么走"。

深度学习的优化算法（SGD 及其变体）有一个隐含假设：**mini-batch 中的样本是独立同分布（i.i.d.）的**。但 RL 的在线交互违反了这个假设，体现在两个层面：

**1. 时间相关性（Temporal Correlation）**

连续时间步的状态转移是高度相关的——$s_{t+1}$ 直接由 $s_t$ 和 $a_t$ 决定。如果直接用连续采样的 $(s_t, a_t, r_t, s_{t+1}), (s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2}), \ldots$ 作为 mini-batch，样本之间的相关性会导致：

- 梯度方向偏向当前局部经验，产生**灾难性遗忘**
- 网络参数在不同区域的经验之间剧烈震荡，训练不收敛

**2. 非平稳分布（Non-stationary Distribution）**

随着策略更新，agent 访问的状态分布也在变化。数据分布不断漂移，进一步加剧训练不稳定。

> [!warning] 函数逼近 + 相关数据 = 发散
> 在表格型 [[Q-Learning]] 中，相关性问题不严重——每个状态-动作对有独立的 Q 值条目，更新一个不影响其他。但换成神经网络后，更新一个样本的参数会影响所有状态的 Q 值估计。相关数据 + 参数共享 = 灾难性的连锁反应。这也是 [[TD Learning]] 与函数逼近结合时的经典不稳定性来源（deadly triad 的一部分）。

## 核心机制

### Replay Buffer 的存储

Replay buffer $\mathcal{D}$ 是一个固定容量 $N$ 的**环形缓冲区**（circular buffer）。Agent 每一步交互后，将 transition $(s_t, a_t, r_t, s_{t+1})$ 存入 buffer；当 buffer 满时，最旧的数据被覆盖。

> [!math] 形式化描述
> - 缓冲区：$\mathcal{D} = \{e_1, e_2, \ldots, e_N\}$，其中 $e_i = (s_i, a_i, r_i, s_i')$
> - 存储：每步将新 transition 写入 $\mathcal{D}[t \mod N]$
> - 采样：均匀随机抽取 mini-batch $B \sim \text{Uniform}(\mathcal{D})$，$|B| \ll N$

### 为什么随机采样有效？

随机采样从时间跨度很大的经验中抽取样本，一个 mini-batch 可能同时包含第 100 步和第 50000 步的经验。这带来两个效果：

1. **打破时间相关性**：mini-batch 中的样本来自不同时间、不同状态区域，近似满足 i.i.d. 假设
2. **平滑数据分布**：buffer 中混合了不同策略版本产生的数据，缓解了非平稳性

> [!example] DQN 原文的参数选择
> Mnih et al. (2015) 在 Atari 实验中使用：
> - Buffer 容量：$N = 10^6$（100 万条 transition）
> - Mini-batch 大小：32
> - 这意味着每次训练只用到 buffer 中 0.003% 的数据，充分打散了时间结构

### 样本效率的提升

没有 replay buffer 时，每条经验只用一次就丢弃（on-policy 的典型做法）。有了 buffer，同一条经验可以被采样多次，**数据利用率大幅提升**。这在样本获取成本高的场景（如机器人控制、真实环境交互）中尤为重要。

## 与 Off-Policy 的本质联系

Experience Replay **天然要求 off-policy 算法**，这不是设计选择，而是逻辑必然。

> [!intuition] 为什么？
> Buffer 中的数据来自过去不同版本的策略 $\pi_{\text{old}}$，而我们要更新的是当前策略 $\pi_{\text{current}}$。数据的收集策略（behavior policy）和优化目标策略（target policy）不同——这正是 [[On-Policy vs Off-Policy|off-policy]] 的定义。

具体来说：

- **Off-policy 算法**（如 [[Q-Learning]]、[[DQN]]、[[SAC]]）：更新规则不依赖数据来自哪个策略，可以自由使用 replay buffer
- **On-policy 算法**（如 SARSA、[[PPO]]、A2C）：要求数据严格来自当前策略，用完即弃，**无法使用 replay buffer**

> [!comparison] On-Policy vs Off-Policy 的数据使用
> | | On-Policy | Off-Policy + Replay |
> |---|---|---|
> | **数据来源** | 当前策略 $\pi_\theta$ | 历史策略混合 |
> | **数据寿命** | 用完即弃 | 反复使用直到被覆盖 |
> | **样本效率** | 低 | 高 |
> | **稳定性** | 高（数据分布匹配） | 需要额外技巧（target network 等） |

这也解释了为什么 [[DQN]] 同时需要 Experience Replay 和 Target Network——前者打破相关性但引入了 off-policy 的分布偏差，后者稳定 off-policy 更新的目标值。

## 重要变体：Prioritized Experience Replay (PER)

标准 Experience Replay 均匀随机采样，隐含假设是所有经验同等重要。但直觉上，**"出乎意料"的经验比"符合预期"的经验包含更多学习信号**。

> [!paper] Schaul et al., 2016
> Prioritized Experience Replay 提出用 TD error 的大小作为采样优先级——TD error 越大，说明当前网络对这条经验的预测越差，越值得学习。

### 优先级定义

对于 transition $i$，其优先级为：

$$p_i = |\delta_i| + \epsilon$$

其中 $\delta_i = r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)$ 是 [[TD Learning|TD error]]，$\epsilon$ 是一个小常数防止优先级为零。

采样概率：

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

$\alpha$ 控制优先级的程度：$\alpha = 0$ 退化为均匀采样，$\alpha = 1$ 完全按优先级采样。

### Importance Sampling 修正

优先级采样改变了数据分布，会引入偏差。PER 用 importance sampling 权重修正：

$$w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta$$

$\beta$ 从较小值逐渐退火到 1，在训练后期完全修正偏差。

> [!intuition] 为什么需要修正？
> 高优先级样本被过度采样，如果不修正，网络会过度拟合到这些"难"样本。Importance sampling 权重降低高频样本的梯度贡献，恢复无偏估计。

## 在现代 Deep RL 和 LLM 训练中的类比

Experience Replay 的核心思想——**存储经验、打破相关性、重复利用数据**——在现代系统中以不同形式延续。

### Deep RL 中的标准配置

几乎所有现代 off-policy 算法都使用 replay buffer：[[SAC]]、TD3、DDPG 等。区别主要在采样策略（均匀 vs 优先级）和 buffer 管理（FIFO vs 分层）。

### LLM RL 中的精神继承

LLM 训练中虽然不直接使用传统 replay buffer（因为 [[PPO]] 和 [[GRPO]] 是 on-policy 或近 on-policy 的），但 Experience Replay 的思想以变体形式出现：

- **Rejection Sampling Buffer**：生成多个候选回复，用 reward model 筛选后存入 buffer，后续训练从中采样。这本质上是一种"精选版"的 experience replay——不是存所有经验，而是只存高质量经验
- **Offline RL for LLM**：[[DPO]] 等 direct alignment 方法直接在离线偏好数据集上训练，可以看作一个"静态的 replay buffer"——数据预先收集好，训练时反复采样
- **[[JACKPOT]]**：通过 Optimal Budget Rejection Sampling 从小模型的 rollout 中筛选与大模型分布接近的样本，本质上也是在管理一个经验池的采样策略

> [!intuition] 共同的底层逻辑
> 无论是 Atari 游戏的 replay buffer 还是 LLM 的 rejection sampling，核心问题都是一样的：**如何从有限的交互经验中最大化学习效率？** 答案都指向同一个方向——存储、筛选、重复利用。

## 局限性

> [!warning] 需要注意的边界

**1. Stale Data（数据过时）**

Buffer 中的旧数据来自过去的策略版本。当策略更新较大时，旧数据的 Q 值估计可能严重偏离当前策略的真实值。这是 off-policy 学习的固有代价，通常通过限制 buffer 大小和策略更新幅度来缓解。

**2. 内存开销**

存储大量 transition 需要可观的内存。DQN 的 100 万条 buffer 在 Atari（84x84 灰度图像）上需要约 7GB 内存。对于高维观测空间（如 RGB 视频），内存压力更大。

**3. 均匀采样的低效**

标准均匀采样对所有经验一视同仁，但大部分经验可能已经被充分学习。PER 部分解决了这个问题，但引入了额外的计算开销（维护优先级队列）和实现复杂度。

**4. 不适用于 On-Policy 算法**

这是结构性限制而非缺陷。On-policy 算法（如 [[PPO]]）的理论保证依赖于数据来自当前策略，使用 replay buffer 会破坏这个前提。

**5. 初始阶段的冷启动**

Buffer 需要先积累足够的经验才能有效训练。DQN 通常在 buffer 中积累 50000 条经验后才开始训练，这段"热身期"agent 只收集数据不学习。

## 面试要点

> [!interview] 高频问题
> **Q: Experience Replay 解决了什么问题？**
> A: 两个问题——(1) 打破连续采样数据的时间相关性，使 mini-batch 近似 i.i.d.，满足 SGD 的假设；(2) 提高样本效率，每条经验可以被多次使用。
>
> **Q: 为什么 Experience Replay 只能用于 off-policy 算法？**
> A: Buffer 中的数据来自历史策略版本（behavior policy），而训练目标是当前策略（target policy）。两者不同，这就是 off-policy 的定义。On-policy 算法要求数据严格来自当前策略，所以无法使用 replay buffer。
>
> **Q: Prioritized Experience Replay 的核心思想是什么？为什么需要 importance sampling 修正？**
> A: 用 TD error 大小作为采样优先级——预测越差的经验越值得学习。但优先级采样改变了数据分布，引入偏差，需要 importance sampling 权重修正以保证无偏估计。

## 延伸阅读

**原始论文**：
- Lin, 1992 — 最早提出 Experience Replay 的概念
- Mnih et al., 2015 — 在 [[DQN]] 中将其发扬光大
- Schaul et al., 2016 — [[Prioritized Experience Replay]]，按 TD error 优先采样

**后续发展**：
- [[Hindsight Experience Replay]] — 从失败经验中学习，将实际到达的状态重新标记为目标
- [[Prioritized Experience Replay]] — 用 TD error 作为采样优先级的改进方案
