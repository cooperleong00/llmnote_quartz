---
description: 强化学习两大学习范式的核心区别——收集数据的策略和优化的策略是否相同，决定了 sample efficiency 与 stability 的根本权衡
type: concept
aliases:
  - On-Policy
  - Off-Policy
  - 在线策略
  - 离线策略
  - on-policy vs off-policy
prerequisites:
  - "[[Policy Gradient]]"
  - "[[TD Learning]]"
tags:
  - reinforcement-learning
  - foundations
created: 2026-02-24
updated: 2026-02-24
---

# On-Policy vs Off-Policy

强化学习中最基本的分类维度之一：**收集数据的策略（behavior policy）和要优化的策略（target policy）是否是同一个？** 如果是，就是 on-policy；如果不是，就是 off-policy。这个看似简单的区别，决定了算法在样本效率、训练稳定性、数据复用能力上的根本权衡。

---

## 核心定义

> [!definition] Behavior Policy vs Target Policy
> - **Behavior policy** $\mu$：实际与环境交互、收集经验数据的策略
> - **Target policy** $\pi$：我们真正想要优化和改进的策略
>
> | 范式 | 关系 | 含义 |
> |------|------|------|
> | **On-Policy** | $\mu = \pi$ | 用当前策略收集数据，用这些数据更新当前策略 |
> | **Off-Policy** | $\mu \neq \pi$ | 用任意策略收集的数据来更新目标策略 |

> [!intuition] 直觉理解
> 想象你在学下棋：
> - **On-policy**：你只从自己下的棋局中学习。每次水平提升后，之前的棋局就"过时"了，必须重新下
> - **Off-policy**：你可以从大师的棋谱、自己的旧棋局、甚至随机对局中学习。数据来源不限于当前水平

---

## 为什么这个区别重要？

这个区别直接影响两个核心问题：

**1. 数据能不能复用？**

On-policy 算法每次更新策略后，之前收集的数据就不再服从当前策略的分布，理论上不能直接使用。Off-policy 算法没有这个限制，历史数据可以反复利用。

**2. 学习信号是否有偏？**

On-policy 的数据天然服从目标分布，梯度估计无偏。Off-policy 的数据来自不同分布，必须通过 [[Importance Sampling]] 等技术修正分布偏差，否则学习信号是有偏的。

---

## On-Policy 算法

**核心特征**：每轮迭代都用当前策略 $\pi_\theta$ 采样新数据，更新后旧数据丢弃。

**代表算法**：
- **SARSA**：[[TD Learning]] 的 on-policy 版本，用当前策略选择的下一个动作来更新 Q 值
- [[A2C]]：用当前策略并行采样，同步更新 actor 和 critic
- [[PPO]]：虽然引入了 importance sampling ratio，但本质仍是 on-policy（见下文）

**更新公式的共同特点**——期望在当前策略下计算：

$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

**优势**：
- 梯度估计无偏，训练更稳定
- 不需要额外的分布修正
- 实现简单，调参相对容易

**劣势**：
- **样本效率极低**：数据用完即弃，每次策略更新都需要重新采样
- 无法利用历史数据或他人经验
- 在 LLM 场景下，每次都要用当前模型做 rollout，计算成本高昂

---

## Off-Policy 算法

**核心特征**：可以用任意策略收集的数据来更新目标策略，包括旧版本的自己、专家策略、甚至随机策略。

**代表算法**：
- **Q-Learning**：[[TD Learning]] 的 off-policy 版本，用 $\max_a Q(s', a)$ 更新（不管实际选了什么动作）
- **DQN**：Q-Learning + 深度网络 + [[Experience Replay]]
- [[SAC]]：最大熵框架下的 off-policy actor-critic

**Q-Learning vs SARSA 的对比**正是 on/off-policy 的经典体现：

> [!comparison] SARSA vs Q-Learning
>
> | | SARSA (On-Policy) | Q-Learning (Off-Policy) |
> |---|---|---|
> | **更新目标** | $Q(s,a) + \alpha[r + \gamma Q(s', a') - Q(s,a)]$ | $Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s,a)]$ |
> | **$a'$ 的来源** | 当前策略实际选择的动作 | 所有动作中 Q 值最大的 |
> | **含义** | 评估"我实际会怎么做" | 评估"最优情况下会怎样" |
> | **行为特点** | 更保守（考虑探索的代价） | 更激进（假设未来总是最优） |

**优势**：
- **样本效率高**：数据可以反复利用（通过 experience replay）
- 可以从离线数据集学习（offline RL）
- 可以利用专家示范数据

**劣势**：
- 需要修正分布偏差（importance sampling），引入额外方差
- 当 behavior policy 和 target policy 差距过大时，修正权重方差爆炸
- 训练可能不稳定，需要更多稳定化技巧（如 target network、双 Q 网络）

---

## Importance Sampling：Off-Policy 的理论基础

Off-policy 学习的核心挑战是：数据来自 $\mu$，但我们要估计 $\pi$ 下的期望。[[Importance Sampling]] 通过重要性权重修正这个分布偏差：

$$
\mathbb{E}_{\pi}[f(x)] = \mathbb{E}_{\mu}\left[\frac{\pi(x)}{\mu(x)} f(x)\right]
$$

在 RL 中，轨迹级的重要性权重是各步的乘积：

$$
\frac{\pi(\tau)}{\mu(\tau)} = \prod_{t=0}^{T} \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}
$$

> [!warning] 方差爆炸问题
> 这个乘积随序列长度指数增长，导致方差极大。这是 off-policy RL 的核心难题，也是为什么实践中需要各种截断和 clip 技巧。

---

## Experience Replay：为什么只能用于 Off-Policy？

Experience Replay 将历史交互 $(s, a, r, s')$ 存入 buffer，训练时随机采样 mini-batch。

这本质上是 off-policy 的：buffer 中的数据来自过去不同版本的策略（旧的 behavior policy），而我们要更新的是当前策略（target policy）。On-policy 算法要求数据严格来自当前策略，所以无法使用 replay buffer。

> [!intuition] 为什么 Experience Replay 重要？
> 1. **打破时间相关性**：连续采样的数据高度相关，随机采样打破这种相关性
> 2. **提高样本效率**：同一条经验可以被多次学习
> 3. **稳定训练**：mini-batch 的分布更均匀

---

## PPO 的特殊位置

[[PPO]] 是一个有趣的边界案例：它是 **on-policy 算法，但借用了 off-policy 的技巧**。

PPO 的 importance sampling ratio：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

这里 $\pi_{\theta_{old}}$ 是上一轮迭代的策略。严格来说，用旧策略的数据更新新策略就是 off-policy。但 PPO 通过两个机制将其限制在"近似 on-policy"的范围内：

1. **Clip 机制**：将 $r_t$ 限制在 $[1-\epsilon, 1+\epsilon]$，防止新旧策略偏离过大
2. **少量复用**：通常只对同一批数据做 3-4 个 epoch 的更新，然后重新采样

> [!intuition] PPO 的设计哲学
> PPO 的本质是在 on-policy 的稳定性和 off-policy 的样本效率之间找到一个实用的折中：允许对同一批数据做几次更新（提高效率），但通过 clip 严格限制策略偏移（保持稳定）。

---

## 在 LLM 训练中的体现

On-policy 和 off-policy 的区别在 LLM post-training 中有直接体现：

> [!comparison] RLHF/PPO vs DPO
>
> | 维度 | RLHF ([[PPO]]) | [[DPO]] |
> |------|------|------|
> | **范式** | On-policy | 类 Off-policy |
> | **数据来源** | 当前模型实时生成 | 预先收集的离线偏好数据 |
> | **采样成本** | 高（每轮都要 rollout） | 低（一次收集，反复使用） |
> | **分布匹配** | 天然匹配 | 可能存在 distribution shift |
> | **迭代速度** | 慢（采样是瓶颈） | 快（纯监督学习） |

[[DPO]] 可以看作一种 off-policy 的 alignment 方法：它直接在离线偏好数据上优化，不需要当前模型的在线采样。这带来了训练效率的巨大提升，但也引入了 distribution shift 的风险——当模型偏离训练数据的分布时，优化信号可能不再准确。

> [!interview] 面试视角
> **Q: PPO 是 on-policy 还是 off-policy？**
> A: 本质是 on-policy。虽然它用了 importance sampling ratio 来复用同一批数据做多次更新，但通过 clip 机制严格限制新旧策略的偏离，保证数据仍然"近似 on-policy"。每轮迭代结束后，旧数据会被丢弃，重新用当前策略采样。
>
> **Q: DPO 为什么可以看作 off-policy？**
> A: DPO 直接在预先收集的偏好数据上做优化，不需要当前模型的在线采样。数据的 behavior policy（生成偏好对的模型）和 target policy（正在训练的模型）通常不同，这正是 off-policy 的定义。代价是可能存在 distribution shift。

---

## 速查

> [!example] 关键对比
>
> | 维度 | On-Policy | Off-Policy |
> |------|-----------|------------|
> | **$\mu$ vs $\pi$** | $\mu = \pi$ | $\mu \neq \pi$ |
> | **样本效率** | 低 | 高 |
> | **稳定性** | 高 | 需要额外技巧 |
> | **数据复用** | 不可以 | 可以（Experience Replay） |
> | **需要 IS** | 不需要 | 需要 |
> | **代表算法** | SARSA, [[A2C]], [[PPO]] | Q-Learning, DQN, [[SAC]] |
> | **LLM 应用** | RLHF/PPO | [[DPO]]（类 off-policy） |
