---
type: concept
description: 强化学习的数学框架，用五元组 (S, A, P, R, γ) 描述序贯决策问题，所有 RL 算法的理论基础
aliases:
  - MDP
  - 马尔可夫决策过程
prerequisites:
  - "[[Probability]]"
tags:
  - reinforcement-learning
  - foundations
created: 2026-02-24
updated: 2026-02-24
---

# Markov Decision Process

马尔可夫决策过程（MDP, Markov Decision Process）是描述**序贯决策问题**的数学框架。它回答一个核心问题：一个 agent 在不确定的环境中，如何通过一系列决策来最大化长期收益？几乎所有 RL 算法——从 [[TD Learning]] 到 [[Policy Gradient]] 到 [[PPO]]——都建立在 MDP 的形式化之上。

## 动机：为什么需要 MDP？

考虑最简单的决策问题：你面前有 $k$ 台老虎机（Multi-Armed Bandit），每次选一台拉，获得随机奖励。你的目标是找到期望奖励最高的那台。

这个问题有一个关键简化：**你的选择不会改变世界的状态**。拉完一台老虎机后，下一次面对的还是同样的 $k$ 台机器，概率分布不变。

但现实中的决策很少这么简单：
- 下棋：每走一步，棋盘状态改变，可选的走法也随之改变
- 机器人导航：每移动一步，位置改变，能看到的环境也不同
- **LLM 生成文本**：每生成一个 token，已有的上下文改变，下一个 token 的概率分布也随之改变

> [!intuition] 从 Bandit 到 MDP
> Bandit 问题是"无状态"的决策——你的行动不影响未来的处境。MDP 引入了**状态**和**状态转移**：你的行动不仅获得即时奖励，还会改变你所处的状态，从而影响未来所有的决策。这就是为什么 MDP 比 Bandit 难得多——你必须考虑行动的**长期后果**。

## 形式化定义：五元组

> [!definition] MDP 五元组
> 一个 MDP 由五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ 定义：
>
> | 符号 | 名称 | 含义 |
> |------|------|------|
> | $\mathcal{S}$ | 状态空间（State Space） | agent 可能处于的所有状态的集合 |
> | $\mathcal{A}$ | 动作空间（Action Space） | agent 可以采取的所有动作的集合 |
> | $P(s'|s, a)$ | 状态转移函数（Transition Function） | 在状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率 |
> | $R(s, a)$ | 奖励函数（Reward Function） | 在状态 $s$ 执行动作 $a$ 获得的即时奖励 |
> | $\gamma \in [0, 1)$ | 折扣因子（Discount Factor） | 未来奖励相对于即时奖励的衰减系数 |

逐个理解这些组件：

**状态空间 $\mathcal{S}$**：描述"世界长什么样"。在棋类游戏中是棋盘布局，在机器人导航中是位置和速度，在 LLM 中是已生成的 token 序列。

**动作空间 $\mathcal{A}$**：描述"agent 能做什么"。可以是离散的（棋类的合法走法、LLM 的词表中的 token）或连续的（机器人关节的角度）。

**状态转移函数 $P(s'|s, a)$**：描述"世界如何变化"。这是环境的动力学模型。注意它是概率性的——同样的状态和动作可能导致不同的下一个状态。

**奖励函数 $R(s, a)$**：描述"什么是好的"。这是 agent 优化的信号。奖励函数的设计直接决定了 agent 学到的行为——这也是为什么 [[Reward Model]] 在 RLHF 中如此关键。

**折扣因子 $\gamma$**：描述"多看重未来"。$\gamma = 0$ 意味着只关心即时奖励（极度短视），$\gamma \to 1$ 意味着几乎同等看重所有未来奖励。

## Markov Property：为什么叫"马尔可夫"？

MDP 的核心假设是 **Markov Property**（马尔可夫性质）：

> [!definition] Markov Property
> 未来只取决于当前状态，与历史无关：
> $$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = P(s_{t+1} | s_t, a_t)$$

> [!intuition] 直觉理解
> Markov Property 说的是：**当前状态已经包含了做决策所需的全部信息**。你不需要回顾整个历史来决定下一步怎么做。
>
> 下棋是一个好例子：你只需要看当前棋盘就能决定下一步，不需要知道这个局面是怎么走出来的。棋盘本身就是"充分统计量"。

**为什么 Markov Property 重要？**

1. **计算可行性**：如果需要考虑完整历史，状态空间会随时间指数增长，问题变得不可解
2. **理论基础**：[[Bellman Equation]] 的推导依赖于 Markov Property——正是因为未来只取决于当前状态，我们才能把长期价值递归地分解为"即时奖励 + 未来价值"
3. **算法设计**：[[TD Learning]] 用 $V(s_{t+1})$ 来估计未来价值，这个 bootstrapping 的合理性来自 Markov Property

> [!warning] Markov Property 不总是成立
> 很多现实问题并不严格满足 Markov Property。解决方案通常是**把历史信息编码进状态**：
> - 使用 RNN/LSTM 将历史压缩为隐状态
> - 使用 frame stacking（如 Atari 游戏用连续 4 帧作为状态）
> - LLM 天然满足：整个 token 序列就是状态，包含了完整历史

## Policy：agent 的行为策略

有了 MDP 的框架，下一个问题是：agent 应该如何行动？这由 **policy**（策略）定义。

> [!definition] Policy
> 策略 $\pi$ 是从状态到动作的映射。分为两种形式：
> - **确定性策略**：$a = \pi(s)$，给定状态直接输出动作
> - **随机策略**：$\pi(a|s) = P(a_t = a | s_t = s)$，给定状态输出动作的概率分布

在 LLM 的语境中，policy 就是模型本身：给定已生成的 token 序列（状态），输出下一个 token 的概率分布。[[Policy Gradient]] 和 [[PPO]] 的目标就是找到最优的 policy。

RL 的核心目标可以简洁地表述为：**找到使长期累积奖励最大化的最优策略 $\pi^*$**。

## Return：如何衡量长期收益？

agent 不应该只看眼前的奖励，而应该最大化**长期累积收益**。这引出了 Return 的定义。

> [!definition] Return（回报）
> 从时间步 $t$ 开始的折扣累积奖励：
> $$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

Return 有一个关键的递归性质，这是 [[Bellman Equation]] 和 [[TD Learning]] 的基础：

$$G_t = r_t + \gamma G_{t+1}$$

这个递归关系说：**当前的长期收益 = 即时奖励 + 折扣后的未来长期收益**。

## Discount Factor γ：为什么要打折？

$\gamma$ 是 MDP 中最容易被低估的组件。它不只是一个"调参用的超参数"，而是有深刻的数学和直觉意义。

**数学角度：保证收敛**

如果 $\gamma = 1$，在无限时间步的情况下，$G_t = \sum_{k=0}^{\infty} r_{t+k}$ 可能发散到无穷大。$\gamma < 1$ 保证了几何级数收敛：

$$G_t \leq \sum_{k=0}^{\infty} \gamma^k r_{\max} = \frac{r_{\max}}{1 - \gamma}$$

这给了 Return 一个有限的上界，使得 [[Value Function]] 有良好的定义。

**直觉角度：不确定性的自然反映**

> [!intuition] 为什么未来的奖励应该"打折"？
> 1. **不确定性**：越远的未来越不可预测。$\gamma$ 反映了对远期预测的不信任
> 2. **时间偏好**：现在的一块钱比明天的一块钱更有价值（经济学中的折现率）
> 3. **有效视野**：$\gamma$ 隐式定义了 agent 的"视野"——$\gamma = 0.99$ 时有效视野约 100 步，$\gamma = 0.9$ 时约 10 步

> [!example] $\gamma$ 的有效视野
> 当 $\gamma^k$ 衰减到可忽略时，超过 $k$ 步的奖励几乎不影响决策：
>
> | $\gamma$ | 有效视野（$\gamma^k < 0.01$） | 适用场景 |
> |----------|-------------------------------|----------|
> | 0.9 | ~22 步 | 短期任务 |
> | 0.99 | ~230 步 | 中等任务 |
> | 0.999 | ~2300 步 | 长期规划 |
>
> 在 LLM 的 RLHF 训练中，$\gamma$ 通常设为 1.0（因为序列长度有限，不存在发散问题），或接近 1.0 的值如 0.99。

## MDP 在 LLM 中的对应

将 LLM 的文本生成过程建模为 MDP 是理解 RLHF 的关键。[[Policy Gradient]] 中讨论的 token 级 vs 序列级视角，本质上就是"是否把文本生成看作 MDP"的问题。

> [!comparison] LLM 生成 = MDP
>
> | MDP 组件 | LLM 对应 | 说明 |
> |----------|----------|------|
> | 状态 $s_t$ | 已生成的 token 序列 $y_{<t}$（含 prompt） | 包含完整历史，天然满足 Markov Property |
> | 动作 $a_t$ | 下一个 token $y_t$ | 离散动作空间，大小 = 词表大小 $|\mathcal{V}|$ |
> | 策略 $\pi(a_t|s_t)$ | LLM 本身 $\pi_\theta(y_t|y_{<t})$ | 模型参数 $\theta$ 定义了策略 |
> | 状态转移 $P$ | 确定性：$s_{t+1} = s_t \oplus a_t$ | 新状态 = 旧序列 + 新 token，无随机性 |
> | 奖励 $R$ | [[Reward Model]] 的输出 | 通常只在序列末尾给出（sparse reward） |
> | 折扣因子 $\gamma$ | 通常为 1.0 | 序列有限长，不需要折扣来保证收敛 |

这个对应关系有几个值得注意的特点：

1. **状态转移是确定性的**：给定当前序列和新 token，下一个状态完全确定。这与传统 RL 中的随机环境不同
2. **奖励是稀疏的**：[[Reward Model]] 通常只在整个序列生成完毕后给出一个分数，中间步骤没有奖励。这使得 [[Credit Assignment]] 成为核心挑战——如何把序列级奖励分配到每个 token？
3. **动作空间巨大但离散**：词表通常有 32k-128k 个 token，远大于传统 RL 的动作空间

> [!intuition] 为什么 LLM 天然满足 Markov Property？
> 因为 LLM 的"状态"就是完整的 token 序列。它不是历史的压缩或近似，而是历史本身。所以 $P(y_t | y_{<t})$ 已经包含了所有历史信息，Markov Property 自动成立。

## 局限性

MDP 是一个强大但有限的框架：

1. **完全可观测假设**：MDP 假设 agent 能观测到完整状态。当状态部分不可观测时，需要扩展为 POMDP（Partially Observable MDP）
2. **单 agent 假设**：MDP 只建模一个决策者。多 agent 场景需要博弈论框架
3. **固定环境假设**：状态转移函数 $P$ 和奖励函数 $R$ 不随时间变化。非平稳环境需要额外处理
4. **状态空间爆炸**：对于复杂问题（如围棋、LLM），状态空间极其庞大，无法枚举。这就是为什么需要函数近似（如神经网络）来表示 [[Value Function]] 和 policy

> [!interview] 面试视角
> **Q: MDP 和 Bandit 问题的核心区别是什么？**
> A: Bandit 是无状态的——动作不影响未来的处境，只需要最大化即时奖励。MDP 引入了状态转移，agent 的动作会改变环境状态，因此必须考虑长期后果。MDP 是 Bandit 的推广：当 $|\mathcal{S}| = 1$ 时，MDP 退化为 Bandit。
>
> **Q: 为什么 LLM 的文本生成可以建模为 MDP？**
> A: 状态是已生成的 token 序列，动作是下一个 token，策略就是 LLM 本身。关键优势是 LLM 天然满足 Markov Property，因为状态包含完整历史。但奖励通常是稀疏的（只在序列末尾），这使得 credit assignment 成为挑战。
>
> **Q: Discount factor $\gamma$ 的作用？**
> A: 数学上保证无限序列的 Return 收敛；直觉上反映对远期奖励的不确定性。$\gamma$ 隐式定义了 agent 的有效视野。在 LLM RLHF 中通常设为 1.0，因为序列有限长。

## 延伸阅读

**后续概念**（基于 MDP 构建）：
- [[Value Function]] — 衡量状态/状态-动作对的长期价值
- [[Bellman Equation]] — 将 Value Function 递归分解的核心方程
- [[REINFORCE]] — 最基础的 Policy Gradient 算法
- [[Actor-Critic]] — 结合 value 估计和 policy 优化的架构
