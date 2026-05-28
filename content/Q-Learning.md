---
type: method
description: 经典 tabular RL 的核心算法，通过 Bellman 最优方程迭代更新 Q 值逼近最优策略，是 off-policy TD 控制的代表方法，也是 DQN 的直接前身
aliases:
  - Q 学习
  - Q-learning
prerequisites:
  - "[[Value Function]]"
  - "[[TD Learning]]"
  - "[[Markov Decision Process]]"
  - "[[On-Policy vs Off-Policy]]"
tags:
  - reinforcement-learning
  - value-based
created: 2026-02-26
updated: 2026-02-26
---

# Q-Learning

Q-Learning 是 value-based RL 中最重要的算法之一。它的核心思想极其简洁：不需要知道环境的转移概率，只需要不断与环境交互，用 [[Value Function#Bellman Optimality Equation|Bellman 最优方程]] 迭代更新每个状态-动作对的 Q 值，最终收敛到最优 $Q^*$。一旦有了 $Q^*$，最优策略就是在每个状态选 Q 值最大的动作——不需要额外的策略网络。

这个算法由 Watkins 在 1989 年提出，是 [[TD Learning]] 从"评估策略"到"寻找最优策略"的关键跨越，也是后来 [[DQN]] 的直接理论基础。

## 动机：从策略评估到策略优化

> [!intuition] 为什么需要 Q-Learning？
> [[TD Learning|TD(0)]] 解决了一个问题：不需要等到 episode 结束，就能逐步评估一个策略的好坏（即估计 $V^\pi$）。但"评估"不等于"优化"——知道当前策略的价值，不代表知道最优策略是什么。
>
> Q-Learning 的关键洞察是：**如果我们直接估计的是最优 Q 函数 $Q^*$，而不是某个固定策略的 $Q^\pi$，那么评估和优化就合二为一了。** 因为 $Q^*$ 本身就编码了最优策略：$\pi^*(s) = \arg\max_a Q^*(s, a)$。

要实现这一点，Q-Learning 做了一个看似微小但意义深远的改动：把 TD target 中的"下一步实际动作的 Q 值"替换为"下一步所有动作中最大的 Q 值"。这个 $\max$ 操作，正是 Bellman 最优方程的体现。

## 核心算法

### Q 值更新公式

> [!definition] Q-Learning 更新规则
> $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big[ \underbrace{r_t + \gamma \max_{a'} Q(s_{t+1}, a')}_{\text{TD target}} - Q(s_t, a_t) \Big]$$
>
> 其中：
> - $Q(s_t, a_t)$：当前对状态-动作对 $(s_t, a_t)$ 的价值估计
> - $\alpha$：学习率，控制更新步长
> - $r_t$：执行 $a_t$ 后获得的即时奖励
> - $\gamma$：折扣因子
> - $\max_{a'} Q(s_{t+1}, a')$：在下一状态 $s_{t+1}$ 中，所有可能动作的最大 Q 值
> - TD target $-$ 当前估计 $=$ [[TD Learning#TD Error|TD error]]

> [!intuition] 这个公式在做什么？
> 想象你在一个迷宫中探索。每走一步，你会更新对当前位置的"价值判断"：
>
> 1. 你在状态 $s$ 选了动作 $a$，拿到奖励 $r$，到达新状态 $s'$
> 2. 你看看 $s'$ 的所有出口，找到"最有前途的那个"（$\max_{a'} Q(s', a')$）
> 3. 用"即时奖励 + 最有前途的未来"作为更好的估计（TD target）
> 4. 把当前估计朝这个更好的估计方向调整一小步
>
> 关键在第 2 步：你评估的是**最优的未来**，而不是你实际会走的那条路。这就是为什么 Q-Learning 能找到最优策略，即使你的探索策略很差。

### 与 Bellman 最优方程的关系

Q-Learning 的更新规则本质上是 [[Value Function#Bellman Optimality Equation|Bellman 最优方程]] 的**采样版本**（stochastic approximation）。

Bellman 最优方程告诉我们 $Q^*$ 满足：

$$Q^*(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

这是一个关于 $Q^*$ 的不动点方程——如果你有完整的环境模型 $P(s'|s,a)$，可以直接用 value iteration 求解。但在 model-free 的设定下，我们没有 $P$，只能通过与环境交互获得样本 $(s, a, r, s')$。

Q-Learning 的做法是：**用单个样本的 $r + \gamma \max_{a'} Q(s', a')$ 替代期望**，然后用增量更新（学习率 $\alpha$）逐步逼近不动点。这正是 [[TD Learning]] 的核心思路——用 bootstrap 估计替代完整期望。

### 完整算法流程

```
初始化 Q(s, a) 为任意值（通常为 0），对所有 (s, a)

对每个 episode:
    初始化状态 s

    对每一步:
        用 ε-greedy 策略从 Q 中选择动作 a    ← 行为策略（探索用）
        执行 a，观察 r, s'

        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
                                        ↑ 目标策略（贪心，取 max）

        s ← s'
    直到 s 是终止状态
```

注意算法中的两个策略：
- **行为策略**（behavior policy）：$\varepsilon$-greedy，负责探索
- **目标策略**（target policy）：greedy（$\max$），负责优化

这两个策略的分离，正是 Q-Learning [[On-Policy vs Off-Policy|off-policy]] 特性的来源。

## Off-Policy 特性

Q-Learning 是 off-policy 算法的经典代表。理解"为什么它是 off-policy 的"，是理解整个 [[On-Policy vs Off-Policy]] 区分的最佳入口。

> [!intuition] 为什么 Q-Learning 是 off-policy 的？
> 关键在于更新公式中的 $\max_{a'} Q(s', a')$。
>
> 当 agent 在状态 $s'$ 时，它**实际选择**的动作（由 $\varepsilon$-greedy 决定）可能是随机探索的动作 $a'_{\text{actual}}$。但更新 Q 值时，我们用的是 $\max_{a'}$——也就是假设未来总是选最优动作。
>
> 换句话说：
> - **收集数据的策略**（行为策略）：$\varepsilon$-greedy，会随机探索
> - **更新目标对应的策略**（目标策略）：纯贪心，总选最优
>
> 这两个策略不同，所以是 off-policy。

这带来一个重要的实际优势：Q-Learning **不需要 [[Importance Sampling|重要性采样]] 修正**。为什么？因为更新公式中根本没有用到行为策略选择的下一步动作 $a'$——它直接对所有动作取 $\max$。无论数据是怎么收集的（随机策略、专家策略、甚至是另一个 agent），Q-Learning 的更新都是一样的。

> [!warning] 常见误区
> "Off-policy 算法都需要 importance sampling" —— 这是错的。
>
> Importance sampling 是 off-policy **策略评估**（如估计 $V^\pi$）时需要的修正。Q-Learning 做的是 **控制**（直接找 $Q^*$），它的 $\max$ 操作天然绕过了分布偏差问题，不需要 IS 修正。
>
> 但当 Q-Learning 扩展到函数逼近（如 [[DQN]]）时，off-policy 带来的分布偏移问题会以其他形式出现（如 experience replay 中的陈旧数据）。

## 与 SARSA 的对比

SARSA（State-Action-Reward-State-Action）是 Q-Learning 的 on-policy 对应物，两者的对比是理解 [[On-Policy vs Off-Policy]] 最经典的案例。

> [!comparison] Q-Learning vs SARSA
>
> | | Q-Learning (Off-Policy) | SARSA (On-Policy) |
> |---|---|---|
> | **更新公式** | $Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s,a)]$ | $Q(s,a) + \alpha[r + \gamma Q(s', a') - Q(s,a)]$ |
> | **下一步动作** | $\max_{a'}$：假设未来总是最优 | $a'$：当前策略实际选择的动作 |
> | **评估的是谁** | 最优策略 $\pi^*$ | 当前行为策略 $\pi_\varepsilon$ |
> | **探索的代价** | 更新时忽略探索的代价 | 更新时考虑探索的代价 |
> | **行为倾向** | 更激进（乐观主义） | 更保守（现实主义） |

> [!example] Cliff Walking 经典案例
> 想象一个悬崖边的网格世界：沿悬崖走是最短路径，但一步走错就掉下去（大负奖励）。
>
> - **Q-Learning** 会学到沿悬崖走的最优路径——因为它假设未来总是选最优动作，不会失足。但在训练过程中，$\varepsilon$-greedy 探索会导致 agent 频繁掉下悬崖。
> - **SARSA** 会学到远离悬崖的安全路径——因为它知道自己会探索（$\varepsilon$ 概率随机走），所以悬崖边的状态被评估为危险的。
>
> 这揭示了一个深层权衡：Q-Learning 学到的策略在**执行时**更优（如果你关掉探索），但 SARSA 学到的策略在**训练时**更安全（考虑了探索的风险）。

两者的区别只有一个字符的差异——$\max_{a'}$ vs $a'$——但这个差异导致了完全不同的学习行为。这也是为什么理解 on-policy vs off-policy 如此重要。

## 收敛性

> [!math] 收敛条件（Tabular 情况）
> 在 tabular 设定下，Q-Learning 保证收敛到 $Q^*$，只要满足以下条件：
>
> 1. **所有状态-动作对被无限次访问**：每个 $(s, a)$ 都要被充分探索（$\varepsilon$-greedy 保证这一点）
> 2. **学习率满足 Robbins-Monro 条件**：
>    $$\sum_{t=0}^{\infty} \alpha_t = \infty, \quad \sum_{t=0}^{\infty} \alpha_t^2 < \infty$$
>    直觉：学习率要足够大以克服初始偏差，又要足够快衰减以保证收敛
> 3. **[[Markov Decision Process|MDP]] 是有限的**：状态空间和动作空间都是有限集

> [!intuition] 为什么这些条件足够？
> Q-Learning 的更新可以看作一个**随机逼近**（stochastic approximation）过程，目标是找到 Bellman 最优算子的不动点。
>
> - 条件 1 保证每个 $(s, a)$ 都有足够的样本来消除噪声
> - 条件 2 保证更新步长既不会太大（导致震荡）也不会太小（导致停滞）
> - 条件 3 保证不动点存在且唯一（$\gamma < 1$ 时 Bellman 最优算子是压缩映射）
>
> 在实践中，通常使用固定学习率（不满足条件 2），这意味着 Q 值会在最优值附近波动而非精确收敛——但对大多数应用来说足够了。

## 局限性：为什么需要 DQN

Q-Learning 在 tabular 设定下有优美的理论保证，但面对真实问题时遇到了根本性的瓶颈。

### Curse of Dimensionality

表格 Q-Learning 为每个 $(s, a)$ 对维护一个独立的 Q 值。当状态空间是高维的（如 Atari 游戏的 $210 \times 160 \times 3$ 像素画面），可能的状态数量是天文数字，表格方法完全不可行：

- 无法存储：状态数量远超内存容量
- 无法学习：大多数状态只会被访问极少次，Q 值估计极不准确
- 无法泛化：相似的状态（如画面只差几个像素）被当作完全不同的条目，学到的知识无法迁移

### 函数逼近的不稳定性

自然的解决方案是用参数化函数 $Q_\theta(s, a)$（如神经网络）替代表格，让网络学会在相似状态间泛化。但直接将神经网络塞进 Q-Learning 会导致训练极不稳定：

1. **样本相关性**：agent 按时间顺序与环境交互，连续样本高度相关（$s_t$ 和 $s_{t+1}$ 几乎一样），违反 SGD 的 i.i.d. 假设
2. **Moving target**：TD target $r + \gamma \max_{a'} Q_\theta(s', a')$ 依赖于正在更新的网络参数 $\theta$，每次更新都改变目标值，导致训练震荡甚至发散
3. **收敛保证失效**：tabular 情况下的收敛证明依赖于每个 $(s, a)$ 独立更新，函数逼近打破了这个前提——更新一个状态的 Q 值会影响所有状态的估计

> [!warning] 致命三角（Deadly Triad）
> Sutton & Barto 指出，以下三个要素同时存在时，TD 学习可能发散：
> 1. **函数逼近**（而非 tabular）
> 2. **Bootstrapping**（用估计值更新估计值，如 TD）
> 3. **Off-policy 学习**
>
> Q-Learning 天然具备后两个要素，加上函数逼近就凑齐了"致命三角"。这正是 [[DQN]] 需要 Experience Replay 和 Target Network 两大创新来稳定训练的根本原因。

## 从 Q-Learning 到 DQN 的演进

Q-Learning 到 [[DQN]] 的演进路径清晰地展示了"理论算法 → 实用算法"的工程化过程：

| 问题 | Q-Learning 的困境 | DQN 的解决方案 |
|------|-------------------|----------------|
| 高维状态空间 | 表格无法扩展 | 用深度神经网络逼近 $Q_\theta(s, a)$ |
| 样本相关性 | 连续样本破坏 i.i.d. | Experience Replay：随机采样历史经验 |
| Moving target | 目标值随参数变化 | Target Network：冻结目标网络，定期同步 |

这个演进的本质是：**Q-Learning 提供了正确的学习目标（Bellman 最优方程），DQN 提供了在大规模问题上稳定逼近这个目标的工程手段。** 算法的"灵魂"没变，变的是实现方式。

> [!interview] 面试要点
> **Q: Q-Learning 和 DQN 的本质区别是什么？**
> A: 学习目标完全相同——都是用 Bellman 最优方程迭代逼近 $Q^*$。区别在于 Q 函数的表示方式（表格 vs 神经网络）和为稳定训练引入的工程技巧（Experience Replay、Target Network）。
>
> **Q: Q-Learning 为什么是 off-policy 的？**
> A: 因为更新时用 $\max_{a'} Q(s', a')$（目标策略是 greedy），而数据收集用 $\varepsilon$-greedy（行为策略有探索）。两个策略不同，所以是 off-policy。关键是 $\max$ 操作让它不需要 importance sampling 修正。
>
> **Q: Q-Learning 和 SARSA 的区别？**
> A: 唯一区别是更新目标中下一步动作的选择：Q-Learning 用 $\max_{a'}$（最优动作），SARSA 用实际选择的 $a'$。这导致 Q-Learning 更激进（学最优路径），SARSA 更保守（考虑探索风险）。Cliff Walking 是经典对比案例。
>
> **Q: 为什么不能直接把神经网络塞进 Q-Learning？**
> A: 三个原因：样本相关性破坏 i.i.d.、moving target 导致训练震荡、函数逼近 + bootstrapping + off-policy 构成"致命三角"。DQN 的两大创新（Experience Replay 和 Target Network）分别解决前两个问题。

## 延伸阅读

**原始论文**：
- Watkins, 1989. *Learning from Delayed Rewards* — Q-Learning 的博士论文
- Watkins & Dayan, 1992. *Q-Learning* — 收敛性证明

**后续发展**：
- [[DQN]] — 用深度网络扩展 Q-Learning 到高维问题
- Double Q-Learning（van Hasselt, 2010）— 解决 Q 值过估计问题，后发展为 Double DQN
