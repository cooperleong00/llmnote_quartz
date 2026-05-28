---
description: 强化学习的核心难题，如何将延迟的序列级奖励分配到每个决策步骤，在 LLM RLHF 中表现为将 sequence-level reward 分配到每个 token
type: concept
aliases:
  - 信用分配
  - 信用分配问题
  - Temporal Credit Assignment
prerequisites:
  - "[[Policy Gradient]]"
  - "[[Actor-Critic]]"
  - "[[Reward Model]]"
tags:
  - reinforcement-learning
  - rlhf
created: 2026-01-29
updated: 2026-01-31T22:42
---

# Credit Assignment

信用分配（Credit Assignment）是强化学习的核心难题之一：当 agent 在一系列决策后获得奖励时，**如何判断哪些决策对最终结果贡献最大**？这个问题在 LLM 的 [[RLHF]] 训练中尤为突出——Reward Model 只在序列末尾给出一个分数，但我们需要更新每个 token 的生成概率。

> [!intuition] 核心直觉
> 想象一场足球比赛，最终比分是 3:2 获胜。这个"胜利"的功劳应该如何分配？是进球的前锋？还是关键传球的中场？还是扑出点球的守门员？
>
> 在 LLM 生成中，一个"好回答"的功劳应该归于哪些 token？开头的格式？中间的关键论点？还是结尾的总结？

---

## 问题定义

> [!definition] Credit Assignment Problem
> 给定一个轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ 和最终奖励 $R(\tau)$，如何估计每个动作 $a_t$ 对 $R(\tau)$ 的贡献？

在 LLM 场景下：
- **轨迹**：生成的 token 序列 $y = (y_1, y_2, \ldots, y_T)$
- **动作**：每个 token 的选择
- **奖励**：Reward Model 对完整回答的评分 $r(x, y)$

**核心困难**：
1. **延迟奖励（Delayed Reward）**：奖励只在序列结束时给出
2. **稀疏反馈（Sparse Feedback）**：中间步骤没有直接监督信号
3. **因果模糊（Causal Ambiguity）**：难以区分"好 token"和"碰巧出现在好序列中的 token"

---

## 为什么 Credit Assignment 在 LLM 中特别困难？

### 1. Sequence-Level Reward 的本质限制

[[Reward Model]] 的设计决定了它只能给出 sequence-level 的评分：

```
输入: prompt x + response y
输出: 单个标量 r(x, y)
```

这个设计有其合理性——人类标注员也是看完整回答后给出偏好判断。但这导致了严重的信息损失：

> [!warning] 信息瓶颈
> 一个 500 token 的回答，只得到 1 bit 的反馈（好/坏）。平均每个 token 只有 0.002 bits 的监督信号。
>
> 对比 [[SFT]]：每个 token 都有明确的目标分布，监督信号密度高几个数量级。

### 2. Token 之间的复杂依赖

LLM 生成是自回归的，token 之间存在强依赖：

- 早期 token 决定了后续 token 的"可能空间"
- 一个关键 token 的错误可能导致整个推理链崩溃
- 但"关键 token"往往不是最后一个 token

> [!example] 数学推理示例
> ```
> 问题：计算 2^10
> 回答：2^10 = 2 × 2 × ... = 1024  ✓
> 错误回答：2^10 = 2 × 10 = 20  ✗
> ```
> 错误发生在"×"这个 token，但 Reward Model 只在看到"20"后才给出低分。如何让模型知道是"×"而非"20"导致了错误？

### 3. 与传统 RL 的差异

| 维度 | 传统 RL（如 Atari） | LLM RLHF |
|------|---------------------|----------|
| **动作空间** | 小（几十个） | 巨大（词表大小，~100K） |
| **序列长度** | 可变，通常较短 | 可变，可能很长（数千 token） |
| **奖励频率** | 每步或频繁 | 仅序列末尾 |
| **状态表示** | 低维或图像 | 高维文本嵌入 |

---

## 解决方案

### 方案 1：Value Function + GAE

这是 [[PPO]] 在 [[RLHF]] 中的标准做法。

**核心思想**：训练一个 Critic 网络 $V(s_t)$ 来估计"从当前状态开始，期望获得多少奖励"。

> [!math] 优势函数估计
> 使用 [[GAE]]（Generalized Advantage Estimation）：
> $$
> \hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}
> $$
> 其中 TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

**直觉**：
- $V(s_t)$ 告诉我们"在生成到第 $t$ 个 token 时，预期最终能得多少分"
- $\hat{A}_t > 0$ 意味着"选择 token $y_t$ 比预期更好"
- $\hat{A}_t < 0$ 意味着"选择 token $y_t$ 比预期更差"

**局限性**：
1. **Value function 训练困难**：在 sparse reward 下，$V(s)$ 很难准确估计
2. **计算开销大**：需要额外的 Critic 网络
3. **仍然是近似**：GAE 只是启发式地分配 credit，不是真正的因果归因

### 方案 2：Group Relative Advantage（GRPO）

[[GRPO]] 的核心洞察：**不需要绝对的 value 估计，只需要相对比较**。

> [!math] GRPO 的 Advantage 计算
> 对同一个 prompt 采样多个回答 $\{y_1, \ldots, y_G\}$，计算：
> $$
> \hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}
> $$

**直觉**：
- 不需要知道"这个回答值多少分"
- 只需要知道"这个回答比同组其他回答好还是差"
- 通过组内比较，隐式地做了 credit assignment

**局限性**：
- 所有 token 共享同一个 advantage（sequence-level）
- 无法区分"关键 token"和"普通 token"

### 方案 3：Process Reward Model（PRM）

直接训练一个能给出 **step-level** 或 **token-level** 奖励的模型。

> [!definition] Process Reward Model
> 不同于 Outcome RM 只评估最终结果，PRM 评估每个中间步骤：
> $$
> r_{PRM}(x, y_{1:t}) \quad \text{for each } t
> $$

**优点**：
- 提供 dense reward signal
- 可以识别"哪一步出错了"
- 更符合人类的评估方式（逐步检查推理）

**缺点**：
- 标注成本极高（需要标注每个步骤）
- 定义"步骤"本身就很困难
- 可能引入新的 [[Reward Hacking]] 模式

### 方案 4：Dense Distillation（On-Policy Distillation）

[[On-Policy Distillation]] 提供了一种绕过 credit assignment 的思路：

> [!intuition] 核心洞察
> 与其让模型自己猜"哪个 token 好"，不如直接告诉它"每个 token 应该怎么改"。

教师模型提供 **token-level** 的监督信号：
$$
\mathcal{L} = \sum_t D_{KL}(\pi_{teacher}(\cdot|x, y_{<t}) \| \pi_{student}(\cdot|x, y_{<t}))
$$

**对比**：

| 方法 | 信号密度 | Credit Assignment |
|------|----------|-------------------|
| RLHF (sparse reward) | O(1) per sequence | 困难，需要 GAE/Value function |
| On-Policy Distillation | O(T) per sequence | 不需要，直接 token-level 监督 |

> [!paper] SDPO 的创新
> [[260120802v1|Self-Distillation Policy Optimization]]通过让模型"回顾"自己的生成，将 sequence-level feedback 转化为 **logit-level** 的 dense credit assignment，无需外部教师。

---

## 信用分配的粒度谱系

```
粗粒度                                                    细粒度
   │                                                        │
   ▼                                                        ▼
Sequence-Level ──► Step-Level ──► Token-Level ──► Logit-Level
   │                   │              │               │
   │                   │              │               │
 GRPO              Process RM    Token Reward    On-Policy
 DPO               Math-Shepherd   Shaping      Distillation
```

**权衡**：
- **粗粒度**：实现简单，但信号稀疏，学习效率低
- **细粒度**：信号密集，但获取成本高，可能引入噪声

---

## 在不同方法中的体现

### PPO + GAE

```python
# 标准 RLHF 中的 credit assignment
def compute_advantages(rewards, values, gamma, lam):
    """
    rewards: [r_T] (只在最后一步有 reward)
    values: [V(s_0), V(s_1), ..., V(s_T)]
    """
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        # 中间步骤 r_t = 0，只有最后一步有 reward
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages
```

### GRPO

```python
# GRPO 的简化 credit assignment
def grpo_advantage(rewards_group):
    """
    rewards_group: [r_1, r_2, ..., r_G] (G 个回答的 reward)
    所有 token 共享同一个 advantage
    """
    mean_r = np.mean(rewards_group)
    std_r = np.std(rewards_group)
    advantages = [(r - mean_r) / (std_r + eps) for r in rewards_group]
    return advantages
```

### Token-Level Reward Shaping

一种启发式方法是将 sequence reward 均匀或按某种规则分配到每个 token：

```python
# 简单的均匀分配
def uniform_credit(sequence_reward, seq_length):
    return [sequence_reward / seq_length] * seq_length

# 基于 KL 的分配（惩罚偏离 reference 的 token）
def kl_shaped_reward(sequence_reward, log_probs, ref_log_probs, beta):
    kl_penalties = beta * (log_probs - ref_log_probs)
    # 最后一个 token 获得 sequence reward
    rewards = [-kl for kl in kl_penalties]
    rewards[-1] += sequence_reward
    return rewards
```

---

## 前沿研究方向

### 1. Forking Tokens

[[250601939v2|80/20 Rule RLVR]] 的研究发现：并非所有 token 对最终结果同等重要。

> [!intuition] 80/20 法则的打破
> 在推理任务中，只有约 20% 的高熵 token（"分叉点"）真正决定了推理的走向。其余 80% 的 token 是"确定性"的，credit assignment 对它们意义不大。

这启发了更精细的 credit assignment 策略：
- 识别高熵 token
- 在这些关键位置投入更多的学习资源

### 2. Hindsight Credit Assignment

让模型"事后回顾"，基于最终结果重新评估每个决策：

- **SDPO**：用 self-teacher 的回顾性分析做 dense credit assignment
- **Outcome-Conditioned Generation**：条件化最终结果来指导中间步骤

### 3. Causal Credit Assignment

使用因果推断的方法来识别"真正导致结果的 token"：

- 反事实分析：如果这个 token 不同，结果会怎样？
- 干预实验：强制改变某个 token，观察结果变化

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: 什么是 Credit Assignment Problem？**
> A: 当 agent 在一系列决策后获得奖励时，如何判断哪些决策对最终结果贡献最大。在 LLM RLHF 中，表现为如何将 sequence-level reward 分配到每个 token。
>
> **Q2: 为什么 Credit Assignment 在 LLM 中特别困难？**
> A: 三个原因：(1) Reward Model 只给 sequence-level 反馈，信号极度稀疏；(2) Token 之间存在复杂依赖，早期错误会传播；(3) 动作空间巨大（词表大小），序列可能很长。
>
> **Q3: PPO 如何解决 Credit Assignment？**
> A: 通过 Value Function + GAE。训练 Critic 估计每个状态的价值 $V(s)$，用 GAE 计算优势函数 $\hat{A}_t$，作为每个 token 的更新权重。但在 sparse reward 下 Value function 很难准确估计。
>
> **Q4: GRPO 如何简化 Credit Assignment？**
> A: 放弃 token-level 的 credit assignment，改用 group relative advantage。对同一 prompt 采样多个回答，用组内相对排名作为 advantage。所有 token 共享同一个 advantage，简单但粗糙。
>
> **Q5: On-Policy Distillation 为什么不需要 Credit Assignment？**
> A: 因为教师模型直接提供 token-level 的监督信号（每个位置的目标分布），不需要从 sparse reward 反推每个 token 的贡献。这是它比 RL 高效 50-100 倍的根本原因。

---

## 相关概念

- [[GAE]] — PPO 中用于 credit assignment 的核心技术
- [[Actor-Critic]] — Value function 在 credit assignment 中的作用
- [[PPO]] — RLHF 中的标准 RL 算法，依赖 GAE 做 credit assignment
- [[GRPO]] — 用 group relative advantage 简化 credit assignment
- [[RLHF]] — Credit assignment 问题的主要应用场景
- [[Reward Model]] — 提供 sequence-level reward 的组件
- [[On-Policy Distillation]] — 通过 dense supervision 绕过 credit assignment
- [[Forking Tokens]] — 识别关键 token 的研究方向
- [[Process Reward Model]] — 提供 step-level reward 的方法
