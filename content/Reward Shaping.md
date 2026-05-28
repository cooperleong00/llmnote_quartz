---
type: concept
description: 通过添加辅助奖励信号引导学习过程，在不改变最优策略的前提下加速收敛
aliases:
  - 奖励塑形
  - Reward Engineering
prerequisites:
  - "[[Reward Model]]"
  - "[[Policy Gradient]]"
  - "[[Credit Assignment]]"
tags:
  - reinforcement-learning
  - rlhf
  - alignment
created: 2026-03-01
updated: 2026-03-01T17:13
---

# Reward Shaping

Reward Shaping（奖励塑形）是一种通过**添加辅助奖励信号**来引导学习过程的技术，目标是在**不改变最优策略**的前提下，加速收敛并改善探索效率。它是设计者有意为之的引导机制，而非模型自发的行为。

---

## 动机

> [!intuition] 为什么需要 Reward Shaping？
>
> 强化学习面临两大核心挑战：
> 1. **稀疏奖励**：只有在任务完成时才有奖励信号，中间步骤没有反馈
> 2. **Credit Assignment 困难**：难以判断哪些决策对最终结果有贡献
>
> 这导致学习效率极低——agent 需要大量随机探索才能偶然获得奖励。
>
> **Reward Shaping 的核心思想**：在原始奖励基础上添加"路标"，告诉 agent "你正在朝正确方向前进"，而不是等到最后才给反馈。

**关键问题**：如何添加辅助奖励而不改变最优策略？

如果随意添加奖励，可能会：
- 引入新的局部最优
- 改变任务的本质目标
- 导致 agent 优化错误的目标

这就需要 **Potential-Based Shaping** 的理论保证。

---

## 核心机制

### Potential-Based Reward Shaping

> [!math] 数学形式
>
> 定义 **shaping function** $F(s, s')$：
> $$F(s, s') = \gamma \Phi(s') - \Phi(s)$$
>
> 其中：
> - $\Phi(s)$：状态 $s$ 的 **potential function**（势函数）
> - $\gamma$：折扣因子
> - $s, s'$：当前状态和下一状态
>
> **Shaped reward**：
> $$\tilde{r}(s, a, s') = r(s, a, s') + F(s, s')$$
>
> **关键性质**：$\tilde{r}$ 和 $r$ 具有**相同的最优策略**。

> [!intuition] 为什么这个形式保证策略不变？
>
> 直觉：$F(s, s')$ 是一个"势能差"——就像物理中的重力势能，只依赖于起点和终点，与路径无关。
>
> 数学上：对于任意轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$，shaping reward 的累积和是：
>
> $$\sum_{t=0}^{T-1} F(s_t, s_{t+1}) = \gamma \Phi(s_T) - \Phi(s_0)$$
>
> 这只依赖于**起点和终点**，与中间路径无关。因此，不同策略之间的相对优劣不变。

### 设计 Potential Function 的原则

好的 $\Phi(s)$ 应该：
1. **反映任务进度**：越接近目标，$\Phi(s)$ 越大
2. **可计算**：能够高效评估
3. **领域知识**：利用人类对任务的理解

**示例**：
- 迷宫导航：$\Phi(s) = -\text{distance\_to\_goal}(s)$
- 游戏：$\Phi(s) = \text{estimated\_score}(s)$
- LLM 生成：$\Phi(s) = \text{KL\_budget\_remaining}(s)$（见下文）

---

## 在 LLM RLHF 中的应用

### 1. KL Penalty 作为 Reward Shaping

在 [[RLHF]] 中，常见的做法是在 reward 中加入 KL 惩罚：

$$r_{\text{shaped}}(x, y) = r_{\text{RM}}(x, y) - \beta \cdot \text{KL}(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x))$$

> [!intuition] KL Penalty 的双重作用
>
> 1. **约束作用**：防止策略偏离 reference policy 太远（避免 [[Reward Hacking]]）
> 2. **Shaping 作用**：提供 token-level 的即时反馈
>
> 每生成一个 token，模型就知道"我偏离了多少"，而不是等到整个序列完成才知道。

**Token-Level 分解**：

$$r_{\text{shaped}}(x, y_{1:T}) = r_{\text{RM}}(x, y_{1:T}) - \beta \sum_{t=1}^{T} \text{KL}_t$$

其中 $\text{KL}_t = \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})}$

这可以视为一种 **potential-based shaping**，其中：
- $\Phi(s_t) = -\beta \cdot \text{KL\_divergence\_so\_far}(s_t)$
- 每个 token 的 shaping reward 反映"这一步偏离了多少"

### 2. Length Penalty

许多 LLM RL 方法（如 [[DAPO]]）使用 length penalty 来控制生成长度：

$$r_{\text{shaped}} = r_{\text{RM}} - \lambda \cdot (\text{length} - \text{target\_length})^2$$

这引导模型生成适当长度的回答，避免过长或过短。

### 3. Process Reward Model (PRM)

[[Process Reward Model]] 可以视为一种**学习到的 shaping function**：

- 传统 RM：只在序列结束时给出 $r(y_{1:T})$
- PRM：在每个推理步骤给出 $r_t$，提供中间反馈

这解决了 [[Credit Assignment]] 问题，但需要额外的 step-level 标注数据。

---

## 与 Reward Hacking 的区别

| 维度 | Reward Shaping | [[Reward Hacking]] |
|------|----------------|-------------------|
| **意图** | 设计者有意引导 | 模型自发钻空子 |
| **目标一致性** | 保持最优策略不变 | 改变了实际优化目标 |
| **可控性** | 可预测、可调节 | 难以预测、难以修复 |
| **示例** | KL penalty 引导探索 | 生成冗长无用的回答获得高分 |

> [!warning] Shaping 也可能出错
>
> 如果 potential function 设计不当，Reward Shaping 也可能引入问题：
> - **过度引导**：模型过度依赖 shaping reward，忽略真实任务目标
> - **引入偏见**：shaping function 反映了设计者的偏见
> - **计算开销**：复杂的 $\Phi(s)$ 可能降低训练效率

---

## 局限性

### 1. 需要领域知识

设计好的 potential function 需要对任务有深入理解，这在复杂任务中很困难。

### 2. 可能引入新的局部最优

如果 $\Phi(s)$ 不是 potential-based 形式，可能改变最优策略。

**反例**：
```python
# ❌ 错误的 shaping（不是 potential-based）
shaped_reward = original_reward + bonus_for_visiting_state_A
```

这会导致 agent 反复访问 state A 来获得 bonus，即使这不是最优策略。

### 3. 超参数敏感

Shaping 的强度（如 KL penalty 的 $\beta$）需要仔细调节：
- 太小：shaping 效果不明显
- 太大：主导了原始 reward，改变了任务目标

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: Reward Shaping 和 Reward Hacking 有什么区别？**
>
> A: Reward Shaping 是设计者有意添加的辅助奖励，目标是加速学习而不改变最优策略（通过 potential-based 形式保证）。Reward Hacking 是模型自发找到的奖励函数漏洞，导致优化了错误的目标。
>
> **Q2: 为什么 Potential-Based Shaping 不改变最优策略？**
>
> A: 因为 $F(s, s') = \gamma \Phi(s') - \Phi(s)$ 在整个轨迹上的累积和只依赖于起点和终点，与路径无关。因此不同策略之间的相对优劣保持不变。
>
> **Q3: KL Penalty 在 RLHF 中是 Reward Shaping 吗？**
>
> A: 是的，KL penalty 既是约束机制（防止偏离 reference policy），也是 reward shaping（提供 token-level 即时反馈）。它可以视为一种 potential-based shaping，其中 potential function 是累积的 KL divergence。
>
> **Q4: Reward Shaping 有什么风险？**
>
> A: 主要风险是设计不当的 shaping function 可能：(1) 引入新的局部最优；(2) 过度主导原始 reward；(3) 反映设计者的偏见。需要通过 potential-based 形式和仔细的超参数调节来缓解。

---

## 延伸阅读

**理论基础**：
- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. ICML.

**在 LLM 中的应用**：
- [[Credit Assignment]] — Reward Shaping 解决的核心问题之一
- [[Process Reward Model]] — 学习到的 shaping function
- [[DAPO]] — 使用 length penalty 和 overlong reward shaping

**相关概念**：
- [[Reward Hacking]] — Shaping 的对立面
- [[KL Divergence]] — KL penalty 的数学基础
- [[Policy Gradient]] — Shaping 作用于的优化目标
