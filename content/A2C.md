---
description: Actor-Critic 的同步并行实现，使用 Advantage function 降低方差，是理解 PPO 的基础
type: method
aliases:
  - Advantage Actor-Critic
  - 优势演员-评论家
prerequisites:
  - "[[Actor-Critic]]"
  - "[[Policy Gradient]]"
tags:
  - reinforcement-learning
  - post-training
created: 2026-02-11
updated: 2026-02-11T00:32
---

# A2C (Advantage Actor-Critic)

A2C（Advantage Actor-Critic）是 [[Actor-Critic]] 架构的标准同步实现。核心改进是使用 **Advantage function** 替代原始 Q 值作为策略梯度的权重，显著降低方差。多个 worker 并行采样、同步更新，比异步版本 A3C 更稳定，也更容易在 GPU 上高效实现。

> [!paper] 论文出处
> Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016
> （A2C 是 A3C 的同步变体，由 OpenAI 等团队在实践中推广）

---

## 动机

> [!intuition] 为什么需要 A2C？
> [[Policy Gradient|策略梯度]]的核心问题是**高方差**：用整条轨迹的累积回报 $G_t$ 作为权重，噪声太大。
>
> [[Actor-Critic]] 用 Critic 估计的 $Q(s,a)$ 替代 $G_t$，降低了方差，但 $Q$ 值本身包含状态的基线价值，仍有优化空间。
>
> A2C 的关键洞察：**只关心动作比平均水平好多少**，而非绝对好坏。这就是 Advantage function 的意义。

---

## 核心机制

### Advantage Function

> [!definition] 定义
> Advantage function 衡量某个动作相对于平均水平的优势：
> $$A(s, a) = Q(s, a) - V(s)$$
> - $Q(s, a)$：在状态 $s$ 采取动作 $a$ 的期望回报
> - $V(s)$：状态 $s$ 的平均价值（所有动作的期望）
> - $A(s, a) > 0$：动作优于平均，应该增加概率
> - $A(s, a) < 0$：动作劣于平均，应该降低概率

> [!intuition] 为什么 Advantage 比 Q 值更好？
> 假设两个状态：
> - 状态 A：所有动作的 Q 值都在 100 左右
> - 状态 B：所有动作的 Q 值都在 10 左右
>
> 用 Q 值作为权重，状态 A 的梯度会被放大 10 倍，但这只是因为状态 A 本身价值高，与动作选择无关。
>
> Advantage 减去了状态的基线价值 $V(s)$，让梯度只反映"这个动作相对于其他动作有多好"。

### 实际估计：TD Error

直接计算 $A = Q - V$ 需要同时估计 Q 和 V，开销大。实践中用 **TD error** 近似：

$$\hat{A}(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$

这是 Advantage 的无偏估计（在期望意义下），只需要一个 Critic 网络估计 $V$。

> [!math] 为什么 TD error 是 Advantage 的估计？
> $$\mathbb{E}[r_t + \gamma V(s_{t+1}) | s_t, a_t] = Q(s_t, a_t)$$
> 因此：
> $$\mathbb{E}[\hat{A}] = Q(s_t, a_t) - V(s_t) = A(s_t, a_t)$$

更好的估计方法是 [[GAE]]，通过 $\lambda$ 参数平衡 bias 和 variance。

---

## 同步并行架构

### A2C vs A3C

| 特性 | A3C (异步) | A2C (同步) |
|------|-----------|-----------|
| **更新方式** | 各 worker 独立更新全局参数 | 所有 worker 同步后统一更新 |
| **梯度一致性** | 可能使用过时梯度 | 梯度始终基于最新参数 |
| **GPU 利用率** | 低（异步难以 batch） | 高（可以 batch 所有 worker 的数据） |
| **实现复杂度** | 高（需要锁/无锁并发） | 低（标准数据并行） |
| **训练稳定性** | 较低 | 较高 |

> [!intuition] 为什么 A2C 在实践中更常用？
> A3C 的异步更新在 CPU 时代有优势（利用多核），但在 GPU 时代反而成为瓶颈：
> - GPU 擅长大 batch 并行计算
> - 异步更新难以形成大 batch
> - 过时梯度引入额外噪声
>
> A2C 的同步更新天然适合 GPU：收集所有 worker 的数据，形成大 batch，一次前向+反向。

### 训练流程

```
1. 初始化：N 个并行环境，共享 Actor-Critic 网络

2. 采样阶段（并行）：
   for each worker i in 1..N:
       收集 T 步轨迹 (s, a, r, s')

3. 同步点：等待所有 worker 完成

4. 更新阶段：
   - 计算所有轨迹的 Advantage（用 TD error 或 GAE）
   - 合并为一个大 batch
   - 计算 Actor loss 和 Critic loss
   - 一次梯度更新

5. 重复 2-4
```

---

## 损失函数

### Actor Loss

$$L_{\text{actor}} = -\mathbb{E}\left[\log \pi_\theta(a|s) \cdot \hat{A}(s, a)\right]$$

这是标准的策略梯度目标，用 Advantage 作为权重。

### Critic Loss

$$L_{\text{critic}} = \mathbb{E}\left[(V_\phi(s) - V_{\text{target}})^2\right]$$

其中 $V_{\text{target}} = r + \gamma V(s')$（TD target）或 $G_t$（Monte Carlo target）。

### Entropy Bonus

为了鼓励探索，通常加入熵正则化：

$$L_{\text{total}} = L_{\text{actor}} + c_1 L_{\text{critic}} - c_2 H(\pi)$$

其中 $H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$ 是策略的熵。

---

## 局限性

> [!warning] A2C 的边界
> 1. **更新幅度无约束**：策略可能在一次更新中变化过大，导致训练不稳定
> 2. **样本效率低**：on-policy 方法，每次更新后数据就过时了
> 3. **超参数敏感**：学习率、entropy 系数等需要仔细调节
>
> 这些问题催生了 [[PPO]]：通过 clip 机制限制更新幅度，显著提升稳定性。

---

## 与 PPO 的关系

A2C 是理解 [[PPO]] 的基础。PPO 在 A2C 的基础上增加了一个关键改进：

> [!comparison] A2C vs PPO
> | 方面 | A2C | PPO |
> |------|-----|-----|
> | **目标函数** | 标准策略梯度 | Clipped surrogate objective |
> | **更新约束** | 无 | 限制概率比在 $[1-\epsilon, 1+\epsilon]$ |
> | **稳定性** | 一般 | 高 |
> | **样本复用** | 不能 | 可以多次更新（mini-batch） |
>
> PPO 的 clip 机制本质上是在说："如果这次更新让策略变化太大，就截断梯度"。

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: A2C 中的 "Advantage" 是什么意思？**
> A: Advantage function $A(s,a) = Q(s,a) - V(s)$，衡量某个动作相对于平均水平的优势。用 Advantage 替代 Q 值作为策略梯度的权重，可以减去状态的基线价值，降低方差。
>
> **Q2: A2C 和 A3C 有什么区别？**
> A: A3C 是异步更新（各 worker 独立更新全局参数），A2C 是同步更新（等所有 worker 采样完再统一更新）。A2C 更稳定、更适合 GPU 并行，是目前更常用的版本。
>
> **Q3: 为什么 PPO 比 A2C 更稳定？**
> A: A2C 的策略更新没有约束，可能一次更新变化太大。PPO 通过 clip 机制限制概率比在 $[1-\epsilon, 1+\epsilon]$，防止策略剧烈变化。

---

## 延伸阅读

**前置与基础**：
- [[Actor-Critic]] — A2C 的架构基础
- [[Policy Gradient]] — 理论基础
- [[GAE]] — 更好的 Advantage 估计方法

**后续发展**：
- [[PPO]] — A2C 的改进版，加入 clip 约束
- [[A3C]] — A2C 的异步版本（stub）
- [[TRPO]] — 用 KL 约束替代 clip 的方法
