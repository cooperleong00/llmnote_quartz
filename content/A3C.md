---
type: method
description: Actor-Critic 的异步并行实现，通过多个 worker 独立更新全局参数加速训练，是深度 RL 的经典算法
aliases:
  - Asynchronous Advantage Actor-Critic
  - 异步优势演员-评论家
prerequisites:
  - "[[Actor-Critic]]"
  - "[[Value Function]]"
  - "[[Policy Gradient]]"
tags:
  - reinforcement-learning
  - optimization
  - distributed-training
created: 2026-03-01
updated: 2026-03-01T16:33
---

# A3C (Asynchronous Advantage Actor-Critic)

A3C（Asynchronous Advantage Actor-Critic，异步优势演员-评论家）是 DeepMind 在 2016 年提出的深度强化学习算法。它通过**异步并行训练多个 agent**，让每个 worker 独立与环境交互并更新全局参数，在 CPU 多核环境下显著加速训练。A3C 是深度 RL 历史上的里程碑算法，证明了无需 Experience Replay 也能稳定训练深度网络。

## 动机

> [!intuition] 为什么需要异步并行？
> 在 A3C 之前，DQN 使用 Experience Replay 打破数据相关性，但这需要大量内存且不适用于 on-policy 算法。A3C 提出了另一种思路：
> - **并行探索**：多个 agent 同时与环境交互，天然产生多样化数据
> - **异步更新**：各 worker 独立更新全局参数，无需等待同步
> - **CPU 友好**：充分利用多核 CPU，无需昂贵的 GPU

核心洞察：**并行性本身就是去相关性的手段**——不同 worker 处于不同状态，产生的梯度自然多样化。

## 核心机制

### 异步架构


```
全局网络（共享参数 θ）
    ↓ 复制
┌───────┬───────┬───────┐
│Worker1│Worker2│Worker3│ ... Worker N
│  θ₁   │  θ₂   │  θ₃   │
└───┬───┴───┬───┴───┬───┘
    │       │       │
    ↓       ↓       ↓
  Env1    Env2    Env3
    │       │       │
    ↓       ↓       ↓
  采样    采样    采样
    │       │       │
    ↓       ↓       ↓
  计算梯度 ∇θ₁, ∇θ₂, ∇θ₃
    │       │       │
    └───────┴───────┴─→ 异步更新全局 θ
```

**关键特性**：
1. **独立采样**：每个 worker 维护自己的环境副本，独立收集轨迹
2. **异步更新**：worker 计算完梯度后立即更新全局参数，无需等待其他 worker
3. **参数同步**：worker 定期从全局网络复制最新参数

### Advantage Function

与 [[A2C]] 相同，使用 Advantage 降低方差：

$$
A(s, a) = Q(s, a) - V(s) = r + \gamma V(s') - V(s)
$$

实际中用 **n-step return** 估计：

$$
A(s_t, a_t) = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n}) - V(s_t)
$$

> [!intuition] 为什么用 n-step？
> - 1-step（TD）：低方差，高偏差
> - Monte Carlo：高方差，低偏差
> - n-step：在两者之间权衡，A3C 论文中 n=5 效果较好

### 训练流程（单个 Worker）

```python
# 伪代码
while True:
    # 1. 从全局网络复制参数
    θ_local = θ_global
    
    # 2. 采样 n 步轨迹
    trajectory = []
    for t in range(n_steps):
        a_t = π(s_t; θ_local)  # 根据当前策略采样动作
        s_t+1, r_t = env.step(a_t)
        trajectory.append((s_t, a_t, r_t))
    
    # 3. 计算梯度
    for (s, a, r) in reversed(trajectory):
        R = r + γ * R  # 累积 return
        A = R - V(s; θ_local)
        
        ∇θ_actor = ∇θ log π(a|s; θ) * A
        ∇θ_critic = ∇θ (R - V(s; θ))²
    
    # 4. 异步更新全局参数（需要加锁）
    θ_global ← θ_global + α * ∇θ
```

## 损失函数

与 [[A2C]] 相同的三部分损失：

### Actor Loss

$$
L_{\text{actor}} = -\log \pi(a_t | s_t; \theta) \cdot A(s_t, a_t)
$$

### Critic Loss

$$
L_{\text{critic}} = (R_t - V(s_t; \theta))^2
$$

### Entropy Bonus

$$
L_{\text{entropy}} = -\beta \sum_a \pi(a|s; \theta) \log \pi(a|s; \theta)
$$

**总损失**：

$$
L = L_{\text{actor}} + c_1 L_{\text{critic}} - c_2 L_{\text{entropy}}
$$

## A3C vs A2C

> [!comparison] 异步 vs 同步

| 特性 | A3C (异步) | A2C (同步) |
|------|-----------|-----------|
| **更新方式** | 各 worker 独立更新全局参数 | 所有 worker 同步后统一更新 |
| **梯度一致性** | 可能使用过时梯度 | 梯度始终基于最新参数 |
| **硬件适配** | CPU 多核友好 | GPU 大 batch 友好 |
| **实现复杂度** | 高（需要锁/无锁并发） | 低（标准数据并行） |
| **训练稳定性** | 较低（过时梯度引入噪声） | 较高 |
| **历史地位** | 2016 年 CPU 时代的突破 | 现代 GPU 时代的主流 |

> [!intuition] 为什么 A2C 在实践中更常用？
> A3C 的异步更新在 CPU 时代有优势（利用多核），但在 GPU 时代反而成为瓶颈：
> - **GPU 擅长大 batch 并行计算**，异步更新难以形成大 batch
> - **过时梯度引入额外噪声**，降低训练稳定性
> - **实现复杂**，需要处理并发锁或无锁数据结构
>
> [[A2C]] 的同步更新天然适合 GPU：收集所有 worker 的数据，形成大 batch，一次前向+反向。

## 局限性

> [!warning] 边界条件

1. **过时梯度问题**
   - Worker 计算梯度时使用的参数可能已被其他 worker 更新
   - 导致梯度方向不准确，训练不稳定
   - 解决：减小学习率，或改用同步更新（[[A2C]]）

2. **实现复杂度高**
   - 需要处理多线程/多进程并发
   - 全局参数更新需要加锁（或使用 Hogwild! 无锁更新）
   - 调试困难（非确定性行为）

3. **GPU 利用率低**
   - 异步更新难以形成大 batch
   - 无法充分发挥 GPU 并行计算优势
   - 现代实践中 [[A2C]] 更高效

4. **超参数敏感**
   - n-step 的选择影响 bias-variance tradeoff
   - 学习率需要仔细调整（过大会因过时梯度发散）
   - Entropy bonus 系数需要随训练调整

## 历史意义

> [!paper] 原始论文
> **Asynchronous Methods for Deep Reinforcement Learning**  
> Mnih et al., ICML 2016  
> [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)

A3C 的重要贡献：
1. **证明了无需 Experience Replay 也能稳定训练**：并行性本身就能打破数据相关性
2. **CPU 友好**：在 GPU 稀缺的 2016 年，让更多研究者能训练深度 RL
3. **启发了后续工作**：[[A2C]]、[[PPO]] 等都继承了 Actor-Critic + 并行采样的思想

> [!intuition] 为什么现在不常用 A3C？
> A3C 是特定时代的产物：
> - **2016 年**：GPU 昂贵，CPU 多核是主流 → 异步更新有优势
> - **2020 年后**：GPU 普及，大 batch 训练成为标准 → 同步更新（[[A2C]]）更高效
>
> 但 A3C 的核心思想（并行采样 + Actor-Critic）仍然是现代 RL 的基础。

## 面试要点

> [!interview] 常见问题

**Q1: A3C 和 A2C 的核心区别是什么？**

A: **更新方式不同**：
- A3C：各 worker 独立异步更新全局参数，可能使用过时梯度
- A2C：所有 worker 同步后统一更新，梯度始终基于最新参数

**实践中 A2C 更常用**，因为同步更新适合 GPU 大 batch 训练，训练更稳定。

**Q2: A3C 如何解决数据相关性问题？**

A: 通过**并行探索**：
- 多个 worker 同时与环境交互，处于不同状态
- 产生的数据天然多样化，打破时间相关性
- 无需像 DQN 那样使用 Experience Replay

**Q3: 为什么 A3C 在现代实践中不常用？**

A: 三个原因：
1. **GPU 利用率低**：异步更新难以形成大 batch
2. **过时梯度**：降低训练稳定性
3. **实现复杂**：需要处理并发，调试困难

[[A2C]] 的同步更新在 GPU 时代更高效。

**Q4: A3C 的 n-step return 如何选择？**

A: 权衡 bias-variance：
- **小 n**（如 n=1）：低方差，高偏差（依赖 V(s) 估计）
- **大 n**（如 n=∞）：高方差，低偏差（Monte Carlo）
- **中等 n**（如 n=5）：平衡两者，A3C 论文中效果较好

## 延伸阅读

**原始论文与深入材料**：
- [[Clippings/Paper/1602.01783v1|A3C 原始论文]] — Mnih et al., ICML 2016

**相关方法**：
- [[A2C]] — 同步版本，现代实践中更常用
- [[PPO]] — 基于 Actor-Critic 的改进，增加了 trust region 约束
- [[Actor-Critic]] — 基础架构
- [[Policy Gradient]] — 理论基础

**前置知识**：
- [[Value Function]] — Advantage function 的基础
- [[Variance Reduction]] — 为什么需要 Advantage
