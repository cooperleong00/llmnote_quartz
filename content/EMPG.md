---
type: method
description: 通过 entropy 调制 policy gradient 解决 long-horizon LLM agent 训练中的 credit assignment 问题，包含 Self-Calibrating Gradient Scaling 和 Future Clarity Bonus 两个核心机制
aliases:
  - Entropy-Modulated Policy Gradients
  - 熵调制策略梯度
prerequisites:
  - "[[Policy Gradient]]"
  - "[[Entropy]]"
  - "[[Credit Assignment]]"
  - "[[GRPO]]"
tags:
  - post-training
  - rl
  - agent
created: 2026-03-06
updated: 2026-03-06T02:43
---

# EMPG

Entropy-Modulated Policy Gradients (EMPG) 是一种专门为 long-horizon LLM agent 训练设计的 policy gradient 框架。它解决了一个基本问题：policy gradient 的幅度与 [[Entropy|entropy]] 天然耦合，导致 confident correct actions 获得小更新（学习慢），而 uncertain exploratory steps 产生大噪声梯度（不稳定）。

EMPG 通过两个互补机制重新校准学习信号：**Self-Calibrating Gradient Scaling** 调整每步更新的幅度，**Future Clarity Bonus** 引导 agent 找到更可预测的解决路径。

## 动机

> [!intuition] 为什么需要 EMPG？
>
> 在 long-horizon agent 任务中（如 WebShop、ALFWorld），agent 需要执行多步推理和行动，但只在任务结束时获得稀疏的 outcome reward。这带来两个核心挑战：
>
> 1. **Credit Assignment 困难**：如何将 sequence-level reward 分配到每个决策步骤？哪些步骤是关键的？
> 2. **Gradient 幅度与 Entropy 的耦合**：标准 policy gradient 中，高 entropy（不确定）的 action 自然产生大梯度，低 entropy（自信）的 action 产生小梯度
>
> 这导致：
> - Confident correct actions 应该被强化，但小梯度限制了学习速度
> - Uncertain exploratory steps 产生大噪声梯度，可能导致训练不稳定
>
> 现有方法（[[GRPO]]、[[DAPO]]）使用 group relative advantage 简化训练，但仍然对所有步骤施加均匀的学习信号，忽略了不同步骤的不确定性差异。

## 理论基础

> [!math] Proposition 1: Gradient Norm 与 Entropy 的耦合
>
> 对于 softmax policy $\pi_\theta$，score function 的期望范数是 entropy 的单调函数：
>
> $$
> \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} \left[ \|\nabla_{z_\theta(s)} \log \pi_\theta(a|s)\|^2 \right] = 1 - \exp(-H_2(\pi))
> $$
>
> 其中 $H_2(\pi)$ 是 Rényi-2 entropy，$z_\theta(s)$ 是 logits。
>
> **含义**：
> - 高 entropy（不确定）→ 大梯度范数
> - 低 entropy（自信）→ 小梯度范数
>
> 这揭示了需要显式地基于 action 的不确定性重新校准学习信号。


## 核心机制

EMPG 通过修改 advantage function 来重新校准学习信号。对于轨迹 $\tau_i$ 中的每个步骤 $t$，定义 modulated advantage：

$$
A_{\text{mod}}(i,t) = \underbrace{A^{(i)} \cdot g(H_t^{(i)})}_{\text{self-calibrating gradient scaling}} + \underbrace{\zeta \cdot f(H_{t+1}^{(i)})}_{\text{future clarity bonus}}
$$

其中：
- $A^{(i)}$：轨迹级别的 advantage（如 [[GRPO]] 中的 group relative advantage）
- $H_t^{(i)}$：步骤 $t$ 的 entropy（量化不确定性）
- $g(H)$：gradient scaling function
- $f(H)$：future clarity bonus function
- $\zeta$：控制 intrinsic bonus 权重的超参数

### 1. Self-Calibrating Gradient Scaling $g(H)$

> [!intuition] 核心思想
>
> 动态调整每步更新的幅度，对抗 entropy-gradient 耦合：
> - **Confident steps**（低 entropy）→ 放大更新（$g(H) > 1$）
> - **Uncertain steps**（高 entropy）→ 衰减更新（$g(H) < 1$）
>
> 关键设计：**self-calibrating** — 函数在 mini-batch 上的均值归一化为 1，确保调制是重新分配学习信号，而非简单地放大或缩小。

> [!math] 数学形式
>
> **Step 1: 量化步骤级别的不确定性**
>
> 对于一个 "reason-then-act" 步骤 $\text{step}_t = \{w_1, \ldots, w_m\}$，计算平均 token-level entropy：
>
> $$
> H_t = \frac{1}{m} \sum_{j=1}^m H(p(\cdot | w_{<j})) = -\frac{1}{m} \sum_{j=1}^m \sum_{v \in V} p(v|w_{<j}) \log p(v|w_{<j})
> $$
>
> **Step 2: Batch-level entropy normalization**
>
> 使用 min-max scaling 将 entropy 归一化到 [0, 1]：
>
> $$
> H_{\text{norm},t}^{(i)} = \frac{H_t^{(i)} - \min_{\text{batch}}(H)}{\max_{\text{batch}}(H) - \min_{\text{batch}}(H) + \epsilon}
> $$
>
> **Step 3: Self-calibrating scaling function**
>
> $$
> g(H_t^{(i)}) = \frac{\exp(-k \cdot H_{\text{norm},t}^{(i)})}{\frac{1}{\sum_{j=1}^{N_B} T_j} \sum_{j=1}^{N_B} \sum_{t'=1}^{T_j} \exp(-k \cdot H_{\text{norm},t'}^{(j)})}
> $$
>
> 其中：
> - $k > 0$：控制调制强度的超参数
> - 分母确保 $\mathbb{E}[g(H)] = 1$（self-calibrating 特性）
>
> **效果**：
> - $H_t < \text{batch average}$ → $g(H_t) > 1$ → 放大更新
>   - 如果 $A^{(i)} > 0$（正确）：加速学习
>   - 如果 $A^{(i)} < 0$（错误）：强惩罚 "hallucinated confidence"
> - $H_t > \text{batch average}$ → $g(H_t) < 1$ → 衰减更新，稳定探索

### 2. Future Clarity Bonus $f(H)$

> [!intuition] 核心思想
>
> 重新校准梯度幅度只是一半的解决方案，还需要引导 agent 朝着有用的方向学习。Future Clarity Bonus 提供内在动机，鼓励 agent 选择能导致**更可预测的下一步状态**的 action。
>
> 这对应于一个局部的、逐步的目标：最小化下一步的 policy entropy
>
> $$
> \min_{a_t} H(\pi_\theta(s_{t+1}))
> $$
>
> 这引导 agent 远离高 entropy、混乱的轨迹，转向具有更高 clarity 的状态。

> [!math] 数学形式
>
> $$
> f(H_{t+1}^{(i)}) = \exp(-k' \cdot H_{\text{norm},t+1}^{(i)})
> $$
>
> 其中：
> - $k' > 0$：控制 bonus 强度的超参数
> - $H_{t+1}$：下一步的 entropy（未来的不确定性）
>
> **效果**：
> - 下一步 entropy 低（清晰）→ 高 bonus → 鼓励这样的 action
> - 下一步 entropy 高（混乱）→ 低 bonus → 不鼓励
>
> 这提供了一个**内在信号**用于有目的的探索，而非依赖外部 reward。

> [!comparison] 与 Empowerment 的联系
>
> Future Clarity Bonus 与 Empowerment 框架的原则一致：agent 应该主动寻求能够最大化未来控制能力的状态。通过鼓励低 entropy 的下一步，agent 学会了一个可泛化的 meta-skill：**在模糊中主动寻求清晰**。

## 算法流程

```
Algorithm: EMPG

Input: Policy π_θ
Hyperparameters: k, k', ζ

for each training iteration:
    1. 收集一批轨迹 B = {τ_i} by running π_θ
    2. 计算 outcome-based advantages A^(i) for each τ_i
    3. 计算所有步骤的 step-level entropies {H_t}
    4. Batch-level entropy normalization: {H_t} → {H_norm,t}
    5. 计算 modulated advantages:
       A_mod(i,t) = A^(i) · g(H_t^(i)) + ζ · f(H_{t+1}^(i))
    6. Final advantage normalization (optional)
    7. 使用 A_mod 更新 policy:
       ∇_θ J ≈ Σ_i Σ_t A_mod(i,t) ∇_θ log π_θ(a_t | s_t)
```

> [!warning] 实现细节
>
> - **Entropy 计算**：在 "reason-then-act" 步骤级别，而非 token 级别
> - **Normalization**：先做 batch-level entropy normalization，再做 final advantage normalization
> - **Hyperparameters**：
>   - $k$：gradient scaling 强度（论文中使用 $k=1$）
>   - $k'$：future clarity bonus 强度
>   - $\zeta$：intrinsic bonus 权重

## 与相关方法的对比

> [!comparison] EMPG vs GRPO/DAPO
>
> - **[[GRPO]]**：使用 group relative advantage 简化训练，但对所有步骤施加均匀学习信号
> - **[[DAPO]]**：改进 GRPO 的稳定性（Clip-Higher、Dynamic Sampling），但仍然是均匀信号
> - **EMPG**：在 GRPO/DAPO 基础上，根据每步的不确定性动态调制学习信号
>
> EMPG 可以作为 GRPO/DAPO 的增强模块，实验显示 "GRPO + EMPG" 和 "DAPO + EMPG" 都显著提升性能。

> [!comparison] EMPG vs EDGE-GRPO
>
> - **EDGE-GRPO**：在单轮数学推理中，用 entropy 调制梯度，解决 confidence misalignment 问题
> - **EMPG**：
>   - **动机不同**：针对 multi-step long-horizon 任务的 credit assignment 问题
>   - **范围不同**：动态分配整个轨迹的 credit，而非单轮响应
>   - **机制不同**：增加 Future Clarity Bonus，引导探索方向
>
> EDGE-GRPO 关注"自信但错误"的响应，EMPG 关注"如何在长轨迹中分配 credit"。

> [!comparison] EMPG vs Seed-GRPO
>
> - **Seed-GRPO**：使用 semantic uncertainty 降权高 entropy 响应（sequence-level）
> - **EMPG**：step-level 的梯度调制，更细粒度的 credit assignment

> [!comparison] EMPG vs Process Reward Models (PRM)
>
> - **PRM**：需要昂贵的人工标注来提供 step-level reward
> - **EMPG**：使用内在的不确定性信号（entropy），无需额外标注

## 实验结果

**Benchmarks**：
- **ALFWorld**：家庭环境中的多步任务（Pick、Clean、Heat 等）
- **WebShop**：电商网站交互任务
- **Deep Search**：复杂搜索任务（ID 和 OOD）

**主要发现**：

1. **性能提升**：
   - ALFWorld (Qwen2.5-1.5B)：GRPO 65.6% → GRPO+EMPG 73.7% (+8.1%)
   - ALFWorld (Qwen2.5-1.5B)：DAPO 80.8% → DAPO+EMPG 88.1% (+7.3%)
   - 在所有 benchmarks 和模型规模上都显著优于 baseline

2. **训练稳定性**：
   - Baseline (DAPO) 在 ~240 steps 后 KL Loss 剧烈波动，出现 policy collapse
   - EMPG 保持低且稳定的 KL Loss，避免训练发散

3. **Ablation Study**（Deep Search）：
   - **Future Clarity Bonus**：强大的 exploitation 信号，ID 任务 +2.6%
   - **Self-Calibrating Gradient Scaling**：强大的 regularization 机制，OOD 任务 +3.9%
   - **Full EMPG**：两者协同，ID 和 OOD 都提升

> [!intuition] 为什么两个组件互补？
>
> - **Future Clarity Bonus**：在训练中强化已知的高质量决策序列 → 掌握 in-domain 分布 → ID 性能提升
> - **Self-Calibrating Gradient Scaling**：教会模型如何在不确定时行为 → 学习鲁棒性 → OOD 泛化提升
>
> EMPG 不是简单地 overfitting，而是学习了一个基本技能：**如何处理不确定性**。

## 局限性

> [!warning] 边界条件
>
> 1. **Entropy 作为不确定性的代理**：
>    - 论文使用 token-level entropy 的平均值作为步骤级别的不确定性度量
>    - 这是计算高效的选择，但可能不完全捕捉真实的不确定性
>    - 未来可以探索其他 uncertainty estimators（如 Monte Carlo dropout、ensemble variance）
>
> 2. **Hyperparameter 敏感性**：
>    - $k$、$k'$、$\zeta$ 需要调优
>    - 虽然 self-calibrating 设计减少了超参数，但仍需要实验确定最佳值
>
> 3. **Step-level vs Token-level**：
>    - 论文在 "reason-then-act" 步骤级别计算 entropy
>    - 对于不同的 agent 框架，步骤的定义可能不同
>
> 4. **适用范围**：
>    - 专门为 long-horizon agent 任务设计
>    - 在单轮任务（如数学推理）中，EDGE-GRPO 可能更合适

## 延伸阅读

**理论基础**：
- [[Policy Gradient]] — EMPG 的理论基础
- [[Entropy]] — 不确定性度量
- [[Credit Assignment]] — Long-horizon RL 的核心挑战

**相关方法**：
- [[GRPO]] — EMPG 的 baseline 方法
- [[DAPO]] — GRPO 的改进版本
- [[PPO]] — 传统 policy gradient 方法

**论文**：
- [[EMPG (2025)]] — 原始论文
