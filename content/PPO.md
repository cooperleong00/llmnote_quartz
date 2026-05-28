---
description: 通过 clip 机制限制策略更新幅度的稳定 RL 算法，是 RLHF 的核心组件
type: method
prerequisites:
  - "[[Policy Gradient|策略梯度]]"
  - "[[Actor-Critic]]"
  - "[[Importance Sampling|重要性采样]]"
  - "[[GAE]]"
tags:
  - post-training
  - reinforcement-learning
created: 2025-01-25
updated: 2026-03-02
---

# PPO (Proximal Policy Optimization)

近端策略优化（PPO, Proximal Policy Optimization）是一种策略梯度强化学习算法，因其稳定性和易用性成为 [[RLHF]] 中最常用的 RL 算法。

> [!paper] 论文出处
> Schulman et al., "Proximal Policy Optimization Algorithms", 2017

---

## 核心思想

> [!intuition] 直觉理解
> [[Policy Gradient|策略梯度]]方法的问题：更新步长难以控制。步长太大，策略可能崩溃；步长太小，学习太慢。
>
> PPO 的解决方案：**限制每次更新的幅度**，确保新策略不会偏离旧策略太远。这就是"Proximal"（近端）的含义。

**朴素 Actor-Critic 的两个核心问题**：

1. **采样效率低**：每次梯度更新都需要重新采样，与环境交互成本高（on-policy）
2. **优势估计有偏**：用 TD error 估计 advantage 时，如果 $V_\phi$ 不准确，会引入系统性偏差

PPO 通过**重要性采样**（Importance Sampling）和 **GAE**（Generalized Advantage Estimation）分别解决这两个问题，并用**信任域约束**确保稳定性。

**两种实现方式**：
1. **PPO-Clip**：直接裁剪概率比（更常用，实现简单）
2. **PPO-Penalty**：用 KL 散度作为惩罚项（需要动态调整 $\beta$）

---

## 从 Actor-Critic 到 PPO

### 重要性采样：从 On-Policy 到 Off-Policy

**On-Policy 的问题**：
- 用 $\pi_\theta$ 采样数据 → 用这批数据更新 $\pi_\theta$ → 数据立即失效
- 每次更新都需要重新采样，训练效率低

**Off-Policy 的思路**：
- 用 $\pi_{old}$ 采样一批数据
- 用这批数据**重复更新 k 次**：$\pi_{old} \to \pi_{\theta_0} \to \pi_{\theta_1} \to \cdots \to \pi_\theta$
- 提升样本利用率

> [!math] 重要性采样推导
>
> 如何用分布 $q(x)$ 的样本估计 $E_{x \sim p(x)}[f(x)]$？
>
> $$
> \begin{aligned}
> E_{x \sim p(x)}[f(x)] &= \int p(x) f(x) dx \\
> &= \int \frac{p(x)}{q(x)} q(x) f(x) dx \\
> &= E_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]
> \end{aligned}
> $$
>
> 应用到策略梯度：
>
> $$
> \nabla J(\pi_\theta) = \underset{\tau \sim \pi_{old}}{E_t} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)} A_\phi(s_t, a_t) \nabla \log \pi_\theta(a_t|s_t) \right]
> $$

> [!warning] 重要性采样的风险
>
> 当 $p(x)$ 和 $q(x)$ 分布差异过大时：
> - 需要**大量采样**才能抵消分布差异的影响
> - 某些低概率事件在 $q(x)$ 下很难采到，但权重 $\frac{p(x)}{q(x)}$ 很大
> - **结论**：$\pi_\theta$ 和 $\pi_{old}$ 不能偏离太远，否则估计不准确

### GAE：平衡方差与偏差

**优势函数的估计困境**：

| 估计方式 | 公式 | 方差 | 偏差 |
|---------|------|------|------|
| **TD error** | $r_t + \gamma V(s_{t+1}) - V(s_t)$ | 低 | 高（依赖 $V$ 的准确性） |
| **Monte Carlo** | $\sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t)$ | 高 | 低（直接用真实回报） |

> [!intuition] 方差与偏差的权衡
>
> - **低方差，高偏差**（TD）：像射击时瞄准偏了，但每次都打在同一个错误位置
> - **高方差，低偏差**（MC）：瞄准是对的，但每次打的位置分散
>
> GAE 通过超参数 $\lambda$ 在两者之间插值。

> [!definition] GAE 公式
>
> $$
> A_t^{GAE} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}
> $$
>
> 其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD error。
>
> - $\lambda \to 0$：退化为 TD error（低方差，高偏差）
> - $\lambda \to 1$：接近 Monte Carlo（高方差，低偏差）
> - **典型值**：$\lambda = 0.95$

### TRPO：信任域约束

引入重要性采样和 GAE 后，优化目标变为：

$$
\arg \max_{\pi_\theta} J(\pi_\theta) = \underset{\tau \sim \pi_{old}}{E_t} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)} A_\phi^{GAE}(s_t, a_t) \right]
$$

**问题**：如何确保 $\pi_\theta$ 和 $\pi_{old}$ 不偏离太远？

**TRPO 的方案**：将分布相似性作为**约束条件**：

$$
\begin{aligned}
&\arg \max_{\pi_\theta} \; \underset{\tau \sim \pi_{old}}{E_t} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)} A_\phi^{GAE}(s_t, a_t) \right] \\
&\text{subject to} \quad E_t[KL(\pi_{old}(\cdot|s_t) \| \pi_\theta(\cdot|s_t))] \leq \delta
\end{aligned}
$$

**缺陷**：约束优化复杂，需要二阶导数（共轭梯度法），计算昂贵。

**PPO 的改进**：将约束**直接放入目标函数**，避免复杂的约束优化。

---

## PPO-Clip 算法

### 概率比

定义新旧策略的概率比：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

- $r_t = 1$：新旧策略对该动作的概率相同
- $r_t > 1$：新策略更倾向于该动作
- $r_t < 1$：新策略更不倾向于该动作

### Clip 目标函数

> [!definition] PPO-Clip 目标
> $$
> L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
> $$

其中：
- $A_t$：advantage（该动作比平均好多少）
- $\epsilon$：clip 范围，通常 0.1 ~ 0.2
- $\text{clip}(x, a, b)$：将 $x$ 限制在 $[a, b]$ 范围内

### Clip 机制的直觉

> [!intuition] 为什么 clip 有效？

考虑两种情况：

**情况 1：$A_t > 0$（好动作，应该增加概率）**
- 目标是增大 $r_t$，但 clip 限制了 $r_t \leq 1 + \epsilon$
- **原因**：不能一味轻信 $A_t^{GAE}$ 而持续提升 $\pi_\theta(a_t|s_t)$，必须保证 $\pi_\theta$ 在 $\pi_{old}$ 的信任域内
- 当 $r_t \geq 1 + \epsilon$ 时，clip 到 $1 + \epsilon$，梯度为 0，停止更新

**情况 2：$A_t < 0$（坏动作，应该减少概率）**
- 目标是减小 $r_t$，但 clip 限制了 $r_t \geq 1 - \epsilon$
- **原因**：同样不能过度降低概率，避免策略崩溃
- 当 $r_t \leq 1 - \epsilon$ 时，clip 到 $1 - \epsilon$，梯度为 0，停止更新

**效果**：无论 advantage 正负，策略更新都被限制在一个"信任域"内，避免因采样不足导致的过度更新。

---

## PPO-Penalty 算法

除了 PPO-Clip，还可以用 **KL 惩罚项**直接约束分布差异：

> [!definition] PPO-Penalty 目标
> $$
> J(\pi_\theta) = \underset{\tau \sim \pi_{old}}{E_t} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)} A_\phi^{GAE}(s_t, a_t) - \beta \, KL(\pi_{old}(\cdot|s_t) \| \pi_\theta(\cdot|s_t)) \right]
> $$

**超参数 $\beta$ 的自适应调整**：

设定阈值 $KL_{min}$ 和 $KL_{max}$：
- 当 $KL \geq KL_{max}$：策略偏离太远 → **增大 $\beta$**，加强约束
- 当 $KL \leq KL_{min}$：策略过于保守（可能只优化 KL 项而忽略 advantage）→ **减小 $\beta$**

**对比 PPO-Clip**：
- PPO-Penalty 需要动态调整 $\beta$，调参复杂
- PPO-Clip 的 $\epsilon$ 是固定的，更简单稳定
- **实践中 PPO-Clip 更常用**

---

## 在 RLHF 中的应用

### Actor-Critic 架构

在 [[RLHF]] 中，PPO 使用 [[Actor-Critic]] 架构：

| 组件 | 作用 | 输出 |
|------|------|------|
| **Actor** ($\pi_\theta$) | 生成回答 | token 概率分布 |
| **Critic** ($V_\psi$) | 估计状态价值 | 标量 value |

### Advantage 计算

使用 [[GAE]]（Generalized Advantage Estimation），详见上文"GAE：平衡方差与偏差"。

### Critic Loss

Critic 需要随 Actor 一起迭代更新——因为 $V_\pi$ 是和 $\pi$ 挂钩的，策略参数更新后，必须有新的 Critic 来衡量新策略的价值。

> [!math] Critic Loss 公式
>
> $$
> V_t^{CLIP} = \text{clip}(V_t^{new}, V_t^{old} - \epsilon, V_t^{old} + \epsilon)
> $$
> $$
> R_t = A_t^{GAE} + V_t^{old}
> $$
> $$
> L(V_\phi) = E_t \left[ \max \left( (V_t^{new} - R_t)^2, \; (V_t^{CLIP} - R_t)^2 \right) \right]
> $$

**关键理解**：
- $R_t = A_t^{GAE} + V_t^{old}$ 本质上是用 GAE 修正后的回报估计
- 在 PPO epochs 迭代中，$R_t$、$V_t^{old}$、$A_t^{GAE}$ 都来自初始经验值，**不随迭代变化**
- Critic 的 clip 和 Actor 一样，限制 $V_t^{new}$ 在 $V_t^{old}$ 的信任域内更新

### RLHF 中的特殊处理

1. **Reward 设计**：只在序列末尾给 reward（sparse reward），这带来 [[Credit Assignment]] 的挑战
2. **KL 惩罚**：加入 $-\beta \log \frac{\pi_\theta}{\pi_{ref}}$ 防止偏离太远
3. **Value head**：通常在 LLM 最后一层加一个线性层作为 Critic

---

## PPO 训练流程

> [!example] PPO 训练伪代码
>
> ```python
> for batch in training_steps:
>     # 1. 用当前策略 π_old 收集经验
>     exps = generate_experience(prompts, actor, critic, reward, ref)
>
>     # 2. 同一批经验重复使用 ppo_epochs 次
>     for epoch in ppo_epochs:
>         actor_loss = cal_actor_loss(exps, actor)   # PPO-Clip loss
>         critic_loss = cal_critic_loss(exps, critic) # Value clip loss
>
>         actor.backward(actor_loss)
>         actor.step()
>
>         critic.backward(critic_loss)
>         critic.step()
>
>     # 3. 更新 π_old = π_θ，进入下一轮
> ```
>
> 这就是重要性采样带来的核心收益：**一批数据用 k 次**，而非每次更新都重新采样。

---

## 超参数

| 参数 | 典型值 | 作用 |
|------|--------|------|
| $\epsilon$ (clip ratio) | 0.1 ~ 0.2 | 控制策略更新幅度 |
| $\gamma$ (discount) | 0.99 ~ 1.0 | 未来 reward 折扣 |
| $\lambda$ (GAE) | 0.95 | advantage 估计的 bias-variance 权衡 |
| batch size | 较大 | 稳定梯度估计 |
| epochs per batch | 3 ~ 10 | 每批数据的更新次数 |

> [!warning] 调参敏感
> PPO 在 RLHF 中对超参数较敏感，尤其是 KL 系数 $\beta$ 和 clip ratio $\epsilon$。这是 [[DPO]] 等方法试图避免的问题。

---

## 为什么 RLHF 选择 PPO？

> [!intuition] PPO 的优势

1. **稳定性**：clip 机制防止灾难性更新
2. **样本效率**：可以对同一批数据多次更新
3. **实现简单**：相比 TRPO 不需要二阶优化
4. **通用性**：适用于连续和离散动作空间

**对比其他算法**：
- **TRPO**：理论更优雅，但需要二阶优化，计算昂贵
- **A2C/A3C**：更简单，但没有信任域约束，不够稳定
- **SAC**：适合连续控制，但在语言任务上不如 PPO

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: PPO 的核心思想是什么？**
> A: 限制策略更新幅度，确保新策略不偏离旧策略太远。通过 clip 机制实现"信任域"约束，平衡探索和稳定性。
>
> **Q2: PPO-Clip 的目标函数是什么？**
> A: $L^{CLIP} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$，其中 $r_t$ 是新旧策略的概率比，$A_t$ 是 advantage。
>
> **Q3: 为什么 RLHF 使用 PPO 而不是其他 RL 算法？**
> A: PPO 稳定、样本效率高、实现简单。相比 TRPO 不需要二阶优化，相比 A2C 有信任域约束更稳定。
>
> **Q4: PPO 在 RLHF 中有什么特殊处理？**
> A: 加入 KL 惩罚防止偏离 reference model；使用 sparse reward（只在末尾给分）；Actor 和 Critic 共享 LLM backbone。
>
> **Q5: PPO 为什么需要重要性采样？**
> A: 朴素 Actor-Critic 是 on-policy 的，每次更新都要重新采样，效率低。重要性采样允许用 $\pi_{old}$ 的数据更新 $\pi_\theta$（off-policy），同一批数据可以重复使用 k 次。代价是需要概率比 $\frac{\pi_\theta}{\pi_{old}}$ 做修正，且两个分布不能差太远。
>
> **Q6: GAE 中 $\lambda$ 的作用是什么？**
> A: $\lambda$ 控制方差-偏差权衡。$\lambda \to 0$ 退化为 TD error（低方差高偏差），$\lambda \to 1$ 接近 Monte Carlo（高方差低偏差）。典型值 0.95，偏向低偏差。
>
> **Q7: PPO-Clip 和 PPO-Penalty 的区别？**
> A: PPO-Clip 用 $\epsilon$ 裁剪概率比，超参固定，简单稳定。PPO-Penalty 用 KL 散度惩罚项，需要动态调整 $\beta$，调参复杂。实践中 PPO-Clip 更常用。

---

## 延伸阅读

**原始论文与深入材料**：
- [[Clippings/Article/人人都能看懂的RL-PPO理论知识|人人都能看懂的RL-PPO理论知识]] — 从策略梯度到 PPO 的完整推导链路

**前身与后续发展**：
- [[TRPO]] — PPO 的前身，理论更严格但计算昂贵
- [[InstructGPT]] — PPO 在 RLHF 中的实践
- [[GRPO]] — 去掉 Critic 的 PPO 变体，用 group reward 估计 baseline

