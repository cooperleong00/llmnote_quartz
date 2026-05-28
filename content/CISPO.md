---
description: 通过 clip importance sampling weights 而非 token updates 保留所有 token 的梯度贡献，解决 PPO/GRPO 中低概率重要 token 被丢弃的问题
type: method
aliases:
  - Clipped IS-weight Policy Optimization
  - 裁剪重要性权重策略优化
prerequisites:
  - "[[GRPO]]"
  - "[[PPO]]"
  - "[[Importance Sampling]]"
tags:
  - post-training
  - reinforcement-learning
  - reasoning
created: 2026-01-29
updated: 2026-04-05T18:22
---

# CISPO (Clipped IS-weight Policy Optimization)

CISPO 是 MiniMax 在 MiniMax-M1 技术报告中提出的 RL 算法，核心创新是**clip importance sampling weights 而非 token updates**。这一改变解决了 [[PPO]]/[[GRPO]] 中低概率但重要的 token（如反思 token：However, Recheck, Wait, Aha）在 off-policy 更新中被丢弃的问题，显著提升了 RL 训练效率。

> [!paper] 论文出处
> MiniMax Team, "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention", 2025
> - 在 Qwen2.5-32B-base 的 zero-RL 实验中，CISPO 用 50% 的训练步数达到 DAPO 的性能
> - MiniMax-M1 完整 RL 训练在 512 H800 GPU 上仅需 3 周，成本约 $534,700

---

## 动机

> [!intuition] 为什么需要 CISPO？

[[PPO]] 和 [[GRPO]] 使用 **token clipping** 来稳定训练：当概率比 $r_t$ 超出 $[1-\epsilon, 1+\epsilon]$ 范围时，梯度被截断。

**问题**：低概率但重要的 token 会被系统性丢弃。

考虑反思 token（如 "However", "Wait", "Recheck"）：
1. 这些 token 在 base model 中概率很低（如 $\pi_{old} = 0.01$）
2. 当 RL 想增加它们的概率时，$r_t$ 很容易超过上界（如 $0.02/0.01 = 2.0 > 1.2$）
3. **第一次 on-policy 更新后**，这些 token 就被 clip 掉，无法贡献后续的 off-policy 梯度更新

这些 token 对于：
- **稳定 entropy**：保持探索多样性
- **可扩展 RL**：支持 long-CoT 推理的涌现

至关重要。

**[[DAPO]] 的尝试**：增大 upper clipping bound（$\epsilon_{high}$），但在多轮 off-policy 更新（如 16 轮）中效果有限。

**CISPO 的思路**：不 clip token updates，而是 clip importance sampling weights。

---

## 核心机制

### 从 REINFORCE 到 CISPO

> [!math] REINFORCE 目标（带 IS 修正）

$$\mathcal{J}_{\text{REINFORCE}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, o_i \sim \pi_{\theta_{\text{old}}}(\cdot | q)} \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \operatorname{sg}(r_{i,t}(\theta)) \hat{A}_{i,t} \log \pi_{\theta}(o_{i,t} | q, o_{i,< t}) \right]$$

其中：
- $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}$ 是 [[Importance Sampling|重要性采样]]权重
- $\operatorname{sg}(\cdot)$ 是 stop-gradient 操作
- $\hat{A}_{i,t}$ 是 advantage

> [!math] CISPO 目标

采用 [[GRPO]] 的 group relative advantage 和 token-level loss：

$$\mathcal{J}_{\text{CISPO}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot | q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \operatorname{sg}(\hat{r}_{i,t}(\theta)) \hat{A}_{i,t} \log \pi_{\theta}(o_{i,t} | q, o_{i,< t}) \right]$$

**关键区别**：使用 **clipped IS weight** $\hat{r}_{i,t}(\theta)$ 而非原始 $r_{i,t}(\theta)$：

$$\hat{r}_{i,t}(\theta) = \text{clip}\left(r_{i,t}(\theta), 1 - \epsilon_{low}^{IS}, 1 + \epsilon_{high}^{IS}\right)$$

### 直觉理解

> [!intuition] 为什么 clip IS weights 而非 token updates？

**PPO/GRPO 的 clip**：
- 当 $r_t > 1 + \epsilon$ 且 $A_t > 0$ 时，**整个 token 的梯度被丢弃**
- 相当于说："这个 token 更新太大了，我们不要它"

**CISPO 的 clip**：
- 当 $r_t > 1 + \epsilon_{high}^{IS}$ 时，**只是限制 IS weight 的大小**
- 相当于说："这个 token 的权重太大了，我们限制它的影响，但仍然保留它的梯度贡献"

**结果**：所有 token 都能贡献梯度，尤其是长响应中的低概率重要 token。

---

## 与其他方法的对比

> [!comparison] PPO vs GRPO vs DAPO vs CISPO

| 方面 | PPO/GRPO | DAPO | CISPO |
|------|----------|------|-------|
| **Clip 对象** | Token updates | Token updates（放宽上界） | IS weights |
| **低概率 token** | 容易被丢弃 | 部分缓解 | 完全保留 |
| **多轮 off-policy** | 效果递减 | 效果有限 | 持续有效 |
| **Entropy 稳定性** | 容易 collapse | 改善 | 更稳定 |

### 统一公式视角

论文提出了一个统一公式，通过 token-wise mask $M_{i,t}$ 来理解不同方法：

$$\mathcal{J}_{\text{unify}}(\theta) = \mathbb{E} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \operatorname{sg}(\hat{r}_{i,t}(\theta)) \hat{A}_{i,t} \log \pi_{\theta}(o_{i,t} | q, o_{i,< t}) M_{i,t} \right]$$

**PPO/GRPO 的隐式 mask**：
$$M_{i,t} = \begin{cases} 0 & \text{if } \hat{A}_{i,t} > 0 \text{ and } r_{i,t}(\theta) > 1 + \epsilon_{\text{high}} \\ 0 & \text{if } \hat{A}_{i,t} < 0 \text{ and } r_{i,t}(\theta) < 1 - \epsilon_{\text{low}} \\ 1 & \text{otherwise} \end{cases}$$

**CISPO**：$M_{i,t} = 1$（永远不丢弃 token）

---

## 实验结果

> [!example] Qwen2.5-32B-base Zero-RL 实验

在 AIME 2024 benchmark 上：
- **CISPO 显著优于 GRPO 和 DAPO**（相同训练步数）
- **CISPO 用 50% 的训练步数达到 DAPO 的性能**

这验证了保留所有 token 梯度贡献的重要性。

---

## 局限性

> [!warning] 边界条件

1. **梯度偏差**：由于 IS weight clipping，$\mathcal{J}_{\text{CISPO}}$ 的梯度是有偏的
   - 但实践中这种偏差被方差减少的收益所抵消

2. **超参数选择**：
   - 论文中 $\epsilon_{low}^{IS}$ 设为很大的值（实际上不限制下界）
   - 只调整 $\epsilon_{high}^{IS}$

3. **与其他技术的结合**：
   - CISPO 仍然使用 [[DAPO]] 的 dynamic sampling 和 length penalty
   - 没有 KL penalty（与 [[DAPO]]、KIMI-K1.5 等一致）

---

## 相关概念

- [[PPO]] - 原始的 clip 机制
- [[GRPO]] - Group relative advantage，CISPO 的基础
- [[DAPO]] - 通过 Clip-Higher 尝试解决同样的问题
- [[DGPO]] - 从概率梯度角度解决 CISPO 的梯度发散问题
- [[Importance Sampling]] - IS weight 的理论基础
- [[Entropy Collapse]] - CISPO 试图解决的核心问题之一

---

> [!interview] 面试视角

**Q: CISPO 和 PPO/GRPO 的核心区别是什么？**

A: PPO/GRPO clip token updates（当概率比超出范围时丢弃整个 token 的梯度），而 CISPO clip importance sampling weights（限制权重大小但保留所有 token 的梯度贡献）。这对于低概率但重要的 token（如反思 token）尤为关键。

**Q: 为什么低概率 token 在 PPO/GRPO 中容易被丢弃？**

A: 因为低概率 token 的概率比 $r_t = \pi_{new}/\pi_{old}$ 很容易超过 clip 上界。例如，$\pi_{old}=0.01$ 的 token 只需要 $\pi_{new}=0.012$ 就会超过 $1.2$ 的上界，导致梯度被截断。

**Q: CISPO 的梯度是有偏的，为什么还能 work？**

A: 虽然 IS weight clipping 引入了偏差，但它显著减少了方差，使训练更稳定。在实践中，方差减少的收益大于偏差带来的损失。这是 bias-variance tradeoff 的经典案例。
