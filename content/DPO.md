---
description: 通过直接优化偏好数据绕过 reward model，将 RLHF 简化为单阶段监督学习
type: method
prerequisites:
  - "[[RLHF]]"
  - "[[Bradley-Terry Model]]"
  - "[[KL Divergence]]"
tags:
  - post-training
  - alignment
  - direct-alignment
created: 2025-01-25
updated: 2026-03-01T17:34
---

# DPO (Direct Preference Optimization)

直接偏好优化（DPO, Direct Preference Optimization）是一种**无需显式奖励模型**的对齐方法，通过直接在偏好数据上优化策略，绕过了传统 [[RLHF]] 中复杂的强化学习过程。

> [!paper] 论文出处
> Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023

---

## 核心思想

> [!intuition] 直觉理解
> RLHF 的流程是：先训练一个 [[Reward Model]]，再用 RL（如 [[PPO]]）优化策略。DPO 的洞察是：**我们可以把这两步合并成一步**。
>
> 具体来说，DPO 证明了：在 [[KL Divergence|KL 约束]] 的 RL 目标下，最优策略和奖励函数之间存在**解析解**关系。因此，我们可以直接用偏好数据优化策略，而不需要先学一个奖励模型。

**RLHF vs DPO 流程对比：**

```
RLHF:  偏好数据 → 训练 Reward Model → RL 优化策略（PPO）
DPO:   偏好数据 → 直接优化策略（一步到位）
```

---

## 数学推导

### 从 RLHF 目标出发

RLHF 的优化目标是最大化奖励的同时，保持策略不要偏离参考策略太远：

$$
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} \left[ r(x, y) \right] - \beta \cdot D_{\text{KL}}\left[ \pi(y|x) \| \pi_{\text{ref}}(y|x) \right]
$$

其中：
- $r(x, y)$ 是奖励函数
- $\pi_{\text{ref}}$ 是参考策略（通常是 SFT 后的模型）
- $\beta$ 控制 KL 惩罚的强度

### 最优策略的解析解

对于上述目标，最优策略有闭式解：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

其中 $Z(x)$ 是归一化常数（partition function）。

### 反解奖励函数

从上式反解 $r(x, y)$：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

> [!intuition] 关键洞察
> 奖励函数可以用**策略的对数概率比**来表示。这意味着：如果我们能直接学到最优策略 $\pi^*$，就隐式地学到了奖励函数。

### Bradley-Terry 偏好模型

假设人类偏好遵循 [[Bradley-Terry Model]]：给定 prompt $x$ 和两个回答 $y_w$（preferred）、$y_l$（rejected），人类偏好 $y_w$ 的概率为：

$$
p(y_w \succ y_l | x) = \sigma\left( r(x, y_w) - r(x, y_l) \right)
$$

其中 $\sigma$ 是 sigmoid 函数。

### DPO Loss 推导

将奖励函数的表达式代入 Bradley-Terry 模型：

$$
p(y_w \succ y_l | x) = \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
$$

注意 $\log Z(x)$ 项在相减时抵消了。

最大化偏好数据的似然，等价于最小化负对数似然：

> [!definition] DPO Loss
> $$
> \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
> $$

简化记号，定义隐式奖励：
$$
\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

则 DPO Loss 可写为：
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma\left( \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l) \right) \right]
$$

---

## 梯度分析

> [!math] 梯度直觉
> DPO 的梯度会：
> 1. **增加** $y_w$（preferred response）的概率
> 2. **降低** $y_l$（rejected response）的概率
> 3. 梯度大小与**当前模型的错误程度**成正比：如果模型已经正确偏好 $y_w$，梯度较小；如果模型错误地偏好 $y_l$，梯度较大

具体地，对 $\pi_\theta$ 的梯度为：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E} \left[ \underbrace{\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))}_{\text{权重：模型犯错的程度}} \left( \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right) \right]
$$

### Token-level 梯度回传机制

语言模型的生成是自回归过程，序列的对数概率等于各 token 对数概率之和：

$$
\log \pi_\theta(y|x) = \sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t})
$$

将其代入梯度公式，可以看到 token 级别的梯度形式：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \underbrace{\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))}_{\text{全局标量权重}} \left[ \sum_{t=1}^{|y_w|} \nabla_\theta \log \pi_\theta(y_{w,t}|x, y_{w,<t}) - \sum_{t=1}^{|y_l|} \nabla_\theta \log \pi_\theta(y_{l,t}|x, y_{l,<t}) \right]
$$

**关键特点**：

1. **统一的全局权重**：$\sigma(\cdot)$ 是序列级别的标量，衡量模型在整个序列层面的错误程度
2. **无差别的 token 更新**：同一序列内所有 token 共享这个全局权重，一荣俱荣、一损俱损

对于具体某个 token 的 logit $z_t$，其梯度大小由两个因素决定：

$$
\frac{\partial \mathcal{L}_{\text{DPO}}}{\partial z_t} = - \underbrace{\beta \sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))}_{\text{全局序列权重}} \times \underbrace{(1 - \pi_\theta(y_t|x, y_{<t}))}_{\text{局部预测惊喜度}}
$$

- **全局序列权重**：决定整体更新步伐，所有 token 共享
- **局部预测概率**：$(1 - \pi_\theta)$ 反映模型对该 token 的置信度
  - 如果 $\pi_\theta$ 接近 1（模型很确定），梯度接近 0
  - 如果 $\pi_\theta$ 接近 0（模型很意外），梯度接近全局权重

> [!warning] 信用分配问题
> 这种梯度分配机制存在根本性缺陷：
> - **坏序列中的好 token 被错误惩罚**：即使前 80% 内容优质，只因结尾有错误，整个序列的所有 token 都会被统一惩罚
> - **好序列中的坏 token 被错误奖励**：chosen 序列中的冗余或次优词汇也会被统一奖励
>
> 这是经典的 [[Credit Assignment|信用分配问题]]——DPO 无法区分序列内部不同 token 对最终结果的贡献。这也是 [[TDPO]] 等 token-level 方法出现的动机。

---

## 与 RLHF 的对比

> [!comparison] DPO vs RLHF

| 方面 | RLHF | DPO |
|------|------|-----|
| **流程** | 两阶段：RM → RL | 单阶段：直接优化 |
| **奖励模型** | 显式训练 | 隐式（策略即奖励） |
| **训练稳定性** | PPO 调参复杂 | 简单的分类 loss |
| **计算开销** | 需要多个模型（actor, critic, RM） | 只需策略 + 参考策略 |
| **理论保证** | 近似最优 | 在 BT 模型假设下等价 |

> [!warning] 常见误区
> **误区**：DPO 完全取代了 RLHF
>
> **事实**：DPO 在某些场景下效果很好，但：
> - 对偏好数据质量更敏感
> - 在 out-of-distribution 泛化上可能不如 RLHF
> - 无法利用过程奖励（process reward）
> - 实际大规模训练中，很多团队仍使用 RLHF 或混合方法

---

## 超参数

- **$\beta$**：KL 惩罚系数，控制偏离参考策略的程度
  - $\beta$ 越大 → 越保守，越接近参考策略
  - $\beta$ 越小 → 越激进，可能过拟合偏好数据
  - 典型值：0.1 ~ 0.5

---

## DPO 的局限性

> [!warning] 核心洞察
> **DPO 对标的是 Reward Model，不是 PPO。** 二者训练数据一样，loss 函数本质上也一致。理解这一点，DPO 的局限性就清晰了。

### 1. 基本假设未被验证

DPO 让模型学习 **evaluate 能力**（判断哪个回答更好），但我们的目标是提升 **generate 能力**（生成好的回答）。

> [!intuition] 类比
> 美食家并不一定做得一手好饭。Evaluate 能力和 Generate 能力是否相互促进，这个假设并非显而易见。

这也是为什么 DPO 训练中常见 **good sentence 和 bad sentence 的 loss 都上升** 的现象——因为 DPO 只在乎 margin 变大，不在乎模型能否生成好句子。

### 2. Offline vs Online 学习

| 方面 | RLHF (Online) | DPO (Offline) |
|------|---------------|---------------|
| 数据来源 | 模型当前生成的 | 固定的偏好数据集 |
| 学习方式 | 修复模型**当前**的问题 | 强制学习**预设**的正确答案 |
| 探索能力 | 有（do_sample 随机性） | 无 |
| 适应性 | 因材施教 | 一刀切 |

**PPO 做了什么 DPO 没做的事？**
- PPO 通过 generate → 打分 → 更新，把 RM 的 evaluate 能力**转化**为 generate 能力
- 这个 generate 过程带来了 **online** 和 **explore** 两个关键特性

### 3. 实践中的 Workarounds

正因为这些局限，DPO 训练中常用的技巧本质上都在**模拟 PPO 的特性**：

| 技巧 | 模拟的特性 |
|------|-----------|
| 先对 good sentence 做 SFT，再 DPO | 模拟 online（训过的知识可以 generate） |
| 用模型自己的 pass@N 构造偏好对 | 模拟 online + explore |
| 魔改 loss 函数（加系数、调权重） | 弥补 evaluate ≠ generate 的问题 |

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: DPO 的核心思想是什么？**
> A: DPO 利用 KL 约束 RL 目标的解析解，将奖励函数重参数化为策略的对数概率比，从而绕过显式奖励模型，直接在偏好数据上优化策略。
>
> **Q2: DPO Loss 是怎么推导出来的？**
> A: 从 RLHF 目标出发 → 写出最优策略的闭式解 → 反解奖励函数 → 代入 Bradley-Terry 偏好模型 → 最大化似然得到 DPO Loss。
>
> **Q3: DPO 相比 RLHF 的优缺点？**
> A: 优点是简单稳定、计算高效；缺点是对数据质量敏感、泛化能力可能较弱、无法利用过程奖励。
>
> **Q4: $\beta$ 参数的作用？**
> A: 控制 KL 惩罚强度。$\beta$ 大则保守，$\beta$ 小则激进。

---

## 延伸阅读

**后续发展**：
- [[IPO]]、[[KTO]]、[[ORPO]]、[[SimPO]] — DPO 的后续改进

**相关工作**：
- [[InstructGPT]] — RLHF 的经典工作
- [[DPO (2023)]] — 原始论文
- [[Constitutional AI]] — 另一种对齐思路

---

## 参考资料

- [[DPO 是如何简化 RLHF 的|朱小霖：DPO 是如何简化 RLHF 的]] — 清晰的数学推导
- [[dpo 的局限性|ybq：DPO 的局限性]] — 深入分析 DPO 的本质问题
