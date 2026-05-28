---
description: 通过 identity mapping 正则化解决 DPO 的过拟合问题，提供更稳健的偏好优化
type: method
aliases:
  - Identity Preference Optimization
  - 恒等偏好优化
prerequisites:
  - "[[DPO]]"
  - "[[Bradley-Terry Model]]"
  - "[[KL Divergence]]"
tags:
  - post-training
  - alignment
  - direct-alignment
created: 2026-01-27
updated: 2026-02-01T00:54
---

# IPO (Identity Preference Optimization)

恒等偏好优化（IPO, Identity Preference Optimization）是 [[DPO]] 的理论改进版本，通过引入 **identity mapping** 正则化，解决了 DPO 在确定性偏好数据上的过拟合问题。作为 [[RLHF]] 的直接对齐方法之一，IPO 继承了 DPO 绕过显式奖励模型的优势，同时提供了更强的理论保证。

> [!paper] 论文出处
> Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Feedback", NeurIPS 2023

---

## 动机：DPO 的理论缺陷

> [!warning] DPO 的核心问题
> DPO 假设偏好数据遵循 [[Bradley-Terry Model]]，但这个假设在实践中存在问题：
> 1. **确定性偏好导致过拟合**：当偏好是确定性的（$y_w$ 总是优于 $y_l$），DPO 会将隐式奖励差推向无穷大
> 2. **Bradley-Terry 假设过强**：真实的人类偏好可能不遵循 BT 模型

### DPO 过拟合的数学解释

回顾 DPO 的损失函数：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

> [!intuition] 过拟合的直觉
> 当偏好数据是确定性的（即 $y_w$ 总是被标注为更好），DPO 的最优解是让：
> $$\beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \to +\infty$$
>
> 这意味着模型会**无限制地**增大 $y_w$ 和 $y_l$ 的隐式奖励差，导致：
> - 策略过度偏离参考策略（违反 [[KL Divergence]] 约束的初衷）
> - 对训练数据过拟合
> - 泛化能力下降

---

## 核心改进：Identity Mapping

> [!definition] IPO 的核心思想
> IPO 不再假设偏好遵循 Bradley-Terry 模型，而是直接优化一个**正则化的偏好目标**，通过 identity mapping 约束隐式奖励差的大小。

### IPO 损失函数

> [!definition] IPO Loss
> $$
> \mathcal{L}_{\text{IPO}}(\pi_\theta; \pi_{\text{ref}}) = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta} \right)^2 \right]
> $$

简化记号，定义隐式奖励差：
$$
h_\theta(x, y_w, y_l) = \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

则 IPO Loss 可写为：
$$
\mathcal{L}_{\text{IPO}} = \mathbb{E} \left[ \left( h_\theta(x, y_w, y_l) - \frac{1}{2\beta} \right)^2 \right]
$$

> [!intuition] 为什么叫 "Identity" Preference Optimization？
> 在 DPO 中，偏好概率通过 **sigmoid** 函数映射：$p = \sigma(\beta \cdot h)$
>
> 在 IPO 中，偏好概率通过 **identity** 函数（恒等映射）直接关联：$p \propto h$
>
> 这个 "identity" 指的是用恒等函数替代 sigmoid，从而避免了 sigmoid 饱和带来的过拟合问题。

---

## 数学推导

### 从 Ψ-Preference Optimization 框架出发

IPO 论文提出了一个通用的偏好优化框架 Ψ-PO：

$$
\mathcal{L}_{\Psi\text{-PO}} = \mathbb{E} \left[ \Psi\left( h_\theta(x, y_w, y_l) \right) \right]
$$

其中 $\Psi$ 是一个凸函数，不同的 $\Psi$ 对应不同的算法：

| 算法 | $\Psi(h)$ | 特点 |
|------|-----------|------|
| DPO | $-\log \sigma(\beta h)$ | 基于 Bradley-Terry |
| IPO | $(h - \frac{1}{2\beta})^2$ | 正则化，防止过拟合 |

### IPO 的最优解

> [!math] 最优解分析
> IPO 的最优解满足：
> $$h^*_\theta(x, y_w, y_l) = \frac{1}{2\beta}$$
>
> 即隐式奖励差被约束在一个**固定值**，而不是像 DPO 那样趋向无穷大。

这个约束确保了：
1. 策略不会过度偏离参考策略
2. 模型不会对确定性偏好数据过拟合
3. 更好的泛化能力

---

## 与 DPO 的对比

> [!comparison] IPO vs DPO

| 方面 | DPO | IPO |
|------|-----|-----|
| **偏好模型假设** | Bradley-Terry | 无需 BT 假设 |
| **损失函数** | 负对数似然（交叉熵） | 均方误差（MSE） |
| **隐式奖励差** | 可能趋向无穷 | 被约束在 $\frac{1}{2\beta}$ |
| **过拟合风险** | 高（确定性偏好时） | 低 |
| **理论保证** | 依赖 BT 假设 | 更通用的理论框架 |
| **实现复杂度** | 简单 | 同样简单 |

### 梯度对比

**DPO 梯度**：
$$
\nabla_\theta \mathcal{L}_{\text{DPO}} \propto \sigma(-\beta h_\theta) \cdot \nabla_\theta h_\theta
$$

当 $h_\theta$ 很大时，$\sigma(-\beta h_\theta) \to 0$，梯度消失。

**IPO 梯度**：
$$
\nabla_\theta \mathcal{L}_{\text{IPO}} \propto \left( h_\theta - \frac{1}{2\beta} \right) \cdot \nabla_\theta h_\theta
$$

梯度始终存在，不会因为 $h_\theta$ 过大而消失。

> [!intuition] 梯度差异的直觉
> - **DPO**：一旦模型"学会"了偏好（$h_\theta$ 很大），梯度就消失了，即使模型过拟合
> - **IPO**：只要 $h_\theta \neq \frac{1}{2\beta}$，就有梯度推动模型调整，防止过拟合

---

## 实验效果

根据原论文的实验：

1. **在确定性偏好数据上**：IPO 显著优于 DPO，避免了过拟合
2. **在随机偏好数据上**：IPO 和 DPO 效果相近
3. **泛化能力**：IPO 在 held-out 数据上表现更稳定

> [!warning] 实践注意
> 虽然 IPO 在理论上更优，但在实际应用中：
> - 如果偏好数据本身就有噪声（非确定性），DPO 和 IPO 差异不大
> - IPO 的 $\beta$ 参数含义与 DPO 不同，需要重新调参
> - 工业界目前仍以 DPO 为主流，IPO 的采用相对较少

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: IPO 解决了 DPO 的什么问题？**
> A: IPO 解决了 DPO 在确定性偏好数据上的过拟合问题。DPO 基于 Bradley-Terry 模型，当偏好是确定性的，会将隐式奖励差推向无穷大。IPO 通过 MSE 损失将奖励差约束在固定值，避免过拟合。
>
> **Q2: IPO 的 "Identity" 指什么？**
> A: 指用恒等函数（identity mapping）替代 DPO 中的 sigmoid 函数来建模偏好概率。这避免了 sigmoid 饱和导致的梯度消失和过拟合。
>
> **Q3: IPO 和 DPO 的损失函数有什么区别？**
> A: DPO 用负对数似然（交叉熵），IPO 用均方误差（MSE）。DPO 最小化 $-\log\sigma(\beta h)$，IPO 最小化 $(h - \frac{1}{2\beta})^2$。
>
> **Q4: 什么时候应该用 IPO 而不是 DPO？**
> A: 当偏好数据是确定性的（标注一致性高）时，IPO 更合适。如果偏好数据本身有噪声，两者差异不大。

---

## 延伸阅读

**原始论文与参考资料**：
- [[DPO (2023)]] — DPO 原始论文
- [[InstructGPT]] — RLHF 的经典工作
- [[Reward Hacking]] — 过拟合偏好数据的一种表现

**基于偏好优化的后续发展**：
- [[KTO]] — 另一种 DPO 变体，使用 Kahneman-Tversky 人类决策模型
- [[SimPO]] — 简化的偏好优化，移除参考模型
- [[ORPO]] — 结合 SFT 和偏好优化的方法
