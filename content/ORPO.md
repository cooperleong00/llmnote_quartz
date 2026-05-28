---
description: 将 SFT 和偏好对齐合并为单阶段训练，使用 Odds Ratio 作为对比信号，无需 reference model
type: method
status: complete
aliases:
  - Odds Ratio Preference Optimization
  - 几率比偏好优化
prerequisites:
  - "[[DPO]]"
  - "[[SFT]]"
tags:
  - post-training
  - alignment
  - direct-alignment
created: 2026-01-27
updated: 2026-02-01T00:56
---

# ORPO (Odds Ratio Preference Optimization)

几率比偏好优化（ORPO, Odds Ratio Preference Optimization）是一种**单阶段对齐方法**，将 [[SFT]] 和偏好对齐合并到一个训练过程中。与 [[DPO]] 不同，ORPO **不需要 reference model**，通过 Odds Ratio 作为对比信号来区分 preferred 和 rejected responses。

> [!paper] 论文出处
> Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model", EMNLP 2024

---

## 动机

> [!intuition] 为什么需要 ORPO？
> 传统的对齐流程是：**SFT → DPO/[[RLHF]]**，两个阶段分开训练。这带来几个问题：
> 1. **流程复杂**：需要先 SFT，再做偏好优化
> 2. **Reference model 开销**：DPO 需要保存一个 frozen reference model，增加显存占用
> 3. **SFT 的副作用**：SFT 会同时增加 preferred 和 rejected responses 的概率（因为都是"正确"的语言）
>
> ORPO 的洞察：**能否在 SFT 的同时就引入偏好信号，一步到位？**

### SFT 的问题

标准 SFT 使用负对数似然（NLL）loss：

$$
\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \log P_\theta(y|x) \right]
$$

问题在于：SFT 只关心"语言是否流畅"，不区分回答的质量。如果用偏好数据做 SFT，模型会**同时学习 preferred 和 rejected responses 的语言模式**。

---

## 核心思想

> [!intuition] Odds Ratio 的直觉
> ORPO 使用 **Odds Ratio（几率比）** 来衡量模型对 preferred vs rejected 的偏好程度。
>
> - **Odds**（几率）：$\frac{P(y|x)}{1 - P(y|x)}$，表示"生成 y 的几率 vs 不生成 y 的几率"
> - **Odds Ratio**：比较两个回答的 odds，如果 OR > 1，说明模型更偏好 $y_w$
>
> 关键洞察：Odds Ratio 是一个**自包含的对比信号**，不需要 reference model 来提供基准。

### 为什么用 Odds Ratio 而不是概率比？

直接比较概率 $\frac{P(y_w|x)}{P(y_l|x)}$ 的问题：
- 概率受序列长度影响（长序列概率更低）
- 没有归一化，难以跨样本比较

Odds Ratio 的优势：
- 对序列长度更鲁棒
- 有统计学上的良好性质（对称性、可解释性）

---

## 数学公式

### ORPO Loss 定义

> [!definition] ORPO Loss
> $$
> \mathcal{L}_{\text{ORPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \underbrace{\mathcal{L}_{\text{SFT}}(y_w)}_{\text{SFT on preferred}} + \lambda \cdot \underbrace{\mathcal{L}_{\text{OR}}(y_w, y_l)}_{\text{Odds Ratio penalty}} \right]
> $$
>
> 其中：
> - $\mathcal{L}_{\text{SFT}}(y_w) = -\log P_\theta(y_w|x)$：只对 preferred response 做 SFT
> - $\mathcal{L}_{\text{OR}}$：Odds Ratio 对比损失
> - $\lambda$：平衡系数

### Odds Ratio Loss

定义 token 级别的平均对数概率：

$$
\log P_\theta(y|x) = \frac{1}{|y|} \sum_{t=1}^{|y|} \log P_\theta(y_t | x, y_{<t})
$$

定义 odds：

$$
\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}
$$

Odds Ratio Loss：

> [!math] OR Loss
> $$
> \mathcal{L}_{\text{OR}} = -\log \sigma \left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right)
> $$
>
> 展开后：
> $$
> \mathcal{L}_{\text{OR}} = -\log \sigma \left( \log \frac{P_\theta(y_w|x)}{1 - P_\theta(y_w|x)} - \log \frac{P_\theta(y_l|x)}{1 - P_\theta(y_l|x)} \right)
> $$

### 梯度行为

> [!intuition] 梯度直觉
> ORPO 的梯度会：
> 1. **SFT 部分**：增加 $y_w$ 的概率（学习好的回答）
> 2. **OR 部分**：拉大 $y_w$ 和 $y_l$ 的 odds 差距
>
> 与 DPO 不同，ORPO **只对 preferred response 做 SFT**，不会意外提升 rejected response 的概率。

---

## 与 DPO 的对比

> [!comparison] ORPO vs DPO

| 方面 | DPO | ORPO |
|------|-----|------|
| **训练阶段** | 需要先 SFT，再 DPO | 单阶段，SFT + 偏好同时进行 |
| **Reference Model** | 需要（frozen） | 不需要 |
| **显存占用** | 较高（需存储 ref model） | 较低 |
| **对比信号** | Log probability ratio | Odds Ratio |
| **SFT 目标** | 对 $y_w$ 和 $y_l$ 都有隐式 SFT | 只对 $y_w$ 做 SFT |
| **理论基础** | [[Bradley-Terry Model]] + KL-constrained RL | 统计学中的 Odds Ratio |

### 为什么 ORPO 不需要 Reference Model？

> [!intuition] 关键区别
> - **DPO** 使用 $\log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}$ 作为隐式奖励，需要 reference model 提供基准
> - **ORPO** 使用 Odds Ratio，这是一个**自包含的对比指标**，不需要外部基准
>
> 类比：DPO 问"相比原来的模型，你更喜欢哪个？"；ORPO 问"在当前模型看来，哪个更好？"

---

## 优势与局限

### 优势

> [!example] ORPO 的优势
> 1. **单阶段训练**：省去 SFT 预训练步骤，流程更简洁
> 2. **无需 Reference Model**：节省显存，简化实现
> 3. **避免 SFT 副作用**：不会提升 rejected response 的概率
> 4. **计算效率**：只需要一个模型，forward pass 更少

### 局限性

> [!warning] 局限性
> 1. **对数据质量敏感**：单阶段训练意味着没有 SFT 的"预热"，对偏好数据质量要求更高
> 2. **超参数敏感**：$\lambda$ 的选择影响 SFT 和偏好对齐的平衡
> 3. **理论基础较弱**：相比 DPO 有严格的 RL 理论推导，ORPO 更多是启发式设计
> 4. **长序列问题**：Odds 计算涉及 $1 - P(y|x)$，当 $P$ 很小时数值不稳定

---

## 超参数

- **$\lambda$**：OR loss 的权重
  - $\lambda$ 越大 → 偏好对齐越强，可能牺牲语言质量
  - $\lambda$ 越小 → 更像纯 SFT，偏好区分度不够
  - 论文推荐值：0.1 ~ 1.0

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: ORPO 的核心思想是什么？**
> A: ORPO 将 SFT 和偏好对齐合并为单阶段训练，使用 Odds Ratio 作为对比信号。Loss = SFT loss（只对 preferred）+ OR penalty。不需要 reference model。
>
> **Q2: ORPO 和 DPO 的主要区别？**
> A: 三个关键区别：(1) ORPO 单阶段，DPO 需要先 SFT；(2) ORPO 不需要 reference model；(3) ORPO 只对 preferred response 做 SFT，DPO 隐式地对两者都有影响。
>
> **Q3: 为什么 ORPO 不需要 reference model？**
> A: DPO 用 log probability ratio 相对于 reference model 作为隐式奖励；ORPO 用 Odds Ratio，这是一个自包含的对比指标，不需要外部基准。
>
> **Q4: ORPO 的 Odds Ratio 是什么？**
> A: Odds = P/(1-P)，表示"发生 vs 不发生"的比值。Odds Ratio 比较两个回答的 odds，用于衡量模型的相对偏好。
>
> **Q5: ORPO 相比 DPO 的优缺点？**
> A: 优点：单阶段、无需 ref model、显存更低。缺点：对数据质量更敏感、理论基础较弱、超参数敏感。

---

## 相关概念

- [[DPO]] — ORPO 的前身，理解 DPO 有助于理解 ORPO 的改进点
- [[SFT]] — ORPO 将 SFT 集成到偏好优化中
- [[IPO]]、[[KTO]]、[[SimPO]] — 其他 DPO 变体，各有不同的改进思路

---

## DPO 变体对比

> [!comparison] Direct Alignment 方法对比

| 方法 | Reference Model | 训练阶段 | 核心改进 |
|------|-----------------|----------|----------|
| [[DPO]] | 需要 | SFT → DPO | 原始方法 |
| [[IPO]] | 需要 | SFT → IPO | 解决 DPO 的过拟合问题 |
| [[KTO]] | 需要 | SFT → KTO | 不需要成对偏好数据 |
| **ORPO** | 不需要 | 单阶段 | 合并 SFT，用 Odds Ratio |
| [[SimPO]] | 不需要 | SFT → SimPO | 简化 DPO，用长度归一化 |

---

## 延伸阅读

- [[DPO]] — 理解 ORPO 的基础
- [[SFT]] — ORPO 集成的监督微调
- [[Reward Hacking]] — 偏好优化中的常见问题
