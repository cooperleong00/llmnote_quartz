---
description: 基于前景理论的对齐方法，只需 pointwise 反馈（thumbs up/down）而非 pairwise 偏好数据
type: method
aliases:
  - Kahneman-Tversky Optimization
prerequisites:
  - "[[DPO]]"
  - "[[KL Divergence]]"
tags:
  - post-training
  - alignment
  - direct-alignment
created: 2026-01-27
updated: 2026-02-01T00:56
---

# KTO (Kahneman-Tversky Optimization)

Kahneman-Tversky Optimization（KTO）是一种**不需要 pairwise 偏好数据**的对齐方法。与 [[RLHF]] 需要训练独立的 [[Reward Model]]、[[DPO]] 需要成对的 (chosen, rejected) 数据不同，KTO 只需要独立的 "好" 或 "坏" 标签（pointwise feedback），大大降低了数据收集的难度。

> [!paper] 论文出处
> Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization", arXiv 2024

---

## 核心动机

> [!intuition] 为什么 Pointwise 数据很重要？
>
> **Pairwise 数据的痛点：**
> - 收集成本高：需要对同一 prompt 生成多个回答，再让标注员比较
> - 标注一致性差：不同标注员对同一对回答的偏好可能不同
> - 数据利用率低：如果只有单个回答的质量标签，无法用于 DPO
>
> **Pointwise 数据的优势：**
> - 收集简单：用户的 thumbs up/down、点赞/踩都是天然的 pointwise 信号
> - 数据量大：互联网上大量存在这类反馈
> - 标注直观：判断"这个回答好不好"比"A 和 B 哪个更好"更容易

---

## 理论基础：前景理论

KTO 的理论基础来自行为经济学中的**前景理论**（Prospect Theory），由 Daniel Kahneman 和 Amos Tversky 于 1979 年提出（Kahneman 因此获得 2002 年诺贝尔经济学奖）。

> [!definition] 前景理论的核心观点
>
> 人类对收益和损失的感知是**非对称**的：
> 1. **损失厌恶**（Loss Aversion）：损失带来的痛苦 > 等量收益带来的快乐
> 2. **参考点依赖**：人们评估结果时，是相对于某个参考点，而非绝对值
> 3. **边际递减**：收益/损失越大，边际效用递减

> [!intuition] 类比理解
>
> 丢了 100 块钱的难受程度，大约是捡到 100 块钱快乐程度的 **2 倍**。这就是损失厌恶。
>
> 在 LLM 对齐中：模型生成一个"坏"回答带来的负面影响，应该比生成一个"好"回答带来的正面影响更大。KTO 通过非对称的损失函数来建模这一点。

### 价值函数

前景理论用**价值函数** $v(z)$ 描述人对结果的主观感知：

$$
v(z) = \begin{cases}
z^\alpha & \text{if } z \geq 0 \text{ (gains)} \\
-\lambda (-z)^\alpha & \text{if } z < 0 \text{ (losses)}
\end{cases}
$$

其中：
- $z$ 是相对于参考点的结果（正为收益，负为损失）
- $\alpha < 1$ 控制边际递减程度
- $\lambda > 1$ 是损失厌恶系数（通常 $\lambda \approx 2$）

---

## KTO 损失函数

### 核心思想

KTO 将 LLM 对齐建模为：最大化模型在"好"回答上的效用，同时最小化在"坏"回答上的效用，且对"坏"回答施加更大的惩罚（损失厌恶）。

### 隐式奖励

与 [[DPO]] 类似，KTO 也使用隐式奖励的概念：

$$
r_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

但 KTO 不需要成对比较，而是将每个回答的奖励与一个**参考点**比较。

### KTO Loss

> [!definition] KTO 损失函数
>
> $$
> \mathcal{L}_{\text{KTO}}(\pi_\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}} \left[ w(y) \cdot \left(1 - v_{\text{KTO}}(r_\theta(x,y) - z_{\text{ref}}) \right) \right]
> $$
>
> 其中：
> - $w(y)$ 是样本权重（desirable 和 undesirable 可以不同）
> - $z_{\text{ref}}$ 是参考点（通常用 KL 散度的期望）
> - $v_{\text{KTO}}$ 是受前景理论启发的价值函数

### 简化形式

实践中，KTO 使用 sigmoid 函数近似价值函数：

> [!math] KTO Loss（实用形式）
>
> 对于 **desirable** 样本 $(x, y_w)$：
> $$
> \mathcal{L}_w = -\mathbb{E} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - z_{\text{ref}} \right) \right]
> $$
>
> 对于 **undesirable** 样本 $(x, y_l)$：
> $$
> \mathcal{L}_l = -\mathbb{E} \left[ \log \sigma\left( z_{\text{ref}} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
> $$
>
> 总损失（带损失厌恶系数 $\lambda$）：
> $$
> \mathcal{L}_{\text{KTO}} = \mathcal{L}_w + \lambda \cdot \mathcal{L}_l
> $$

> [!intuition] 直觉理解
>
> - **Desirable 样本**：希望 $r_\theta(x, y_w)$ 高于参考点 → 增加好回答的概率
> - **Undesirable 样本**：希望 $r_\theta(x, y_l)$ 低于参考点 → 降低坏回答的概率
> - **$\lambda > 1$**：对坏回答的惩罚更重，体现损失厌恶

---

## 与 DPO 的对比

> [!comparison] KTO vs DPO

| 方面 | DPO | KTO |
|------|-----|-----|
| **数据格式** | Pairwise $(x, y_w, y_l)$ | Pointwise $(x, y, \text{label})$ |
| **理论基础** | [[Bradley-Terry Model|Bradley-Terry 偏好模型]] | 前景理论（Prospect Theory） |
| **损失函数** | 对称（chosen vs rejected） | 非对称（损失厌恶） |
| **数据收集** | 需要成对比较 | 只需 thumbs up/down |
| **数据利用** | 必须有配对 | 可以独立使用每个样本 |
| **参考点** | 隐式（通过 pair 相减抵消） | 显式（$z_{\text{ref}}$） |

> [!intuition] 核心区别
>
> **DPO** 问的是："A 比 B 好多少？"（相对比较）
>
> **KTO** 问的是："A 好不好？"（绝对判断）
>
> 这使得 KTO 可以利用大量只有单一标签的数据，而 DPO 必须有成对数据。

### 数学联系

当 KTO 的数据恰好是成对的（每个 prompt 有一个 desirable 和一个 undesirable），且 $\lambda = 1$（无损失厌恶）时，KTO 与 DPO 有相似的行为。但 KTO 的优势在于：
1. 不要求数据成对
2. 通过 $\lambda$ 引入损失厌恶的归纳偏置

---

## 优势与局限

### 优势

> [!example] KTO 的实际优势
>
> 1. **数据收集更简单**
>    - 用户反馈天然是 pointwise：点赞、踩、thumbs up/down
>    - 不需要复杂的 A/B 比较标注流程
>
> 2. **数据利用率更高**
>    - 可以利用只有单一标签的数据
>    - 不需要为每个 prompt 生成多个回答
>
> 3. **理论基础更贴近人类行为**
>    - 前景理论是经过大量实验验证的人类决策模型
>    - 损失厌恶是人类的普遍心理特征
>
> 4. **实验效果**
>    - 论文显示 KTO 在多个 benchmark 上与 DPO 相当甚至更好
>    - 尤其在数据量较少时表现更稳定

### 局限

> [!warning] 潜在问题
>
> 1. **参考点选择**：$z_{\text{ref}}$ 的设定会影响训练效果
> 2. **数据不平衡**：desirable 和 undesirable 样本比例需要平衡
> 3. **超参数敏感**：$\lambda$（损失厌恶系数）需要调优
> 4. **与 DPO 共享的问题**：仍然是 offline 方法，有 [[DPO]] 类似的局限性

---

## 超参数

- **$\beta$**：KL 惩罚系数，与 DPO 中的 $\beta$ 作用相同
  - 控制策略偏离参考策略的程度
  - 典型值：0.1 ~ 0.5

- **$\lambda$**：损失厌恶系数
  - $\lambda > 1$ 表示对坏回答惩罚更重
  - 典型值：1.0 ~ 2.0（论文建议 $\lambda \approx 1.33$）

- **$z_{\text{ref}}$**：参考点
  - 通常设为 KL 散度的期望值
  - 可以用 batch 内的统计量估计

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: KTO 与 DPO 的核心区别是什么？**
> A: DPO 需要 pairwise 偏好数据（A 比 B 好），KTO 只需要 pointwise 反馈（A 好/不好）。这使得 KTO 可以利用更广泛的数据来源，如用户的 thumbs up/down。
>
> **Q2: KTO 的理论基础是什么？**
> A: 前景理论（Prospect Theory），核心是损失厌恶——人对损失的敏感度高于对等量收益的敏感度。KTO 通过非对称的损失函数（$\lambda > 1$）来建模这一点。
>
> **Q3: 为什么 pointwise 数据比 pairwise 数据更容易收集？**
> A: Pairwise 需要对同一 prompt 生成多个回答并比较，而 pointwise 只需要判断单个回答的好坏。用户的点赞/踩、thumbs up/down 都是天然的 pointwise 信号。
>
> **Q4: KTO 的损失函数与 DPO 有什么联系？**
> A: 当数据恰好成对且 $\lambda = 1$ 时，KTO 与 DPO 行为相似。但 KTO 不要求数据成对，且通过 $\lambda$ 引入损失厌恶的归纳偏置。

---

## 相关概念

- [[IPO]] — 另一种 DPO 变体，解决 DPO 的过拟合问题
- [[SimPO]] — 无需参考模型的简化方法

---

## 延伸阅读

- [[InstructGPT]] — RLHF 的经典工作
- [[Constitutional AI]] — 另一种对齐思路
- Kahneman & Tversky (1979), "Prospect Theory: An Analysis of Decision under Risk" — 前景理论原始论文
