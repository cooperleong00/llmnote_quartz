---
description: 衡量两个概率分布差异的非对称度量，在 RLHF 中用于约束策略偏离
type: concept
aliases:
  - KL 散度
  - 相对熵
  - Relative Entropy
prerequisites:
  - "[[Probability Distribution]]"
  - "[[Entropy]]"
tags:
  - math
  - information-theory
created: 2025-01-25
updated: 2026-02-01T01:04
---

# KL Divergence

KL 散度（Kullback-Leibler Divergence），也称相对熵（Relative Entropy），是衡量两个概率分布差异的指标。它在 [[RLHF]] 和 [[DPO]] 中扮演核心角色，用于约束模型不要偏离参考分布太远。

---

## 定义

> [!definition] KL 散度
> 对于两个概率分布 $P$ 和 $Q$，$P$ 相对于 $Q$ 的 KL 散度定义为：
>
> **离散情况**：
> $$
> D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
> $$
>
> **连续情况**：
> $$
> D_{KL}(P \| Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx
> $$

也可以写成期望形式：

$$
D_{KL}(P \| Q) = \mathbb{E}_{x \sim P} \left[ \log \frac{P(x)}{Q(x)} \right]
$$

---

## 直觉理解

> [!intuition] 为什么叫"散度"而非"距离"？
> KL 散度衡量的是：**用分布 $Q$ 来近似分布 $P$ 时，损失了多少信息**。
>
> - $D_{KL}(P \| Q) = 0$ 当且仅当 $P = Q$
> - $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$（不对称！）
>
> 因为不对称，所以不是严格的"距离"，而叫"散度"。

### 信息论视角

$$
D_{KL}(P \| Q) = H(P, Q) - H(P)
$$

其中：
- $H(P)$ 是 $P$ 的[[Entropy|熵]]（用最优编码表示 $P$ 所需的平均比特数）
- $H(P, Q)$ 是[[Cross-Entropy|交叉熵]]（用基于 $Q$ 的编码表示 $P$ 所需的平均比特数）

**直觉**：KL 散度 = 用错误编码（$Q$）的额外代价。

---

## 核心性质

### 1. 非负性

$$
D_{KL}(P \| Q) \geq 0
$$

等号成立当且仅当 $P = Q$（几乎处处）。

> [!math] 证明（Jensen 不等式）
> $$
> D_{KL}(P \| Q) = -\mathbb{E}_P \left[ \log \frac{Q}{P} \right] \geq -\log \mathbb{E}_P \left[ \frac{Q}{P} \right] = -\log 1 = 0
> $$

### 2. 非对称性

$$
D_{KL}(P \| Q) \neq D_{KL}(Q \| P)
$$

这两个方向有不同的含义：
- $D_{KL}(P \| Q)$：**前向 KL**，$Q$ 需要覆盖 $P$ 的所有支撑
- $D_{KL}(Q \| P)$：**反向 KL**，$Q$ 倾向于集中在 $P$ 的高概率区域

> [!intuition] Mode-seeking vs Mode-covering
> - **前向 KL** $D_{KL}(P \| Q)$：mode-covering，$Q$ 会变得更分散
> - **反向 KL** $D_{KL}(Q \| P)$：mode-seeking，$Q$ 会集中在 $P$ 的某个峰

### 3. 与交叉熵的关系

$$
D_{KL}(P \| Q) = H(P, Q) - H(P)
$$

当 $P$ 固定时，最小化 KL 散度等价于最小化交叉熵。

---

## 在 LLM 中的应用

### 1. RLHF 中的 KL 惩罚

[[RLHF]] 的优化目标：

$$
\max_{\pi_\theta} \mathbb{E}_{x, y \sim \pi_\theta} \left[ r(x, y) \right] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})
$$

**作用**：
- 防止 [[Reward Hacking]]：模型不能为了高分而生成奇怪的回答
- 保持生成质量：约束模型在"正常回答"的分布附近
- $\beta$ 控制约束强度：$\beta$ 大则保守，$\beta$ 小则激进

**实现方式**：通常将 KL 惩罚加到 reward 上：

$$
r_{total}(x, y) = r(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

### 2. DPO 中的隐式 KL

[[DPO]] 的推导基于 KL 约束 RL 目标的解析解。最优策略为：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left( \frac{r(x,y)}{\beta} \right)
$$

这个形式本身就包含了对 $\pi_{ref}$ 的 KL 约束。DPO 虽然没有显式计算 KL，但通过 $\beta$ 参数隐式控制了偏离程度。

### 3. 知识蒸馏

用 KL 散度让学生模型 $Q$ 模仿教师模型 $P$：

$$
\mathcal{L}_{KD} = D_{KL}(P_{teacher} \| Q_{student})
$$

---

## 与其他度量的对比

| 度量 | 对称性 | 三角不等式 | 特点 |
|------|--------|------------|------|
| **KL 散度** | ❌ | ❌ | 信息论基础，计算简单 |
| **[[JS Divergence|JS 散度]]** | ✅ | ❌ | $JS = \frac{1}{2}KL(P\|M) + \frac{1}{2}KL(Q\|M)$，$M=\frac{P+Q}{2}$ |
| **Wasserstein** | ✅ | ✅ | 考虑分布的几何结构，计算较贵 |
| **Total Variation** | ✅ | ✅ | $TV = \frac{1}{2}\sum|P-Q|$，上界 |

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: KL 散度的定义是什么？**
> A: $D_{KL}(P \| Q) = \mathbb{E}_P[\log \frac{P}{Q}]$，衡量用 $Q$ 近似 $P$ 时损失的信息量。
>
> **Q2: KL 散度为什么不是距离？**
> A: 因为不对称：$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$。前向 KL 是 mode-covering，反向 KL 是 mode-seeking。
>
> **Q3: RLHF 中 KL 惩罚的作用？**
> A: 防止 reward hacking，约束模型不要偏离参考策略太远，保持生成质量。
>
> **Q4: KL 散度和交叉熵的关系？**
> A: $D_{KL}(P \| Q) = H(P, Q) - H(P)$。当 $P$ 固定时，最小化 KL 等价于最小化交叉熵。

---

## 相关概念

- [[RLHF]] — KL 惩罚的主要应用
- [[DPO]] — 隐式 KL 约束
