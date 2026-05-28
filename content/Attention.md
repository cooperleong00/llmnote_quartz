---
description: Transformer 的核心机制，通过 Query-Key-Value 动态计算序列中各位置的相关性
type: concept
prerequisites: []
tags:
  - transformer
  - architecture
  - foundations
created: 2025-01-26
updated: 2026-02-01T00:57
---

# Attention

注意力机制（Attention）是 [[Transformer]] 架构的核心组件，允许模型在处理序列时动态地关注不同位置的信息。它解决了 RNN 的长距离依赖问题，成为现代 LLM 的基础。

Attention 本身不包含位置信息（对输入顺序不敏感），因此需要配合 [[Positional Encoding]] 使用。实践中，通常使用 [[Multi-Head Attention]] 来增强表达能力，让模型同时关注不同子空间的信息。

> [!paper] 论文出处
> Vaswani et al., "Attention Is All You Need", NeurIPS 2017

---

## 核心思想

> [!intuition] 直觉理解
> Attention 的本质是**加权求和**：给定一个查询（Query），从一组键值对（Key-Value）中检索相关信息。
>
> 类比：在图书馆找书时，你的问题是 Query，书的标题是 Key，书的内容是 Value。你根据问题和标题的匹配程度，决定从哪些书中提取多少内容。

---

## 数学定义

### Scaled Dot-Product Attention

> [!definition] Scaled Dot-Product Attention
> $$
> \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
> $$
>
> 其中：
> - $Q \in \mathbb{R}^{n \times d_k}$：Query 矩阵
> - $K \in \mathbb{R}^{m \times d_k}$：Key 矩阵
> - $V \in \mathbb{R}^{m \times d_v}$：Value 矩阵
> - $d_k$：Key 的维度
> - $n$：Query 序列长度，$m$：Key/Value 序列长度

### Q/K/V 的直觉

> [!intuition] Q/K/V 分别是什么？
> - **Query (Q)**：当前位置想要查询的信息，"我在找什么？"
> - **Key (K)**：每个位置的索引/标签，"我有什么？"
> - **Value (V)**：每个位置的实际内容，"我能提供什么？"
>
> 注意力权重 $\text{softmax}(QK^T/\sqrt{d_k})$ 衡量 Query 和每个 Key 的相关性，然后用这个权重对 Value 加权求和。

### 为什么要除以 $\sqrt{d_k}$？

> [!math] Scaling 的原因
> 当 $d_k$ 较大时，$QK^T$ 的方差会随 $d_k$ 增大。假设 $Q$ 和 $K$ 的元素独立同分布，均值为 0，方差为 1：
>
> $$
> \text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k
> $$
>
> 点积的方差是 $d_k$，标准差是 $\sqrt{d_k}$。除以 $\sqrt{d_k}$ 使方差归一化为 1，避免 softmax 进入饱和区（梯度消失）。

---

## 复杂度分析

### 时间复杂度

> [!definition] 时间复杂度
> $$
> O(n^2 \cdot d)
> $$
>
> 其中 $n$ 是序列长度，$d$ 是特征维度。

**分解**：
1. $QK^T$ 计算：$O(n \times d) \times (d \times n) = O(n^2 d)$
2. Softmax：$O(n^2)$
3. 与 $V$ 相乘：$O(n^2 d)$

> [!warning] 二次复杂度的问题
> 当序列长度 $n$ 增大时，计算量和内存都以 $O(n^2)$ 增长。这是 Transformer 处理长序列的主要瓶颈，催生了各种高效 Attention 变体（如 [[Flash Attention]]、[[Sparse Attention]]）。推理阶段可以使用 [[KV Cache]] 避免重复计算历史 token 的 Key/Value。

### 空间复杂度

> [!definition] 空间复杂度
> $$
> O(n^2)
> $$

主要来自存储 $n \times n$ 的注意力矩阵。对于长序列，这是显存的主要消耗。

---

## Self-Attention vs Cross-Attention

| 类型 | Q 来源 | K/V 来源 | 用途 |
|------|--------|----------|------|
| **Self-Attention** | 同一序列 | 同一序列 | 序列内部建模 |
| **Cross-Attention** | 序列 A | 序列 B | 序列间交互 |

### Self-Attention

在 Self-Attention 中，Q、K、V 都来自同一个输入序列 $X$：

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

**作用**：让序列中的每个位置都能关注到其他所有位置，捕捉全局依赖。

### Cross-Attention

在 Cross-Attention 中，Q 来自一个序列，K/V 来自另一个序列：

$$
Q = X_1 W^Q, \quad K = X_2 W^K, \quad V = X_2 W^V
$$

**典型应用**：
- Encoder-Decoder 架构中，Decoder 用 Cross-Attention 关注 Encoder 的输出
- 多模态模型中，文本关注图像特征

---

## Causal Attention (Masked Self-Attention)

在自回归语言模型中，需要防止当前位置看到未来的信息：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
$$

其中 $M$ 是 causal mask：

$$
M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}
$$

> [!intuition] Causal Mask 的作用
> 将未来位置的注意力权重设为 $-\infty$，经过 softmax 后变为 0，确保位置 $i$ 只能看到位置 $\leq i$ 的信息。

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: Attention 的核心公式是什么？**
> A: $\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k}) V$
>
> **Q2: 为什么要除以 $\sqrt{d_k}$？**
> A: 防止点积过大导致 softmax 饱和。点积的方差与 $d_k$ 成正比，除以 $\sqrt{d_k}$ 归一化方差。
>
> **Q3: Attention 的时间复杂度是多少？为什么是瓶颈？**
> A: $O(n^2 d)$，对序列长度是二次的。长序列时计算量和内存都会爆炸。
>
> **Q4: Self-Attention 和 Cross-Attention 的区别？**
> A: Self-Attention 的 Q/K/V 来自同一序列，用于序列内部建模；Cross-Attention 的 Q 和 K/V 来自不同序列，用于序列间交互。
>
> **Q5: Causal Mask 的作用是什么？**
> A: 在自回归模型中，防止当前位置看到未来信息，保证生成时的因果性。
