---
description: 并行运行多个注意力头关注不同子空间信息，是 Transformer 的核心组件
type: concept
prerequisites:
  - "[[Attention]]"
tags:
  - transformer
  - architecture
  - foundations
created: 2025-01-26
updated: 2026-02-01T00:57
---

# Multi-Head Attention

多头注意力（Multi-Head Attention, MHA）是 [[Transformer]] 的核心组件，通过并行运行多个 [[Attention|注意力]]头，让模型能够同时关注不同子空间的信息。

> [!paper] 论文出处
> Vaswani et al., "Attention Is All You Need", NeurIPS 2017

---

## 为什么需要多头？

> [!intuition] 单头 Attention 的局限
> 单头 Attention 只能学习一种"关注模式"。但语言中的依赖关系是多样的：
> - 语法依赖：主语-谓语-宾语
> - 语义依赖：指代消解、同义词
> - 位置依赖：相邻词、远距离依赖
>
> **多头的作用**：让不同的头学习不同的关注模式，然后综合起来。

> [!example] 多头的直觉
> 想象你在读一篇文章：
> - 头 1：关注语法结构（主谓宾）
> - 头 2：关注指代关系（"他"指谁）
> - 头 3：关注相邻词（局部上下文）
> - 头 4：关注关键词（主题相关）
>
> 每个头专注于不同的"视角"，最后综合得到完整理解。

---

## 数学定义

> [!definition] Multi-Head Attention
> $$
> \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
> $$
>
> 其中每个头：
> $$
> \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
> $$
>
> 参数：
> - $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
> - $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
> - $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
> - $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

### 维度设计

通常设置 $d_k = d_v = d_{model} / h$，使得：
- 每个头的维度较小（如 64）
- 所有头拼接后恢复原始维度
- 总计算量与单头大维度 Attention 相当

> [!example] 典型配置
> GPT-3 175B：$d_{model} = 12288$，$h = 96$，$d_k = d_v = 128$

---

## 参数量分析

对于一个 Multi-Head Attention 层：

| 参数 | 形状 | 数量 |
|------|------|------|
| $W^Q$ | $d_{model} \times d_{model}$ | $d_{model}^2$ |
| $W^K$ | $d_{model} \times d_{model}$ | $d_{model}^2$ |
| $W^V$ | $d_{model} \times d_{model}$ | $d_{model}^2$ |
| $W^O$ | $d_{model} \times d_{model}$ | $d_{model}^2$ |
| **总计** | | $4 d_{model}^2$ |

> [!math] 为什么 $W^Q$ 是 $d_{model} \times d_{model}$？
> 虽然每个头的 $W_i^Q$ 是 $d_{model} \times d_k$，但 $h$ 个头的参数可以合并：
> $$
> W^Q = [W_1^Q; W_2^Q; ...; W_h^Q] \in \mathbb{R}^{d_{model} \times (h \cdot d_k)} = \mathbb{R}^{d_{model} \times d_{model}}
> $$

---

## 与单头 Attention 的对比

| 方面 | 单头 Attention | Multi-Head Attention |
|------|----------------|----------------------|
| **表达能力** | 单一关注模式 | 多种关注模式 |
| **子空间** | 在完整空间计算 | 在多个子空间并行计算 |
| **计算量** | $O(n^2 d)$ | $O(n^2 d)$（相同） |
| **参数量** | $3d^2$（Q/K/V） | $4d^2$（Q/K/V/O） |

> [!intuition] 为什么计算量相同？
> 假设单头用 $d_{model}$ 维，多头用 $h$ 个 $d_{model}/h$ 维的头：
> - 单头：$O(n^2 \cdot d_{model})$
> - 多头：$h \times O(n^2 \cdot d_{model}/h) = O(n^2 \cdot d_{model})$
>
> 多头只是把计算"分散"到多个子空间，总量不变。

---

## 多头的可视化理解

```
输入 X ─┬─→ W^Q ─→ Q ─┬─→ head_1 ─┐
        │             │           │
        ├─→ W^K ─→ K ─┼─→ head_2 ─┼─→ Concat ─→ W^O ─→ 输出
        │             │           │
        └─→ W^V ─→ V ─┴─→ head_h ─┘
```

每个头在不同的子空间中计算注意力，最后拼接并投影回原始维度。

---

## 多头学到了什么？

研究表明，不同的头确实学到了不同的模式：

> [!example] 头的专业化
> - **语法头**：关注依存关系（如动词关注其主语）
> - **位置头**：关注相邻位置
> - **稀有头**：关注特定的罕见模式
> - **冗余头**：一些头可能是冗余的（可以剪枝）

这也是 [[Multi-Query Attention]] 和 [[Grouped-Query Attention]] 的理论基础：既然有些头是冗余的，能否共享参数？

---

## 实现细节

### 高效实现

实际实现中，不会真的分开计算每个头，而是：

```python
# 合并计算 Q/K/V
Q = X @ W_Q  # (batch, seq, d_model)
K = X @ W_K
V = X @ W_V

# 重塑为多头形式
Q = Q.view(batch, seq, h, d_k).transpose(1, 2)  # (batch, h, seq, d_k)
K = K.view(batch, seq, h, d_k).transpose(1, 2)
V = V.view(batch, seq, h, d_v).transpose(1, 2)

# 并行计算所有头的注意力
attn = softmax(Q @ K.T / sqrt(d_k)) @ V  # (batch, h, seq, d_v)

# 合并头并投影
out = attn.transpose(1, 2).view(batch, seq, d_model) @ W_O
```

进一步的优化包括：使用 [[Flash Attention]] 减少内存访问开销，以及在推理时使用 [[KV Cache]] 避免重复计算历史 token 的 K/V。

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: 为什么要用 Multi-Head Attention？**
> A: 让模型能同时关注不同子空间的信息，学习多种关注模式（语法、语义、位置等）。
>
> **Q2: Multi-Head Attention 的计算量比单头大吗？**
> A: 不大。每个头的维度是 $d_{model}/h$，$h$ 个头的总计算量与单头 $d_{model}$ 维相同。
>
> **Q3: 多头的参数量是多少？**
> A: $4d_{model}^2$（Q/K/V/O 各 $d_{model}^2$）。
>
> **Q4: 不同的头学到了什么？**
> A: 研究表明不同头会专业化：有的关注语法、有的关注位置、有的关注语义。但也存在冗余头。
>
> **Q5: 为什么 $d_k = d_{model}/h$？**
> A: 保持总计算量不变。每个头在较小的子空间计算，所有头拼接后恢复原始维度。

---

## 相关概念

**MHA 的变体**：
- [[Multi-Query Attention]] — 所有头共享 K/V，减少 KV Cache 大小
- [[Grouped-Query Attention]] — 分组共享 K/V，平衡效率与表达能力
