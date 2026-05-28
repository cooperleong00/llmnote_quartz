---
description: 局部注意力机制，每个 token 只关注固定大小的窗口，将复杂度从 O(N^2) 降至 O(N*W)
type: method
aliases:
  - SWA
  - 滑动窗口注意力
prerequisites:
  - "[[Attention]]"
  - "[[KV Cache]]"
tags:
  - attention
  - efficient-attention
  - inference
created: 2026-01-27
updated: 2026-02-01T01:12
---

# Sliding Window Attention

Sliding Window Attention（SWA，滑动窗口注意力）是一种高效注意力机制，每个 token 只关注固定大小的局部窗口，而非整个序列。它将 Attention 的复杂度从 $O(N^2)$ 降至 $O(N \cdot W)$，是处理长序列的重要方法。

---

## 动机

> [!intuition] 为什么需要 SWA？
> 标准 [[Attention]] 的 $O(N^2)$ 复杂度在长序列时不可接受。但很多任务中，**局部上下文已足够**——语言建模中相邻词的依赖通常比远距离依赖更重要。SWA 利用这一先验，用局部窗口换取效率。

### 标准 Attention 的问题

| 序列长度 | Attention 计算量 | [[KV Cache]] 大小 |
|----------|------------------|---------------|
| 1K | $1 \times$ | $1 \times$ |
| 4K | $16 \times$ | $4 \times$ |
| 32K | $1024 \times$ | $32 \times$ |
| 128K | $16384 \times$ | $128 \times$ |

长序列时，计算量和内存都会爆炸。

---

## 核心机制

> [!definition] Sliding Window Attention
> 给定窗口大小 $W$，位置 $i$ 的 token 只能 attend 到位置 $[i-W+1, i]$ 的 tokens：
> $$
> \text{SWA}(Q, K, V)_i = \text{softmax}\left(\frac{q_i K_{[i-W+1:i]}^T}{\sqrt{d_k}}\right) V_{[i-W+1:i]}
> $$

### 直觉理解

> [!intuition] 滑动窗口的本质
> 想象一个固定大小的"视野窗口"随着序列滑动。每个 token 只能看到窗口内的内容，窗口外的信息被遮蔽。
>
> ```
> 序列:  [t1] [t2] [t3] [t4] [t5] [t6] [t7] [t8]
>                        ↑
>                    当前位置 t5
>
> W=4:   [t2] [t3] [t4] [t5]  ← t5 只能看到这 4 个 token
>              └─────────┘
>                窗口范围
> ```

### 复杂度分析

> [!math] 复杂度对比
> | 指标 | 标准 Attention | SWA |
> |------|----------------|-----|
> | 时间复杂度 | $O(N^2 \cdot d)$ | $O(N \cdot W \cdot d)$ |
> | KV Cache | $O(N \cdot d)$ | $O(W \cdot d)$ |
>
> 当 $W \ll N$ 时，SWA 的效率优势显著。

---

## 信息传播与有效感受野

> [!intuition] 多层堆叠扩展感受野
> 虽然单层 SWA 只能看到 $W$ 个 token，但多层堆叠后，信息可以逐层传播：
>
> - 第 1 层：每个 token 看到 $W$ 个 token
> - 第 2 层：每个 token 间接看到 $2W$ 个 token
> - 第 $L$ 层：有效感受野达到 $L \cdot W$ 个 token
>
> ```
> Layer 3:  [─────────────────────────────]  感受野: 3W
> Layer 2:  [─────────────────────]          感受野: 2W
> Layer 1:  [───────────]                    感受野: W
>                 ↑
>             当前位置
> ```

这意味着即使窗口很小，深层网络仍能捕捉长程依赖——只是需要更多层来传播信息。

---

## Hybrid SWA 架构

纯 SWA 在某些任务上会有性能损失。现代架构通常采用 **Hybrid SWA**：交替使用 SWA 和 Global Attention (GA)。

### MiMo-V2-Flash 的设计

> [!example] MiMo-V2-Flash 架构
> - **比例**：5:1（5 层 SWA + 1 层 GA）
> - **窗口大小**：仅 128 tokens
> - **效果**：约 6x KV Cache 节省
>
> ```
> Layer 1-5:   SWA (W=128)
> Layer 6:     Global Attention
> Layer 7-11:  SWA (W=128)
> Layer 12:    Global Attention
> ...
> ```

### 为什么 Hybrid 更好？

> [!intuition] 分工协作
> - **SWA 层**：专注局部模式（语法、短语结构）
> - **GA 层**：处理长程依赖（指代消解、全局语义）
>
> 小窗口**强制** SWA 专注局部，让 GA 层承担长程依赖的责任，形成更好的分工。

---

## Attention Sink Bias

> [!warning] 纯 SWA 的问题
> 纯 SWA 会导致性能退化。原因之一是模型有时需要"不关注任何 token"（如 padding 或无关位置），但 softmax 强制注意力权重和为 1。

### 解决方案：Learnable Sink Bias

引入可学习的 sink bias，允许模型在不需要时"不关注任何 token"：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + b_{\text{sink}}\right) V
$$

> [!example] MiMo-V2-Flash 的发现
> - W=128 + sink bias **优于** W=512（无 sink bias）
> - Hybrid SWA 甚至可以**超越** All-GA baseline
>
> 这说明 sink bias 不只是修复问题，而是让模型学到更好的注意力模式。

---

## 与其他高效 Attention 的对比

> [!comparison] 方法对比
> | 方法 | 类型 | 核心思想 | 特点 |
> |------|------|----------|------|
> | **SWA** | 稀疏模式 | 局部窗口 | 简单、保留 softmax |
> | [[Sparse Attention]] | 稀疏模式 | 多种稀疏模式组合 | 更灵活、更复杂 |
> | [[Linear Attention]] | 近似 | 去掉 softmax | 线性复杂度、可能不稳定 |
> | [[Flash Attention]] | 实现优化 | IO-aware 计算 | 精确、不改变结果 |

**关键区别**：
- SWA 是**架构设计**，改变了 Attention 的计算范围
- Flash Attention 是**实现优化**，不改变计算结果
- 两者可以**同时使用**：用 Flash Attention 高效实现 SWA

---

## 实际应用

### 使用 SWA 的模型

| 模型 | 窗口大小 | 架构 |
|------|----------|------|
| Longformer | 512 | SWA + Global (特定位置) |
| BigBird | 64 | SWA + Random + Global |
| Mistral 7B | 4096 | SWA |
| MiMo-V2-Flash | 128 | Hybrid SWA (5:1) |

### 适用场景

> [!intuition] 何时使用 SWA？
> - **长文档处理**：文档长度远超训练长度
> - **流式生成**：需要固定内存的实时生成
> - **资源受限**：KV Cache 是瓶颈时

---

## 局限性

> [!warning] SWA 的边界条件
> 1. **全局依赖任务**：需要全局信息的任务（如文档级 QA）可能受损
> 2. **窗口大小选择**：太小损失信息，太大失去效率优势
> 3. **训练-推理不一致**：如果训练用全局 Attention，推理用 SWA，可能有性能差距
> 4. **需要配合技巧**：纯 SWA 效果不佳，通常需要 sink bias 或 hybrid 架构

---

## 面试要点

> [!interview] 面试视角
> **Q: SWA 的核心思想是什么？**
> A: 每个 token 只关注固定大小的局部窗口，将复杂度从 $O(N^2)$ 降至 $O(N \cdot W)$。
>
> **Q: SWA 如何处理长程依赖？**
> A: 通过多层堆叠，信息逐层传播。$L$ 层后有效感受野达到 $L \cdot W$。也可以用 Hybrid 架构，交替 SWA 和 Global Attention。
>
> **Q: SWA 和 Flash Attention 的区别？**
> A: SWA 是架构设计，改变 Attention 的计算范围；Flash Attention 是实现优化，不改变计算结果。两者正交，可以同时使用。
>
> **Q: 什么是 Attention Sink？**
> A: 模型有时需要"不关注任何 token"，但 softmax 强制权重和为 1。Sink bias 提供一个可学习的"垃圾桶"位置，让模型可以把不需要的注意力放到那里。

---

## 相关概念

- [[Attention]] — 标准注意力机制
- [[Flash Attention]] — 可以高效实现 SWA
- [[Sparse Attention]] — 更通用的稀疏注意力
- [[Linear Attention]] — 另一种高效 Attention 方案
- [[Length Extrapolation]] — 长度外推问题

---

## 参考资料

- Longformer: The Long-Document Transformer (Beltagy et al., 2020) — 首次系统提出 SWA
- BigBird: Transformers for Longer Sequences (Zaheer et al., 2020) — SWA + Random + Global
- Mistral 7B (Jiang et al., 2023) — 在主流模型中使用 SWA
- MiMo-V2-Flash (2025) — Hybrid SWA + Attention Sink Bias
