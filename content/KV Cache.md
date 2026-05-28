---
description: 缓存已计算的 Key/Value 向量避免重复计算，是 LLM 推理的核心优化
type: concept
prerequisites:
  - "[[Attention]]"
  - "[[Multi-Head Attention]]"
tags:
  - inference
  - attention
  - optimization
created: 2025-01-26
updated: 2026-01-31T22:42
---

# KV Cache

KV Cache（Key-Value Cache）是 LLM 推理中的核心优化技术，通过缓存已计算的 Key 和 Value 向量，避免自回归生成时的重复计算。

---

## 为什么需要 KV Cache

> [!intuition] 直觉理解
> 自回归生成时，每生成一个新 token 都需要对所有历史 token 做 Attention。如果每次都重新计算所有 token 的 K/V，计算量会随序列长度平方增长。KV Cache 把历史 token 的 K/V 缓存起来，每步只需计算新 token 的 K/V。

### 自回归生成的计算模式

生成第 $t$ 个 token 时，需要计算：

$$
\text{Attention}(q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}
$$

其中 $K_{1:t}$ 和 $V_{1:t}$ 是前 $t$ 个 token 的 Key 和 Value。

**无 KV Cache**：每步重新计算所有 K/V
- 第 1 步：计算 $K_1, V_1$
- 第 2 步：计算 $K_1, K_2, V_1, V_2$（$K_1, V_1$ 重复计算）
- 第 $t$ 步：计算 $K_{1:t}, V_{1:t}$（前 $t-1$ 个重复计算）

**有 KV Cache**：缓存历史 K/V，每步只计算新 token
- 第 1 步：计算 $K_1, V_1$，缓存
- 第 2 步：计算 $K_2, V_2$，拼接缓存
- 第 $t$ 步：计算 $K_t, V_t$，拼接缓存

### 计算量对比

生成长度为 $L$ 的序列：

| 方式 | 总计算量 |
|------|----------|
| 无 KV Cache | $O(L^2 \cdot d)$ |
| 有 KV Cache | $O(L \cdot d)$ |

---

## KV Cache 的内存占用

> [!math] 内存计算
> 对于一个 batch size 为 $B$、序列长度为 $L$ 的推理任务：
>
> $$
> \text{KV Cache 大小} = 2 \times B \times L \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes}
> $$
>
> 其中 2 表示 K 和 V 两个缓存。

### 实际例子：Llama 2 70B

| 参数 | 值 |
|------|-----|
| 层数 | 80 |
| 注意力头数 | 64 (GQA: 8 KV heads) |
| 头维度 | 128 |
| 精度 | FP16 (2 bytes) |

单个请求、序列长度 4096：
$$
2 \times 1 \times 4096 \times 80 \times 8 \times 128 \times 2 = 1.34 \text{ GB}
$$

> [!warning] 常见误区
> KV Cache 的内存占用与 batch size 和序列长度**线性相关**，在长序列或大 batch 场景下会成为主要瓶颈，甚至超过模型参数本身的显存占用。

---

## KV Cache 的核心问题

### 1. 内存碎片

传统实现为每个请求预分配最大长度的连续内存：

```
请求 A: [████████░░░░░░░░] 实际用 8，预留 16
请求 B: [██████░░░░░░░░░░] 实际用 6，预留 16
请求 C: [████░░░░░░░░░░░░] 实际用 4，预留 16
```

**问题**：
- 内部碎片：预分配但未使用的空间
- 外部碎片：请求结束后留下的不连续空闲块
- 内存利用率低，限制并发请求数

### 2. 无法共享

不同请求可能有相同的前缀（如 system prompt），但传统实现无法共享：

```
请求 A: [System Prompt][User Query A]
请求 B: [System Prompt][User Query B]
         ↑ 相同内容，重复存储
```

### 3. 动态长度

生成长度事先未知，难以精确分配内存：
- 分配过多：浪费内存
- 分配过少：需要重新分配和拷贝

---

## 解决方案

### 架构层面：MQA / GQA

通过减少 KV heads 数量来降低 KV Cache 大小：

| 方法 | KV heads | Cache 大小 |
|------|----------|------------|
| [[Multi-Head Attention\|MHA]] | $h$ | $1\times$ |
| [[Grouped-Query Attention\|GQA]] | $g$ | $g/h \times$ |
| [[Multi-Query Attention\|MQA]] | $1$ | $1/h \times$ |

### 系统层面：内存管理优化

| 方法 | 核心思想 | 解决问题 |
|------|----------|----------|
| [[Paged Attention]] | 分页内存管理 | 内存碎片 |
| [[Radix Attention]] | 前缀树共享 | 重复存储 |

---

## 与 Flash Attention 的关系

> [!comparison] 对比
> - **KV Cache**：解决"存什么"——避免重复计算，缓存历史 K/V
> - **[[Flash Attention]]**：解决"怎么算"——优化 Attention 计算的 IO 效率
>
> 两者是正交的优化，可以同时使用。

---

## 面试要点

> [!interview] 面试要点
> **Q: 为什么需要 KV Cache？**
> A: 自回归生成时避免重复计算历史 token 的 K/V，将计算复杂度从 $O(L^2)$ 降到 $O(L)$。
>
> **Q: KV Cache 的主要问题是什么？**
> A: 内存占用大（与序列长度线性相关）、内存碎片（预分配导致）、无法共享（相同前缀重复存储）。
>
> **Q: 如何减少 KV Cache 的内存占用？**
> A: 架构层面用 MQA/GQA 减少 KV heads；系统层面用 Paged Attention 减少碎片、Radix Attention 实现共享。

---

## 参考资料

- vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)
- Efficient Memory Management for Large Language Model Serving with PagedAttention
- SGLang: Efficient Execution of Structured Language Model Programs (2024)
