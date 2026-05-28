---
description: Attention 机制导航：核心机制、效率变体、位置编码
type: moc
created: 2025-01-26
updated: 2026-01-28T21:07
---

# MOC - Attention

注意力机制（Attention）是 Transformer 架构的核心，也是现代 LLM 的基础。本 MOC 组织 Attention 相关的所有概念，包括核心机制、效率变体和位置编码。

---

## 概览

```
Attention 机制
├── 核心机制
│   ├── [[Attention]] — Scaled Dot-Product Attention
│   └── [[Multi-Head Attention]] — 多头注意力
│
├── 效率变体（KV Cache 优化）
│   ├── [[Multi-Query Attention]] — 所有头共享 K/V
│   ├── [[Grouped-Query Attention]] — 分组共享 K/V
│   └── [[Multi-head Latent Attention]] — 低秩压缩 K/V (DeepSeek)
│
├── Efficient Attention
│   ├── 计算层: [[Flash Attention]] — IO-aware 算法
│   └── 服务层: [[KV Cache]] → [[Paged Attention]] / [[Radix Attention]]
│
└── 位置编码
    ├── [[Positional Encoding]] — 基础概念
    ├── [[RoPE]] — 旋转位置编码
    └── [[ALiBi]] — 注意力线性偏置
```

---

## 核心机制

### Scaled Dot-Product Attention

[[Attention]] 是 Transformer 的基础组件：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**关键特性**：
- 时间复杂度：$O(n^2 d)$
- 空间复杂度：$O(n^2)$
- 置换不变性（需要位置编码）

### Multi-Head Attention

[[Multi-Head Attention]] 通过并行多个注意力头，让模型学习不同的关注模式：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

**设计动机**：
- 不同头关注不同子空间
- 学习多种依赖关系（语法、语义、位置）
- 计算量与单头相同

---

## 效率变体

### KV Cache 问题

自回归生成时，需要缓存历史 token 的 Key 和 Value。标准 MHA 的 KV Cache 随头数线性增长，成为长序列推理的瓶颈。

### 解决方案对比

| 方法 | 核心思想 | KV Cache | 质量 | 代表模型 |
|------|----------|----------|------|----------|
| [[Multi-Head Attention\|MHA]] | 每头独立 K/V | $h \times$ | 最好 | GPT-3, Llama 1 |
| [[Grouped-Query Attention\|GQA]] | 分组共享 K/V | $g \times$ | 很好 | Llama 2/3, Mistral |
| [[Multi-Query Attention\|MQA]] | 全部共享 K/V | $1 \times$ | 较好 | PaLM, Falcon |
| [[Multi-head Latent Attention\|MLA]] | 低秩压缩 K/V | 极小 | 很好 | DeepSeek-V2/V3 |

**主流选择**：
- 小模型：MHA（Llama 2 7B/13B）
- 大模型：GQA（Llama 2 70B、Llama 3、Mistral）
- 超大模型：MLA（DeepSeek-V3 671B）

> [!intuition] MQA/GQA vs MLA 的本质区别
> - **MQA/GQA**：减少 K/V 的**数量**（共享）
> - **MLA**：减少 K/V 的**维度**（压缩）
>
> MLA 通过低秩分解保留每个头的独立性，同时实现更激进的压缩（DeepSeek-V3 约 57 倍）。

---

## 位置编码

### 为什么需要位置编码？

[[Attention]] 是置换不变的，无法区分 token 顺序。[[Positional Encoding]] 注入位置信息。

### 主流方案

| 方法 | 类型 | 长度外推 | 代表模型 |
|------|------|----------|----------|
| Sinusoidal | 绝对 | 差 | 原始 Transformer |
| 可学习 PE | 绝对 | 无 | BERT, GPT-2 |
| [[RoPE]] | 相对 | 中等 | Llama, Qwen, Mistral |
| [[ALiBi]] | 相对 | 好 | BLOOM, MPT |

**当前主流**：[[RoPE]]（Llama 系列的成功推动）

---

## 复杂度总结

| 操作 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| Attention 计算 | $O(n^2 d)$ | $O(n^2)$ |
| RoPE | $O(nd)$ | $O(1)$ |
| ALiBi | $O(n^2)$ | $O(n^2)$ |
| KV Cache (MHA) | — | $O(Lhnd_k)$ |
| KV Cache (GQA) | — | $O(Lgnd_k)$ |
| KV Cache (MLA) | — | $O(L(d_c + d_h^R))$ |

---

## 学习路径建议

**入门路线**：
1. [[Attention]] — 理解核心机制
2. [[Multi-Head Attention]] — 理解多头设计
3. [[Positional Encoding]] — 理解位置信息注入

**进阶路线**：
1. [[RoPE]] — 当前主流位置编码
2. [[Multi-Query Attention]] / [[Grouped-Query Attention]] — 推理优化
3. [[Multi-head Latent Attention]] — DeepSeek 的低秩压缩方案
4. [[ALiBi]] — 长度外推方案

**面试重点**：
- Attention 的时间复杂度及其瓶颈
- 为什么需要 Multi-Head
- MQA/GQA/MLA 的动机和权衡
- RoPE vs ALiBi 的对比

---

## Efficient Attention

Attention 的效率优化可以从两个层面理解：

```
Efficient Attention
├── 计算层优化（Attention 计算本身）
│   └── [[Flash Attention]] — IO-aware 算法，优化 HBM 访问
│
└── 服务层优化（KV Cache 内存管理）
    ├── [[KV Cache]] — 基础概念，避免重复计算
    ├── [[Paged Attention]] — 分页内存管理，减少碎片 (vLLM)
    └── [[Radix Attention]] — 前缀树共享，跨请求复用 (SGLang)
```

### 计算层：Flash Attention

[[Flash Attention]] 解决 Attention 计算的 **HBM 带宽瓶颈**：
- 核心技术：Tiling + Online Softmax + Recomputation
- 效果：2-4x 加速，内存从 $O(N^2)$ 降到 $O(N)$
- 特点：**精确算法**，结果与标准 Attention 完全相同

### 服务层：KV Cache 管理

[[KV Cache]] 是自回归推理的核心优化，但带来内存管理挑战：

| 问题 | 解决方案 |
|------|----------|
| 内存碎片 | [[Paged Attention]] — 借鉴 OS 分页机制 |
| 重复存储 | [[Radix Attention]] — Radix Tree 前缀共享 |
| KV 数量多 | [[Grouped-Query Attention\|GQA]] / [[Multi-Query Attention\|MQA]] — 架构层减少 KV heads |
| KV 维度大 | [[Multi-head Latent Attention\|MLA]] — 低秩压缩 |

---

## 延伸话题

### 其他高效 Attention 变体

- [[Sparse Attention]] — 稀疏注意力
- [[Linear Attention]] — 线性复杂度注意力

### 推理优化

- [[Speculative Decoding]] — 投机解码
- [[Continuous Batching]] — 连续批处理

### 长序列处理

- [[长度外推]] — 超出训练长度的推理
- [[Sliding Window Attention]] — 滑动窗口注意力
- [[Attention Sink]] — 解决 SWA 性能退化的 learnable bias
- [[Ring Attention]] — 分布式长序列处理

---

## 相关 MOC

- [[MOC - Foundations]] — Transformer 基础
- [[MOC - Inference]] — 推理优化
- [[MOC - Post-training]] — 后训练方法
