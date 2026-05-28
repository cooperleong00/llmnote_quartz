---
description: vLLM 的分页 KV Cache 管理，借鉴 OS 虚拟内存解决内存碎片问题
type: method
prerequisites:
  - "[[KV Cache]]"
tags:
  - inference
  - attention
  - optimization
  - serving
created: 2025-01-26
updated: 2026-01-31T22:42
---

# Paged Attention

Paged Attention 是 vLLM 提出的 KV Cache 内存管理方法，借鉴操作系统虚拟内存的分页机制，解决 LLM 推理服务中的内存碎片问题。

---

## 动机：KV Cache 内存碎片

> [!intuition] 直觉理解
> 传统 LLM 推理为每个请求预分配一块连续内存存储 KV Cache。由于生成长度未知，通常按最大长度预分配，导致大量内存浪费。Paged Attention 把 KV Cache 切成小块（pages），按需分配，像操作系统管理内存一样。

### 传统方法的问题

```
预分配方式（最大长度 2048）：
请求 A: [████████████░░░░░░░░░░░░░░░░░░░░] 实际 512，预留 2048
请求 B: [██████░░░░░░░░░░░░░░░░░░░░░░░░░░] 实际 256，预留 2048
请求 C: [████████████████░░░░░░░░░░░░░░░░] 实际 768，预留 2048

内存利用率: (512+256+768) / (2048×3) = 25%
```

**三类浪费**：
1. **预留浪费**：预分配但未使用的空间
2. **内部碎片**：请求结束前，已分配但暂未使用的空间
3. **外部碎片**：请求结束后，留下的不连续空闲块

---

## 核心思想：分页内存管理

### 类比操作系统虚拟内存

| OS 概念 | Paged Attention 对应 |
|---------|---------------------|
| 虚拟页 (Virtual Page) | 逻辑 KV Block |
| 物理页帧 (Physical Frame) | 物理 KV Block |
| 页表 (Page Table) | Block Table |
| 按需分页 (Demand Paging) | 按需分配 KV Block |

### Block 设计

将 KV Cache 划分为固定大小的 Block：

```
Block 大小: B 个 token 的 KV
每个 Block: [K₁ K₂ ... Kᵦ | V₁ V₂ ... Vᵦ]

一个请求的 KV Cache:
逻辑视图: [Block 0][Block 1][Block 2][Block 3]
              ↓        ↓        ↓        ↓
物理存储: [Block 7][Block 2][Block 5][Block 9]  (非连续)
```

### Block Table

每个请求维护一个 Block Table，记录逻辑块到物理块的映射：

```
请求 A 的 Block Table:
┌─────────┬─────────────┐
│ 逻辑块  │ 物理块地址   │
├─────────┼─────────────┤
│    0    │     7       │
│    1    │     2       │
│    2    │     5       │
│    3    │     9       │
└─────────┴─────────────┘
```

---

## 内存分配流程

### 初始化

1. 将 GPU 内存划分为固定大小的物理 Block 池
2. 维护空闲 Block 列表

### 请求处理

```
新请求到达:
1. 分配第一个 Block（存储 prompt 的 KV）
2. 创建 Block Table

生成过程:
3. 当前 Block 填满 → 从空闲池分配新 Block
4. 更新 Block Table

请求结束:
5. 释放所有 Block 回空闲池
```

### 动态分配示例

```
时刻 T1: 请求 A 开始，分配 Block 0
空闲池: [1,2,3,4,5,6,7,8,9]
请求 A: [0]

时刻 T2: 请求 A 的 Block 0 填满，分配 Block 1
空闲池: [2,3,4,5,6,7,8,9]
请求 A: [0,1]

时刻 T3: 请求 B 开始，分配 Block 2
空闲池: [3,4,5,6,7,8,9]
请求 A: [0,1]
请求 B: [2]

时刻 T4: 请求 A 结束，释放 Block 0,1
空闲池: [0,1,3,4,5,6,7,8,9]
请求 B: [2]
```

---

## Attention 计算的修改

### 标准 Attention

```python
# 连续内存
output = attention(query, keys, values)
```

### Paged Attention

```python
# 分页内存
output = paged_attention(
    query,
    key_cache,      # 所有物理 Block 的 Key
    value_cache,    # 所有物理 Block 的 Value
    block_table,    # 逻辑块 → 物理块映射
    context_len     # 当前序列长度
)
```

**计算过程**：
1. 根据 Block Table 找到物理块地址
2. 从非连续的物理块中 gather K/V
3. 执行标准 Attention 计算

---

## 内存利用率提升

> [!math] 量化分析
> 设 Block 大小为 $B$ 个 token，请求的实际长度为 $L$：
>
> **传统方法**：预分配 $L_{\max}$，浪费 $L_{\max} - L$
>
> **Paged Attention**：分配 $\lceil L/B \rceil$ 个 Block，浪费最多 $B-1$
>
> 当 $B \ll L_{\max}$ 时，内存利用率显著提升。

### 实验数据（vLLM 论文）

| 场景 | 传统方法 | Paged Attention |
|------|----------|-----------------|
| 内存利用率 | ~20-40% | ~95%+ |
| 吞吐量提升 | 1x | 2-4x |

---

## 高级特性

### Copy-on-Write

当多个请求共享相同前缀时，可以共享 Block：

```
请求 A: [System Prompt] → [Block 0, Block 1]
请求 B: [System Prompt] → [Block 0, Block 1] (共享)

请求 B 开始生成不同内容时:
请求 B: [Block 0, Block 1, Block 2'] (Block 2' 是新分配的)
```

### Preemption（抢占）

当内存不足时，可以：
1. **Swap**：将低优先级请求的 KV Cache 换出到 CPU 内存
2. **Recomputation**：丢弃 KV Cache，需要时重新计算

---

## 与其他方法的对比

> [!comparison] 对比
> | 方法 | 解决问题 | 层次 |
> |------|----------|------|
> | Paged Attention | 内存碎片 | 内存管理 |
> | [[Radix Attention]] | 前缀共享 | 缓存复用 |
> | [[Flash Attention]] | 计算 IO | 算子优化 |
> | [[Grouped-Query Attention\|GQA]] | KV 数量 | 模型架构 |
>
> Paged Attention 和 Radix Attention 都是 KV Cache 管理方法，但侧重点不同：
> - Paged Attention：减少碎片，提高单请求内存效率
> - Radix Attention：跨请求共享，减少重复存储

---

## 面试要点

> [!interview] 面试要点
> **Q: Paged Attention 解决什么问题？**
> A: KV Cache 的内存碎片问题。传统方法按最大长度预分配连续内存，导致大量浪费；Paged Attention 按需分配非连续的小块。
>
> **Q: Paged Attention 的核心思想是什么？**
> A: 借鉴 OS 虚拟内存的分页机制：将 KV Cache 划分为固定大小的 Block，通过 Block Table 管理逻辑块到物理块的映射，实现按需分配和非连续存储。
>
> **Q: Paged Attention 有什么开销？**
> A: (1) Block Table 的存储和查找开销；(2) 非连续内存访问的 gather 操作；(3) Block 粒度导致的少量内部碎片。但这些开销远小于节省的内存。
>
> **Q: vLLM 的吞吐量为什么能提升 2-4x？**
> A: 内存利用率从 ~30% 提升到 ~95%，可以同时处理更多请求（更大的 batch size），从而提高 GPU 利用率和吞吐量。

---

## 参考资料

- Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Blog: PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
