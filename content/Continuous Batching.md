---
description: 迭代级调度的批处理策略，允许请求随时加入/退出 batch，解决静态批处理中短序列等待长序列的效率问题
type: concept
aliases:
  - 连续批处理
  - Dynamic Batching
  - Iteration-level Scheduling
prerequisites:
  - "[[KV Cache]]"
tags:
  - inference
  - serving
  - optimization
created: 2026-01-29
updated: 2026-02-01T01:01
---

# Continuous Batching

Continuous Batching（连续批处理）是 LLM Serving 的核心调度策略，通过迭代级（iteration-level）调度允许请求在任意时刻加入或退出 batch，解决传统静态批处理中短序列等待长序列的效率问题。

---

## 动机：Static Batching 的问题

> [!intuition] 为什么需要 Continuous Batching？
> 传统 Static Batching 将多个请求打包成一个 batch 一起处理，但必须等待 batch 中最长的序列完成才能释放资源。这导致短序列生成完成后空等，GPU 利用率低下。

### Static Batching 的工作方式

```
Static Batching（batch size = 3）:

请求 A: [████████████████████████████████████████] 生成 200 tokens
请求 B: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 生成 50 tokens，等待 150 tokens
请求 C: [████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 生成 100 tokens，等待 100 tokens

时间 ──────────────────────────────────────────────────────────────────────►
      │                                                                    │
      └── Batch 开始                                          Batch 结束 ──┘
                                                              (等 A 完成)
```

**问题**：
1. **资源浪费**：B 和 C 完成后，其 GPU 计算资源空闲
2. **延迟增加**：B 和 C 的响应延迟被 A 拖累
3. **吞吐量受限**：新请求必须等待整个 batch 完成才能开始

---

## 核心思想：Iteration-level Scheduling

> [!intuition] 直觉理解
> 把调度粒度从"整个生成过程"细化到"每次 forward pass"。每个 iteration 结束后，调度器检查哪些请求完成了，哪些新请求可以加入，动态调整 batch 组成。

### Continuous Batching 的工作方式

```
Continuous Batching:

Iter 1: [A, B, C]     ← 三个请求同时处理
Iter 2: [A, B, C]
  ...
Iter 50: [A, C, D]    ← B 完成退出，D 加入
Iter 51: [A, C, D]
  ...
Iter 100: [A, D, E]   ← C 完成退出，E 加入
  ...
Iter 200: [D, E, F]   ← A 完成退出，F 加入

时间 ──────────────────────────────────────────────────────────────────────►
      │    │    │    │    │    │    │    │    │    │    │    │    │    │
      每个 iteration 都可以调整 batch 组成
```

### 关键机制

1. **请求完成即释放**：序列生成 EOS 或达到最大长度后，立即从 batch 移除
2. **新请求即时加入**：只要有空位，等待队列中的请求可以立即加入当前 batch
3. **资源动态分配**：[[KV Cache]] 按需分配和释放，不需要预留最大长度

---

## 实现原理

### 调度器设计

```python
class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size, max_tokens):
        self.running = []      # 正在处理的请求
        self.waiting = []      # 等待队列
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens  # 内存限制

    def schedule(self):
        """每个 iteration 调用一次"""
        # 1. 移除已完成的请求
        self.running = [r for r in self.running if not r.is_finished()]

        # 2. 释放完成请求的 KV Cache
        for r in finished_requests:
            self.free_kv_cache(r)

        # 3. 尝试加入新请求
        while (len(self.running) < self.max_batch_size
               and self.waiting
               and self.has_memory_for(self.waiting[0])):
            new_request = self.waiting.pop(0)
            self.allocate_kv_cache(new_request)
            self.running.append(new_request)

        return self.running
```

### 内存管理

Continuous Batching 需要动态管理 [[KV Cache]]：

```
传统方式：
┌─────────────────────────────────────────────────────────┐
│ 请求 A: [预分配 max_len 的 KV Cache                    ] │
│ 请求 B: [预分配 max_len 的 KV Cache                    ] │
│ 请求 C: [预分配 max_len 的 KV Cache                    ] │
└─────────────────────────────────────────────────────────┘

Continuous Batching + Paged Attention：
┌─────────────────────────────────────────────────────────┐
│ 请求 A: [Block 0][Block 1][Block 2]                     │
│ 请求 B: [Block 3]                    ← 按需分配         │
│ 请求 C: [Block 4][Block 5]                              │
│ 空闲池: [Block 6][Block 7][Block 8]...                  │
└─────────────────────────────────────────────────────────┘
```

---

## 与 Static Batching 的对比

> [!comparison] 对比

| 维度 | Static Batching | Continuous Batching |
|------|-----------------|---------------------|
| **调度粒度** | Request-level | Iteration-level |
| **请求加入** | Batch 开始时 | 任意 iteration |
| **请求退出** | Batch 结束时 | 完成即退出 |
| **GPU 利用率** | 低（短序列空等） | 高（持续满载） |
| **延迟** | 受最长序列影响 | 各请求独立 |
| **实现复杂度** | 简单 | 需要动态内存管理 |
| **内存管理** | 静态预分配 | 动态分配/释放 |

### 效率分析

假设 3 个请求，生成长度分别为 50、100、200 tokens：

**Static Batching**：
- 总时间：200 iterations
- GPU 利用率：(50 + 100 + 200) / (200 × 3) = 58%

**Continuous Batching**：
- 请求 B 完成后，新请求 D 加入
- 请求 C 完成后，新请求 E 加入
- GPU 利用率接近 100%（假设有足够的等待请求）

---

## 与 Paged Attention 的配合

> [!intuition] 为什么 Continuous Batching 需要 Paged Attention？
> Continuous Batching 要求动态分配和释放 KV Cache。传统的连续内存分配会导致严重的内存碎片。[[Paged Attention]] 通过分页机制解决这个问题，两者是天然的搭档。

### vLLM 的架构

```
┌─────────────────────────────────────────────────────────┐
│                    vLLM 架构                            │
├─────────────────────────────────────────────────────────┤
│  Scheduler (Continuous Batching)                        │
│    ├── 管理请求队列                                      │
│    ├── 决定每个 iteration 的 batch 组成                  │
│    └── 触发 KV Cache 分配/释放                          │
├─────────────────────────────────────────────────────────┤
│  Memory Manager (Paged Attention)                       │
│    ├── Block 分配器                                      │
│    ├── Block Table 管理                                  │
│    └── Copy-on-Write 支持                               │
├─────────────────────────────────────────────────────────┤
│  Execution Engine                                       │
│    ├── Paged Attention Kernel                           │
│    └── 模型 Forward Pass                                │
└─────────────────────────────────────────────────────────┘
```

---

## 吞吐量提升分析

> [!math] 量化分析
> 设请求到达率为 $\lambda$，平均生成长度为 $\bar{L}$，最大生成长度为 $L_{\max}$。
>
> **Static Batching 吞吐量**：
> $$\text{Throughput}_{\text{static}} = \frac{B}{L_{\max}} \cdot \text{tokens/s}$$
>
> **Continuous Batching 吞吐量**：
> $$\text{Throughput}_{\text{continuous}} \approx \frac{B}{\bar{L}} \cdot \text{tokens/s}$$
>
> 提升比例：$\frac{L_{\max}}{\bar{L}}$
>
> 当 $L_{\max} = 2048$，$\bar{L} = 256$ 时，理论提升可达 **8 倍**。

### 实际提升因素

1. **生成长度方差**：方差越大，Continuous Batching 优势越明显
2. **请求到达率**：高负载时优势更显著
3. **内存限制**：配合 Paged Attention 可支持更大 batch size

---

## 局限性

> [!warning] 边界条件
> 1. **实现复杂度**：需要动态内存管理，增加系统复杂性
> 2. **Prefill 阶段**：新请求加入时的 prefill 可能影响正在 decode 的请求
> 3. **内存碎片**：没有 Paged Attention 时仍会有碎片问题
> 4. **调度开销**：每个 iteration 都需要调度决策

### Chunked Prefill

为解决 prefill 影响 decode 的问题，可以将 prefill 分块：

```
传统方式：
Iter N:   [Decode A, B, C] + [Prefill D 全部 1000 tokens]  ← 延迟spike

Chunked Prefill：
Iter N:   [Decode A, B, C] + [Prefill D chunk 1: 100 tokens]
Iter N+1: [Decode A, B, C, D] + [Prefill D chunk 2: 100 tokens]
...
```

### 与其他优化技术的关系

Continuous Batching 是调度层的优化，可以与其他层次的优化技术正交组合：

- **计算层优化**：[[Flash Attention]] 优化注意力计算的内存访问模式，减少 HBM 读写，与 Continuous Batching 的调度策略互不干扰
- **推理加速**：[[Speculative Decoding]] 通过小模型预测 + 大模型验证来加速生成，可以在 Continuous Batching 框架内使用
- **前缀共享**：[[Radix Attention]]（SGLang）通过 radix tree 管理 KV Cache，实现多请求间的前缀共享，是对 Paged Attention 的进一步优化

---

## 面试要点

> [!interview] 面试视角
> **Q: 什么是 Continuous Batching？解决什么问题？**
> A: Continuous Batching 是迭代级调度策略，允许请求在任意 iteration 加入或退出 batch。解决 Static Batching 中短序列等待长序列的问题，显著提升 GPU 利用率和吞吐量。
>
> **Q: Continuous Batching 和 Paged Attention 的关系？**
> A: Continuous Batching 需要动态分配/释放 KV Cache，Paged Attention 提供了高效的分页内存管理。两者配合使用：Continuous Batching 负责调度，Paged Attention 负责内存管理。vLLM 就是这种架构。
>
> **Q: Continuous Batching 能带来多大的吞吐量提升？**
> A: 理论提升比例约为 $L_{\max}/\bar{L}$。实际中，配合 Paged Attention，vLLM 报告了 2-4 倍的吞吐量提升。提升幅度取决于生成长度的方差和系统负载。
>
> **Q: Continuous Batching 有什么挑战？**
> A: (1) 需要动态内存管理，增加实现复杂度；(2) 新请求的 prefill 可能影响正在 decode 的请求（可用 Chunked Prefill 缓解）；(3) 每个 iteration 都有调度开销。

---

## 参考资料

- Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., 2022) - 首次提出 iteration-level scheduling
- Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023) - vLLM 论文
- [vLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
