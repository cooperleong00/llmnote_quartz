---
description: IO-aware 的精确 Attention 算法，通过优化 GPU 内存访问模式大幅提升效率
type: method
prerequisites:
  - "[[Attention]]"
tags:
  - inference
  - attention
  - optimization
  - gpu
created: 2025-01-26
updated: 2026-01-31T22:42
---

# Flash Attention

Flash Attention 是一种 IO-aware 的精确 Attention 算法，通过优化 GPU 内存访问模式，在不改变计算结果的前提下大幅提升 Attention 的计算效率。

---

## 核心问题：HBM 带宽瓶颈

> [!intuition] 直觉理解
> 标准 Attention 实现的瓶颈不是计算量，而是内存读写。GPU 的计算能力远超内存带宽，大量时间花在等待数据从 HBM 传输到计算单元。Flash Attention 通过减少 HBM 访问次数来加速。

### GPU 内存层次

```
┌─────────────────────────────────────┐
│           SRAM (On-chip)            │
│  容量: ~20 MB    带宽: ~19 TB/s     │
│  (A100: 192KB per SM × 108 SMs)     │
└─────────────────────────────────────┘
                 ↑↓ 快
┌─────────────────────────────────────┐
│           HBM (Off-chip)            │
│  容量: 40-80 GB   带宽: ~2 TB/s     │
└─────────────────────────────────────┘
```

**关键洞察**：SRAM 带宽是 HBM 的 ~10 倍，但容量只有 HBM 的 ~0.1%。

### 标准 Attention 的 IO 问题

标准实现需要多次读写 HBM：

```python
# 标准实现（伪代码）
S = Q @ K.T          # 写入 HBM: O(N²)
P = softmax(S)       # 读 S，写 P: O(N²) × 2
O = P @ V            # 读 P，写 O: O(N²) + O(N·d)
```

**HBM 访问量**：$O(N^2 + Nd)$，其中 $N^2$ 项来自存储注意力矩阵。

---

## Flash Attention 的核心思想

### Tiling（分块计算）

将 Q、K、V 分成小块，每次只在 SRAM 中处理一小块：

```
Q: [Q₁][Q₂][Q₃]...    每块大小: Bᵣ × d
K: [K₁][K₂][K₃]...    每块大小: Bᶜ × d
V: [V₁][V₂][V₃]...    每块大小: Bᶜ × d
```

**关键**：块大小选择使得 $Q_i$、$K_j$、$V_j$ 和中间结果能同时放入 SRAM。

### Online Softmax

> [!math] 数学推导
> 标准 softmax 需要先计算所有元素的 max 和 sum，再归一化。Flash Attention 使用 online softmax，边计算边更新：
>
> 对于分块 $j$，维护：
> - $m^{(j)}$：当前见过的最大值
> - $\ell^{(j)}$：当前的 softmax 分母（缩放后）
> - $O^{(j)}$：当前的输出（缩放后）
>
> 更新公式：
> $$
> m^{(j)} = \max(m^{(j-1)}, \max(S_{ij}))
> $$
> $$
> \ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + \sum_k e^{S_{ijk} - m^{(j)}}
> $$
> $$
> O^{(j)} = \text{diag}(e^{m^{(j-1)} - m^{(j)}}) O^{(j-1)} + \tilde{P}_{ij} V_j
> $$

### Recomputation（反向传播时重算）

前向传播不保存 $O(N^2)$ 的注意力矩阵，反向传播时重新计算：
- 节省内存：不需要存储 $N \times N$ 的 softmax 输出
- 额外计算：反向传播多一次前向计算
- 净收益：减少 HBM 访问带来的加速 > 重算的开销

---

## IO 复杂度分析

> [!math] 复杂度对比
> 设序列长度 $N$，头维度 $d$，SRAM 大小 $M$：
>
> | 方法 | HBM 访问量 |
> |------|------------|
> | 标准 Attention | $O(Nd + N^2)$ |
> | Flash Attention | $O(N^2 d^2 / M)$ |
>
> 当 $M = O(Nd)$（典型情况）时，Flash Attention 的 IO 复杂度为 $O(N^2 d / M)$，相比标准实现减少了 $O(M/d)$ 倍。

---

## 版本演进

### Flash Attention V1 (2022)

- 提出 tiling + online softmax + recomputation
- 相比标准实现：2-4x 加速，内存从 $O(N^2)$ 降到 $O(N)$
- 支持 dropout 和 causal masking

### Flash Attention V2 (2023)

**主要改进**：
1. **减少非矩阵乘法操作**：重新组织算法，减少 rescaling 操作
2. **并行化优化**：
   - V1：外层循环遍历 K/V 块，内层遍历 Q 块
   - V2：外层循环遍历 Q 块，内层遍历 K/V 块
   - 好处：不同 Q 块可以并行，更好利用 GPU
3. **更好的 work partitioning**：在 warps 之间更均匀分配工作

**性能提升**：相比 V1 提升约 2x，达到理论峰值的 50-73%。

### Flash Attention V3 (2024)

针对 Hopper 架构（H100）的优化：

1. **异步执行**：利用 Tensor Memory Accelerator (TMA) 实现数据加载与计算重叠
2. **低精度支持**：FP8 量化，进一步提升吞吐
3. **Warp specialization**：不同 warp 专门负责不同任务（生产者/消费者模式）

**性能**：在 H100 上达到 740 TFLOPs（FP16），接近理论峰值。

---

## 实际加速效果

以 A100 GPU、序列长度 2048 为例：

| 实现 | 速度 (TFLOPs) | 相对加速 |
|------|---------------|----------|
| PyTorch naive | ~50 | 1x |
| Flash Attention V1 | ~120 | 2.4x |
| Flash Attention V2 | ~220 | 4.4x |
| 理论峰值 | 312 | — |

---

## 与其他优化的关系

> [!comparison] 对比
> | 优化方法 | 优化目标 | 层次 |
> |----------|----------|------|
> | Flash Attention | Attention 计算的 IO | 算子层 |
> | [[KV Cache]] | 避免重复计算 | 算法层 |
> | [[Paged Attention]] | KV Cache 内存管理 | 系统层 |
> | [[Multi-Query Attention\|MQA]]/[[Grouped-Query Attention\|GQA]] | 减少 KV 数量 | 架构层 |
>
> 这些优化是**正交**的，可以同时使用。

---

## 面试要点

> [!interview] 面试要点
> **Q: Flash Attention 解决什么问题？**
> A: 标准 Attention 的 HBM 带宽瓶颈。通过 tiling 减少 HBM 访问，利用高带宽的 SRAM。
>
> **Q: Flash Attention 的核心技术是什么？**
> A: 三个关键技术：(1) Tiling 分块计算；(2) Online softmax 增量计算；(3) Recomputation 反向重算。
>
> **Q: Flash Attention 是近似算法吗？**
> A: 不是，Flash Attention 是**精确**算法，计算结果与标准 Attention 完全相同，只是改变了计算顺序和内存访问模式。
>
> **Q: V1 到 V2 的主要改进是什么？**
> A: 改变循环顺序（外层遍历 Q 而非 K/V），使不同 Q 块可以并行；减少非矩阵乘法操作；更好的 warp 间工作分配。

---

## 参考资料

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)
- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Dao et al., 2024)
- [Tri Dao's Blog](https://tridao.me/blog/)
