---
type: concept
description: 将单个 Transformer 层的矩阵运算切分到多张 GPU，使超大模型的单层计算能跨卡并行，是 Megatron-LM 提出的层内模型并行方案
aliases:
  - 张量并行
  - TP
  - Intra-layer Parallelism
  - Megatron-style Parallelism
prerequisites:
  - "[[Data Parallelism]]"
tags:
  - distributed-training
  - efficiency
created: 2026-02-24
updated: 2026-02-24
---

# Tensor Parallelism

Tensor Parallelism（张量并行，TP）将 Transformer 层内的矩阵运算切分到多张 GPU 上并行执行。与 [[Data Parallelism]] 复制完整模型不同，TP 让每张卡只持有每一层的部分参数，从而突破单卡显存对模型规模的限制。这一方案由 Megatron-LM（Shoeybi et al., 2019）系统提出，至今仍是大模型训练的核心并行策略。

## 动机：为什么需要层内切分

[[Data Parallelism]] 要求每张卡放得下完整模型。[[ZeRO]] 通过切分 optimizer states / gradients / parameters 缓解了显存冗余，但每张卡仍然执行完整的 forward/backward——当单层的参数矩阵或 activation 就超过单卡显存时，ZeRO 无能为力。

这时需要把**单个层的计算本身**拆开。TP 的核心问题是：如何切分矩阵乘法，使得每张卡独立计算一部分，最后只需要一次简单的通信就能拼出正确结果？

## 核心思想：Column Parallel 与 Row Parallel

矩阵乘法 $Y = XA$ 有两种自然的切分方式：

**Column Parallel（列切分）**：把权重 $A$ 按列切成 $N$ 份 $[A_1, A_2, \dots, A_N]$，每张卡拿到完整输入 $X$ 和部分列 $A_i$，独立计算 $Y_i = XA_i$。结果 $Y_i$ 是输出的一部分列——拼接（Concatenate）即可得到完整输出。

**Row Parallel（行切分）**：把权重 $A$ 按行切成 $N$ 份，对应地输入 $X$ 也按列切成 $N$ 份 $[X_1, X_2, \dots, X_N]$，每张卡计算 $Y_i = X_i A_i$。结果需要**求和**（AllReduce）才能得到完整输出：$Y = \sum_i Y_i$。

> [!intuition] 为什么这两种切分是互补的？
> Column Parallel 的输出是"拼接"的（每卡持有部分列），而 Row Parallel 需要的输入恰好是"按列切分"的。把两者串联起来，Column Parallel 的输出可以直接作为 Row Parallel 的输入，**中间不需要任何通信**。这就是 Megatron-LM 的核心洞察。

## Megatron-LM 的切分策略

### MLP 层的切分

Transformer 的 MLP 由两个线性层组成：

$$\text{MLP}(X) = \text{GeLU}(XA) \cdot B$$

Megatron-LM 的切分方案：

1. **第一个线性层 $A$：Column Parallel** — 按列切分为 $[A_1, A_2]$（以 2 卡为例），每卡计算 $\text{GeLU}(XA_i)$
2. **第二个线性层 $B$：Row Parallel** — 按行切分为 $[B_1; B_2]$，每卡计算 $\text{GeLU}(XA_i) \cdot B_i$
3. **最后做一次 AllReduce** 求和得到完整输出

> [!math] 为什么 GeLU 不影响切分？
> GeLU 是逐元素操作。Column Parallel 后每卡持有输出的部分列，对这些列独立做 GeLU 等价于对完整输出做 GeLU 再切分。这个性质对 ReLU、SiLU 等逐元素激活函数都成立。
>
> 但如果激活函数涉及跨列的操作（如 Softmax），就不能这样切分了。

### Self-Attention 层的切分

Multi-Head Attention 天然适合 TP：每个 attention head 的计算是独立的。

切分方案：将 $Q, K, V$ 的投影矩阵按列切分（Column Parallel），每张卡负责部分 attention heads 的完整计算。输出投影矩阵 $W_O$ 按行切分（Row Parallel），最后 AllReduce 求和。

以 $h$ 个 head、$N$ 张卡为例，每卡负责 $h/N$ 个 head：

$$\text{head}_i = \text{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)})$$

每卡独立计算自己负责的 heads，然后通过 Row Parallel 的输出投影 + AllReduce 合并结果。整个 attention 层只需要**一次 AllReduce**。

> [!intuition] 为什么按 head 切分最自然？
> Multi-Head Attention 的设计本身就是"多个独立的 attention 计算 → 拼接 → 线性投影"。TP 只是把"拼接"这一步分布到了多张卡上。每个 head 内部的 $QK^T$ 和 softmax 都是完整的，不需要跨卡通信。这也是为什么 TP 的并行度通常不超过 head 数量。

### 完整 Transformer 层的通信模式

一个 Transformer 层 = Self-Attention + MLP，每个子层各需要一次 AllReduce：

```
Input X
  │
  ├─→ [GPU 0] Attention (heads 0..h/N-1) ──┐
  ├─→ [GPU 1] Attention (heads h/N..2h/N-1)─┤  AllReduce (f)
  └─→ ...                                   ──┘
  │
  ├─→ [GPU 0] MLP (column 0..d/N-1) ──┐
  ├─→ [GPU 1] MLP (column d/N..2d/N-1)─┤  AllReduce (g)
  └─→ ...                              ──┘
  │
Output Y
```

**Forward 阶段**：2 次 AllReduce（attention 一次 + MLP 一次）
**Backward 阶段**：2 次 AllReduce（梯度的反向传播同样需要同步）

> [!math] 通信量分析
> 每次 AllReduce 的通信量约为 $2 \cdot b \cdot s \cdot d$（$b$ = batch size，$s$ = sequence length，$d$ = hidden dimension），与切分数 $N$ 无关（Ring AllReduce 的特性）。
>
> 一个 Transformer 层每个训练 step 需要 **4 次 AllReduce**，总通信量约 $8bsd$。对于 $L$ 层模型，总通信量为 $8Lbsd$。
>
> 关键观察：通信量与 $N$ 无关，但**通信延迟**（latency）与 $N$ 成正比。这意味着增加 TP 并行度不会增加带宽需求，但会增加同步等待时间——这就是为什么 TP 对延迟敏感，需要高带宽低延迟的互联（NVLink）。

## 与 Data Parallelism 的对比

> [!comparison] TP vs DP：切什么、怎么通信

| 维度 | Data Parallelism | Tensor Parallelism |
|------|------------------|--------------------|
| **切分对象** | 数据（每卡完整模型） | 模型层内参数（每卡部分参数） |
| **通信内容** | 梯度（AllReduce） | activation（AllReduce） |
| **通信频率** | 每个 step 一次（可与 backward 重叠） | 每层 2 次 forward + 2 次 backward |
| **通信量** | $\approx 2M$（$M$ = 模型大小） | $\approx 8Lbsd$（$L$ = 层数） |
| **对互联的要求** | 中等（可跨机） | 极高（通常限制在 NVLink 互联的单机内） |
| **扩展性** | 理论上无限（加机器加数据） | 受限于 head 数和 hidden dim 的可整除性 |
| **解决的问题** | 加速训练 | 突破单卡显存限制 |

## 实际部署：3D 并行中的 TP

实际的大模型训练很少单独使用 TP，而是与 DP 和 [[Pipeline Parallelism]] 组合成 **3D 并行**：

- **节点内**：TP（利用 NVLink 的高带宽低延迟，通常 8 卡）
- **节点间**：[[Pipeline Parallelism]]（按层切分，通信量小，容忍较高延迟）
- **跨节点组**：[[Data Parallelism]] + [[ZeRO]]（切分数据和 optimizer states）

例如 DeepSeek-V3 的训练使用了 DualPipe 流水线并行算法，通过计算-通信重叠隐藏大部分通信开销，配合节点内的 TP 实现 2048 卡集群上的高效训练。

## Sequence Parallelism：对非张量并行部分的补充

TP 只切分了矩阵乘法（Attention 和 MLP），但 Transformer 层中还有 **LayerNorm** 和 **Dropout** 这些操作。在标准 TP 中，这些操作在每张卡上独立执行完整计算——这意味着它们的 activation 没有被切分，造成显存浪费。

Sequence Parallelism（Korthikanti et al., 2022）的解决方案：在 LayerNorm 和 Dropout 处，沿 **sequence 维度**切分 activation。每张卡只处理部分 token 的 LayerNorm/Dropout，然后在进入 Attention/MLP 前通过通信原语（AllGather / ReduceScatter）转换回 TP 的切分方式。

这样整个 Transformer 层的 activation 都被切分了，显存节省更彻底。代价是引入了额外的通信操作，但可以与计算重叠。

> [!warning] 名称混淆
> Sequence Parallelism 有多个含义。这里指的是 Megatron-LM 论文中作为 TP 补充的 SP，沿 sequence 维度切分 LayerNorm/Dropout 的 activation。另一种 Sequence Parallelism（如 Ring Attention）是沿 sequence 维度切分 attention 计算本身，用于处理超长序列，是不同的技术。

## 局限性

> [!warning] TP 的边界条件
>
> 1. **通信频率极高**：每个 Transformer 层的 forward 和 backward 各需要 2 次 AllReduce，对于 100+ 层的模型意味着数百次同步。这要求 GPU 间有极高带宽、极低延迟的互联
> 2. **通常限制在单机内**：NVLink 提供 ~900 GB/s 带宽（H100），而跨机 InfiniBand 通常只有 ~50-400 GB/s。跨机做 TP 会严重拖慢训练
> 3. **并行度受限**：TP 度数必须整除 attention head 数量和 hidden dimension。常见配置是 TP=8（单机 8 卡），很少超过 16
> 4. **负载均衡**：如果 head 数不能被 TP 度数整除，会导致部分卡计算量更大，其他卡空等
> 5. **与 [[Flash Attention]] 的交互**：Flash Attention 优化的是单卡内的 attention 计算效率，与 TP 的跨卡切分是正交的——TP 按 head 切分后，每张卡内部仍然可以用 Flash Attention 加速各自负责的 heads

## 面试要点

> [!interview] 面试视角
>
> **Q: Tensor Parallelism 和 Data Parallelism 的核心区别是什么？**
> A: DP 切数据、每卡完整模型；TP 切模型层内参数、每卡部分参数。DP 通信梯度（每 step 一次），TP 通信 activation（每层多次）。DP 可以跨机，TP 通常限制在 NVLink 互联的单机内。
>
> **Q: Megatron-LM 为什么选择 Column + Row Parallel 的组合？**
> A: Column Parallel 的输出（按列切分）恰好是 Row Parallel 需要的输入格式，两者串联时中间不需要通信。这样 MLP 的两个线性层只需要最后一次 AllReduce，而非每个线性层都通信。
>
> **Q: 为什么 TP 通常限制在 8 卡？**
> A: TP 每层需要 4 次 AllReduce（forward 2 次 + backward 2 次），通信频率极高。NVLink 提供的高带宽低延迟是必要条件，而 NVLink 通常连接同一台机器内的 8 张 GPU。跨机的 InfiniBand 带宽不足以支撑如此频繁的同步。
>
> **Q: TP 的通信量和并行度 N 的关系？**
> A: 每次 AllReduce 的通信量约 $2bsd$，与 $N$ 无关（Ring AllReduce 特性）。但通信延迟与 $N$ 成正比，所以增加 TP 度数主要增加的是延迟开销而非带宽开销。

> [!paper] 论文出处
> - Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", 2019 — 提出 Transformer 的 TP 切分策略
> - Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models", 2022 — 提出 Sequence Parallelism 作为 TP 的补充

## 延伸阅读

**后续发展**：
- [[Pipeline Parallelism]] — 层间切分，与 TP 的层内切分互补
- [[Expert Parallelism]] — [[Mixture of Experts|MoE]] 模型中按专家切分，又一种模型并行维度
- [[Activation Checkpointing]] — 用重计算换显存，与 TP 配合进一步降低显存需求
