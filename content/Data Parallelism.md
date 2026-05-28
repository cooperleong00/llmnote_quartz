---
type: concept
description: 最基础的分布式训练范式，每个 GPU 持有完整模型副本、数据切分后并行计算梯度再同步，是 ZeRO 和模型并行的起点
aliases:
  - 数据并行
  - DP
  - DDP
  - DistributedDataParallel
prerequisites:
  - "[[Backpropagation]]"
tags:
  - distributed-training
  - efficiency
  - optimization
created: 2026-02-24
updated: 2026-02-24
---

# Data Parallelism

数据并行（Data Parallelism）是分布式训练最基础的范式：每张 GPU 持有一份完整的模型副本，将一个大 batch 切分到多张卡上并行计算梯度，然后同步梯度使所有副本保持一致。它的核心假设很简单——**模型足够小，单卡放得下**；瓶颈在于数据量大、训练慢，需要多卡加速。

这个假设在 LLM 时代被打破了（一个 70B 模型的 optimizer states 就需要上百 GB），由此催生了 [[ZeRO]]、[[Tensor Parallelism]]、[[Pipeline Parallelism]] 等更复杂的并行策略。但理解 Data Parallelism 是理解所有后续方案的起点。

## 核心思想

> [!intuition] 一句话理解
> 数据并行就是"人多力量大"——同一个模型复制 $N$ 份，每份处理 $1/N$ 的数据，最后把大家算出的梯度汇总取平均。

训练一个 step 的流程：

1. **数据切分**：将 mini-batch $B$ 均匀分成 $N$ 份（$N$ = GPU 数量），每张卡拿到 $B/N$ 个样本
2. **前向传播**：每张卡用自己的模型副本独立计算 loss
3. **反向传播**：每张卡独立计算梯度 $g_i$
4. **梯度同步**：所有卡交换梯度，计算平均值 $\bar{g} = \frac{1}{N}\sum_{i=1}^{N} g_i$
5. **参数更新**：每张卡用相同的 $\bar{g}$ 更新参数，保持模型一致

数学上，这等价于在完整 batch 上计算梯度——因为梯度对样本是线性可加的。所以数据并行不改变优化语义，只是加速计算。

## 从 DP 到 DDP：通信架构的演进

### PyTorch DP（DataParallel）：Parameter Server 模式

PyTorch 最早的 `torch.nn.DataParallel` 采用 Parameter Server 架构：

- 一张"主卡"（通常是 GPU 0）负责收集所有卡的梯度、求平均、更新参数，再把新参数广播回去
- 所有通信都经过主卡

> [!warning] DP 的致命瓶颈
> 1. **主卡通信瓶颈**：所有梯度汇聚到一张卡，通信量 $O(N)$ 集中在单点
> 2. **主卡显存更高**：主卡需要额外存储汇总的梯度，容易 OOM
> 3. **只能单机**：基于 Python 多线程，受 GIL 限制，无法跨机器
> 4. **负载不均衡**：主卡承担了额外的 reduce + broadcast 计算

### PyTorch DDP（DistributedDataParallel）：AllReduce 模式

`torch.nn.parallel.DistributedDataParallel` 用 AllReduce 替代了 Parameter Server，彻底消除了单点瓶颈：

- 每张卡地位平等，没有"主卡"
- 梯度同步通过 AllReduce 集合通信完成
- 基于多进程（每个 GPU 一个进程），支持多机多卡

> [!comparison] DP vs DDP
> | 维度 | DP (DataParallel) | DDP (DistributedDataParallel) |
> |------|-------------------|-------------------------------|
> | 通信模式 | Parameter Server（主卡汇总） | AllReduce（对等通信） |
> | 通信瓶颈 | 主卡带宽 | 均摊到所有卡 |
> | 多机支持 | 不支持 | 支持（基于 NCCL） |
> | 进程模型 | 单进程多线程 | 多进程（每 GPU 一个） |
> | 实际性能 | 随卡数增加急剧下降 | 近线性扩展 |

实际工程中，DP 已经基本被淘汰，DDP 是数据并行的标准实现。

## AllReduce：核心通信原语

AllReduce 的语义是：每张卡都有一个向量（梯度），操作完成后每张卡都拿到所有向量的和（或平均）。

### Ring AllReduce

Ring AllReduce 是最经典的实现，将 $N$ 张卡组成一个环：

**阶段一：Reduce-Scatter**
- 每张卡将自己的梯度切成 $N$ 块
- 经过 $N-1$ 轮传递，每张卡负责一块的"部分和"逐步累加
- 结束时，每张卡持有一块的完整求和结果

**阶段二：All-Gather**
- 再经过 $N-1$ 轮传递，每张卡把自己持有的完整块广播给其他卡
- 结束时，每张卡都拥有完整的求和结果

> [!math] 通信量分析
> 设每张卡的梯度大小为 $M$（字节）：
>
> - **Reduce-Scatter**：每轮每张卡发送 $M/N$，共 $N-1$ 轮 → 每卡发送 $\frac{(N-1)}{N} \cdot M$
> - **All-Gather**：同上 → 每卡发送 $\frac{(N-1)}{N} \cdot M$
> - **总计**：每卡发送 $\frac{2(N-1)}{N} \cdot M \approx 2M$（当 $N$ 较大时）
>
> 关键洞察：**每卡的通信量与 GPU 数量 $N$ 几乎无关**。这就是 Ring AllReduce 优于 Parameter Server 的根本原因——Parameter Server 的主卡通信量是 $O(N \cdot M)$，而 Ring AllReduce 每卡只有 $O(M)$。

实际中 NVIDIA 的 NCCL 库会根据拓扑（NVLink、PCIe、InfiniBand）自动选择最优的 AllReduce 实现，不一定是纯 Ring 结构。

## DDP 的关键实现细节

DDP 不只是简单地在反向传播后做一次 AllReduce，它有两个重要的工程优化：

### Gradient Bucketing（梯度分桶）

如果对每个参数单独做 AllReduce，通信启动开销（latency）会非常大。DDP 将参数的梯度按反向传播的计算顺序分成若干个 bucket（默认 25MB），对每个 bucket 做一次 AllReduce。

- Bucket 按反向传播顺序填充：最后一层的梯度最先算出来，最先填入 bucket
- Bucket 满了就立即触发 AllReduce，不等所有梯度算完

### 通信-计算重叠（Overlap）

这是 DDP 性能的关键：**一个 bucket 的 AllReduce 和下一个 bucket 的梯度计算可以同时进行**。

```
时间 →
GPU 计算:  [反向传播 layer N] [反向传播 layer N-1] [反向传播 layer N-2] ...
通信:                         [AllReduce bucket 1]  [AllReduce bucket 2]  ...
```

因为反向传播是从后往前逐层计算的，最后几层的梯度最先就绪。DDP 在这些梯度填满一个 bucket 后立即启动通信，同时 GPU 继续计算前面层的梯度。理想情况下，当反向传播全部完成时，AllReduce 也几乎同时完成，通信开销被"藏"在了计算时间里。

> [!example] 实际效果
> 在通信带宽足够的情况下（如 NVLink 连接的同机多卡），DDP 的通信开销可以被几乎完全重叠，实现接近线性的扩展效率。跨机通信（InfiniBand / RoCE）带宽较低时，重叠效果会打折扣。

## 显存分析：为什么需要 ZeRO

理解 Data Parallelism 的显存占用是理解 [[ZeRO]] 动机的关键。

以一个参数量为 $\Phi$ 的模型、使用 [[Adam]] optimizer、混合精度训练为例，每张卡需要存储：

> [!math] 每张卡的显存占用（Mixed Precision Training）
>
> | 组件 | 精度 | 大小 |
> |------|------|------|
> | 模型参数（fp16 训练副本） | FP16 | $2\Phi$ bytes |
> | 模型参数（fp32 主副本） | FP32 | $4\Phi$ bytes |
> | 梯度 | FP16 | $2\Phi$ bytes |
> | Adam 一阶动量 (m) | FP32 | $4\Phi$ bytes |
> | Adam 二阶动量 (v) | FP32 | $4\Phi$ bytes |
> | **合计** | | **$16\Phi$ bytes** |
>
> 其中 parameters + gradients = $4\Phi$（称为 residual states），optimizer states = $12\Phi$（称为 model states 中的主要部分）。

以 GPT-2（1.5B 参数）为例：$16 \times 1.5 \times 10^9 = 24$ GB，一张 A100 (80GB) 勉强放得下（还要留空间给 activations）。

但对于 LLaMA-70B：$16 \times 70 \times 10^9 = 1120$ GB，**单卡完全不可能**。

> [!warning] Data Parallelism 的核心矛盾
> 在 Data Parallelism 中，**每张卡都存储完全相同的 model states**。8 张卡就有 8 份完全一样的 optimizer states，这是巨大的显存浪费。
>
> 这正是 [[ZeRO]]（Zero Redundancy Optimizer）的出发点：既然 $N$ 张卡存了 $N$ 份相同的东西，为什么不让每张卡只存 $1/N$？

## 局限性与边界

> [!warning] Data Parallelism 的适用边界
>
> 1. **模型必须单卡放得下**：这是最根本的限制。当模型参数 + optimizer states 超过单卡显存时，纯数据并行无法工作
> 2. **显存利用率低**：$N$ 张卡存 $N$ 份相同的 model states，冗余度 100%
> 3. **Batch Size 隐式增大**：$N$ 张卡意味着有效 batch size 变为 $N$ 倍，可能影响收敛性（需要配合 learning rate warmup / scaling）
> 4. **通信带宽依赖**：跨机通信时，AllReduce 的通信量（$\approx 2M$）可能成为瓶颈，尤其是模型较大时

当模型太大单卡放不下时，需要引入模型并行：
- [[Tensor Parallelism]] — 将单个算子（如矩阵乘法）切分到多卡
- [[Pipeline Parallelism]] — 将模型按层切分到多卡
- [[ZeRO]] — 在数据并行框架内切分 model states，减少冗余

实际的大模型训练通常采用 **3D 并行**：Data Parallelism + Tensor Parallelism + Pipeline Parallelism 的组合，有时还加上 [[Mixture of Experts|MoE]] 的 [[Expert Parallelism]]。

## 与 Gradient Accumulation 的关系

[[Gradient Accumulation]] 是数据并行的"穷人版"替代方案：当 GPU 数量不够时，可以在单卡上多次前向/反向传播、累积梯度后再更新，模拟大 batch 的效果。

区别在于：
- Data Parallelism：$N$ 张卡同时算 $N$ 个 micro-batch → **时间不变，吞吐量 $\times N$**
- Gradient Accumulation：1 张卡串行算 $N$ 个 micro-batch → **时间 $\times N$，吞吐量不变**

两者可以组合使用：DDP + Gradient Accumulation，在有限 GPU 数量下实现更大的有效 batch size。

> [!interview] 面试要点
> **Q: DP 和 DDP 的区别是什么？**
> A: DP 用 Parameter Server（主卡汇总梯度），有单点通信瓶颈、显存不均衡、只能单机。DDP 用 AllReduce（对等通信），通信量均摊、支持多机，是现在的标准方案。
>
> **Q: Ring AllReduce 的通信量是多少？为什么比 Parameter Server 好？**
> A: 每卡通信量约 $2M$（$M$ 为梯度大小），与 GPU 数量 $N$ 无关。Parameter Server 主卡通信量是 $O(NM)$，随卡数线性增长。
>
> **Q: DDP 如何实现通信-计算重叠？**
> A: 梯度按反向传播顺序分桶（gradient bucketing），一个桶满了就立即启动 AllReduce，同时 GPU 继续计算前面层的梯度。理想情况下通信完全被计算掩盖。
>
> **Q: 数据并行的显存瓶颈是什么？**
> A: 每张卡存完整的 model states（参数 + 梯度 + optimizer states），以混合精度 Adam 为例需要 $16\Phi$ bytes。$N$ 张卡有 $N$ 份完全相同的冗余，这是 ZeRO 要解决的问题。

## 延伸阅读

**后续发展**：
- [[ZeRO]] — 在数据并行框架内消除 model states 冗余（待创建）
- [[Tensor Parallelism]] — 算子级模型并行（待创建）
- [[Pipeline Parallelism]] — 层级模型并行（待创建）
- [[FSDP]] — PyTorch 对 ZeRO Stage 3 的原生实现（待创建）

**相关背景**：
- [[Flash Attention]] — GPU 内存层级优化的另一个视角
- [[Quantization|量化]] — 通过降低精度减少显存占用
