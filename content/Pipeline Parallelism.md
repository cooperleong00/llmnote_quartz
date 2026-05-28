---
type: concept
description: 将模型按层切分到多个设备顺序执行的并行策略，核心挑战是减少 GPU 空闲时间（pipeline bubble），GPipe 和 1F1B 是两种经典调度方案
aliases:
  - 流水线并行
  - PP
prerequisites:
  - "[[Data Parallelism]]"
tags:
  - distributed-training
  - efficiency
created: 2026-02-24
updated: 2026-02-24
---

# Pipeline Parallelism

流水线并行（Pipeline Parallelism, PP）将模型的不同层分配到不同设备上，数据像流水线一样依次流过各个阶段（stage）。它解决的核心问题是：**当模型太大、单张 GPU 放不下时，如何跨设备训练**。与 [[Tensor Parallelism]] 切分单个算子不同，PP 的切分粒度是"层"——直觉上就是把一个很深的网络从中间切几刀，每段放一张卡。

PP 的通信量很小（只需在 stage 边界传递 activation），但代价是引入了 **pipeline bubble**——当某些 stage 在等待上游数据时，GPU 处于空闲状态。如何减少 bubble 是 PP 研究的核心问题。

## 动机：为什么需要按层切分

[[Data Parallelism]] 要求每张卡都持有完整的模型副本。当模型参数量达到数十亿甚至上千亿时，单卡显存根本装不下完整的 model states（参数 + 梯度 + optimizer states）。

模型并行的两种思路：
- **层内切分**（[[Tensor Parallelism]]）：把单个矩阵乘法拆到多卡，通信频繁但延迟低，适合高带宽互联（如 NVLink）
- **层间切分**（Pipeline Parallelism）：把不同层放到不同卡，通信少但有 bubble，适合低带宽互联（如 InfiniBand 跨机）

PP 的优势在于通信开销极小——每个 stage 只需要在边界处传递一次 activation tensor，而 TP 在每个 Transformer 层内部就需要多次 AllReduce。

## 朴素方法与 Bubble 问题

最简单的实现：把模型分成 $p$ 个 stage，数据从 stage 0 流到 stage $p-1$（forward），再从 stage $p-1$ 流回 stage 0（backward）。

问题显而易见：**在任意时刻，只有一个 stage 在工作，其余 $p-1$ 个 stage 全部空闲**。GPU 利用率仅为 $1/p$，这就是所谓的 pipeline bubble。对于 $p=8$ 的配置，87.5% 的计算资源被浪费了。

## GPipe：Micro-batch 调度

> [!paper] Huang et al., 2019
> GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

GPipe 的核心思想很简单：**把一个 mini-batch 切成 $m$ 个 micro-batch，让它们像流水线一样依次进入 pipeline**。当 micro-batch 1 在 stage 1 做 forward 时，micro-batch 0 已经到了 stage 2——多个 stage 可以同时工作。

所有 micro-batch 的 forward 全部完成后，再统一做 backward（F-then-B 调度）。梯度在所有 micro-batch 上累积，最后做一次参数更新——这本质上就是 [[Gradient Accumulation]] 在 pipeline 上的应用。

### Bubble 分析

GPipe 的 bubble 出现在两个地方：pipeline 填充阶段（ramp-up）和排空阶段（ramp-down）。

> [!math] Bubble Ratio
> $$\text{Bubble Ratio} = \frac{p - 1}{p - 1 + m}$$
>
> 其中 $p$ 是 pipeline stage 数，$m$ 是 micro-batch 数。
>
> - $m = 1$（无切分）：bubble ratio = $(p-1)/p$，退化为朴素方法
> - $m \gg p$：bubble ratio $\to 0$，趋近理想利用率
> - 实践中通常取 $m \geq 4p$ 使 bubble 降到可接受水平

### GPipe 的显存问题

GPipe 需要**同时缓存所有 $m$ 个 micro-batch 的中间 activation**（因为 forward 全做完才开始 backward）。这意味着显存占用随 $m$ 线性增长——为了减少 bubble 而增大 $m$，反而加剧了显存压力。

这是 GPipe 最大的局限：**bubble 和显存之间存在 trade-off**。

## 1F1B：交替调度

> [!paper] Harlap et al., 2018 (PipeDream)
> PipeDream: Fast and Efficient Pipeline Parallel DNN Training

1F1B（One Forward One Backward）调度的核心改进：**不等所有 forward 做完，而是尽早开始 backward**。

具体来说，pipeline 填充阶段（warmup）先执行若干 forward，之后每个 stage 交替执行一次 forward 和一次 backward。这样做的关键好处是：

> [!intuition] 为什么 1F1B 省显存
> 在 GPipe 中，stage 0 做完所有 $m$ 个 micro-batch 的 forward 后才开始 backward，所以必须同时保存 $m$ 份 activation。
>
> 在 1F1B 中，stage 0 做完一个 forward 后很快就做对应的 backward，activation 用完即释放。**任意时刻每个 stage 最多只缓存 $p$ 份 activation**（而非 $m$ 份），显存峰值大幅降低。

1F1B 的 bubble ratio 与 GPipe 相同（都是 $(p-1)/(p-1+m)$），但显存峰值从 $O(m)$ 降到 $O(p)$。由于实践中 $m \gg p$，这是一个显著的改进。

## 与其他并行方式的对比

> [!comparison] DP vs TP vs PP
>
> | 维度 | [[Data Parallelism\|DP]] | [[Tensor Parallelism\|TP]] | Pipeline Parallelism (PP) |
> |------|------|------|------|
> | **切分对象** | 数据 | 单个算子（矩阵） | 模型层 |
> | **通信模式** | AllReduce 梯度 | AllReduce activation（每层 2 次） | 点对点传 activation（stage 边界） |
> | **通信频率** | 每步 1 次 | 每层多次 | 每 micro-batch 每 stage 1 次 |
> | **通信量** | 大（全部梯度） | 中 | 小（单层 activation） |
> | **Bubble** | 无 | 无 | 有（核心瓶颈） |
> | **显存冗余** | 高（完整模型副本） | 低 | 低 |
> | **适合互联** | 任意 | 高带宽（NVLink） | 低带宽也可（InfiniBand） |

三种并行方式解决不同维度的问题，实际训练中通常组合使用。

## 3D Parallelism：DP + TP + PP

大模型训练的标准范式是 **3D 并行**——在三个维度同时切分：

- **TP**（机内）：同一节点内的 GPU 通过 NVLink 互联（带宽 ~600 GB/s），适合高频通信的层内切分
- **PP**（机间）：不同节点间通过 InfiniBand 互联（带宽 ~50 GB/s），适合低频通信的层间切分
- **DP**（全局）：在 TP + PP 构成的"模型副本"之间做数据并行，通常结合 [[ZeRO]] 减少显存冗余

> [!example] 典型配置示例
> 训练一个 175B 模型，使用 512 张 A100（64 个节点，每节点 8 卡）：
> - TP = 8（节点内 8 卡做 tensor 并行）
> - PP = 8（8 个节点串成 pipeline）
> - DP = 8（8 组 pipeline 副本做数据并行）
> - 总计：$8 \times 8 \times 8 = 512$ 卡

对于 [[Mixture of Experts|MoE]] 模型，还会加入 [[Expert Parallelism]]（EP），将不同 expert 分布到不同设备，形成 4D 并行。

## Bubble 优化的前沿进展

### Zero Bubble Pipeline Parallelism

> [!paper] Qi et al., 2023
> Zero Bubble Pipeline Parallelism

Zero Bubble（ZB）的核心洞察：**backward 可以拆成两个独立的部分**——backward for input（$B$）和 backward for weights（$W$）。$B$ 需要尽快执行（下游 stage 在等），但 $W$ 可以延后执行，用来填充 bubble。

通过精心调度 $F$、$B$、$W$ 三种操作，ZB 理论上可以将 bubble 降到接近零。代价是需要更复杂的调度逻辑和略高的显存（缓存更多中间状态）。

### DualPipe（DeepSeek-V3）

[[DeepSeek-V3 (2024)]] 提出的 DualPipe 算法在 Zero Bubble 的基础上更进一步：

- **双向调度**：从 pipeline 两端同时注入 micro-batch，进一步减少 bubble
- **计算-通信重叠**：将 attention、MLP、all-to-all dispatch/combine 等操作重新排列，使通信完全被计算掩盖
- 代价是需要保存两份模型参数（2x parameter memory），但在大 EP size 下这不是主要瓶颈

DualPipe 的设计动机来自 DeepSeek-V3 的 MoE 架构——跨节点的 expert parallelism 导致 all-to-all 通信开销与计算量接近 1:1，必须通过 pipeline 调度来隐藏通信。

## 局限性

> [!warning] Pipeline Parallelism 的边界
>
> 1. **Bubble 不可完全消除**：即使是 Zero Bubble 方案，实际中也难以达到理论零 bubble（调度开销、负载不均衡等）
> 2. **负载均衡困难**：不同层的计算量可能不同（如 MoE 层 vs 普通层），均匀切分会导致某些 stage 成为瓶颈
> 3. **Micro-batch 约束**：$m$ 必须足够大才能摊薄 bubble，但这增加了 batch size 的下限
> 4. **调试复杂**：跨设备的异步执行使得 debug 和 profiling 比单卡训练困难得多
> 5. **不适合浅模型**：PP 的收益来自"层数多"，对于层数少的模型，bubble 比例过高

> [!interview] 面试要点
> **Q: Pipeline Parallelism 和 Tensor Parallelism 的核心区别是什么？**
> A: TP 在层内切分（一个矩阵乘法拆到多卡），通信频繁但无 bubble；PP 在层间切分（不同层放不同卡），通信少但有 bubble。TP 需要高带宽互联（NVLink），PP 对带宽要求低，适合跨机。
>
> **Q: GPipe 和 1F1B 的主要区别？**
> A: Bubble ratio 相同，但 1F1B 通过交替执行 F 和 B 将显存峰值从 $O(m)$ 降到 $O(p)$。GPipe 是 F-then-B（先做完所有 forward 再 backward），1F1B 是尽早开始 backward。
>
> **Q: 为什么大模型训练要用 3D 并行而不是只用一种？**
> A: 每种并行有不同的 trade-off。TP 通信重但无 bubble，适合机内高带宽；PP 通信轻但有 bubble，适合机间；DP 无 bubble 但有显存冗余。3D 并行在不同维度组合它们，匹配硬件拓扑（NVLink 机内 + InfiniBand 机间）。

## 延伸阅读

**原始论文**：
- GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Huang et al., 2019)
- PipeDream: Generalized Pipeline Parallelism for DNN Training (Harlap et al., 2018)
- Zero Bubble Pipeline Parallelism (Qi et al., 2023)

**实际应用**：
- [[DeepSeek-V3 (2024)]] — DualPipe 算法，结合 PP 与计算-通信重叠
- Megatron-LM — NVIDIA 的 3D 并行训练框架
