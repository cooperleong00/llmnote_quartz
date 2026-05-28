---
type: concept
description: 将数据并行、张量并行、流水线并行三种策略组合使用的分布式训练方案，是训练千亿参数模型的标准范式
aliases:
  - 三维并行
  - 3D 并行
  - Multi-dimensional Parallelism
prerequisites:
  - "[[Data Parallelism]]"
  - "[[Tensor Parallelism]]"
  - "[[Pipeline Parallelism]]"
tags:
  - distributed-training
created: 2026-02-24
updated: 2026-02-24
---

# 3D Parallelism

三维并行（3D Parallelism）将 [[Data Parallelism]]、[[Tensor Parallelism]] 和 [[Pipeline Parallelism]] 三种并行策略组合使用，在不同维度同时切分数据和模型。这是训练千亿参数模型（如 GPT-3、Llama 3）的标准工程范式，由 NVIDIA 的 Megatron-LM 框架率先系统化实现。

## 动机：为什么单一并行不够

> [!intuition] 每种并行策略都有自己的瓶颈
>
> - **Data Parallelism**：每张卡需要存完整模型副本，当模型大到单卡放不下时直接失效
> - **Tensor Parallelism**：每层需要多次 AllReduce 同步，通信频率极高，跨节点带宽撑不住
> - **Pipeline Parallelism**：流水线气泡（bubble）浪费算力，且单独使用时并行度受限于层数
>
> 单独使用任何一种，要么放不下模型，要么通信成为瓶颈，要么效率太低。组合使用的核心思路是：**让每种并行策略在它最擅长的维度上工作，匹配硬件拓扑的通信特性**。

把问题具体化：假设你要在 512 张 GPU（64 个节点，每节点 8 卡）上训练一个 175B 模型。

- 纯 DP？175B 模型单卡放不下（即使 A100 80GB 也不够）
- 纯 TP？512 卡做 TP 意味着跨节点的 AllReduce，每层 4 次同步，延迟爆炸
- 纯 PP？512 个 pipeline stage，bubble 比例 $\frac{p-1}{m+p-1}$ 极高

3D 并行的答案是：**在三个正交维度上各取所长**。

## 核心机制：三维切分

### 三个维度各切什么

| 维度 | 切分对象 | 通信模式 | 通信需求 |
|------|----------|----------|----------|
| **TP**（层内） | 单层的矩阵运算 | AllReduce（每层 forward/backward 各 2 次） | 高带宽 + 低延迟 |
| **PP**（层间） | 模型按层分段 | 点对点传输（stage 间传 activation） | 低延迟，带宽要求不高 |
| **DP**（数据） | 训练数据 | AllReduce 梯度（每 step 一次） | 带宽要求相对宽松 |

三者的关系是正交的——TP 在一层内部切分计算，PP 把不同层分到不同设备组，DP 在多个"模型副本"之间切分数据：

$$N_{\text{total}} = N_{\text{TP}} \times N_{\text{PP}} \times N_{\text{DP}}$$

### 设备拓扑映射：关键设计决策

3D 并行的核心工程问题不是"要不要组合"，而是**怎么把三个维度映射到物理硬件拓扑上**。现代 GPU 集群的通信带宽是分层的：

- **节点内**（NVLink）：~600-900 GB/s（H100），延迟极低
- **节点间**（InfiniBand）：~50-400 GB/s，延迟较高

> [!intuition] 拓扑映射的核心原则
>
> **通信需求最高的维度，放在带宽最高的互联上。**
>
> - TP 每层需要多次 AllReduce → 放在节点内（NVLink）
> - PP 只需要 stage 间传 activation → 可以跨节点（InfiniBand）
> - DP 每 step 同步一次梯度 → 跨节点组，通常结合 [[ZeRO]] 减少通信量

以 512 卡（64 节点 × 8 卡/节点）训练 175B 模型为例：

```
节点内 (NVLink, ~900 GB/s)          节点间 (InfiniBand, ~400 GB/s)
┌──────────────────────┐
│  GPU0 GPU1 ... GPU7  │──── PP Stage 0, DP Replica 0
│  ←── TP = 8 ────→    │
└──────────────────────┘
         │ PP (跨节点)
┌──────────────────────┐
│  GPU0 GPU1 ... GPU7  │──── PP Stage 1, DP Replica 0
│  ←── TP = 8 ────→    │
└──────────────────────┘
         │ ...
    (共 8 个 PP stage)

× 8 组 DP 副本 = 8 × 8 × 8 = 512 卡
```

### 通信量分析

理解三个维度的通信开销有助于做出正确的并行度选择：

**Tensor Parallelism**：每个 Transformer 层的 forward 和 backward 各需要 2 次 AllReduce，通信量约 $4 \times 2bsh$（$b$ = micro-batch size，$s$ = sequence length，$h$ = hidden size）。对于 100+ 层的模型，这意味着每个 training step 数百次同步。

**Pipeline Parallelism**：stage 之间只需要传递 activation tensor，通信量为 $bsh$（每个 micro-batch 一次）。通信频率远低于 TP，但 pipeline bubble 会浪费算力。

**Data Parallelism**：每个 training step 结束时做一次梯度 AllReduce，通信量约 $2M$（$M$ 为模型参数量）。如果结合 [[ZeRO]]，还可以进一步切分 optimizer states 和梯度，减少显存占用。

> [!comparison] 通信特性对比
>
> | 维度 | 通信频率 | 单次通信量 | 对延迟敏感度 | 对带宽敏感度 |
> |------|----------|-----------|-------------|-------------|
> | TP | 极高（每层多次） | 中等 | 高 | 高 |
> | PP | 中等（每 micro-batch） | 较小 | 中 | 低 |
> | DP | 低（每 step 一次） | 大（全部梯度） | 低 | 中 |

## 扩展到更多维度

3D 并行是基础框架，实际的大规模训练往往需要更多维度：

### 4D 并行：+ Expert Parallelism

对于 [[Mixture of Experts|MoE]] 模型，[[Expert Parallelism]]（EP）将不同 Expert 分布到不同设备上，通过 All-to-All 通信路由 token。EP 和 DP 的关系比较微妙——EP 实际上**替代了 MoE 层的数据并行**，因为每个设备处理的 token 来自多个 DP rank。

典型的 4D 配置：TP + EP + PP + DP，其中 TP 和 EP 共享节点内的 GPU（EP 通常需要跨节点通信）。

### 5D 并行：+ Sequence Parallelism

Sequence Parallelism（SP）沿序列维度切分 LayerNorm 和 Dropout 的 activation，作为 TP 的补充。在 Megatron-LM 中，SP 与 TP 紧密耦合——TP 切分 Attention/MLP 的计算，SP 切分其余部分，使整个 Transformer 层的 activation 都被切分，显存节省更彻底。

### + ZeRO：混合策略

[[ZeRO]] 可以叠加在 DP 维度上，将 optimizer states（Stage 1）、梯度（Stage 2）甚至参数（Stage 3）切分到 DP 组内的各卡。这不增加新的并行维度，但显著降低每卡的显存占用。实践中 ZeRO Stage 1 几乎是标配，因为它只增加极少通信开销就能大幅节省显存。

## 实际配置案例

### Megatron-LM 的经典 3D 配置

Megatron-LM（NVIDIA）是 3D 并行的标杆实现。以训练 GPT-3 175B 为例：

| 参数 | 值 | 说明 |
|------|-----|------|
| TP | 8 | 节点内 8 卡，NVLink 互联 |
| PP | 8 | 8 个节点串成 pipeline |
| DP | 8 | 8 组 pipeline 副本 |
| 总 GPU | 512 | $8 \times 8 \times 8$ |
| Micro-batch | 较大 | 摊薄 pipeline bubble |

配合 [[Activation Checkpointing]] 减少 activation 显存，[[Mixed Precision Training]] 加速计算和减少通信量。

### DeepSeek-V3 的多维并行

[[DeepSeek-V3 (2024)]] 在 2048 张 H800 上训练 671B MoE 模型，采用了更复杂的多维并行：

- **EP = 64**（跨节点分布 256 个 routed expert）
- **PP = 16**（DualPipe 双向调度，大幅减少 bubble）
- **DP = 2**（ZeRO-1 切分 optimizer states）
- **TP = 1**（未使用 TP，因为 MoE 的单个 Expert 足够小）

> [!intuition] 为什么 DeepSeek-V3 不用 TP？
>
> MoE 模型的特点是单个 Expert 参数量小（每个 Expert 约 2B），单卡完全放得下，不需要 TP 做层内切分。但 Expert 数量多（256 个），需要大规模 EP 来分布。这与 Dense 模型（如 GPT-3）的 3D 配置形成鲜明对比——Dense 模型单层参数大，必须用 TP 切分。

DeepSeek-V3 的关键创新是 DualPipe 算法：从 pipeline 两端同时注入 micro-batch，并将 attention、MLP、All-to-All 通信重新排列，使 EP 的跨节点通信几乎完全被计算掩盖。

## 如何选择并行度：决策框架

给定 $N$ 张 GPU 和模型大小，如何确定 $N_{\text{TP}}$、$N_{\text{PP}}$、$N_{\text{DP}}$？

### Step 1：确定 TP 度数

**约束**：TP 必须在节点内（NVLink），所以 $N_{\text{TP}} \leq$ 节点内 GPU 数（通常 8）。

**决策**：
- 单层参数能单卡放下 → $N_{\text{TP}} = 1$（如 MoE 的单个 Expert）
- 单层参数需要切分 → $N_{\text{TP}} = 2, 4, 8$（必须整除 attention head 数）

### Step 2：确定 PP 度数

**约束**：$N_{\text{PP}}$ 不宜太大，否则 bubble 比例 $\frac{p-1}{m+p-1}$ 过高。

**决策**：
- 模型层数 / PP 度数 = 每个 stage 的层数，应尽量均匀
- 通常 $N_{\text{PP}} = 4 \sim 16$，配合足够多的 micro-batch 摊薄 bubble
- 如果模型不大（<30B），可以不用 PP（$N_{\text{PP}} = 1$）

### Step 3：DP 度数自动确定

$$N_{\text{DP}} = \frac{N_{\text{total}}}{N_{\text{TP}} \times N_{\text{PP}}}$$

DP 度数越大，训练吞吐量越高，但有效 batch size 也越大（需要配合 learning rate scaling）。

### Step 4：叠加优化

- **ZeRO Stage 1**：几乎必选，切分 optimizer states
- **[[Activation Checkpointing]]**：显存不够时启用，代价约 33% 额外计算
- **[[Mixed Precision Training]]**：BF16/FP16 加速计算，FP8 进一步压缩通信量
- **Gradient Accumulation**：在 DP 度数有限时模拟更大 batch size

> [!example] 决策示例
>
> **场景**：在 128 张 H100（16 节点 × 8 卡）上训练 70B Dense 模型
>
> 1. TP = 8（节点内，70B 单层约 1B 参数，需要切分）
> 2. PP = 4（4 个节点串成 pipeline，每 stage 约 20 层）
> 3. DP = 4（$128 / (8 \times 4) = 4$ 组副本）
> 4. 叠加 ZeRO-1 + Activation Checkpointing + BF16

## 局限性

> [!warning] 3D 并行的边界
>
> 1. **配置复杂度高**：三个维度的并行度选择相互耦合，最优配置依赖于模型架构、硬件拓扑、batch size 等多个因素，通常需要 profiling 实验
> 2. **负载均衡困难**：PP 的不同 stage 计算量可能不均（如 embedding 层 vs Transformer 层），TP 要求 head 数整除并行度
> 3. **故障恢复代价大**：任何一张卡故障都会影响整个训练，需要从 checkpoint 恢复。集群越大，单卡故障概率越高
> 4. **框架依赖**：不同框架（Megatron-LM、DeepSpeed、PyTorch FSDP）对 3D 并行的支持程度和实现方式不同，迁移成本高
> 5. **不是万能的**：对于小模型（<10B），单纯的 DP + ZeRO 可能比 3D 并行更高效，引入 TP/PP 的通信开销反而得不偿失

> [!interview] 面试要点
>
> **Q: 为什么大模型训练要用 3D 并行而不是只用一种？**
> A: 每种并行有不同的 trade-off。TP 通信重但无 bubble，适合机内高带宽；PP 通信轻但有 bubble，适合机间；DP 无 bubble 但有显存冗余。3D 并行在不同维度组合它们，匹配硬件拓扑（NVLink 机内 + InfiniBand 机间），让每种策略在最合适的层级工作。
>
> **Q: 如何决定 TP、PP、DP 的并行度？**
> A: 先确定 TP（受限于节点内 GPU 数和 attention head 数），再确定 PP（受限于层数和可接受的 bubble 比例），DP 由剩余 GPU 数决定。核心原则是 TP 放节点内（需要高带宽），PP 跨节点（通信量小），DP 最外层。
>
> **Q: MoE 模型的并行策略和 Dense 模型有什么不同？**
> A: MoE 模型单个 Expert 参数量小，通常不需要 TP；但 Expert 数量多，需要 EP 跨设备分布。所以 MoE 常用 EP + PP + DP（如 DeepSeek-V3），而 Dense 模型常用 TP + PP + DP。EP 引入的 All-to-All 通信是 MoE 训练的主要瓶颈。

## 延伸阅读

**原始论文与框架**：
- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", 2019
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", 2021 — 系统化提出 3D 并行的拓扑映射策略

**后续发展**：
- [[Expert Parallelism]] — MoE 模型的第四维并行
- [[FSDP]] — PyTorch 原生的 ZeRO 实现，可与 TP/PP 组合
- [[DualPipe]] — DeepSeek-V3 的双向流水线调度，专为 EP 通信重叠设计
