---
description: 分布式训练导航：数据并行、模型并行、精度优化与组合策略
type: moc
tags:
  - distributed-training
  - efficiency
  - optimization
created: 2026-02-24
updated: 2026-02-24
---

# MOC - Distributed Training

当模型参数量从数十亿增长到万亿级别，单张 GPU 既装不下完整模型（显存墙），也无法在合理时间内完成训练（计算墙）。分布式训练（Distributed Training）的核心挑战就是在多设备间高效地切分显存负担和计算负载，同时尽量减少通信开销和 GPU 空闲时间。不同的并行策略切分的对象不同——数据并行切分数据、张量并行切分层内计算、流水线并行切分层间计算——而实际的大规模训练往往需要将它们组合使用。

---

## 概览

```
分布式训练
├── 数据并行系列（切分数据，复制模型）
│   ├── [[Data Parallelism]] — DP/DDP + AllReduce ✅
│   ├── [[ZeRO]] — 消除 DDP 显存冗余 (Stage 1/2/3) ✅
│   └── [[FSDP]] — PyTorch 原生 ZeRO-3 实现 🔗
│
├── 模型并行系列（切分模型，分摊显存）
│   ├── [[Tensor Parallelism]] — 层内矩阵切分 (Megatron-LM) ✅
│   ├── [[Pipeline Parallelism]] — 层间流水线调度 (GPipe/1F1B) ✅
│   └── [[Expert Parallelism]] — MoE 专家分布 🔗
│
├── 精度优化（降低每个参数的显存/计算成本）
│   ├── [[Mixed Precision Training]] — FP16/BF16/FP8 混合精度 ✅
│   └── [[Quantization]] — 量化（主要用于推理） ✅
│
├── 计算优化（提升单卡效率）
│   ├── [[Flash Attention]] — IO-aware Attention 算法 ✅
│   └── [[Gradient Checkpointing]] — 用重计算换显存 🔗
│
└── 组合策略
    └── [[3D Parallelism]] — DP + TP + PP 的工程实践 🔗
```

> ✅ 已有笔记 | 🔗 待创建（stub）

---

## 数据并行系列

数据并行是分布式训练的起点：每张 GPU 持有完整模型副本，各自处理不同的数据 mini-batch，然后同步梯度。这条路线的演进核心是**消除冗余显存**。

### 演进路线：DP → DDP → ZeRO → FSDP

[[Data Parallelism]] 从最朴素的 DP（Parameter Server 架构）发展到 DDP（AllReduce 架构），解决了通信瓶颈问题。但 DDP 的根本限制是每张卡都存储完整的 model states（参数 + 梯度 + 优化器状态），显存利用率很低。

[[ZeRO]] 正是为了解决这个问题：通过渐进式切分 model states，在数据并行框架内消除冗余。三个 Stage 分别切分 optimizer states（Stage 1）、gradients（Stage 2）和 parameters（Stage 3），显存节省从 4x 到线性扩展。

[[FSDP]]（Fully Sharded Data Parallel）是 PyTorch 对 ZeRO Stage 3 的原生实现，提供了更易用的 API 和与 PyTorch 生态的深度集成。

> [!intuition] 核心直觉
> 数据并行系列的演进逻辑：DDP 让每张卡都存一份完整模型 → 太浪费 → ZeRO 把冗余的部分切开分给不同卡 → 需要时再通信取回。本质是**用通信换显存**。

### 关键权衡

| 方案 | 显存效率 | 通信量 | 实现复杂度 |
|------|----------|--------|------------|
| DDP | 低（全冗余） | AllReduce 梯度 | 低 |
| ZeRO-1 | 中 | 同 DDP | 低 |
| ZeRO-2 | 中高 | 同 DDP | 中 |
| ZeRO-3 / FSDP | 高（线性扩展） | AllGather 参数 + ReduceScatter 梯度 | 高 |

---

## 模型并行系列

当单个模型大到即使用 ZeRO-3 也无法高效训练时，需要将模型本身切分到多个设备。模型并行有两个正交的维度：层内切分和层间切分。

### 层内并行：Tensor Parallelism

[[Tensor Parallelism]] 将单个 Transformer 层的矩阵运算切分到多张 GPU。Megatron-LM 提出的列切分（Column Parallel）和行切分（Row Parallel）方案，使得 MLP 和 Self-Attention 的计算可以跨卡并行，每层只需两次 AllReduce 通信。

TP 的通信发生在每一层的前向和反向传播中，因此对卡间带宽要求极高，通常限制在同一节点内的 NVLink 连接的 GPU 之间。

### 层间并行：Pipeline Parallelism

[[Pipeline Parallelism]] 将模型按层切分到不同设备，数据以流水线方式依次通过各 stage。核心挑战是 pipeline bubble——GPU 空闲等待的时间。GPipe 通过 micro-batch 切分减少 bubble，1F1B（One Forward One Backward）调度进一步优化了显存占用，DualPipe 则在 DeepSeek-V3 中实现了计算与通信的重叠。

PP 的通信只发生在相邻 stage 之间（点对点），带宽需求远低于 TP，适合跨节点部署。

### 专家并行：Expert Parallelism

[[Mixture of Experts]] 架构天然适合并行化：不同专家可以放在不同 GPU 上。[[Expert Parallelism]] 将 MoE 层的专家分布到多个设备，token 通过 All-to-All 通信路由到对应专家。这是 MoE 模型（如 Mixtral、DeepSeek-V2/V3）训练和推理的关键并行策略。

> [!comparison] 三种模型并行的对比
>
> | 维度 | Tensor Parallelism | Pipeline Parallelism | Expert Parallelism |
> |------|-------------------|---------------------|-------------------|
> | 切分对象 | 层内矩阵 | 层间（按层分组） | MoE 专家 |
> | 通信模式 | AllReduce（每层） | P2P（相邻 stage） | All-to-All（路由） |
> | 带宽需求 | 极高（NVLink） | 中等 | 高 |
> | 适用场景 | 节点内 | 跨节点 | MoE 架构 |

---

## 精度优化

降低每个参数的数值精度，可以同时减少显存占用和加速计算。

[[Mixed Precision Training]] 是现代 LLM 训练的标准配置：前向和反向传播使用 FP16/BF16 加速计算，master weights 保持 FP32 确保数值稳定。BF16 因其更大的指数范围（不易溢出）已成为大模型训练的首选格式。FP8 训练（如 DeepSeek-V3）是最新的前沿方向，进一步将精度降低到 8-bit。

[[Quantization]] 主要用于推理阶段，将权重和激活值量化到 INT8/INT4 以减少内存和加速计算。训练中的量化（Quantization-Aware Training）也在探索中。

---

## 计算优化

在不改变并行策略的前提下，提升单卡的计算效率。

[[Flash Attention]] 通过 tiling 和 kernel fusion 优化 Attention 计算的内存访问模式，将 HBM 访问量从 $O(N^2)$ 降到 $O(N)$，实现 2-4x 加速。这是一个精确算法（非近似），已成为所有主流训练框架的标配。

[[Gradient Checkpointing]]（也称 Activation Checkpointing）是经典的**时间换空间**策略：前向传播时不保存中间激活值，反向传播时重新计算。可以将激活值显存从 $O(L)$ 降到 $O(\sqrt{L})$（$L$ 为层数），代价是约 33% 的额外计算。

---

## 组合策略：3D Parallelism

实际的大规模训练（如 GPT-3、Llama 3、DeepSeek-V3）不会只用单一并行策略，而是将多种策略组合使用。

[[3D Parallelism]] 是最经典的组合：
- **TP**（节点内）：利用 NVLink 高带宽，切分层内计算
- **PP**（跨节点）：利用较低的跨节点带宽，切分层间计算
- **DP/ZeRO**（最外层）：扩展数据吞吐量

```
                    ┌─────────────────────────────────┐
                    │         3D Parallelism           │
                    │                                  │
  DP/ZeRO 维度     │  Replica 0        Replica 1      │
  (跨组复制)       │  ┌────────────┐  ┌────────────┐  │
                    │  │ Stage 0    │  │ Stage 0    │  │
  PP 维度          │  │ ┌───┬───┐  │  │ ┌───┬───┐  │  │
  (跨节点分层)     │  │ │GPU│GPU│  │  │ │GPU│GPU│  │  │
                    │  │ │ 0 │ 1 │  │  │ │ 4 │ 5 │  │  │
  TP 维度          │  │ └───┴───┘  │  │ └───┴───┘  │  │
  (节点内切分)     │  │ Stage 1    │  │ Stage 1    │  │
                    │  │ ┌───┬───┐  │  │ ┌───┬───┐  │  │
                    │  │ │GPU│GPU│  │  │ │GPU│GPU│  │  │
                    │  │ │ 2 │ 3 │  │  │ │ 6 │ 7 │  │  │
                    │  │ └───┴───┘  │  │ └───┴───┘  │  │
                    │  └────────────┘  └────────────┘  │
                    └─────────────────────────────────┘
```

DeepSeek-V3 等最新模型还加入了 Expert Parallelism，形成 4D 甚至 5D 并行。

---

## 学习路径建议

**入门路线**（理解基本概念）：
1. [[Data Parallelism]] — 从最简单的 DP/DDP 开始
2. [[Mixed Precision Training]] — 理解精度与效率的权衡
3. [[ZeRO]] — 理解显存优化的核心思路

**进阶路线**（理解模型并行）：
1. [[Tensor Parallelism]] — 层内切分的原理
2. [[Pipeline Parallelism]] — 层间切分与调度
3. [[Flash Attention]] — 计算层优化

**高阶路线**（理解工程实践）：
1. [[3D Parallelism]] — 组合策略的设计选择
2. [[FSDP]] — PyTorch 原生方案
3. [[Expert Parallelism]] — MoE 并行化

---

## 相关 MOC

- [[MOC - Inference]] — 推理优化（与训练优化互补）
- [[MOC - Attention]] — Attention 机制（Flash Attention 的理论基础）
- [[MOC - Foundations]] — Transformer 基础架构
