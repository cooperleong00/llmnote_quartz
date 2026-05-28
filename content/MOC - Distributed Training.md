---
description: 分布式训练导航：显存账本、数据并行、模型并行、精度优化与 RL 训练系统
type: moc
tags:
  - distributed-training
  - efficiency
  - optimization
created: 2026-02-24
updated: 2026-05-29T00:00
---

# MOC - Distributed Training

分布式训练解决两个核心约束：模型和激活装不进单卡显存，训练吞吐无法满足规模需求。理解这件事要先建立显存账本，再看不同并行策略分别切分什么对象、引入什么通信、适合放在集群拓扑的哪个层级。

---

## 概览

```
分布式训练
├── 显存与 batch 账本
│   ├── [[GPU Memory Calculation]] — 参数/梯度/优化器/激活/KV cache
│   ├── [[Gradient Accumulation]] — 小显存模拟大 batch
│   └── [[Activation Checkpointing]] — 重计算换激活显存
│
├── 数据并行系列
│   ├── [[Data Parallelism]] — 复制模型，切分数据
│   ├── [[ZeRO]] — 切分 optimizer/gradient/parameter states
│   └── [[FSDP]] — PyTorch 原生全分片数据并行
│
├── 模型并行系列
│   ├── [[Tensor Parallelism]] — 层内矩阵切分
│   ├── [[Pipeline Parallelism]] — 层间流水线切分
│   └── [[Expert Parallelism]] — MoE 专家并行
│
├── 数值与 kernel 优化
│   ├── [[Mixed Precision Training]] — FP16/BF16/FP8
│   ├── [[Flash Attention]] — IO-aware attention kernel
│   └── [[Quantization]] — 低精度表示，推理为主
│
└── 组合与 RL 训练系统
    ├── [[3D Parallelism]] — DP + TP + PP
    ├── [[ROLL (2025)]] / [[ROLL Flash (2025)]]
    └── [[DORA]] — 异步 rollout 与资源编排
```

---

## 显存与 Batch 账本

[[GPU Memory Calculation]] 是进入分布式训练的第一站。训练显存通常由参数、梯度、优化器状态、激活值和临时 buffer 组成；推理显存还要重点计算 [[KV Cache]]。

| 技术 | 解决的问题 | 核心代价 |
|------|------------|----------|
| [[Gradient Accumulation]] | 显存装不下目标 global batch | 更新频率降低，训练时间增加 |
| [[Activation Checkpointing]] | 激活值显存随层数增长 | 反向传播需要重算前向片段 |
| [[Mixed Precision Training]] | 参数和激活的字节数过高 | 需要处理 loss scaling 和数值稳定 |

这三类技巧通常在引入跨卡并行之前就会使用，因为它们不改变模型切分方式。

---

## 数据并行系列

数据并行的基本模式是每张 GPU 持有一份模型副本，处理不同 mini-batch，再同步梯度。它扩展吞吐最直接，也最容易遇到 model states 冗余。

### DP/DDP → ZeRO → FSDP

[[Data Parallelism]] 的核心通信是梯度同步。随着模型增大，每张卡保存完整参数、梯度和优化器状态会造成显存浪费。

[[ZeRO]] 将这些 model states 分阶段切分：

| 方案 | 切分对象 | 显存收益 | 通信变化 |
|------|----------|----------|----------|
| ZeRO-1 | Optimizer states | 中等 | 接近 DDP |
| ZeRO-2 | Optimizer states + gradients | 更高 | ReduceScatter 梯度 |
| ZeRO-3 | Optimizer states + gradients + parameters | 最高 | 需要 AllGather 参数 |
| [[FSDP]] | PyTorch 原生 ZeRO-3 风格分片 | 高 | 与 PyTorch 生态集成更深 |

> [!intuition] 核心直觉
> 数据并行系列把每张卡重复保存的状态切开，需要时再通过通信取回。它用通信成本换显存容量。

---

## 模型并行系列

模型并行在模型本身超过单卡可承载范围时使用。三种常见切分维度彼此正交。

### Tensor Parallelism

[[Tensor Parallelism]] 在单个 Transformer 层内部切分矩阵计算，例如 Megatron-LM 的列并行和行并行。它通信频繁，通常依赖节点内 NVLink。

### Pipeline Parallelism

[[Pipeline Parallelism]] 按层把模型切成多个 stage。核心挑战是 pipeline bubble，GPipe 用 micro-batch 减少空闲时间，1F1B 调度进一步降低激活驻留。

### Expert Parallelism

[[Expert Parallelism]] 服务于 [[Mixture of Experts]]。不同 expert 分布在不同设备，token 通过 All-to-All 路由到目标 expert。它和 [[Loss-Free Load Balancing]]、[[Rollout Routing Replay]]、[[IcePop]] 一起构成 MoE 训练效率和稳定性的关键背景。

| 并行方式 | 切分对象 | 典型通信 | 适合拓扑 |
|----------|----------|----------|----------|
| [[Tensor Parallelism]] | 层内矩阵 | AllReduce / AllGather | 节点内高带宽 |
| [[Pipeline Parallelism]] | 层间 stage | P2P | 跨节点 |
| [[Expert Parallelism]] | MoE expert | All-to-All | 高带宽集群 |

---

## 数值与 Kernel 优化

[[Mixed Precision Training]] 通过 FP16/BF16/FP8 降低显存和提升吞吐。BF16 因指数范围更大，常用于大模型训练；FP8 在新一代模型报告中越来越常见。

[[Flash Attention]] 是 attention kernel 层的核心优化，通过 tiling、online softmax 和 recomputation 减少 HBM 访问。它与 [[MOC - Attention]] 中的高效 attention 主线相连。

[[Quantization]] 主要服务推理，但 QAT、FP8 训练和低精度 optimizer states 会把量化问题带回训练系统。

---

## 组合策略：3D Parallelism

[[3D Parallelism]] 将 DP、TP、PP 组合使用，是训练百亿到万亿参数模型的常见组织方式：

1. TP 放在节点内，利用高带宽连接切分层内矩阵。
2. PP 放在跨节点维度，按层切分模型并流水线执行。
3. DP/ZeRO/FSDP 放在最外层，扩展数据吞吐并切分 model states。

MoE 模型还会叠加 [[Expert Parallelism]]，形成包含 DP、TP、PP、EP 的多维并行。

---

## RL 训练系统

LLM RL 训练引入了 rollout、reward、policy update 之间的系统耦合。这里的瓶颈通常来自长序列生成、actor/critic/reference/reward model 的资源竞争，以及 online 数据生成带来的同步等待。

| 系统 | 核心贡献 | 关联问题 |
|------|----------|----------|
| [[ROLL (2025)]] | 单控制器与模块化 RL 训练框架 | 大规模 RL 编排 |
| [[ROLL Flash (2025)]] | 异步 RL 架构和细粒度并行优化 | RLVR 与 agentic 任务加速 |
| [[DORA]] | 多版本流式 rollout、动态资源编排、KV-cache 迁移 | 消除 long-tail generation bubble |

这部分与 [[MOC - Reinforcement Learning]] 和 [[MOC - Post-training]] 的 online RL 算法主线互相补充。

---

## 学习路径建议

**入门路线**：
1. [[GPU Memory Calculation]] — 先会算显存。
2. [[Data Parallelism]] → [[ZeRO]] → [[FSDP]] — 理解数据并行如何扩展。
3. [[Mixed Precision Training]] → [[Activation Checkpointing]] → [[Gradient Accumulation]] — 掌握单机效率技巧。

**进阶路线**：
1. [[Tensor Parallelism]] → [[Pipeline Parallelism]] — 理解模型切分。
2. [[Expert Parallelism]] → [[Mixture of Experts]] — 理解 MoE 并行。
3. [[3D Parallelism]] — 把 DP、TP、PP 放到同一个集群视角里。

**RL 系统路线**：
1. [[PPO]] / [[GRPO]] — 先理解训练环路。
2. [[ROLL (2025)]] → [[ROLL Flash (2025)]] → [[DORA]] — 理解 rollout 和 update 的系统优化。
3. [[Training-Inference Mismatch]] → [[Rollout Routing Replay]] — 理解训练系统与模型行为的耦合。

---

## 相关 MOC

- [[MOC - Inference]] — 推理端显存、serving 与吞吐优化
- [[MOC - Attention]] — Flash Attention、KV Cache 与 attention 架构
- [[MOC - Reinforcement Learning]] — Online RL 算法与 RL 训练环路
- [[MOC - Post-training]] — SFT、RLHF、OPD 与 Agentic RL 的训练流程
