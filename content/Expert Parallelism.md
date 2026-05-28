---
type: concept
description: MoE 模型中将不同 Expert 分布到不同设备上的并行策略，通过 All-to-All 通信实现 token 的跨设备路由
aliases:
  - 专家并行
  - EP
prerequisites:
  - "[[Mixture of Experts]]"
  - "[[Data Parallelism]]"
tags:
  - distributed-training
  - moe
created: 2026-02-24
updated: 2026-02-24
---

# Expert Parallelism

Expert Parallelism（专家并行）是 [[Mixture of Experts]] 模型的分布式并行策略：将不同 Expert 分布到不同设备上，token 通过 All-to-All 通信路由到目标 Expert 所在设备完成计算，再将结果收集回来。与 [[Tensor Parallelism]] 切分单层矩阵、[[Pipeline Parallelism]] 切分层不同，EP 切分的是 MoE 层中的专家集合。

## 动机

> [!intuition] 为什么需要 Expert Parallelism？
> MoE 模型的参数量远超同等计算量的 Dense 模型——[[DeepSeek-V3 (2024)]] 有 256 个 Expert，总参数 671B，但每个 token 只激活 37B。这些 Expert 不可能全部放在一张卡上，必须分布到多个设备。
>
> 关键在于：MoE 的稀疏激活天然适合并行化。每个 token 只需要访问少数几个 Expert，不同 Expert 之间没有计算依赖，可以完全独立地在不同设备上执行。问题只在于——如何高效地把 token 送到正确的设备上。

## 核心机制

### All-to-All 通信

EP 的核心通信原语是 All-to-All，整个过程分为两个阶段：

**Token Dispatch（分发）**：Router 决定每个 token 去哪些 Expert 后，通过 All-to-All 将 token 发送到对应 Expert 所在的设备。每个设备既是发送方（把本地 token 发给其他设备的 Expert）也是接收方（接收其他设备发来的 token）。

**Token Combine（收集）**：Expert 计算完成后，再通过一次 All-to-All 将结果发回 token 的原始设备，按 Router 权重加权合并。

```
Device 0 (Expert 0,1)    Device 1 (Expert 2,3)    Device 2 (Expert 4,5)
     │                        │                        │
     ├── token A → Expert 2 ──┤                        │
     │                        ├── token B → Expert 5 ──┤
     ├── token C → Expert 0 ──┤                        │
     │        (local)         │                        │
     │                        │                        │
     │◄── All-to-All Dispatch ──────────────────────────┤
     │                                                  │
     │        [Expert 计算]                              │
     │                                                  │
     │◄── All-to-All Combine ───────────────────────────┤
```

> [!warning] All-to-All 的通信特点
> 与 [[Tensor Parallelism]] 的 AllReduce 不同，All-to-All 的通信量取决于 token 的路由分布，而非固定的参数大小。如果路由不均匀，部分链路会成为瓶颈。
>
> 更关键的是，All-to-All 通常需要**跨节点**通信（节点内 GPU 数量有限，Expert 数量远超单节点容量），带宽受限于节点间互联（如 InfiniBand/RDMA），远低于节点内的 NVLink。

### Expert 分组策略

将 $E$ 个 Expert 分布到 $N$ 个设备上，每个设备持有 $E/N$ 个 Expert。分组策略影响通信模式：

- **均匀分配**：最简单的方式，每个设备放相同数量的 Expert
- **亲和性分组**：将经常被同时激活的 Expert 放在同一设备，减少跨设备通信。但这依赖于路由模式的稳定性，实际中较难利用

EP size（参与 Expert Parallelism 的设备数）是关键超参数。EP size 越大，每个设备上的 Expert 越少、计算量越小，但 All-to-All 通信涉及的设备越多、通信开销越大。

### 通信与计算的 Overlap

All-to-All 通信开销是 EP 的主要瓶颈。在大规模 MoE 模型中，跨节点的 All-to-All 通信时间可能与 Expert 计算时间接近 1:1，如果串行执行会严重拖慢训练。

主要的 overlap 策略：

1. **Shared Expert 掩盖通信**：MoE 架构中的 shared expert（所有 token 都经过的 Expert）在本地计算，可以与 routed expert 的 All-to-All dispatch 并行执行。但单个 shared expert 的计算量有限，掩盖窗口较小。

2. **DualPipe 调度**：[[DeepSeek-V3 (2024)]] 提出的方案，将 attention、MoE dispatch、Expert 计算、combine 等操作重新编排到流水线中，使前向和后向的通信与计算交错执行，几乎完全隐藏 All-to-All 开销。

3. **Shortcut-Connected MoE**：[[LongCat-Flash (2025)]] 的方案，通过跨层快捷连接重排执行流水线，扩大计算-通信重叠窗口。

## 与其他并行的组合

EP 很少单独使用，通常与其他并行策略组合形成多维并行：

> [!comparison] EP 与其他并行策略的组合
>
> | 组合 | 作用 | 典型场景 |
> |------|------|----------|
> | EP + [[Tensor Parallelism\|TP]] | Expert 内部再做张量切分 | 单个 Expert 仍然太大时 |
> | EP + [[Pipeline Parallelism\|PP]] | 不同层的 Expert 在不同 pipeline stage | 减少 pipeline bubble，隐藏通信 |
> | EP + [[Data Parallelism\|DP]] | 数据并行与专家并行正交组合 | 扩大训练吞吐量 |

EP 和 DP 的关系值得特别说明：在纯 DP 中，每个设备持有完整模型副本；加入 EP 后，MoE 层的 Expert 不再复制而是分布。因此 EP 实际上**替代了 MoE 层的数据并行**——每个设备处理的 token 来自多个 DP rank，有效 batch size 变为：

$$\text{effective\_batch} = \text{micro\_batch} \times \text{ep\_data\_parallel\_size}$$

这也是 [[Loss-Free Load Balancing]] 在大规模 EP 场景下效果更好的原因之一：更大的有效 batch 使全局负载统计更稳定。

## 负载均衡问题

> [!warning] Expert 热度不均是 EP 的核心挑战
> Router 的路由决策可能导致部分 Expert 接收远多于平均数量的 token（"热门 Expert"），而其他 Expert 几乎空闲。这带来两个问题：
>
> 1. **计算不平衡**：持有热门 Expert 的设备成为瓶颈，其他设备空等
> 2. **通信不平衡**：发往热门 Expert 所在设备的数据量远大于其他设备，链路拥塞
>
> 传统方案使用辅助损失（auxiliary loss）惩罚不均匀路由，但会干扰主任务梯度。[[Loss-Free Load Balancing]] 通过动态调整 expert-level bias 实现无损均衡，已成为大规模 MoE 训练的标准做法。

## 实际案例：DeepSeek-V3

[[DeepSeek-V3 (2024)]] 的 EP 配置是目前公开的最大规模实践之一：

- 256 个 routed expert + 1 个 shared expert
- 在 2048 张 H800 GPU 上训练
- 采用 DualPipe 实现计算-通信重叠，几乎零 All-to-All 通信开销
- 配合 [[Loss-Free Load Balancing]] 实现无辅助损失的负载均衡
- FP8 混合精度训练进一步降低通信量

> [!interview] 面试视角
> **Q: Expert Parallelism 和 Tensor Parallelism 有什么区别？**
> A: TP 切分的是层内的矩阵运算（每层都需要 AllReduce 同步），EP 切分的是 MoE 层中的 Expert 集合（通过 All-to-All 路由 token）。TP 要求极高带宽（通常限于节点内 NVLink），EP 的通信模式取决于路由分布，可以跨节点但需要处理负载均衡。两者可以组合使用。
>
> **Q: EP 的主要瓶颈是什么？**
> A: All-to-All 跨节点通信。在大规模 MoE 中，通信时间可能与计算时间接近 1:1。解决方案是通过流水线调度（如 DualPipe）实现计算-通信重叠，以及通过负载均衡减少通信热点。

## 延伸阅读

**后续发展**：
- [[DualPipe]] — DeepSeek-V3 的流水线调度，专为 EP 通信重叠设计
- [[LongCat-Flash (2025)]] — Shortcut-Connected MoE 扩大重叠窗口

**导航**：
- [[MOC - Distributed Training]] — 分布式训练全景
