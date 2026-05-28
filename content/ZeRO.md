---
type: method
description: 在数据并行框架内渐进式切分 model states（optimizer states → gradients → parameters），消除 DDP 的显存冗余，使数十亿参数模型无需模型并行即可训练
aliases:
  - Zero Redundancy Optimizer
  - ZeRO-DP
prerequisites:
  - "[[Data Parallelism]]"
tags:
  - distributed-training
  - efficiency
  - optimization
created: 2026-02-24
updated: 2026-02-24
---

# ZeRO

ZeRO（Zero Redundancy Optimizer）解决了 [[Data Parallelism]] 中最大的显存浪费问题：$N$ 张卡存 $N$ 份完全相同的 model states。核心思想极其简单——既然每张卡都存了一样的东西，为什么不让每张卡只存 $1/N$？ZeRO 通过三个渐进式阶段逐步切分 optimizer states、gradients 和 parameters，在**不改变数据并行通信模式**的前提下，将单卡显存占用从 $16\Phi$ 降低到接近 $16\Phi / N$。

## 动机：DDP 的显存冗余

> [!intuition] 为什么需要 ZeRO？
> 在 [[Data Parallelism|DDP]] 中，每张卡持有完整的 model states 用于独立计算梯度，然后通过 AllReduce 同步。但仔细想想——AllReduce 之后每张卡的 optimizer states、gradients 都是一样的，为什么每张卡都要存一份？
>
> 这就像一个团队里每个人都打印了一份完整的文档，但每个人只负责编辑其中几页。ZeRO 的做法是：每个人只保留自己负责的那几页。

回顾 [[Data Parallelism#显存分析：为什么需要 ZeRO|DDP 的显存分析]]，以混合精度 + [[Adam]] 为例，每张卡需要存储 $16\Phi$ bytes：

| 组件 | 精度 | 大小 | 类别 |
|------|------|------|------|
| 模型参数（fp16） | FP16 | $2\Phi$ | Parameters |
| 梯度（fp16） | FP16 | $2\Phi$ | Gradients |
| 模型参数（fp32 主副本） | FP32 | $4\Phi$ | Optimizer States |
| Adam 一阶动量 $m$ | FP32 | $4\Phi$ | Optimizer States |
| Adam 二阶动量 $v$ | FP32 | $4\Phi$ | Optimizer States |
| **合计** | | **$16\Phi$** | |

其中 Optimizer States 占了 $12\Phi$（75%），是最大的冗余来源。ZeRO 的策略就是从最"胖"的部分开始切。

## 三个阶段：渐进式切分

ZeRO 的设计哲学是**渐进式**的：每个阶段在前一个阶段的基础上多切一类数据，显存节省越来越多，但通信模式也逐渐变化。

### Stage 1：切分 Optimizer States（$P_{os}$）

最保守也最实用的阶段。每张卡只保存 $1/N$ 的 optimizer states，但仍持有完整的 gradients 和 parameters。

训练流程：
1. Forward + Backward：和 DDP 完全一样，每张卡独立计算梯度
2. AllReduce Gradients：和 DDP 一样同步梯度（实际用 Reduce-Scatter，每卡只收集自己负责的那部分）
3. Optimizer Step：每张卡只更新自己负责的 $1/N$ 参数（用自己持有的 optimizer states）
4. AllGather Parameters：将更新后的参数广播给所有卡，恢复完整模型

> [!math] Stage 1 显存分析
>
> | 组件 | 每卡存储 |
> |------|----------|
> | Parameters（fp16） | $2\Phi$（完整） |
> | Gradients（fp16） | $2\Phi$（完整） |
> | Optimizer States | $12\Phi / N$（切分） |
> | **合计** | $4\Phi + 12\Phi / N$ |
>
> 当 $N = 64$ 时：$4\Phi + 0.19\Phi \approx 4.2\Phi$，相比 DDP 的 $16\Phi$ 节省了 **~4x**。

> [!intuition] 为什么从 Optimizer States 开始切？
> 因为 optimizer states 占比最大（75%），而且它们只在 optimizer step 时使用。Forward 和 backward 过程完全不需要 optimizer states，所以切分它们对计算流程的影响最小。

### Stage 2：+ 切分 Gradients（$P_{os+g}$）

在 Stage 1 基础上，梯度也不再需要每张卡都存完整的。既然每张卡只更新 $1/N$ 的参数，那它也只需要这 $1/N$ 参数对应的梯度。

训练流程的关键变化：
- Backward 时，梯度计算完成后立即通过 Reduce-Scatter 分发（而非先 AllReduce 再丢弃）
- 每张卡只保留自己负责的那 $1/N$ 梯度，其余立即释放

> [!math] Stage 2 显存分析
>
> | 组件 | 每卡存储 |
> |------|----------|
> | Parameters（fp16） | $2\Phi$（完整） |
> | Gradients（fp16） | $2\Phi / N$（切分） |
> | Optimizer States | $12\Phi / N$（切分） |
> | **合计** | $2\Phi + 14\Phi / N$ |
>
> 当 $N = 64$ 时：$2\Phi + 0.22\Phi \approx 2.2\Phi$，相比 DDP 节省了 **~8x**。

### Stage 3：+ 切分 Parameters（$P_{os+g+p}$）

最激进的阶段。连模型参数本身也切分了——每张卡只存 $1/N$ 的参数。这意味着 forward 和 backward 时需要临时从其他卡收集完整参数。

训练流程的关键变化：
- Forward：每计算一层前，通过 AllGather 收集该层的完整参数；计算完后立即释放非本卡负责的参数
- Backward：同样需要 AllGather 收集参数来计算梯度；梯度通过 Reduce-Scatter 分发后释放

> [!math] Stage 3 显存分析
>
> | 组件 | 每卡存储 |
> |------|----------|
> | Parameters（fp16） | $2\Phi / N$（切分） |
> | Gradients（fp16） | $2\Phi / N$（切分） |
> | Optimizer States | $12\Phi / N$（切分） |
> | **合计** | $16\Phi / N$ |
>
> 当 $N = 64$ 时：$0.25\Phi$，相比 DDP 节省了 **~64x**。理论上 $N$ 越大，单卡显存越接近零。

> [!warning] Stage 3 的代价
> 参数切分意味着 forward 和 backward 的每一层都需要额外的 AllGather 通信。这是用**通信换显存**的典型 trade-off。

### 三阶段总览

| | DDP | Stage 1 | Stage 2 | Stage 3 |
|------|-----|---------|---------|---------|
| **切分内容** | 无 | Optimizer States | + Gradients | + Parameters |
| **每卡显存** | $16\Phi$ | $4\Phi + 12\Phi/N$ | $2\Phi + 14\Phi/N$ | $16\Phi/N$ |
| **N=64 时** | $16\Phi$ | $\approx 4.2\Phi$ | $\approx 2.2\Phi$ | $0.25\Phi$ |
| **通信量** | $2\Phi$ | $2\Phi$ | $2\Phi$ | $\approx 3\Phi$ |
| **通信模式** | AllReduce | Reduce-Scatter + AllGather | Reduce-Scatter + AllGather | 逐层 AllGather × 2 + Reduce-Scatter |

## 通信量分析

这是理解 ZeRO 实用性的关键：Stage 1 和 Stage 2 的通信量和 DDP 完全相同。

> [!math] 通信量推导
>
> **DDP（AllReduce）**：
> AllReduce 可以分解为 Reduce-Scatter + AllGather，总通信量为 $2\Phi$（每个元素发送一次 + 接收一次）。
>
> **ZeRO Stage 1 & 2**：
> - Reduce-Scatter（梯度同步）：$\Phi$（每个元素发送一次）
> - AllGather（参数同步）：$\Phi$（每个元素接收一次）
> - 合计：$2\Phi$，**和 DDP 完全一样**
>
> **ZeRO Stage 3**：
> - Forward AllGather：$\Phi$（收集参数）
> - Backward AllGather：$\Phi$（再次收集参数）
> - Reduce-Scatter（梯度）：$\Phi$
> - 合计：$3\Phi$，约为 DDP 的 **1.5 倍**

> [!intuition] 为什么 Stage 1/2 不增加通信？
> DDP 的 AllReduce 本质上就是 Reduce-Scatter + AllGather。ZeRO Stage 1/2 只是把这两步拆开，在中间插入了"只保留自己那份"的逻辑。总的数据传输量没变，只是每张卡保留的数据变少了。

## ZeRO-Offload 与 ZeRO-Infinity

当 GPU 显存仍然不够时，ZeRO 提供了进一步的 offload 策略：

**ZeRO-Offload**：将 optimizer states 和部分计算 offload 到 CPU。
- Optimizer step 在 CPU 上执行（Adam 更新是 element-wise 的，CPU 也能做）
- GPU 只负责 forward/backward 计算
- 代价：CPU-GPU 之间的 PCIe 带宽成为瓶颈

**ZeRO-Infinity**：在 ZeRO-Offload 基础上进一步扩展到 NVMe SSD。
- 利用 NVMe 的大容量存储 optimizer states 甚至 parameters
- 通过预取（prefetch）和流水线化隐藏 I/O 延迟
- 理论上可以在单个 GPU 节点上训练万亿参数模型

> [!warning] Offload 的实际考量
> Offload 策略的效果高度依赖硬件配置：
> - PCIe 4.0 x16 带宽约 32 GB/s，对于大模型可能成为瓶颈
> - NVMe SSD 的随机读写性能差异很大
> - 实践中通常只 offload optimizer states，避免 offload parameters（通信开销太大）

## 与 FSDP 的关系

[[FSDP]]（Fully Sharded Data Parallel）是 PyTorch 对 ZeRO Stage 3 的原生实现。

> [!comparison] ZeRO Stage 3 vs FSDP
>
> | 维度 | ZeRO (DeepSpeed) | FSDP (PyTorch) |
> |------|------------------|----------------|
> | **实现框架** | DeepSpeed（微软） | PyTorch 原生 |
> | **对应阶段** | Stage 3 | ≈ Stage 3 |
> | **切分粒度** | 按参数 flatten 后均分 | 按 module 切分（更灵活） |
> | **API 风格** | 配置文件驱动 | Python API，wrap model |
> | **生态集成** | HuggingFace Trainer 支持 | PyTorch 原生，Lightning 支持 |
> | **混合并行** | 与 DeepSpeed 的 PP/TP 集成 | 与 PyTorch 的 DeviceMesh 集成 |
>
> 核心算法思想一致，选择主要取决于技术栈偏好。DeepSpeed 的配置更简单（JSON 文件），FSDP 的灵活性更高（可以控制哪些 module 切分）。

## 实际使用：DeepSpeed 配置

ZeRO 最常见的使用方式是通过微软的 DeepSpeed 框架。

> [!example] DeepSpeed ZeRO Stage 2 配置
> ```json
> {
>   "zero_optimization": {
>     "stage": 2,
>     "allgather_partitions": true,
>     "allgather_bucket_size": 2e8,
>     "overlap_comm": true,
>     "reduce_scatter": true,
>     "reduce_bucket_size": 2e8,
>     "contiguous_gradients": true
>   },
>   "fp16": {
>     "enabled": true
>   }
> }
> ```
>
> 关键参数：
> - `stage`：1/2/3，对应三个阶段
> - `overlap_comm`：通信与计算重叠，减少等待时间
> - `offload_optimizer`：启用 ZeRO-Offload（Stage 2/3 可用）

> [!interview] 面试视角
>
> **Q: ZeRO 的三个 Stage 分别切分什么？显存节省多少？**
> A: Stage 1 切分 optimizer states（$4\Phi + 12\Phi/N$），Stage 2 额外切分 gradients（$2\Phi + 14\Phi/N$），Stage 3 连 parameters 也切分（$16\Phi/N$）。Stage 1/2 通信量和 DDP 相同（$2\Phi$），Stage 3 增加到约 $3\Phi$。
>
> **Q: 为什么 Stage 1/2 不增加通信量？**
> A: DDP 的 AllReduce 本质是 Reduce-Scatter + AllGather。ZeRO 只是把这两步拆开使用——Reduce-Scatter 后每卡只保留自己的 $1/N$，optimizer step 后再 AllGather 恢复完整参数。总传输量不变。
>
> **Q: ZeRO 和 FSDP 是什么关系？**
> A: FSDP 是 PyTorch 对 ZeRO Stage 3 的原生实现，核心算法一致。区别在于切分粒度（FSDP 按 module，ZeRO 按 flatten tensor）和 API 风格（FSDP 是 Python API，DeepSpeed 是 JSON 配置）。
>
> **Q: 实际训练中怎么选择 Stage？**
> A: Stage 1 最稳妥，通信不增加且实现简单，适合大多数场景（如 [[Kimi K2 (2025)]] 使用 ZeRO-1）。Stage 2 在显存紧张时使用。Stage 3 / FSDP 在模型单卡放不下时使用，但需要注意通信开销。

## 局限性

> [!warning] ZeRO 的边界
>
> 1. **仍然是数据并行**：ZeRO 不切分计算图，每张卡仍然执行完整的 forward/backward。当单层的 activation 就超过显存时，ZeRO 无能为力——需要 [[Tensor Parallelism]]
> 2. **Stage 3 通信开销**：每层 forward/backward 都需要 AllGather，对网络带宽要求高。跨节点使用 Stage 3 时性能下降明显
> 3. **与模型并行的互补关系**：实际大模型训练通常组合使用——节点内用 [[Tensor Parallelism]]（高带宽 NVLink），节点间用 ZeRO Stage 1 + [[Pipeline Parallelism]]（低带宽容忍）
> 4. **Activation Memory 不在 ZeRO 范围内**：ZeRO 只处理 model states（参数、梯度、optimizer states），activation 的显存需要通过 [[Activation Checkpointing]] 等技术单独处理

> [!paper] 论文出处
> Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020
> - 提出 ZeRO 三阶段切分框架
> - 在 DeepSpeed 中实现，支持训练 100B+ 参数模型

## 延伸阅读

**后续发展**：
- [[FSDP]] — PyTorch 原生的 ZeRO Stage 3 实现（待创建）
- [[Tensor Parallelism]] — 切分计算图而非 states，与 ZeRO 互补（待创建）
- [[Pipeline Parallelism]] — 按层切分模型，与 ZeRO 互补（待创建）
- [[Activation Checkpointing]] — 解决 ZeRO 不覆盖的 activation 显存问题（待创建）
