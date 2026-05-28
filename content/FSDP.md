---
type: method
description: PyTorch 原生的全分片数据并行，将 ZeRO 的参数/梯度/优化器分片策略集成到 PyTorch 分布式框架中
aliases:
  - 全分片数据并行
  - Fully Sharded Data Parallelism
  - PyTorch FSDP
prerequisites:
  - "[[ZeRO]]"
  - "[[Data Parallelism]]"
tags:
  - distributed-training
created: 2026-02-24
updated: 2026-02-24T14:36
---

# FSDP

FSDP（Fully Sharded Data Parallel）是 PyTorch 对 [[ZeRO]] 思想的原生实现。它将模型参数、梯度和优化器状态分片到多张 GPU 上，使得单卡只需持有 $1/N$ 的 model states，从而在 [[Data Parallelism]] 框架内突破显存瓶颈。与 DeepSpeed ZeRO 不同，FSDP 不需要引入额外框架——它就是 `torch.distributed` 的一部分。

## 动机

> [!intuition] 为什么 PyTorch 需要自己的 ZeRO？
>
> DeepSpeed ZeRO 证明了"在数据并行中切分 model states"是训练大模型的有效路径。但 DeepSpeed 是微软维护的独立框架，与 PyTorch 生态存在摩擦：
>
> - **依赖管理**：需要额外安装 DeepSpeed，版本兼容性问题频发
> - **调试困难**：DeepSpeed 的抽象层让 PyTorch 原生工具（profiler、hooks）难以直接使用
> - **扩展受限**：想要组合 ZeRO 与 PyTorch 的其他并行策略（如 DeviceMesh）需要额外适配
>
> PyTorch 社区需要一个**原生方案**，让分片数据并行像 DDP 一样成为标准工具。这就是 FSDP 的由来——2022 年在 PyTorch 1.11 中正式发布，基于 FairScale 的 `FullyShardedDataParallel` 重写。

## 核心机制

### Sharding Strategy：三种分片模式

FSDP 通过 `ShardingStrategy` 参数控制分片粒度，对应 [[ZeRO]] 的不同阶段：

| Strategy | 对应 ZeRO | 切分内容 | 显存节省 | 通信量 |
|----------|-----------|----------|----------|--------|
| `FULL_SHARD` | Stage 3 | 参数 + 梯度 + 优化器 | 最大 | 最高 |
| `SHARD_GRAD_OP` | Stage 2 | 梯度 + 优化器 | 中等 | 中等 |
| `NO_SHARD` | — | 不切分（≈ DDP） | 无 | 最低 |

实际使用中，`FULL_SHARD` 是默认也是最常用的策略。`SHARD_GRAD_OP` 适合显存压力不大但希望减少通信的场景——forward 期间参数不释放，省去一次 all-gather。

### Forward/Backward 中的通信

以 `FULL_SHARD` 为例，FSDP 在训练的每一步中执行以下流程：

```
Forward pass:
  对每个 FSDP unit:
    1. all-gather: 从所有 rank 收集完整参数
    2. 执行该 unit 的 forward 计算
    3. 释放非本地分片的参数（腾出显存）

Backward pass:
  对每个 FSDP unit（逆序）:
    1. all-gather: 再次收集完整参数（计算梯度需要）
    2. 执行该 unit 的 backward 计算
    3. reduce-scatter: 每个 rank 只保留自己负责的梯度分片
    4. 释放完整参数
```

> [!intuition] 核心 trade-off
>
> FSDP 用**通信换显存**：每个 FSDP unit 在计算前临时拼装完整参数，计算后立即释放。这意味着同一份参数在 forward 和 backward 中各被 all-gather 一次——通信量是 DDP 的 1.5 倍（DDP 只需一次 all-reduce），但显存降到 $1/N$。这和 [[ZeRO]] Stage 3 的 trade-off 完全一致。

### FSDP Unit：Wrapping 粒度

FSDP 的一个关键设计决策是**以什么粒度切分模型**。每个被 wrap 的 module 成为一个独立的 FSDP unit，拥有自己的分片参数和通信调度。

**粒度选择的影响**：

- **太粗**（整个模型一个 unit）：all-gather 一次性收集所有参数，峰值显存高，失去分片的意义
- **太细**（每个 Linear 层一个 unit）：通信次数过多，小消息的通信效率低
- **合适**（每个 Transformer layer 一个 unit）：平衡显存和通信效率

PyTorch 提供了自动 wrap 策略：

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 按 Transformer layer 自动 wrap
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerDecoderLayer},
)

model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
```

> [!warning] Wrapping 的常见陷阱
>
> - **Shared parameters**：如果两个 module 共享参数但被 wrap 到不同 FSDP unit，会导致参数被重复分片。需要确保共享参数在同一个 unit 内。
> - **Buffer 处理**：非参数的 buffer（如 BatchNorm 的 running_mean）默认不分片，需要注意显存占用。

## FSDP vs DeepSpeed ZeRO

> [!comparison] 两种 ZeRO 实现的选择
>
> | 维度 | DeepSpeed ZeRO | FSDP |
> |------|----------------|------|
> | **API 风格** | JSON 配置文件驱动，改配置不改代码 | Python API，需要 wrap model |
> | **学习曲线** | 配置简单，但出问题时黑盒难调 | 需要理解 wrapping，但可调试性强 |
> | **性能** | ZeRO-3 通信优化更成熟（prefetch、pin memory） | FSDP1 略逊，FSDP2 在追赶 |
> | **混合并行** | 与 DeepSpeed 自己的 PP/TP 集成 | 与 PyTorch DeviceMesh、DTensor 集成 |
> | **生态** | HuggingFace Trainer 深度集成 | HuggingFace Accelerate、PyTorch Lightning 支持 |
> | **Offload** | ZeRO-Offload/Infinity 成熟 | FSDP1 支持 CPU offload，FSDP2 改进中 |
>
> **选择建议**：如果团队已经深度使用 DeepSpeed 且运行稳定，没有迁移的必要。新项目或重度依赖 PyTorch 生态（compile、DTensor、自定义 hook）的场景，FSDP 是更自然的选择。

## FSDP2：下一代实现

PyTorch 从 2.x 开始推进 FSDP2（`torch.distributed._composable.fsdp`），是对 FSDP1 的重写：

- **Composable API**：不再需要 wrap 整个 model，可以对单个参数或 module 声明分片策略，与其他并行策略（TP、PP）自由组合
- **基于 DTensor**：分片参数用 `DTensor` 表示，与 PyTorch 的 tensor 抽象统一
- **Per-parameter sharding**：更细粒度的控制，不再受限于 FSDP unit 的 module 边界
- **更好的 `torch.compile` 兼容性**：FSDP1 的动态 wrap 与 compile 的静态图假设冲突，FSDP2 从设计上解决了这个问题

> [!warning] FSDP2 的状态
>
> 截至 2025 年底，FSDP2 已在 torchtitan 等项目中验证，Meta 内部大规模使用。但 API 仍标记为 experimental，社区生态（HuggingFace、Lightning）的适配还在进行中。新项目建议关注 FSDP2，但生产环境需评估稳定性。

## 实际使用注意事项

### Wrap Policy 选择

对 Transformer 模型，推荐按 layer 粒度 wrap：

```python
# 最常见的 wrap 方式
model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

如果模型结构不规则，可以用 `size_based_auto_wrap_policy` 按参数量阈值自动 wrap。

### 与 Mixed Precision 配合

FSDP 原生支持 [[Mixed Precision Training]]，通过 `MixedPrecision` 配置：

```python
from torch.distributed.fsdp import MixedPrecision

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,      # forward/backward 计算精度
    reduce_dtype=torch.float32,       # 梯度 reduce 精度（保持数值稳定）
    buffer_dtype=torch.bfloat16,
)

model = FSDP(model, mixed_precision=mp_policy)
```

> [!warning] reduce_dtype 的选择
>
> 梯度 reduce 建议用 FP32。如果用 BF16 做 reduce-scatter，大规模训练中梯度累积的精度损失会导致训练不稳定。这和 DDP 中 all-reduce 用 FP32 是同一个道理。

### 与 Activation Checkpointing 配合

大模型训练中，FSDP 通常与 [[Activation Checkpointing]] 联合使用——FSDP 解决 model states 的显存问题，activation checkpointing 解决 activation 的显存问题：

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

# 对每个 Transformer layer 启用 activation checkpointing
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda module: isinstance(module, TransformerDecoderLayer),
)
```

### Checkpoint 保存与加载

FSDP 的 checkpoint 需要特殊处理，因为参数是分片的：

- **分布式 checkpoint**（推荐）：每个 rank 保存自己的分片，加载时按需 reshard。使用 `torch.distributed.checkpoint` API。
- **全量 checkpoint**：先 `FSDP.full_state_dict()` 收集完整参数到 rank 0，再保存。简单但显存峰值高，大模型可能 OOM。

```python
# 推荐：分布式 checkpoint
from torch.distributed.checkpoint import save, load

save({"model": model.state_dict()}, checkpoint_id=path)
load({"model": model.state_dict()}, checkpoint_id=path)
```

## 延伸阅读

**原始论文与文档**：
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) — Meta 团队的 FSDP 设计与实践总结
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — 官方教程

**后续发展**：
- [[Tensor Parallelism]] — 与 FSDP 正交的另一种并行策略，FSDP2 支持两者组合
- [[Pipeline Parallelism]] — 大模型训练中常与 FSDP 组成 3D 并行
