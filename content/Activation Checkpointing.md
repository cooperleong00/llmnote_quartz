---
type: concept
description: 用计算换显存的训练优化技巧，只保存部分层的激活值，反向传播时重新计算被丢弃的部分
aliases:
  - 激活检查点
  - Gradient Checkpointing
  - 梯度检查点
  - Rematerialization
prerequisites:
  - "[[Data Parallelism]]"
tags:
  - distributed-training
created: 2026-02-24
updated: 2026-02-24
---

# Activation Checkpointing

Activation Checkpointing（激活检查点，也称 Gradient Checkpointing）是一种**用计算换显存**的训练优化技巧：前向传播时只保存少数"检查点"层的激活值，丢弃其余中间激活；反向传播需要这些激活时，从最近的检查点重新 forward 计算出来。代价是约 33% 的额外计算，换来的是激活显存从 $O(L)$ 降到 $O(\sqrt{L})$。

## 动机：激活值是大模型训练的显存瓶颈

训练时 GPU 显存主要被三类数据占据：

1. **Model states**：参数 + 梯度 + optimizer states（[[ZeRO]] 专门解决这部分）
2. **Activation memory**：前向传播保存的中间结果，反向传播计算梯度时需要
3. **临时缓冲区**：通信 buffer、workspace 等

对于大模型，激活显存与 $\text{seq\_len} \times \text{hidden\_size} \times \text{num\_layers}$ 成正比。以一个 70B 模型为例，单条长序列的激活值可以轻松占满数十 GB 显存。ZeRO 能切分 model states，但**不处理 activation memory**——这正是 Activation Checkpointing 的用武之地。

> [!intuition] 为什么不能直接丢掉所有激活？
> 反向传播计算梯度时，链式法则要求知道每一层的输入激活值。如果全部丢掉，就无法计算梯度了。Checkpointing 的巧妙之处在于：**不需要保存所有层的激活，只要保存足够多的"锚点"，就能从锚点重新算出任意层的激活**。

## 核心机制

### 标准反向传播 vs Checkpointing

**标准做法**：前向传播时保存每一层的输入激活，反向传播直接使用。

$$\text{显存} = O(L) \quad \text{（L 为层数）}$$

**Checkpointing**：每隔 $K$ 层保存一个 checkpoint，其余层的激活在前向传播后丢弃。反向传播到某一层时，从最近的 checkpoint 重新 forward 到该层，恢复所需的激活值。

$$\text{显存} = O(L/K + K) \quad \text{（checkpoint 本身 + 两个 checkpoint 之间的临时激活）}$$

当 $K = \sqrt{L}$ 时取最优：

$$\text{显存} = O(\sqrt{L})$$

> [!math] 计算开销分析
> 标准训练中，每层做 1 次 forward + 1 次 backward。使用 checkpointing 后，被丢弃的层在 backward 阶段需要额外做 1 次 forward（从 checkpoint 重算）。最坏情况下，forward 计算量翻倍，但由于 backward 本身的计算量约为 forward 的 2 倍，总计算量从 $3F$ 变为 $4F$，即**增加约 33%**。

### 具体流程

以 4 层网络、每 2 层设一个 checkpoint 为例：

```
Forward（保存 checkpoint）:
  Layer 1 → [保存 ✓] → Layer 2 → [丢弃 ✗]
  Layer 3 → [保存 ✓] → Layer 4 → [丢弃 ✗]

Backward（需要时重算）:
  反向到 Layer 4 → 从 Layer 3 的 checkpoint 重新 forward 到 Layer 4，恢复激活
  反向到 Layer 2 → 从 Layer 1 的 checkpoint 重新 forward 到 Layer 2，恢复激活
```

## Checkpoint 粒度选择

不同粒度在显存节省和计算开销之间有不同的 trade-off：

### Per-layer Checkpointing（最常用）

每个 Transformer 层的边界设一个 checkpoint。这是最常见的做法，实现简单，显存节省显著。在 Megatron-LM、DeepSpeed 等框架中是默认选项。

### Selective Checkpointing

并非所有操作的激活值占用相同显存。Attention 的中间激活（$O(\text{seq\_len}^2)$）远大于 MLP。Selective checkpointing 只对显存占用大的操作（如 self-attention）做 checkpoint，保留计算便宜的操作的激活。

> [!intuition] 选择性的智慧
> 如果一个操作的激活值很大但重算很快（如 attention score），checkpoint 它很划算；如果激活值很小但重算很慢，保留它更好。Selective checkpointing 就是在这个 trade-off 上做更细粒度的优化。

### Full vs Partial Checkpointing

- **Full checkpointing**：所有层都参与 checkpoint 策略，显存节省最大
- **Partial checkpointing**：只对部分层做 checkpoint（如前 N/2 层），在显存和速度之间折中

## 与其他显存优化技术的配合

Activation Checkpointing 解决的是 activation memory，而大模型训练中的显存压力来自多个方面。实际训练通常组合多种技术：

**与 [[ZeRO]] 配合**：ZeRO 切分 model states（参数、梯度、optimizer states），Checkpointing 减少 activation memory。两者解决的是显存的不同组成部分，天然互补。DeepSpeed 中两者经常一起使用。

**与 [[Tensor Parallelism]] 配合**：TP 将每层的计算切分到多卡，每卡只持有部分 activation。Megatron-LM 的 Sequence Parallelism 进一步沿 sequence 维度切分 LayerNorm/Dropout 的 activation，与 checkpointing 叠加可以大幅降低单卡激活显存。

**与 [[Mixed Precision Training]] 配合**：使用 FP16/BF16 存储激活值可以直接将 activation memory 减半。Checkpointing 在此基础上进一步减少需要保存的激活数量，两者效果叠加。

> [!comparison] 各技术解决的显存组成
> | 技术 | 解决的显存部分 |
> |------|---------------|
> | [[ZeRO]] | Model states（参数、梯度、optimizer states） |
> | Activation Checkpointing | Activation memory |
> | [[Mixed Precision Training]] | 所有部分（通过降低精度） |
> | [[Tensor Parallelism]] + SP | Activation memory（通过切分到多卡） |

## PyTorch 实现

PyTorch 提供了 `torch.utils.checkpoint.checkpoint` 接口：

> [!example] 基本用法
> ```python
> from torch.utils.checkpoint import checkpoint
>
> class TransformerBlock(nn.Module):
>     def forward(self, x):
>         # 将 self-attention 包装为 checkpoint
>         x = checkpoint(self.attention, x, use_reentrant=False)
>         x = checkpoint(self.mlp, x, use_reentrant=False)
>         return x
> ```
>
> `use_reentrant=False` 是推荐的新接口，支持更多场景（如非确定性操作）且行为更可预测。

在 Hugging Face Transformers 中，可以通过 `model.gradient_checkpointing_enable()` 一键开启。

## 局限性

> [!warning] 需要注意的边界
> - **计算开销不可忽视**：33% 的额外计算在大规模训练中意味着显著的时间和成本增加
> - **不适合计算瓶颈场景**：如果训练已经是计算瓶颈（而非显存瓶颈），checkpointing 只会让训练更慢
> - **与某些操作不兼容**：有副作用的操作（如 in-place 操作、全局状态修改）在重算时可能产生不同结果
> - **调试困难**：重算路径中的错误可能难以定位，因为同一段 forward 代码被执行了两次

> [!interview] 面试视角
> **Q: Activation Checkpointing 的核心 trade-off 是什么？**
> A: 用约 33% 的额外计算换取 $O(\sqrt{L})$ 的激活显存，本质是时间换空间。
>
> **Q: 为什么 ZeRO 不能替代 Activation Checkpointing？**
> A: ZeRO 只切分 model states（参数、梯度、optimizer states），不处理 activation memory。两者解决的是显存的不同组成部分。
>
> **Q: 实际训练中 checkpointing 的计算开销真的是 33% 吗？**
> A: 33% 是理论上界（假设所有层都 checkpoint 且 backward 计算量是 forward 的 2 倍）。实际中通过 selective checkpointing、与通信重叠等优化，开销通常更低。

## 延伸阅读

**原始论文**：
- Chen et al., "Training Deep Nets with Sublinear Memory Cost", 2016 — 提出 $O(\sqrt{N})$ 的 checkpointing 策略

**工程实现**：
- [[Pipeline Parallelism]] — 也涉及激活值的跨设备管理，与 checkpointing 有交互
- [[Flash Attention]] — 通过 IO-aware 算法减少 attention 的显存占用，与 selective checkpointing 互补
