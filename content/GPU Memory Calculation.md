---
type: concept
description: 训练和推理时 GPU 显存占用的系统性计算方法，涵盖参数、梯度、优化器状态、激活值和 KV cache 的显存分析
aliases:
  - GPU 显存计算
  - 模型显存估算
  - Memory Footprint
prerequisites:
  - "[[Transformer]]"
  - "[[Adam]]"
  - "[[Mixed Precision Training]]"
tags:
  - foundations
  - distributed-training
created: 2026-03-01
updated: 2026-03-01T23:17
---

# GPU Memory Calculation

GPU 显存计算（GPU Memory Calculation）是估算模型训练和推理时显存占用的系统性方法。准确估算显存需求是选择硬件配置、设计分布式策略、应用优化技巧的前提——显存不够，模型跑不起来；显存浪费，成本白白增加。

核心问题：**一个 $M$ 参数的模型，训练/推理时需要多少显存？**

---

## 动机：为什么需要精确计算显存

> [!intuition] 显存是 LLM 训练的第一瓶颈
>
> 训练一个 7B 模型，FP32 精度下：
> - 参数本身：7B × 4 bytes = 28GB
> - 梯度：28GB
> - Adam 优化器状态（m, v）：56GB
> - 激活值（batch_size=8, seq_len=2048）：~100GB+
>
> **总计 >200GB**——单张 A100 (80GB) 根本装不下。
>
> 不精确估算显存会导致：
> - **OOM (Out of Memory)**：训练到一半崩溃
> - **资源浪费**：买了 8×A100 但只用到 20% 显存
> - **优化盲目**：不知道瓶颈在哪，乱用优化技巧

精确计算显存的价值：
1. **硬件选型**：需要几张卡？A100 还是 H100？
2. **优化决策**：是用 [[Mixed Precision Training]] 还是 [[Activation Checkpointing]]？
3. **Batch Size 调优**：在显存限制下最大化吞吐量

---

## 精度与字节数

不同数值精度占用的字节数：

| 精度 | 字节数 | 典型用途 |
|------|--------|----------|
| **FP32** | 4 bytes | 传统训练、Master Weights |
| **FP16** | 2 bytes | 混合精度训练（计算） |
| **BF16** | 2 bytes | 混合精度训练（主流） |
| **FP8** | 1 byte | 前沿研究（DeepSeek-V3） |
| **INT8** | 1 byte | 量化推理 |

> [!warning] 精度不是全局统一的
> 混合精度训练中：
> - 参数存储：FP32（Master Weights）
> - 前向/反向计算：FP16/BF16
> - 梯度：FP16/BF16
> - 优化器状态：FP32
>
> 显存计算时需要分别考虑每个组件的精度。

---

## 训练时显存组成

训练时显存分为四大部分：

$$
\text{Total Memory} = \text{Model States} + \text{Activations} + \text{Temp Buffers}
$$

### Model States（模型状态）

Model States 包括：

| 组件 | 大小 | 说明 |
|------|------|------|
| **Parameters** | $M \times b_p$ | 模型参数 |
| **Gradients** | $M \times b_g$ | 每个参数的梯度 |
| **Optimizer States** | $M \times b_o$ | 优化器维护的额外状态 |

其中：
- $M$：模型参数量
- $b_p$：参数精度的字节数
- $b_g$：梯度精度的字节数
- $b_o$：优化器状态的字节数（取决于优化器类型）

#### 优化器状态的显存占用

不同优化器的状态大小：

| 优化器 | 状态 | 显存占用 |
|--------|------|----------|
| **SGD** | 无 | 0 |
| **SGD + Momentum** | $m$ | $M \times 4$ bytes (FP32) |
| **Adam / AdamW** | $m, v$ | $M \times 8$ bytes (FP32) |

[[Adam]] 和 [[AdamW]] 需要存储：
- $m_t$：一阶矩估计（梯度的指数移动平均）
- $v_t$：二阶矩估计（梯度平方的指数移动平均）

两者都是 FP32 精度，每个参数额外占用 8 bytes。

> [!example] FP32 训练的 Model States
> 对于 $M$ 参数的模型，使用 Adam 优化器：
> - Parameters: $M \times 4$ bytes
> - Gradients: $M \times 4$ bytes
> - Optimizer States: $M \times 8$ bytes
> - **总计**: $M \times 16$ bytes
>
> 7B 模型：7B × 16 = **112GB**

> [!example] 混合精度训练的 Model States
> FP16 计算 + FP32 Master Weights：
> - Parameters (FP32 master): $M \times 4$ bytes
> - Parameters (FP16 copy): $M \times 2$ bytes
> - Gradients (FP16): $M \times 2$ bytes
> - Optimizer States (FP32): $M \times 8$ bytes
> - **总计**: $M \times 16$ bytes
>
> 混合精度训练的 Model States 显存与 FP32 相同！节省主要来自激活值。

### Activations（激活值）

激活值是前向传播中每一层的输出，需要保存用于反向传播计算梯度。

**激活值显存 ≈ 层数 × batch_size × seq_len × hidden_dim × 字节数**

更精确的公式（对于 Transformer）：

$$
\text{Activations} \approx L \times b \times s \times h \times (10 \sim 12) \times \text{bytes}
$$

其中：
- $L$：层数
- $b$：batch size
- $s$：序列长度
- $h$：hidden dimension
- 系数 10~12：每层有多个激活值（Attention、FFN、LayerNorm 等）

> [!intuition] 为什么激活值这么大？
>
> 参数是固定的（7B 就是 7B），但激活值随 batch size 和 seq_len 线性增长：
> - batch_size 翻倍 → 激活值翻倍
> - seq_len 翻倍 → 激活值翻倍
>
> 这就是为什么大 batch / 长序列训练容易 OOM。

> [!example] 7B 模型的激活值
> 假设：
> - 层数 $L = 32$
> - batch_size $b = 8$
> - seq_len $s = 2048$
> - hidden_dim $h = 4096$
> - 精度：FP16 (2 bytes)
>
> $$
> \text{Activations} \approx 32 \times 8 \times 2048 \times 4096 \times 12 \times 2 \text{ bytes} \approx 50 \text{GB}
> $$

### Temporary Buffers（临时缓冲区）

训练过程中的临时显存：
- **Gradient All-Reduce 缓冲区**（[[Data Parallelism]]）
- **通信缓冲区**（分布式训练）
- **中间计算结果**（如 Softmax 的临时张量）

通常占总显存的 5-10%，实际使用中可忽略或按经验值估算。

---

## 推理时显存组成

推理时不需要梯度和优化器状态，显存组成简化为：

$$
\text{Inference Memory} = \text{Parameters} + \text{KV Cache} + \text{Activations}
$$

### Parameters（参数）

推理时参数通常使用低精度：

| 精度 | 显存占用 | 说明 |
|------|----------|------|
| FP16/BF16 | $M \times 2$ bytes | 标准推理 |
| INT8 | $M \times 1$ byte | 量化推理 |
| INT4 | $M \times 0.5$ bytes | 极致压缩（GPTQ, AWQ） |

> [!example] 7B 模型参数显存
> - FP16: 7B × 2 = **14GB**
> - INT8: 7B × 1 = **7GB**
> - INT4: 7B × 0.5 = **3.5GB**

### KV Cache

[[Transformer]] 自回归生成时，为避免重复计算，会缓存每一层的 Key 和 Value：

$$
\text{KV Cache} = 2 \times L \times b \times s \times h \times \text{bytes}
$$

其中：
- 系数 2：Key 和 Value 各一份
- $L$：层数
- $b$：batch size
- $s$：序列长度（生成的 token 数）
- $h$：hidden dimension（或 KV head dimension）

> [!warning] KV Cache 随生成长度线性增长
> 生成 1024 tokens 和 4096 tokens，KV Cache 相差 4 倍。
>
> 这就是为什么长文本生成容易 OOM，也是 [[Multi-Query Attention]] 和 [[Grouped-Query Attention]] 的动机——减少 KV Cache。

> [!example] 7B 模型的 KV Cache
> 假设：
> - 层数 $L = 32$
> - batch_size $b = 1$
> - 生成长度 $s = 2048$
> - hidden_dim $h = 4096$
> - 精度：FP16 (2 bytes)
>
> $$
> \text{KV Cache} = 2 \times 32 \times 1 \times 2048 \times 4096 \times 2 \text{ bytes} \approx 1 \text{GB}
> $$
>
> 如果 batch_size = 16，KV Cache 增加到 **16GB**。

### Activations（激活值）

推理时只需要当前 token 的激活值，显存占用远小于训练：

$$
\text{Activations} \approx L \times b \times h \times 10 \times \text{bytes}
$$

（不依赖序列长度，因为是逐 token 生成）

通常可忽略，主要瓶颈是 Parameters 和 KV Cache。

---

## 实战案例：7B 模型显存估算

### 训练场景

**配置**：
- 模型：7B 参数
- 优化器：AdamW
- 精度：BF16 混合精度
- Batch size：8
- Seq len：2048
- 架构：32 层，hidden_dim = 4096

**显存计算**：

| 组件 | 计算 | 显存 |
|------|------|------|
| Model States | 7B × 16 bytes | 112 GB |
| Activations | 32 × 8 × 2048 × 4096 × 12 × 2 | ~50 GB |
| **总计** | | **~162 GB** |

**结论**：
- 单张 A100 (80GB) **装不下**
- 需要 2×A100 或使用优化技巧

### 优化后的训练

应用 [[ZeRO]] Stage 2 + [[Activation Checkpointing]]：

| 优化 | 效果 |
|------|------|
| ZeRO Stage 2 (4 卡) | Model States: 112GB / 4 = 28GB/卡 |
| Activation Checkpointing | Activations: 50GB → 10GB |
| **单卡显存** | **38GB** |

**结论**：4×A100 可以训练，每卡显存占用 <50%。

### 推理场景

**配置**：
- 模型：7B 参数
- 精度：FP16
- Batch size：1
- 生成长度：2048

**显存计算**：

| 组件 | 计算 | 显存 |
|------|------|------|
| Parameters | 7B × 2 bytes | 14 GB |
| KV Cache | 2 × 32 × 1 × 2048 × 4096 × 2 | ~1 GB |
| Activations | 忽略 | <1 GB |
| **总计** | | **~16 GB** |

**结论**：单张 RTX 4090 (24GB) 可以推理。

---

## 显存优化技巧的影响

| 技巧 | 影响组件 | 效果 |
|------|----------|------|
| [[Mixed Precision Training]] | Activations | 减半（FP32 → FP16） |
| [[Activation Checkpointing]] | Activations | 减少 70-80% |
| [[ZeRO]] Stage 1 | Optimizer States | 切分到 $N$ 卡 |
| [[ZeRO]] Stage 2 | Optimizer States + Gradients | 切分到 $N$ 卡 |
| [[ZeRO]] Stage 3 | 全部 Model States | 切分到 $N$ 卡 |
| [[Gradient Accumulation]] | Activations | 减少（小 batch） |
| [[Flash Attention]] | Activations | 减少 Attention 显存 |

> [!intuition] 优化策略的选择
>
> - **Model States 是瓶颈** → 用 [[ZeRO]]
> - **Activations 是瓶颈** → 用 [[Activation Checkpointing]] 或减小 batch size
> - **两者都是瓶颈** → 组合使用 + 增加 GPU 数量

---

## 速查表

### 训练显存公式

**FP32 训练（Adam）**：
$$
\text{Memory} = M \times 16 + L \times b \times s \times h \times 40
$$

**混合精度训练（Adam）**：
$$
\text{Memory} = M \times 16 + L \times b \times s \times h \times 24
$$

### 推理显存公式

**FP16 推理**：
$$
\text{Memory} = M \times 2 + 2 \times L \times b \times s \times h \times 2
$$

### 常见模型的显存需求

| 模型 | 参数量 | 训练（混合精度） | 推理（FP16） |
|------|--------|------------------|--------------|
| 1B | 1B | ~20 GB | ~2 GB |
| 7B | 7B | ~160 GB | ~16 GB |
| 13B | 13B | ~300 GB | ~28 GB |
| 70B | 70B | ~1.5 TB | ~150 GB |

（假设 batch_size=8, seq_len=2048 训练；batch_size=1, seq_len=2048 推理）

---

## 局限性

> [!warning] 实际显存可能更高
>
> 1. **框架开销**：PyTorch/CUDA 本身占用 1-2GB
> 2. **碎片化**：显存分配不连续导致浪费
> 3. **动态图开销**：PyTorch 动态图比静态图多占 10-20%
> 4. **通信缓冲区**：分布式训练的临时缓冲
>
> 实际使用中，预留 10-20% 的显存余量。

> [!warning] 公式是近似值
>
> - Activation 的系数（10~12）是经验值，不同架构有差异
> - 忽略了 LayerNorm、Dropout 等小组件
> - MoE 模型的计算更复杂（只激活部分专家）
>
> 精确显存需要 profiling 工具（如 `torch.cuda.memory_summary()`）。

---

## 延伸阅读

**显存优化技巧**：
- [[Mixed Precision Training]] — 用 FP16/BF16 减少激活值显存
- [[Activation Checkpointing]] — 用计算换显存
- [[ZeRO]] — 切分 Model States 到多卡
- [[Gradient Accumulation]] — 用小 batch 减少激活值

**分布式训练**：
- [[Data Parallelism]] — 理解 Model States 的冗余
- [[Pipeline Parallelism]] — 按层切分模型
- [[Tensor Parallelism]] — 按张量切分模型

**推理优化**：
- [[Multi-Query Attention]] — 减少 KV Cache
- [[Grouped-Query Attention]] — MQA 和 MHA 的折中
- [[Flash Attention]] — 减少 Attention 显存和计算

**工具**：
- `torch.cuda.memory_summary()` — PyTorch 显存分析
- `nvidia-smi` — GPU 显存监控
- DeepSpeed Memory Estimator — 训练显存估算工具

