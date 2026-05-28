---
description: 将高精度权重/激活值映射到低精度表示，以减少内存占用和加速推理
type: method
aliases:
  - 量化
  - 模型量化
prerequisites:
  - "[[Transformer]]"
  - "[[KV Cache]]"
tags:
  - inference
  - optimization
  - compression
created: 2026-01-28
updated: 2026-02-01T01:09
---

# Quantization

量化（Quantization）是将模型的高精度浮点数（如 FP32/FP16）映射到低精度表示（如 INT8/INT4）的技术。核心目标是在可接受的精度损失下，大幅减少内存占用和计算开销，使大模型能在资源受限的环境中部署。

---

## 动机

> [!intuition] 为什么需要量化？
> LLM 的参数量巨大（7B-70B+），以 FP16 存储一个 70B 模型需要 140GB 显存。量化到 INT4 可以将显存需求降至 35GB，使单卡部署成为可能。同时，低精度计算在现代硬件上通常更快。

### 量化的收益

| 精度 | 每参数字节 | 70B 模型大小 | 相对 FP16 |
|------|-----------|-------------|-----------|
| FP32 | 4 bytes | 280 GB | 2x |
| FP16/BF16 | 2 bytes | 140 GB | 1x |
| INT8 | 1 byte | 70 GB | 0.5x |
| INT4 | 0.5 bytes | 35 GB | 0.25x |

量化是 LLM 推理优化的核心技术之一，通常与其他优化手段配合使用：[[Flash Attention]] 优化注意力计算效率，[[Paged Attention]] 优化 KV Cache 内存管理，[[Speculative Decoding]] 加速自回归生成。这些技术相互正交，可以叠加使用。

---

## 核心概念

### 量化的数学形式

> [!math] 线性量化
> 将浮点数 $x$ 映射到整数 $x_q$：
>
> $$x_q = \text{round}\left(\frac{x}{s}\right) + z$$
>
> 反量化（dequantization）：
>
> $$\hat{x} = s \cdot (x_q - z)$$
>
> 其中：
> - $s$（scale）：缩放因子，决定量化精度
> - $z$（zero-point）：零点偏移，用于非对称量化

### 对称量化 vs 非对称量化

| 类型 | Zero-point | 适用场景 | 特点 |
|------|------------|----------|------|
| 对称量化 | $z = 0$ | 权重（近似对称分布） | 计算简单，无需存储 $z$ |
| 非对称量化 | $z \neq 0$ | 激活值（常有偏移） | 更精确，但需额外存储 |

### 量化粒度

量化粒度决定了 scale 和 zero-point 的共享范围：

| 粒度 | 描述 | 精度 | 开销 |
|------|------|------|------|
| Per-tensor | 整个张量共享一组参数 | 低 | 最小 |
| Per-channel | 每个输出通道一组参数 | 中 | 中等 |
| Per-group | 每 $g$ 个元素一组参数 | 高 | 较大 |
| Per-token | 每个 token 一组参数 | 最高 | 最大 |

> [!intuition] 粒度权衡
> 粒度越细，量化误差越小，但需要存储更多的 scale/zero-point，且计算更复杂。实践中 per-group（如 group size=128）是常见的平衡点。

---

## 量化类型分类

### PTQ vs QAT

| 类型 | 全称 | 描述 | 优缺点 |
|------|------|------|--------|
| **PTQ** | Post-Training Quantization | 训练后直接量化 | 快速、无需训练数据，但精度可能下降 |
| **QAT** | Quantization-Aware Training | 训练时模拟量化 | 精度更高，但需要训练资源 |

> [!comparison] PTQ vs QAT
> - **PTQ**：适合快速部署，INT8 通常无明显精度损失，INT4 需要更精细的方法（如 GPTQ、AWQ）
> - **QAT**：适合追求极致精度，如 [[QLoRA]] 在微调时引入量化（结合了量化与 [[LoRA]]）

### Weight-only vs Weight-Activation

| 类型 | 量化对象 | 典型精度 | 适用场景 |
|------|----------|----------|----------|
| **Weight-only** | 仅权重 | W4A16, W8A16 | 内存受限，batch size 小 |
| **Weight-Activation** | 权重 + 激活 | W8A8, W4A4 | 追求计算加速 |

> [!intuition] 为什么 Weight-only 更常见？
> 1. 权重是静态的，可以离线精细量化
> 2. 激活值动态变化，且存在 outlier 问题
> 3. LLM 推理通常是 memory-bound，减少权重内存比加速计算更重要

---

## 精度格式

### 整数格式

| 格式 | 范围 | 典型应用 |
|------|------|----------|
| INT8 | [-128, 127] | 通用量化，精度损失小 |
| INT4 | [-8, 7] | 激进量化，需要特殊方法 |
| UINT4 | [0, 15] | 非对称量化 |

### 浮点格式

| 格式 | 指数位 | 尾数位 | 特点 |
|------|--------|--------|------|
| FP16 | 5 | 10 | 标准半精度 |
| BF16 | 8 | 7 | 更大动态范围，训练常用 |
| FP8 (E4M3) | 4 | 3 | 推理优化，H100 原生支持 |
| FP8 (E5M2) | 5 | 2 | 更大范围，适合梯度 |

### NF4 (NormalFloat 4-bit)

> [!definition] NF4
> [[QLoRA]] 提出的 4-bit 格式，专为正态分布的权重设计。将 [-1, 1] 范围按正态分布的分位数划分为 16 个 bin，使量化误差在统计意义上最小化。

---

## 主流量化方法

### [[GPTQ]]

> [!definition] GPTQ
> 基于 Hessian 信息的 weight-only PTQ 方法，通过最小化量化后的输出误差来确定最优量化参数。

**核心思想**：
1. 逐层量化，最小化该层输出的重建误差
2. 使用 Hessian 矩阵的近似来衡量每个权重的重要性
3. 按重要性顺序量化，并用剩余权重补偿误差

**优点**：
- 无需训练，仅需少量校准数据（~128 samples）
- 支持 4-bit 甚至 3-bit 量化
- 广泛支持（Hugging Face、vLLM、llama.cpp）

### [[AWQ]] (Activation-aware Weight Quantization)

> [!definition] AWQ
> 观察到少数"重要"权重通道对输出影响巨大，通过保护这些通道来减少量化误差。

**核心思想**：
1. 用激活值的统计信息识别重要权重通道
2. 对重要通道使用更大的 scale（等效于更高精度）
3. 通过 per-channel scaling 实现，无需混合精度

> [!comparison] GPTQ vs AWQ
> | 方面 | GPTQ | AWQ |
> |------|------|-----|
> | 核心思想 | 最小化重建误差 | 保护重要通道 |
> | 校准数据 | 需要 | 需要 |
> | 速度 | 较慢（逐列量化） | 较快 |
> | 精度 | 略高 | 接近 |
> | 硬件友好 | 一般 | 更好（无混合精度） |

### [[SmoothQuant]]

> [!definition] SmoothQuant
> 解决激活值量化的 outlier 问题，通过数学等价变换将激活的难度"转移"到权重上。

**问题**：激活值存在 outlier（极端值），直接量化会导致大部分值被压缩到很小的范围。

**解决方案**：
$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}$$

通过 per-channel 的缩放因子 $s$，将激活的动态范围转移到权重上。权重是静态的，可以离线处理。

### GGUF / llama.cpp 量化

llama.cpp 提供了一系列实用的量化格式：

| 格式 | 描述 | 典型用途 |
|------|------|----------|
| Q4_0 | 4-bit，per-block 量化 | 基础 4-bit |
| Q4_K_M | 4-bit，K-quant，中等质量 | 推荐的 4-bit |
| Q5_K_M | 5-bit，K-quant，中等质量 | 质量-大小平衡 |
| Q8_0 | 8-bit，per-block 量化 | 高质量 |

> [!intuition] K-quant 的含义
> K-quant 使用不同的量化策略处理不同层：对敏感层（如 attention）使用更高精度，对不敏感层使用更低精度。

---

## 量化的挑战

### 1. Outlier 问题

> [!warning] Outlier 问题
> LLM 的激活值中存在少量极端值（outlier），可能比正常值大 100 倍以上。这些 outlier 会"霸占"量化范围，导致正常值的精度严重下降。

**表现**：
- 某些通道的激活值异常大
- 通常出现在特定的 attention head 或 FFN 通道
- 随着模型规模增大，outlier 问题更严重

**解决方案**：
- [[SmoothQuant]]：将 outlier 转移到权重
- Mixed-precision：对 outlier 通道保持高精度
- Clipping：截断极端值（会损失信息）

### 2. 层敏感度差异

不同层对量化的敏感度不同：

| 层类型 | 敏感度 | 建议 |
|--------|--------|------|
| Embedding | 高 | 保持 FP16 或 INT8 |
| Attention QKV | 中-高 | 谨慎量化 |
| Attention Output | 中 | 可以量化 |
| FFN | 低-中 | 可以激进量化 |
| LM Head | 高 | 保持 FP16 或 INT8 |

> [!intuition] 为什么 Embedding 和 LM Head 敏感？
> 这两层直接与 vocabulary 交互，量化误差会直接影响 token 的选择概率。

### 3. 精度-效率权衡

| 精度 | 内存节省 | 速度提升 | 精度损失 |
|------|----------|----------|----------|
| INT8 | 2x | 1.5-2x | 通常可忽略 |
| INT4 | 4x | 2-3x | 需要精细方法 |
| INT3/INT2 | 5-8x | 理论更高 | 显著，需要特殊处理 |

---

## KV Cache 量化

除了模型权重，[[KV Cache]] 也可以量化以减少推理时的内存占用：

> [!intuition] 为什么量化 KV Cache？
> 长序列推理时，KV Cache 可能占用比模型参数更多的显存。量化 KV Cache 可以支持更长的上下文或更大的 batch size。

**常见方案**：
- **KV Cache INT8**：通常精度损失可接受
- **KV Cache INT4**：需要更精细的方法，如 per-head 量化
- **FP8 KV Cache**：H100 等新硬件原生支持

**注意**：KV Cache 量化与权重量化是独立的，可以组合使用（如 W4A16 + KV8）。

---

## 实践建议

> [!example] 量化选择指南
>
> **场景 1：单卡部署 7B 模型**
> - 推荐：GPTQ/AWQ 4-bit 或 GGUF Q4_K_M
> - 精度损失：通常 <1% perplexity 增加
>
> **场景 2：单卡部署 70B 模型**
> - 推荐：GPTQ/AWQ 4-bit（需要 24GB+ 显存）
> - 或 GGUF Q4_K_M + CPU offload
>
> **场景 3：追求最高精度**
> - 推荐：INT8 量化或 FP8（如果硬件支持）
> - 考虑 QAT 如果有微调需求
>
> **场景 4：边缘设备部署**
> - 推荐：GGUF 格式 + llama.cpp
> - 可以尝试更激进的量化（Q3、Q2）

---

## 面试要点

> [!interview] 面试要点
> **Q: 什么是量化？为什么 LLM 需要量化？**
> A: 量化是将高精度浮点数映射到低精度表示的技术。LLM 参数量巨大，量化可以减少内存占用（如 4-bit 量化减少 4 倍），使大模型能在消费级硬件上部署。
>
> **Q: PTQ 和 QAT 的区别？**
> A: PTQ 是训练后量化，快速但可能损失精度；QAT 在训练时模拟量化，精度更高但需要训练资源。LLM 通常用 PTQ，因为重新训练成本太高。
>
> **Q: 为什么 Weight-only 量化比 Weight-Activation 更常见？**
> A: 1) 权重静态，可离线精细量化；2) 激活有 outlier 问题；3) LLM 推理是 memory-bound，减少权重内存比加速计算更重要。
>
> **Q: 什么是 outlier 问题？如何解决？**
> A: 激活值中存在极端值，会压缩正常值的量化精度。解决方案包括 SmoothQuant（转移到权重）、混合精度（outlier 通道保持高精度）、clipping（截断）。
>
> **Q: GPTQ 和 AWQ 的核心区别？**
> A: GPTQ 基于 Hessian 最小化重建误差，AWQ 通过保护重要权重通道减少误差。AWQ 更快且硬件友好，GPTQ 精度略高。

---

## 延伸阅读

**导航**：
- [[MOC - Inference]] — 推理优化导航
