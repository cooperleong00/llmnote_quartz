---
type: concept
description: 在训练中混合使用不同精度的浮点格式，用 FP16/BF16 加速计算、用 FP32 保持数值稳定，是现代 LLM 训练的标准配置
aliases:
  - 混合精度训练
  - AMP
  - Automatic Mixed Precision
prerequisites:
  - "[[Data Parallelism]]"
tags:
  - efficiency
  - training
  - optimization
created: 2026-02-24
updated: 2026-02-24
---

# Mixed Precision Training

混合精度训练（Mixed Precision Training）的核心思想很直接：**用低精度做计算加速，用高精度保数值稳定**。纯 FP32 训练太慢太耗显存，纯 FP16 训练会数值崩溃——混合精度在两者之间找到了平衡点，是现代 LLM 训练的标准配置。

---

## 动机：为什么不能只用一种精度？

> [!intuition] 精度的两难
> FP32 训练一个 70B 模型需要 280GB 显存光存参数，计算也慢。FP16 把显存砍半、计算翻倍——但 FP16 的数值范围太小，梯度经常 underflow 变成 0，训练直接崩。
>
> 混合精度的洞察是：**不是所有计算都需要高精度**。矩阵乘法占了 90%+ 的计算量，用低精度做完全够；但梯度累加、参数更新这些对精度敏感的操作，必须用高精度保护。

这正是 [[Data Parallelism]] 显存分析中 $16\Phi$ 的来源——混合精度训练下，每个参数需要同时维护 FP16 训练副本和 FP32 主副本，加上 optimizer states，总共 $16\Phi$ bytes。

---

## 数值格式对比

理解混合精度训练，首先要理解不同浮点格式的 trade-off。

IEEE 浮点数由三部分组成：**符号位（sign）+ 指数位（exponent）+ 尾数位（mantissa）**。指数位决定**动态范围**（能表示多大/多小的数），尾数位决定**精度**（相邻两个可表示数之间的间距）。

> [!math] 浮点格式对比
>
> | 格式 | 总位数 | 指数位 | 尾数位 | 动态范围 | 精度 | 典型用途 |
> |------|--------|--------|--------|----------|------|----------|
> | FP32 | 32 | 8 | 23 | $\sim 10^{\pm 38}$ | 高 | Master weights, optimizer states |
> | FP16 | 16 | 5 | 10 | $\sim 10^{\pm 5}$ | 中 | 早期混合精度训练 |
> | BF16 | 16 | 8 | 7 | $\sim 10^{\pm 38}$ | 低 | 现代 LLM 训练主流 |
> | FP8 E4M3 | 8 | 4 | 3 | $\sim 10^{\pm 2}$ | 极低 | Forward/activation 存储 |
> | FP8 E5M2 | 8 | 5 | 2 | $\sim 10^{\pm 4}$ | 极低 | 梯度存储 |

### FP16 vs BF16：为什么 BF16 成为主流

FP16 和 BF16 都是 16-bit，但设计哲学完全不同：

- **FP16**（5 位指数 + 10 位尾数）：精度高，但动态范围只有 $\sim 10^{\pm 5}$。LLM 训练中梯度值经常小于 $10^{-5}$，直接 underflow 为 0。必须配合 Loss Scaling 才能用。
- **BF16**（8 位指数 + 7 位尾数）：精度低一些，但动态范围和 FP32 一样大（$\sim 10^{\pm 38}$）。梯度再小也不会 underflow，**不需要 Loss Scaling**，训练更稳定。

> [!warning] BF16 的精度代价
> BF16 只有 7 位尾数（vs FP16 的 10 位），意味着相邻可表示数之间的间距更大。对于需要高精度的操作（如 softmax、layer norm），BF16 可能引入可观的舍入误差。这就是为什么即使用 BF16 训练，某些操作仍需保持 FP32。

### FP8：下一代训练精度

FP8 有两种变体，分别优化不同场景：

- **E4M3**（4 位指数 + 3 位尾数）：精度相对更高，适合 forward pass 的激活值和权重
- **E5M2**（5 位指数 + 2 位尾数）：范围更大，适合 backward pass 的梯度

传统做法是 forward 用 E4M3、backward 用 E5M2。但 [[DeepSeek-V3 (2024)]] 的实践表明，配合 fine-grained quantization，**全部使用 E4M3** 也能保持稳定——因为 tile/block-wise scaling 有效扩展了动态范围。

与推理时的 [[Quantization]] 不同，FP8 训练需要在训练过程中持续保持数值稳定，对 scaling 策略的要求更高。

---

## 核心策略：三大支柱

经典的混合精度训练（Micikevicius et al., 2018）建立在三个支柱上：

### 1. FP32 Master Weights

> [!intuition] 为什么需要两份权重？
> 低精度格式的最小可表示间距可能大于学习率 × 梯度。比如 FP16 在数值 1.0 附近的精度约为 $10^{-3}$，如果 `lr × gradient = 1e-5`，这个更新直接被舍入为 0——权重永远不会更新。
>
> 解决方案：在 FP32 中维护一份"主权重"（master weights），所有参数更新在 FP32 上进行，然后将更新后的权重转换为低精度用于下一轮 forward/backward。

训练循环变成：
1. 将 FP32 master weights 转换为 FP16/BF16
2. 用低精度做 forward + backward，得到低精度梯度
3. 将梯度转换为 FP32
4. 在 FP32 上更新 master weights（optimizer step）
5. 回到步骤 1

### 2. Forward/Backward 用低精度

所有 GEMM（General Matrix Multiplication）操作用 FP16/BF16 执行，这是计算加速的主要来源。现代 GPU 的 Tensor Cores 对低精度矩阵乘法有专门的硬件加速：

| 精度 | A100 TFLOPS | H100 TFLOPS |
|------|-------------|-------------|
| FP32 | 19.5 | 67 |
| FP16/BF16 | 312 | 990 |
| FP8 | — | 1,979 |

低精度计算的吞吐量是 FP32 的 **16-30 倍**。

但并非所有操作都适合低精度。以下操作通常保持 FP32：
- **Softmax**：涉及指数运算，对精度极其敏感
- **Layer Normalization**：涉及方差计算，低精度会累积误差
- **Loss 计算**：直接影响梯度方向
- **Embedding / Output head**：直接与 vocabulary 交互

### 3. Loss Scaling（FP16 专用）

> [!math] Loss Scaling 的原理
> FP16 的最小正规数约为 $6 \times 10^{-8}$，但训练中的梯度值经常在 $10^{-8}$ 到 $10^{-5}$ 之间——大量梯度 underflow 为 0。
>
> Loss Scaling 的做法：在 backward 之前将 loss 乘以一个大的 scaling factor $S$（如 1024），这样所有梯度也被放大 $S$ 倍，避免 underflow。在 optimizer step 之前再除以 $S$ 恢复原始值。
>
> $$\text{scaled\_loss} = S \cdot \mathcal{L}$$
> $$\text{scaled\_gradients} = \nabla_\theta (S \cdot \mathcal{L}) = S \cdot \nabla_\theta \mathcal{L}$$
> $$\text{true\_gradients} = \frac{\text{scaled\_gradients}}{S}$$

**Dynamic Loss Scaling**：固定的 $S$ 不够灵活——太大会 overflow，太小仍然 underflow。动态策略从一个大的 $S$ 开始，如果检测到 overflow（出现 `inf`/`NaN`），就将 $S$ 减半并跳过这个 step；如果连续多个 step 没有 overflow，就将 $S$ 翻倍。

> [!warning] BF16 不需要 Loss Scaling
> BF16 的动态范围和 FP32 一样大，梯度不会 underflow。这是 BF16 在 LLM 训练中取代 FP16 的关键原因之一——少了 Loss Scaling 这个不稳定因素，训练更简单可靠。

---

## AMP（Automatic Mixed Precision）

PyTorch 的 AMP 将上述策略封装为两个核心 API：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # 管理 dynamic loss scaling

for data, target in dataloader:
    optimizer.zero_grad()

    # autocast 自动选择每个操作的精度
    with autocast(dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)

    # GradScaler 处理 loss scaling（BF16 时可省略）
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

`autocast` 的核心逻辑是维护一个**白名单/黑名单**：
- **白名单**（用低精度）：`torch.mm`, `torch.matmul`, `conv2d` 等 GEMM 操作
- **黑名单**（保持 FP32）：`softmax`, `layer_norm`, `cross_entropy`, `log`, `exp`
- **灰名单**（跟随输入精度）：element-wise 操作如 `relu`, `dropout`

> [!intuition] AMP 的本质
> AMP 不是"自动把所有东西变成 FP16"，而是**自动决定每个操作用什么精度**。它的价值在于开发者不需要手动管理每个操作的精度——框架根据数值稳定性需求自动选择。

---

## 显存节省分析

混合精度训练的显存影响需要分组件分析。以 Adam optimizer 训练一个参数量为 $\Phi$ 的模型为例：

> [!math] 显存占用对比
>
> | 组件 | 纯 FP32 | Mixed Precision (FP16/BF16) |
> |------|---------|----------------------------|
> | 模型参数 | $4\Phi$ | $2\Phi$（低精度）+ $4\Phi$（FP32 master） |
> | 梯度 | $4\Phi$ | $2\Phi$（低精度） |
> | Adam $m$ | $4\Phi$ | $4\Phi$（FP32） |
> | Adam $v$ | $4\Phi$ | $4\Phi$（FP32） |
> | **合计** | $16\Phi$ | $16\Phi$ |

等等——显存没省？对于 model states 确实如此。但混合精度的显存收益主要来自：

1. **Activation memory**：forward pass 的中间激活值用低精度存储，直接减半。对于长序列、大 batch 的训练，activation memory 往往远大于 model states
2. **通信量减半**：分布式训练中梯度同步的通信量从 $4\Phi$ 降到 $2\Phi$
3. **计算速度翻倍**：Tensor Core 的低精度吞吐量远高于 FP32

> [!comparison] 训练 vs 推理的精度优化
> - **训练**（Mixed Precision）：需要维护 FP32 master weights，显存节省有限，主要收益是**计算加速**
> - **推理**（[[Quantization]]）：不需要 optimizer states 和梯度，可以激进地量化到 INT4/INT8，主要收益是**显存节省**
>
> 两者的共同挑战是处理 activation outliers，但解决方案不同：训练用 loss scaling / fine-grained scaling，推理用 SmoothQuant / GPTQ 等。

---

## FP8 训练：最新进展

FP8 训练是混合精度的下一步演进，[[DeepSeek-V3 (2024)]] 首次在 671B 参数的超大规模模型上验证了其可行性，[[Ling 2.0 (2025)]] 进一步将其扩展到 1T 参数。

### 核心挑战

FP8 的动态范围极小（E4M3 只有 $\sim 10^{\pm 2}$），传统的 per-tensor scaling（整个 tensor 共享一个 scaling factor）无法处理 activation outliers——一个极端值会压缩其他所有值的精度。

### Fine-Grained Quantization

DeepSeek-V3 提出的解决方案是**更细粒度的 scaling**：

- **Activation**：per-tile scaling，每 $1 \times 128$ 个元素（每个 token 的 128 个 channel）共享一个 scaling factor
- **Weight**：per-block scaling，每 $128 \times 128$ 个元素共享一个 scaling factor

这种 tile/block-wise scaling 让每组元素有独立的动态范围，有效隔离了 outlier 的影响。

> [!intuition] 为什么 fine-grained scaling 有效？
> 想象一个 tensor 中 99% 的值在 $[-1, 1]$，但有一个 outlier 值为 1000。Per-tensor scaling 会把 scaling factor 设为 1000/max_fp8，导致 $[-1, 1]$ 范围内的值全部被压缩到极少的几个可表示数上。Per-tile scaling 让 outlier 所在的 tile 单独处理，其他 tile 不受影响。

### DeepSeek-V3 的 FP8 框架

DeepSeek-V3 的 FP8 混合精度框架有几个关键设计：

1. **GEMM 全部 FP8**：Forward（Fprop）、activation backward（Dgrad）、weight backward（Wgrad）三个 GEMM 都用 FP8，理论上计算速度翻倍
2. **全部使用 E4M3**：不像传统做法区分 E4M3/E5M2，因为 fine-grained scaling 已经扩展了动态范围
3. **Online quantization**：实时计算每个 tile/block 的 scaling factor，而非用历史最大值延迟估计
4. **高精度累加**：FP8 GEMM 的中间结果每 128 个元素提升到 CUDA Cores 做 FP32 累加，避免 Tensor Core 有限的累加精度（仅约 14 bits）
5. **敏感操作保持高精度**：embedding、output head、MoE gating、normalization、attention 仍用 BF16/FP32

> [!math] DeepSeek-V3 的显存优化
> 在 FP8 框架基础上进一步压缩：
> - **Optimizer states**：Adam 的 $m$ 和 $v$ 用 BF16 而非 FP32（从 $8\Phi$ 降到 $4\Phi$）
> - **Activation cache**：backward 需要的激活值用 FP8 存储
> - **通信**：MoE dispatch 的激活值用 FP8 传输
>
> 最终结果：相比 BF16 baseline，loss 的相对误差始终低于 0.25%。

[[Flash Attention]] 的最新版本（FlashAttention-3）也开始支持 FP8，利用 H100 的 TMA（Tensor Memory Accelerator）实现 FP8 attention 计算，进一步提升吞吐。

---

## 局限性

> [!warning] 边界条件
> 1. **不是所有模型都能无损混合精度**：某些架构（如早期的 Transformer 变体）对精度更敏感，需要仔细调试哪些层保持高精度
> 2. **FP8 训练仍不成熟**：目前只有少数团队（DeepSeek、Inclusion AI）在超大规模上验证过，小模型的收益不明显
> 3. **硬件依赖**：BF16 需要 Ampere+（A100），FP8 需要 Hopper+（H100）。老硬件只能用 FP16 + Loss Scaling
> 4. **Debugging 更难**：精度问题导致的 NaN/Inf 可能在训练后期才暴露，定位困难
> 5. **Master weights 的显存开销**：FP32 master weights 是必须的，这限制了 model states 的显存节省

> [!interview] 面试要点
> **Q: 混合精度训练为什么能加速但不损失精度？**
> A: 加速来自 Tensor Core 对低精度 GEMM 的硬件加速（16-30x throughput）。不损失精度是因为参数更新在 FP32 master weights 上进行，低精度只用于 forward/backward 的矩阵乘法——这些操作对精度不敏感。
>
> **Q: BF16 和 FP16 的区别？为什么 LLM 训练更倾向 BF16？**
> A: 同样 16-bit，BF16 用 8 位指数换来和 FP32 一样的动态范围，代价是精度低一些（7 位尾数 vs FP16 的 10 位）。LLM 训练中梯度值跨度极大，BF16 的大动态范围避免了 underflow，不需要 Loss Scaling，训练更稳定。
>
> **Q: $16\Phi$ 是怎么算出来的？**
> A: FP16 参数 $2\Phi$ + FP32 master weights $4\Phi$ + FP16 梯度 $2\Phi$ + FP32 Adam $m$ $4\Phi$ + FP32 Adam $v$ $4\Phi$ = $16\Phi$。这也是 [[ZeRO]] 优化的起点。

---

## 延伸阅读

**原始论文**：
- Micikevicius et al., "Mixed Precision Training" (ICLR 2018) — 奠基性工作，提出 master weights + loss scaling + FP16 compute 三大支柱

**FP8 训练的实践**：
- [[DeepSeek-V3 (2024)]] — 首次在 671B 模型上验证 FP8 训练，提出 fine-grained quantization
- [[Ling 2.0 (2025)]] — 最大规模的开源 FP8 训练模型（1T 参数），精度损失 <= 0.25%
