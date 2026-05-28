---
type: paper
description: 小米 309B MoE 模型，通过 Hybrid SWA 架构、MOPD 多教师蒸馏和 MTP 推理加速，以 15B 激活参数达到 DeepSeek-V3 级别性能
aliases:
  - MiMo-V2-Flash
prerequisites:
  - "[[Mixture of Experts]]"
  - "[[Sliding Window Attention]]"
  - "[[Multi-Token Prediction]]"
  - "[[MOPD]]"
tags:
  - technical-report
  - moe
  - efficient-inference
  - agentic
created: 2026-02-04
updated: 2026-02-04T18:57
---

# MiMo-V2-Flash (2025)

MiMo-V2-Flash 是小米发布的 [[Mixture of Experts|MoE]] 模型，总参数 309B、激活参数仅 15B，专注于高效推理和 agentic 能力。其核心创新在于三个方面：Hybrid [[Sliding Window Attention|SWA]] 架构实现长上下文建模、[[MOPD]] 范式解决 post-training 能力失衡、[[Multi-Token Prediction|MTP]] 实现 2.6x 推理加速。尽管参数量仅为 DeepSeek-V3 的 1/2、Kimi-K2 的 1/3，MiMo-V2-Flash 在多数 benchmark 上达到可比性能。

> [!paper] 论文信息
> - **标题**: MiMo-V2-Flash Technical Report
> - **机构**: Xiaomi
> - **发布**: 2025
> - **开源**: 模型权重 + 3 层 MTP 权重

## 核心贡献

### 1. Hybrid Sliding Window Attention 架构

MiMo-V2-Flash 采用 SWA 与 Global Attention (GA) 交替的混合架构，以 5:1 的比例堆叠（每 5 层 SWA 后接 1 层 GA），共 48 层（39 SWA + 9 GA）。

**关键设计选择**：

| 配置 | 值 |
|------|-----|
| 滑动窗口大小 | 128 tokens |
| SWA:GA 比例 | 5:1 |
| SWA 层数 / GA 层数 | 39 / 9 |
| GQA 配置 (SWA) | 64 Q heads / 8 KV heads |
| GQA 配置 (GA) | 64 Q heads / 4 KV heads |

> [!intuition] 为什么 128 token 窗口 + 5:1 比例？
> 实验表明，配合 [[Attention Sink|learnable attention sink bias]]，小窗口 (W=128) 反而优于大窗口 (W=512)。Attention sink 让模型在不需要关注任何 token 时可以"丢弃"注意力，解决了 SWA 的信息瓶颈问题。

**实验验证**（32B dense 模型）：

| 配置 | MMLU | BBH | GSM8K | MATH |
|------|------|-----|-------|------|
| All GA (baseline) | 57.3 | 54.7 | 34.2 | 9.5 |
| Hybrid SWA (W=128, w/o sink) | 54.9 | 52.4 | 36.9 | 8.9 |
| Hybrid SWA (W=128, w/ sink) | **58.3** | **56.1** | 36.9 | **10.3** |
| Hybrid SWA (W=512, w/ sink) | 58.3 | 54.9 | **37.9** | 10.0 |

在长上下文任务上，W=128 + sink 的配置在 GSM-Infinite、NoLiMa 等 benchmark 上也优于 W=512。

### 2. Lightweight Multi-Token Prediction

MiMo-V2-Flash 的 MTP 模块设计为轻量级，用于 [[Speculative Decoding]] 加速推理。

**设计特点**：
- 使用 dense FFN（非 MoE）减少参数量
- 使用 SWA（非 GA）减少 KV cache 和计算开销
- 每个 MTP block 仅 0.33B 参数
- 预训练时使用 1 层 MTP，post-training 时复制为 3 层

**推理加速效果**：

| Acceptance Length | 2.8 | 3.0 | 3.2 | 3.4 | 3.6 |
|-------------------|-----|-----|-----|-----|-----|
| Speedup (batch=64) | 1.97x | 2.11x | 2.25x | 2.39x | 2.53x |

> [!intuition] MTP 的双重价值
> 1. **推理加速**：作为 draft model 用于 speculative decoding，低熵任务（如代码生成）可达 3.6 acceptance length
> 2. **RL 训练加速**：缓解 rollout 阶段的 GPU 空闲问题，使小 batch on-policy 训练更实用

### 3. MOPD Post-Training 范式

MiMo-V2-Flash 采用 [[MOPD|Multi-Teacher On-Policy Distillation]] 进行 post-training，这是一个三阶段框架：

**Stage 1: SFT** - 建立基础指令遵循能力

**Stage 2: Domain-Specialized Training** - 训练多个领域专家教师：
- Agentic: search、coding、general tool use
- Non-agentic: math reasoning、general reasoning、safety

**Stage 3: Multi-Teacher Distillation** - 学生模型从自身分布采样，接收教师的 token-level KL 奖励

> [!math] MOPD 损失函数
> $$\mathcal{L}_{\text{MOPD}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, y \sim \mu_{\theta}(\cdot|x)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} w_t \hat{A}_{\text{MOPD},t} \log \pi_{\theta}(y_t|x, y_{< t}) \right]$$
>
> 其中 $\hat{A}_{\text{MOPD},t} = \log\frac{\pi_{\text{domain}_x}(y_t|x,y_{< t})}{\pi_{\theta}(y_t|x,y_{< t})}$ 是 token-level 的 KL advantage。

**MOPD 的优势**：
- 避免 capability imbalance（"跷跷板效应"）
- On-policy 采样避免 distribution mismatch
- 支持 teacher-student 迭代共演化

### 4. RL 基础设施创新

**[[Rollout Routing Replay|R3]]**：记录推理时的 MoE 路由分布，训练时重放，解决 [[Training-Inference Mismatch]] 问题。

## 模型配置

| 配置 | 值 |
|------|-----|
| 总参数量 | 309B |
| 激活参数量 | 15B |
| 层数 | 48 (39 SWA + 9 GA) |
| Hidden dim | 4096 |
| MoE 专家数 | 256 total, 8 activated |
| 预训练 tokens | 27T |
| 原生上下文长度 | 32K (扩展至 256K) |

## 性能亮点

**与同级模型对比**：

| Benchmark | MiMo-V2-Flash | DeepSeek-V3.2 | Kimi-K2 |
|-----------|---------------|---------------|---------|
| 激活参数 | 15B | 37B | 32B |
| 总参数 | 309B | 671B | 1043B |
| AIME 2025 | 94.1 | - | - |
| SWE-Bench Verified | 73.4 | - | - |
| LiveCodeBench | 83.2 | - | - |

**Agentic 能力**：
- SWE-Bench Verified: 73.4%（超越所有开源模型）
- SWE-Bench Multilingual: 71.7%
- BrowseComp: 45.4 (with context management: 58.3)

## 局限性

> [!warning] 已知局限
> - **知识容量受限**：SimpleQA 等知识密集型任务表现较弱，受限于参数量
> - **与闭源模型差距**：与 GPT-5 等顶级闭源模型仍有差距
> - **架构探索初步**：Hybrid SWA 的设计权衡分析尚不充分

## 延伸阅读

**相关 Technical Reports**：
- [[DeepSeek-V3 (2024)]] - 另一个高效 MoE 模型
- [[Kimi K2 (2025)]] - 更大规模的 MoE 模型

**核心技术详解**：
- [[MOPD]] - 多教师在线蒸馏的完整方法论
- [[Multi-Token Prediction]] - MTP 训练目标和推理加速
- [[Rollout Routing Replay]] - MoE RL 训练稳定性技术
