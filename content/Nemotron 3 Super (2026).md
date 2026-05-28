---
type: paper
description: NVIDIA 提出 120B/12B-active 的 Hybrid Mamba-Attention MoE 模型，通过 LatentMoE 架构优化 accuracy per parameter，结合 MTP 和 NVFP4 预训练实现 2.2-7.5x 推理吞吐提升
aliases:
  - Nemotron 3 Super
  - Nemotron Super
prerequisites:
  - "[[Mixture of Experts]]"
  - "[[Speculative Decoding]]"
tags:
  - architecture
  - inference-efficiency
  - post-training
  - agentic
created: 2026-03-13
updated: 2026-03-13T23:53
---

# Nemotron 3 Super (2026)

Nemotron 3 Super 是 NVIDIA 2026 年发布的开源 LLM，核心目标是在保持前沿精度的同时大幅提升推理效率。它将三个正交的效率方向——[[Mixture of Experts|稀疏 MoE]]、Hybrid Mamba-Attention、低精度预训练——统一到一个 120B total / 12B active 的架构中，在 8k-in/64k-out 场景下比 GPT-OSS-120B 吞吐高 2.2x，比 Qwen3.5-122B 高 7.5x。

> [!paper] 论文信息
> NVIDIA, 2026. *Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning*
> 开源：base / post-trained / quantized checkpoints + 训练数据

## 架构概览

88 层 hybrid stack，三大支柱：

| 组件 | 作用 | 关键参数 |
|------|------|----------|
| **LatentMoE** | 稀疏扩展，优化 accuracy/byte | 512 experts, top-22, latent dim 1024 |
| **Hybrid Mamba-Attention** | 线性复杂度序列建模 + 全局锚点 | Mamba-2 为主，少量 GQA attention |
| **MTP** | 推理加速（native speculative decoding） | 2 层 shared-weight heads |


> [!example] 模型规格速查
> - **Total params**: 120.6B / **Active**: 12.7B (12.1B excl. embeddings)
> - **Layers**: 88 / **Model dim**: 4096 / **Head dim**: 128
> - **Attention**: GQA, 32 Q-heads, 2 KV-heads
> - **Mamba**: state dim 128, 8 groups, 128 heads, head dim 64
> - **Expert FFN dim**: 2688 / **Shared expert**: 5376
> - **Pretraining**: 25T tokens (Phase 1: 20T 多样性, Phase 2: 5T 高质量)
> - **Context**: up to 1M tokens
> - **Precision**: NVFP4 (E2M1, 16-element micro-blocks)

## LatentMoE：核心架构创新

### 动机：accuracy per parameter 被忽视了

传统 [[Mixture of Experts|MoE]] 设计主要优化 accuracy per FLOP（计算效率），但忽略了实际部署中的关键瓶颈：

- **低延迟场景**：推理被 expert 权重的内存带宽读取主导（每个 expert 矩阵 $d \times m$）
- **高吞吐场景**：分布式推理被 all-to-all routing 通信主导（通信量 $\propto d \times K$）

这意味着一个在 FLOP 上看起来高效的 MoE，在实际部署中可能因为内存带宽和通信开销而效率低下。

### 核心思想：在 latent space 做 expert 计算

> [!intuition] 直觉理解
> 标准 MoE 中，每个 expert 在完整的 hidden dimension $d$ 上操作。LatentMoE 的洞察是：expert 计算不需要完整的 $d$ 维信息。先把 token 投影到低维 latent space $\ell$，在那里做 routing 和 expert 计算，最后再投影回来。省下来的带宽预算用来增加 expert 数量和 top-K，反而提升了精度。

**具体流程**：

$$x \in \mathbb{R}^d \xrightarrow{W_\downarrow} z \in \mathbb{R}^\ell \xrightarrow{\text{route + experts}} z' \in \mathbb{R}^\ell \xrightarrow{W_\uparrow} y \in \mathbb{R}^d$$


### 设计原则

论文从 hardware-software co-design 视角提炼了五条原则：

1. 低延迟推理被 expert 权重读取主导 → 需要减小 $d$ 或 $m$
2. 高吞吐推理被 all-to-all routing 主导 → 需要减小 $d$ 或 $K$
3. 保持模型质量需要保持 nonlinear budget $K \cdot m$ → $K$ 和 $m$ 不能动
4. 存在 effective feature rank $r_{\text{eff}}$ 下界 → $d$ 不能无限缩小
5. 增加 expert 总数 $N$ 和 top-$K$ 能指数级扩展 expert 组合空间

**关键推导**：原则 1-3 指向 $d$ 是最有前途的缩减轴。将 $d$ 缩减 $\alpha$ 倍（投影到 latent space），同时将 $N$ 和 $K$ 增加 $\alpha$ 倍，在近似恒定的推理成本下获得更高精度。

在 Nemotron 3 Super 中：$d = 4096$，$\ell = 1024$（$\alpha = 4$），因此 expert 数从标准设计的 ~128 扩展到 512，top-K 从 ~5-6 扩展到 22。

> [!warning] 保留在全维度的计算
> 并非所有计算都在 latent space 进行。Routing gate、shared expert、非 expert 层仍在完整 $d$ 维度操作——因为它们不是带宽瓶颈，降维反而会损害质量。

## Hybrid Mamba-Attention

### 为什么不全用 Attention？

标准 Transformer 的 KV cache 随序列长度二次增长，是长上下文推理的主要系统瓶颈。Mamba-2 使用固定大小的 state，生成时内存开销恒定，延迟显著降低。

### 混合策略

88 层采用周期性交错模式：

- **主体**：Mamba-2 blocks + LatentMoE layers（高效线性时间序列建模）
- **锚点**：少量 self-attention layers 作为 global anchors（全 token 交互，长距离信息路由）

这种设计让大部分计算由高效的 Mamba 和 sparse MoE 承担，同时通过 attention anchors 保持全局依赖建模能力，支持 1M token 上下文。


## Multi-Token Prediction (MTP)

MTP 让模型在每个位置预测多个未来 token，带来两个好处：

1. **训练质量**：鼓励 representation 捕获多步依赖和长程结构，降低 validation loss
2. **推理加速**：辅助预测头作为内置 draft model，实现 native [[Speculative Decoding|speculative decoding]]，无需外部 draft model

### Shared-Weight Head 设计

标准 MTP 用 $N$ 个独立 head 分别预测固定 offset（$n+2, \ldots, n+N+1$），限制了 draft 长度且存在 train-inference mismatch：head 训练时看到 ground-truth hidden states，推理时却要 condition on 自己生成的 states。

Nemotron 3 Super 的解决方案：**跨 MTP head 共享参数**，让统一的 prediction head 暴露于多个 offset。这种 shared-weight 设计：
- 正则化 head 跨预测 horizon 的行为
- 提升对 self-generated hidden states 的鲁棒性
- 允许递归应用同一 head 生成更长 draft

> [!example] Speculative Decoding 性能
> 在 SPEED-Bench 上（draft length = 7），Nemotron 3 Super 平均 acceptance length 3.45，超过 DeepSeek-R1 (2.70) 和 Qwen3-Next (3.33)。在 Blackwell 硬件上，MTP draft depth 从 1 增加到 3 显著改善了 throughput-latency Pareto frontier。

## NVFP4 预训练

Nemotron 3 Super 是首个在 FP4 精度下完成全部 25T token 预训练的大规模模型。

**精度分配策略**（并非所有层都用 FP4）：

| 层类型 | 精度 | 原因 |
|--------|------|------|
| 大部分 Linear Layers | NVFP4 | 主体计算 |
| 最后 15% 网络 | BF16 | 训练稳定性 |
| Latent Projections | BF16 | 对 step-time 影响可忽略 |
| MTP Layers | BF16 | 保持多 token 预测能力 |
| QKV & Attention | BF16 | 保持少量 attention 层的精度 |
| Mamba Output Projection | MXFP8 | 避免 NVFP4 下的 underflow |

> [!warning] FP4 训练的已知现象
> 训练过程中观察到 expert 层出现 channel magnitude patterns：FC1 输出通道和 FC2 输入通道的 norm 趋向零，导致约 7% 参数的 weight gradient 为零。分析表明这是 NVFP4 量化将本已很小的梯度 underflow 为零的结果，BF16 训练在更长 horizon 后也会出现类似模式。尝试在 annealing 前切换到 MXFP8 未带来下游任务提升，最终全程使用 NVFP4。


## Post-Training：大规模 Agentic RL

Post-training 流程强调 agentic 能力，包含 [[SFT]] + 多阶段 RL：

### SFT：两阶段 Loss 设计

单阶段 SFT 会导致 long-input-short-output 场景退化，因此采用两阶段：

**Stage 1 — Token-level average**：标准的全局 token 平均 loss，诱导强推理行为：

$$\mathcal{L}_{\text{tok}} = \frac{\sum_{c \in \mathcal{B}} \sum_{t \in \mathcal{O}_c} \ell_t}{\sum_{c \in \mathcal{B}} |\mathcal{O}_c|}$$

**Stage 2 — Sample-level average**：切换到 per-conversation 归一化，防止长输出主导 loss：

$$\mathcal{L}_{\text{samp}} = \frac{1}{|\mathcal{B}|} \sum_{c \in \mathcal{B}} \left( \frac{1}{|\mathcal{O}_c|} \sum_{t \in \mathcal{O}_c} \ell_t \right)$$

> [!intuition] 为什么需要两阶段？
> Stage 1 让模型学会推理（长输出获得更多梯度信号），Stage 2 恢复对短输出任务的表现（每个 conversation 权重相等，不被长输出淹没）。

SFT 数据大幅扩展了 agentic 数据集，包括 Software Engineering（从 SWE-Gym/R2E-Gym 蒸馏）、Agentic CLI Programming（~15k 合成任务 + 3k SWE 任务 + 10k web 开发任务）、多种 CLI 环境（Claude Code, OpenCode, Codex）的交互轨迹蒸馏。

### RL：四阶段流水线

| 阶段 | 内容 | 特点 |
|------|------|------|
| **Stage 1: RLVR** | 21 个环境、37 个数据集的多环境 RL | 统一训练防止单任务退化 |
| **Stage 2: SWE-RL** | 端到端软件工程任务 | 长 horizon rollout，独立阶段 |
| **Stage 3: RLHF** | Principle-following GenRM | 改善指令遵循和交互质量 |
| **Stage 4: MTP Healing** | 冻结主模型，训练 MTP heads | 恢复 RL 后的 speculative decoding 性能 |

**算法**：异步 [[GRPO]]，训练和推理解耦到不同 GPU。推理 worker 持续生成 trajectory，训练 worker 收集 batch 后更新。支持 in-flight weight updates（不等待进行中的 rollout 完成就推送新权重），单个 trajectory 可能包含不同 model version 生成的 token。通过 importance sampling ratio masking 稳定训练。

**PivotRL**（Agentic RL 的关键创新）：长 horizon agentic 任务面临 SFT（便宜但 OOD 退化）vs end-to-end RL（准确但昂贵）的矛盾。PivotRL 复用离线 SFT expert trajectory，聚焦于 policy 有不确定性的 "pivot" turns 进行 RL，用 domain-appropriate reward 匹配 policy action 到 expert action，大幅提升 agentic RL 效率。

> [!example] RL 基础设施规模
> - 1000+ GPU 异步训练
> - NeMo Gym + NeMo RL 框架
> - SWE-RL 使用 Apptainer 容器隔离（无 root 权限下替代 Docker）
> - 内存 watchdog 防止 agent 进程 OOM
> - 命令 blocklist 防止 agent 执行危险命令（如 `killall`）


## 评估亮点

> [!comparison] 与同级模型对比（post-trained）
>
> | 维度 | Nemotron 3 Super | GPT-OSS-120B | Qwen3.5-122B |
> |------|-----------------|--------------|--------------|
> | AIME25 | 90.21 | 92.50 | 90.36 |
> | HMMT Feb25 (tools) | 94.73 | - | 89.55 |
> | SWE-Bench (OpenHands) | 60.47 | 41.9 | 66.40 |
> | RULER 1M | 91.75 | 22.30 | 91.33 |
> | 推理吞吐 (8k/64k) | **baseline** | 0.45x | 0.13x |
>
> 精度上与 GPT-OSS-120B 和 Qwen3.5-122B 竞争力相当，但推理吞吐显著领先。长上下文（RULER 1M: 91.75）和 agentic 能力（SWE-Bench Multilingual: 45.78 vs GPT-OSS 30.80）是突出优势。

## 局限性

> [!warning] 边界与不足
> - **精度 vs 效率 tradeoff**：在部分 benchmark 上落后于 Qwen3.5-122B（如 GPQA 79.23 vs 86.60, TauBench Telecom 64.36 vs 95.00），效率优势以部分精度为代价
> - **NVFP4 训练的 channel attenuation**：7% 参数梯度归零，长期影响尚不完全清楚
> - **Mamba 的局限**：虽然推理高效，但 Mamba 在需要精确 token-level attention 的任务上可能不如纯 Transformer
> - **LatentMoE 的 latent rank 下界**：$\ell$ 不能无限缩小，存在 task-specific $r_{\text{eff}}$ 约束

## 延伸阅读

**独立技术贡献**（各有单独的 technical report）：
- [[LatentMoE]] — 完整的 hardware-aware MoE 设计分析（Elango et al., 2026）
- [[Mamba]] — SSM 架构基础

**相关模型**：
- [[Nemotron 3 Nano]] — 同系列小模型，共享 hybrid Mamba-Attention 架构
- [[DeepSeek-V3]] — 另一个大规模 MoE 模型，使用 MTP 但架构不同

**Post-training 相关**：
- [[GRPO]] — RL 阶段使用的核心算法
- [[RLVR]] — 多环境可验证奖励 RL
- [[PivotRL]] — 长 horizon agentic RL 的效率方法
