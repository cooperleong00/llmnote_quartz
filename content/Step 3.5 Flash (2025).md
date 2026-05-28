---
type: paper
description: StepFun 196B MoE 模型（11B 活跃参数），通过 MIS-PO 实现稳定的大规模 off-policy RL 训练，结合 3:1 Hybrid SWA 和 MTP-3 达到前沿级 agentic 智能
aliases:
  - Step 3.5 Flash
  - StepFun 3.5 Flash
tags:
  - technical-report
  - moe
  - post-training
  - agentic
created: 2026-02-12
updated: 2026-02-12T19:43
---

# Step 3.5 Flash (2025)

Step 3.5 Flash 是 StepFun 发布的 [[Mixture of Experts|MoE]] 模型，以 196B 总参数、11B 活跃参数的高效配置，在 agent、coding、math 任务上达到与 GPT-5.2 xHigh 和 Gemini 3.0 Pro 相当的前沿水平。核心创新包括：用于稳定大规模 off-policy RL 训练的 [[MIS-PO]] 算法、3:1 [[Sliding Window Attention|SWA]]/Full Attention 混合架构、以及 [[Multi-Token Prediction|MTP-3]] 推理加速。

## 核心贡献

### 1. 架构：高效的 MoE + Hybrid Attention

**MoE 配置**：
- 45 层 Transformer（3 dense + 42 MoE）
- 每层 288 routed experts + 1 shared expert
- Top-8 routing，11B 活跃参数 / 196B 总参数

**3:1 Hybrid Attention**：
- 三层 SWA（window=512）+ 一层 Full GQA-8 的交替布局
- SWA 层 query heads 从 64 增加到 96，弥补性能损失
- 采用 [[Gated Attention|Head-wise Gated Attention]] 替代 sink token，性能更优

> [!intuition] 为什么 3:1 比 1:1 更好？
> 1:1 布局（SWA/Full 交替）在长文本质量上最优，但 prefill/decode FLOPs 是 3:1 的 1.6 倍。3:1 配合增加的 query heads 能以极低的额外成本（~1%）接近 1:1 的质量。

**MTP-3**：
- 三个轻量级 MTP heads（各含 SWA + dense FFN），仅增加 0.81B 参数（0.41%）
- 训练时主要优化 MTP-1，后期初始化 MTP-2/3 并联合训练
- 支持 [[Speculative Decoding]]，在 Hopper GPU 上达到 ~170 tokens/s

### 2. MIS-PO：稳定的大规模 Off-Policy RL

[[MIS-PO]]（Metropolis Independence Sampling-Filtered Policy Optimization）是本文的核心方法创新，解决了大规模 off-policy RL 训练的稳定性问题。

> [!intuition] 核心思想
> 传统 [[Importance Sampling]] 通过 bounded ratio 缩放梯度，但方差很高。MIS-PO 改用**二值 mask** 过滤 off-distribution 样本，将保留的样本视为 on-policy，大幅降低梯度方差。

**双层过滤机制**：

1. **Token-level**：过滤概率比 $x_t = \pi_{\theta_{\text{old}}}(a_t|s_t) / \pi_{\theta_{\text{vllm}}}(a_t|s_t)$
2. **Trajectory-level**：过滤几何平均比 $\bar{\rho}(\tau) = (\prod_t x_t)^{1/T}$

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{\tau \sim \pi_{\theta_{\text{vllm}}}} \left[ \mathbb{I}(x_t) \cdot \mathbb{I}(\bar{\rho}(\tau)) \cdot \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t \right]$$

其中 $\mathbb{I}(x) = \mathbb{I}[\rho_{\min} \le x \le \rho_{\max}]$，token-level 边界为 [0.5, 2]，trajectory-level 边界为 [0.996, 1.001]。

**配套技术**：
- **Truncation-Aware Value Bootstrapping**：用 value estimate 替代截断轨迹的零奖励，避免惩罚长链推理
- **Routing Confidence 监控**：低 $\Sigma_k$（激活专家的平均概率质量）预示训练不稳定

### 3. Reward System

**三类奖励信号**：

| 类型 | 方法 | 应用场景 |
|------|------|----------|
| Verifiable | Rule-based / Model-based verifiers | Math, Code, Logic |
| Non-verifiable | GenRM（pairwise generative RM）| 开放式任务 |
| Agent | Entity-matching / Rubric-based judge | Search, Research |

**GenRM 设计**：
- 输出 confidence score，转换为 [[Bradley-Terry Model|Bradley-Terry]] win rate 作为奖励
- 内置 length penalty 抑制长度膨胀
- **MetaRM** 检测 spurious reasoning（正确偏好但推理有缺陷），降低其训练奖励

### 4. 训练稳定性技术

**Pre-training 挑战**：
- **Muon 优化器的数值敏感性**：需要精细的超参数调整
- **Expert Collapse**：不仅是 routing collapse，还包括专家权重退化
- **Localized Activation Blow-up**：MoE 层的局部激活爆炸

**解决方案**：
- EP-level load balancing loss 促进 rank 级别的均匀利用
- [[Loss-Free Load Balancing]] 实现全局 token 平衡

## 性能亮点

| Benchmark | Score | 对比 |
|-----------|-------|------|
| IMO-AnswerBench | 85.4% | 接近 GPT-5.2 xHigh (86.3%) |
| LiveCodeBench-v6 | 86.4% | 超越 Claude Opus 4.5 (84.8%) |
| $\tau^2$-Bench | 88.2% | Agent 任务 SOTA |
| BrowseComp (w. Ctx Manage) | 69.0% | 超越 GPT-5.2 xHigh (65.8%) |
| Terminal-Bench 2.0 | 51.0% | 开源 SOTA |

**Test-time Scaling**：采用 Parallel Coordinated Reasoning (PaCoRe) 范式，配置 [4,4,4,4] 多轮并行推理，进一步提升性能。

## 局限性

1. **Token Efficiency**：生成轨迹比 Gemini 3.0 Pro 更长才能达到相当质量
2. **Universal Mastery**：通用能力与领域专精的平衡仍需改进，计划通过 [[On-Policy Distillation]] 解决

## 与其他模型的对比

| 模型 | 总参数 | 活跃参数 | 特点 |
|------|--------|----------|------|
| Step 3.5 Flash | 196B | 11B | MIS-PO, 3:1 Hybrid SWA |
| [[MiMo-V2-Flash (2025)|MiMo-V2-Flash]] | 309B | 15B | [[MOPD]], 4:1 Hybrid SWA |
| [[DeepSeek-V3 (2024)|DeepSeek-V3]] | 671B | 37B | [[Multi-head Latent Attention|MLA]], 无辅助损失 LB |
| [[GLM-4.5 (2025)|GLM-4.5]] | 355B | 32B | Expert Model Iteration |

Step 3.5 Flash 以最小的活跃参数（11B）达到前沿水平，验证了高效架构 + 稳定 RL 训练的组合价值。

## 参考资料

> [!paper] 原始论文
> StepFun Team. "Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters." 2025.
> [[Clippings/Paper/260210604v1/260210604v1|Paper Clipping]]
