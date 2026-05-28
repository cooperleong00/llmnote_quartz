---
type: overview
description: 2025-2026 开源大模型技术全景图，深度分析 12 个旗舰模型的架构演进、训练范式、推理能力和效率优化的趋势与权衡
aliases:
  - 开源 LLM 全景
  - Open-Source LLM Landscape
tags:
  - technical-report
  - landscape
created: 2026-02-04
updated: 2026-02-26T22:23
---

# Open-Source LLM Landscape (2025-2026)

开源模型如何逼近 frontier：12 个旗舰模型的技术全景

---

## 引言

这篇文章综合了 2025-2026 年间 11 篇开源大模型 technical report 的核心洞察，旨在帮助读者建立对前沿 LLM 技术的整体认知框架。

> [!intuition] 核心发现
> 2025-2026 年开源 LLM 的发展呈现三大主线：
> 1. **MoE 成为主流架构**：所有旗舰模型都采用 MoE，稀疏度从 8x 提升到 48x
> 2. **RL 训练从 PPO 演进到 GRPO 家族**：解决 value function 开销和训练稳定性问题
> 3. **Thinking Mode 成为标配**：test-time compute scaling 从实验走向产品化

**本文结构**：
- **Part I: 模型巡礼** — 11 个模型的深度介绍，每个模型的背景、核心技术、关键数据
- **Part II: 技术深度分析** — 架构、训练、推理三个维度的横向对比
- **Part III: Benchmark 全景** — 数学、代码、Agent 能力的详细对比
- **Part IV: 洞察与趋势** — 技术共识、差异化路线、未来方向

---

## Part I: 模型巡礼

| 模型 | 发布时间 | 机构 | 总参数 | 激活参数 | 稀疏度 | 核心创新 |
|------|----------|------|--------|----------|--------|----------|
| [[DeepSeek-V3 (2024)]] | 2024.12 | DeepSeek | 671B | 37B | 32x | MLA + Loss-Free Balancing + MTP |
| [[MiniMax-M1 (2025)]] | 2025.01 | MiniMax | 456B | 46B | 10x | Lightning Attention + CISPO |
| [[Qwen3 (2025)]] | 2025.04 | Alibaba | 235B | 22B | 11x | Thinking/Non-Thinking 双模式统一 |
| [[Kimi K2 (2025)]] | 2025.06 | Moonshot | 1043B | 32B | 48x | MuonClip 优化器 + Agentic 数据合成 |
| [[GLM-4.5 (2025)]] | 2025.06 | Zhipu | 355B | 32B | 11x | Expert Model Iteration |
| [[MiMo-V2-Flash (2025)]] | 2025.06 | Xiaomi | 309B | 15B | 32x | Hybrid SWA + MOPD |
| [[GLM-5 (2026)]] | 2026.02 | Zhipu | 744B | 40B | 19x | 异步 Agentic RL + DSA + Environment Scaling |
| [[LongCat-Flash (2025)]] | 2025.xx | Meituan | 560B | 27B | 21x | Zero-Computation Experts |
| [[LongCat-Flash-Thinking (2025)]] | 2025.xx | Meituan | 560B | 27B | 21x | DORA 异步 RL + Domain-Parallel |
| [[Ling 2.0 (2025)]] | 2025.xx | Inclusion | 1000B | 51B | 28x | LPO 句子级优化 + Ling Scaling Laws |
| [[Ring-1T (2025)]] | 2025.xx | Inclusion | 1000B | 50B | 28x | IcePop + C3PO++ |
| [[Step 3.5 Flash (2025)]] | 2026.02 | StepFun | 196B | 11B | 18x | MIS-PO + 3:1 Hybrid SWA + MTP-3 |

> [!note] 模型详细介绍
> 以下各节将深入介绍每个模型的背景、核心技术选择、关键数据和重要性。

### 1.1 DeepSeek-V3 (2024.12)

**背景与定位**：[[DeepSeek-V3 (2024)]] 是 2024 年底开源 LLM 领域的标志性事件，671B 参数的 MoE 模型以极致的训练效率（仅 $5.576M）和稳定性（零回滚）证明了中国团队在大模型工程上的突破。它不仅在性能上超越所有开源模型，更重要的是树立了"高效训练"的新标杆，让万亿参数级模型的训练成本从数千万美元降至百万美元级别。

**核心技术选择**：DeepSeek-V3 的架构继承了 V2 验证过的 [[Multi-head Latent Attention|MLA]] 和 DeepSeekMoE，但在训练方法上实现了三大创新：(1) **[[Loss-Free Load Balancing]]**——首创无辅助损失的负载均衡策略，通过动态 bias 调整专家负载，避免辅助损失对模型性能的干扰；(2) **[[Multi-Token Prediction|MTP]]**——在每个位置预测多个未来 token，既增强训练信号，又可用于 [[Speculative Decoding]] 加速推理；(3) **FP8 混合精度训练 + DualPipe 算法**——首次在超大规模模型上验证 FP8 训练的可行性，配合高效的流水线并行算法，实现训练加速和显存节省。

**关键数据**：总参数 671B，激活参数 37B（256 专家，激活 8 个），预训练 14.8T tokens，仅需 2.788M H800 GPU 小时（每 1T tokens 仅需 180K GPU 小时，在 2048 卡集群上约 3.7 天）。性能上，MMLU 88.5 超越所有开源模型，MATH-500 超越 o1-preview，LiveCodeBench 达到开源 SOTA。Post-training 使用 [[GRPO]]，从 DeepSeek-R1 蒸馏推理数据，通过 rejection sampling 保留准确性的同时保持简洁。

**为什么重要**：DeepSeek-V3 的意义不仅在于性能，更在于它证明了"效率优先"的技术路线是可行的。Loss-Free Balancing 和 MTP 被后续几乎所有 2025 年模型采用，成为事实上的标准方案。它的训练成本数据（$5.576M）让学术界和小团队看到了训练超大模型的可能性，推动了开源 LLM 的民主化进程。

### 1.2 MiniMax-M1 (2025.01)

**背景与定位**：[[MiniMax-M1 (2025)]] 是首个开源的大规模 hybrid-attention 推理模型，代表了与 DeepSeek 系不同的技术路线。在 DeepSeek-V3 证明 MLA 的有效性后，MiniMax 选择了更激进的方案——Lightning Attention，原生支持 1M token 上下文（DeepSeek R1 的 8 倍）和 80K token 生成长度，在长上下文任务上建立了显著优势。

**核心技术选择**：M1 的核心是 **Lightning Attention** 架构：每 7 层 Transnormer block（使用 linear attention）后接 1 层标准 Transformer block（使用 softmax attention），实现 7:1 的 hybrid 配比。这种设计将复杂度从 $O(n^2)$ 降至接近 $O(n)$，在 100K token 生成时 FLOPs 仅为 DeepSeek R1 的 25%。RL 训练上，M1 提出 [[CISPO]] 算法，核心思想是 clip importance sampling weights 而非 token updates，保留低概率反思 token（如 "However", "Wait"）的梯度贡献，训练效率提升 2x。为解决 Lightning Attention 的训练-推理精度不匹配问题，将 LM output head 精度提升到 FP32，使概率相关性从 ~0.9x 提升到 ~0.99x。

**关键数据**：总参数 456B，激活参数 45.9B（32 专家），在 MiniMax-Text-01 基础上用 7.5T reasoning-intensive tokens 继续预训练。RL 训练使用 512 H800 GPUs，3 周完成，成本约 $534,700。性能上，AIME 2024 达到 86.0%（仅次于 DeepSeek R1 的 91.4%），SWE-bench Verified 56.0%，但在长上下文任务上表现突出：OpenAI-MRCR (128k) 76.1% 超越 o3 和 Claude 4，OpenAI-MRCR (1M) 58.6% 与 Gemini 2.5 Pro 持平，LongBench-v2 61.5% 超越所有开源模型。

**为什么重要**：M1 证明了 linear attention 在超长上下文场景下的实用性，为 MLA 之外提供了另一条可行路线。CISPO 算法的提出丰富了 GRPO 家族，其"保留低概率 token 梯度"的思想对 RL 训练中的反思能力培养有重要启发。M1 的成功表明，在特定场景（如长上下文、长输出）下，架构选择可以带来显著的差异化优势。

### 1.3 Qwen3 (2025.04)

**背景与定位**：[[Qwen3 (2025)]] 是阿里巴巴 Qwen 团队的最新力作，其核心创新在于**首次将 thinking mode 和 non-thinking mode 统一到单一模型中**。在 DeepSeek R1 和 QwQ 证明推理模式的价值后，Qwen3 解决了一个关键的产品化问题：如何让用户在同一个模型中灵活切换快速响应和深度推理，而不需要部署多个模型。旗舰模型 235B-A22B 在 AIME'24 达到 85.7，AIME'25 达到 81.5，展现出与 o1、o3-mini 竞争的数学推理能力。

**核心技术选择**：Qwen3 的核心是 **Thinking Mode Fusion** 技术，通过设计 `/think` 和 `/no_think` chat template，让模型学会在两种模式间切换。训练流程采用四阶段设计：(1) Long-CoT Cold Start——用 QwQ-32B 生成推理数据，严格过滤；(2) Reasoning RL——使用 [[GRPO]]，在 3995 个 query-verifier pairs 上训练，AIME'24 从 70.1 提升到 85.1（仅 170 steps）；(3) Thinking Mode Fusion——融合 thinking 和 non-thinking 数据，训练模式切换能力；(4) General RL——20+ 任务的 reward system，提升通用能力。一个意外发现是 **Thinking Budget** 能力：模型可以在推理过程中被手动打断，基于已有推理生成答案，这种能力未经显式训练，是模式融合的涌现结果。对于轻量级模型（0.6B-14B），Qwen3 采用 [[On-Policy Distillation]] 而非完整 RL，仅需 1/10 GPU 小时，AIME'24 性能反而更高（74.4 vs 67.6）。

**关键数据**：旗舰模型 235B 总参数，22B 激活参数（128 专家，激活 8 个），预训练使用 36T tokens，覆盖 119 种语言（Qwen2.5 仅 29 种）。架构上采用标准 [[Grouped-Query Attention|GQA]]，移除 QKV-bias，引入 QK-Norm 确保训练稳定性，无共享专家（与 Qwen2.5-MoE 不同）。性能上，AIME'24 85.7 竞争 o1/o3-mini，LiveCodeBench v5 70.7，CodeForces 2056，BFCL v3 70.8（Agent 任务）。ThinkFollow 指标从 88.7 提升到 98.9，证明模式切换的准确性。

**为什么重要**：Qwen3 解决了推理模型的产品化难题——如何在单一模型中平衡推理深度和响应速度。Thinking Mode Fusion 的成功证明，通过精心设计的训练流程，模型可以学会"元认知"能力（知道何时需要深度思考）。On-Policy Distillation 的发现为轻量级模型提供了比 RL 更高效的训练路径。Qwen3 的多语言能力（119 种语言）也为开源模型的全球化提供了重要参考。

### 1.4 Kimi K2 (2025.06)

**背景与定位**：[[Kimi K2 (2025)|Kimi K2]] 是月之暗面在 2025 年 6 月发布的 1.04T 参数 MoE 模型，定位于 **Agentic Intelligence**——让模型具备自主感知、规划、推理和行动的能力。在 LMSYS Arena 上排名开源模型第 1、总榜第 5，标志着开源模型在智能体任务上的重大突破。

**核心技术选择**：K2 在架构上延续了 [[DeepSeek-V3 (2024)|DeepSeek-V3]] 的 ultra-sparse MoE + [[Multi-head Latent Attention|MLA]] 路线，但做了两个关键优化：(1) **更高稀疏度**——384 个专家（稀疏度 48），Scaling law 分析表明这比稀疏度 8 节省 1.69x FLOPs 达到相同 loss；(2) **更少注意力头**——从 128 头减至 64 头，在 128K 长上下文场景下节省 83% 推理 FLOPs，loss 仅退化 0.5%-1.2%。训练方面，K2 首创 **MuonClip 优化器**，将 token-efficient 的 [[Muon]] 优化器与 QK-Clip 稳定性机制结合，通过在优化器更新后 rescale Q/K 投影权重，从根源约束 attention logits 增长，在 15.5T tokens 训练中实现**零 loss spike**。

**关键数据**：总参数 1.04T，激活参数 32.6B，384 个专家（激活 8 个），稀疏度 48x。预训练使用 15.5T tokens，采用 synthetic rephrasing 提升 token utility。在 Agentic 任务上表现突出：SWE-bench Verified 65.8%（开源第一），Tau2-Bench 66.1%，LiveCodeBench 53.7%。数学推理能力也达到开源顶级：AIME 2025 Avg@64 为 49.5%，GPQA-Diamond 75.1%。

**为什么重要**：K2 证明了三个重要趋势：(1) **更高稀疏度的可行性**——48x 稀疏度是当时所有开源模型中最高的，为后续模型探索更激进的稀疏化提供了实证支持；(2) **Agentic 数据合成的价值**——通过系统性的智能体数据合成管线（覆盖 20,000+ 合成工具和 3,000+ 真实 MCP 工具），K2 在 tool use 和软件工程任务上显著超越其他开源模型；(3) **训练稳定性的新方案**——MuonClip 为大规模 MoE 训练提供了除 AdamW 之外的高效稳定选择。

### 1.5 GLM-4.5 (2025.06)

**背景与定位**：[[GLM-4.5 (2025)|GLM-4.5]] 是智谱 AI 在 2025 年 6 月发布的开源 MoE 模型，核心目标是在单一模型中统一 **Agentic**（智能体）、**Reasoning**（推理）、**Coding**（编程）三大能力。在 ARC 任务上达到开源模型 SOTA，整体排名第 3（仅次于 o3 和 Claude Sonnet 4），证明了"小而精"的架构设计和精细化 post-training 的价值。

**核心技术选择**：GLM-4.5 采取了与 DeepSeek-V3 和 Kimi K2 不同的架构路线——**更深而非更宽**。总参数 355B（仅为 K2 的 1/3），但 MoE 层数达到 89 层（vs DeepSeek-V3 的 58 层），hidden dim 仅 5120（vs 7168）。关键设计包括：(1) **更多 attention heads**——96 heads for 5120 hidden dim，是常规配置的 2.5 倍，虽然不改善 training loss，但在 MMLU、BBH 等推理 benchmark 上表现更好；(2) **Expert Model Iteration**——这是 GLM-4.5 最核心的训练创新，分两阶段进行：先分别训练 Reasoning、Agent、General 三个专家模型，再通过 self-distillation 整合专家能力，训练出支持 hybrid reasoning 的统一模型，可在 thinking mode 和 non-thinking mode 之间切换。

**关键数据**：总参数 355B，激活参数 32B，89 层 MoE。预训练 23T tokens（4K context），mid-training 阶段逐步扩展至 128K context。性能上，TAU-Bench 70.1%（≈ Claude Sonnet 4），SWE-bench Verified 64.2%（> GPT-4.1），AIME 24 达到 91.0%。值得注意的是，GLM-4.5 在参数效率上表现突出：仅用 DeepSeek-R1 一半、Kimi K2 三分之一的参数，达到相近或更好的性能。

**为什么重要**：GLM-4.5 证明了两个重要观点：(1) **架构多样性的价值**——在 MoE 成为共识的背景下，"更深而非更宽"的设计路线仍然可行，为资源受限的团队提供了另一种选择；(2) **Post-training 的关键作用**——Expert Model Iteration 范式通过分阶段训练和 self-distillation，在不增加推理成本的前提下整合多种能力，避免了传统多任务训练的"跷跷板效应"。此外，GLM-4.5 的 RL 训练策略（Difficulty-based Curriculum、Dynamic Sampling Temperature、Iterative Self-distillation）为后续模型提供了丰富的工程经验。

### 1.6 MiMo-V2-Flash (2025.06)

**背景与定位**：[[MiMo-V2-Flash (2025)|MiMo-V2-Flash]] 是小米在 2025 年 6 月发布的 309B 参数 MoE 模型，激活参数仅 15B，是所有 2025 年旗舰模型中最小的。其核心定位是**极致效率**——用最少的激活参数达到 SOTA 性能。在 SWE-bench Verified 上达到 73.4%（超越所有开源模型），AIME 2025 达到 94.1%，证明了"小而强"的可行性。

**核心技术选择**：MiMo-V2-Flash 的技术路线围绕"效率"展开，包含三个核心创新：(1) **Hybrid Sliding Window Attention**——采用 5:1 的 SWA 与 Global Attention 混合架构（48 层中 39 层 SWA + 9 层 GA），窗口大小仅 128 tokens。关键发现是配合 [[Attention Sink|learnable attention sink bias]]，小窗口反而优于大窗口，因为 attention sink 让模型在不需要关注任何 token 时可以"丢弃"注意力；(2) **[[MOPD|Multi-Teacher On-Policy Distillation]]**——先训练多个领域专家教师（Agentic、Math、Safety 等），再让学生模型从自身分布采样，接收教师的 token-level KL 奖励，避免传统多任务训练的能力失衡问题；(3) **Lightweight Multi-Token Prediction**——使用 dense FFN + SWA 的轻量级 MTP 模块（每层仅 0.33B 参数），实现 2.6x 推理加速。

**关键数据**：总参数 309B，激活参数 15B（256 专家，激活 8 个），稀疏度 32x。预训练 27T tokens，原生 32K context（扩展至 256K）。性能上，SWE-bench Verified 73.4%（开源第一），AIME 2025 94.1%，LiveCodeBench 83.2%，BrowseComp 45.4%（with context management 58.3%）。推理效率上，MTP speculative decoding 在低熵任务（如代码生成）可达 3.6 acceptance length，实现 2.53x 加速。

**为什么重要**：MiMo-V2-Flash 证明了"效率极限"的可能性——15B 激活参数可以达到 37B（DeepSeek-V3）甚至 32B（Kimi K2）的性能水平。这对资源受限的部署场景（边缘设备、实时应用）具有重要意义。MOPD 范式为解决 post-training 的能力失衡问题提供了新思路，避免了传统 RL 训练中"提升数学能力会损害对话能力"的跷跷板效应。Hybrid SWA + Attention Sink 的组合也为长上下文建模提供了比 MLA 更轻量的替代方案。

### 1.7 LongCat-Flash & LongCat-Flash-Thinking (2025)

美团 LongCat 团队的双子星模型，展示了从高效基础模型到推理模型的完整路径。

**[[LongCat-Flash (2025)|LongCat-Flash]]** 是 560B 参数 MoE 基础模型，核心创新在于**动态计算预算分配**。通过 Zero-Computation Experts 机制，模型根据 token 的上下文重要性动态激活 18.6B-31.3B 参数（平均 27B），实现"不是所有 token 都需要同等计算量"的直觉。配合 Shortcut-Connected MoE 架构扩大计算-通信重叠窗口，在数万张加速卡上仅用 30 天完成 20T token 训练。这种动态计算分配的思想与 [[Speculative Decoding]] 的经验性发现一致：小模型能可靠预测大模型在大多数"简单"token 上的输出。

**[[LongCat-Flash-Thinking (2025)|LongCat-Flash-Thinking]]** 基于 LongCat-Flash 构建，通过 **DORA 异步 RL 框架**实现 3x 训练加速。DORA 的核心思想是保留多个旧版本 Actor，让每个响应由同一版本完整生成，避免传统 partial rollout 的 prefill 开销和策略不一致问题。更重要的是，LongCat-Flash-Thinking 提出 **Domain-Parallel RL Training**：针对 STEM、Code、Agentic 三个领域分别训练专家模型，再通过 task vector 融合为单一模型。这一策略解决了异步训练中 mixed-domain 的负迁移问题（不同领域的响应长度分布差异巨大）。

2601 版本进一步强化 agentic 能力，通过 **Environment Scaling**（20+ 领域，每个领域 60+ tools）和 **Noisy Environment Training**（curriculum-based 注入噪声）提升真实世界泛化能力。Heavy Thinking Mode 同时扩展推理深度和宽度：并行生成多个候选轨迹，再用 Summary Model 反思聚合，在 AIME-25 上 token 消耗降低 64.5% 而准确率不降。在 τ²-Bench 上达到 88.2%，BrowseComp（w/ context management）达到 73.1%，展示了开源模型在 agentic reasoning 上的新高度。

**为什么重要**：LongCat 系列证明了动态计算分配和分域训练的价值，为 MoE 模型的效率优化和 RL 训练提供了新思路。DORA 框架解决了大规模异步 RL 训练的核心挑战，Environment Scaling 为 agentic 能力的系统性提升提供了可行路径。

### 1.8 Ling 2.0 & Ring-1T (2025)

Inclusion AI 的 Ling/Ring 双系列代表了万亿参数模型的两条路径：高效基础模型和深度推理模型。

**[[Ling 2.0 (2025)|Ling 2.0]]** 是推理导向的基础模型系列，核心目标是在万亿参数规模下实现高效推理。通过高稀疏度 MoE 架构（256 专家，8 激活，3.5% 激活率）和系统性的 **Ling Scaling Laws**，Ling-1T 以 51B 激活参数实现了与 dense 模型相当的性能，计算效率提升约 7 倍。Ling Scaling Laws 基于超过 1000 个实验，揭示了激活率、专家粒度、计算预算三者之间的幂律关系，为万亿参数模型设计提供了理论指导。

Ling 2.0 的后训练创新包括 **Linguistic-unit Policy Optimization (LPO)**——将句子（而非 token 或整个序列）作为策略更新的基本单元，在语义边界上进行优化。这一粒度选择在 [[GRPO]]（token 级，噪声大）和 [[GSPO]]（sequence 级，粒度过粗）之间找到了平衡点。配合 **Evolutionary Chain-of-Thought (Evo-CoT)** 渐进式深化推理能力，Ling-1T 在 AIME 2025 上达到 43.75% Pass@1，显著领先同规模模型。

**[[Ring-1T (2025)|Ring-1T]]** 是首个开源万亿参数 thinking model，基于 Ling-1T-base 构建。论文的核心贡献不在于模型架构本身，而在于解决万亿参数规模 RL 训练的三大挑战：(1) **IcePop** 通过双边校准和 masking 解决训练-推理概率不一致导致的训练不稳定；(2) **C3PO++** 通过 token budget 机制动态分割 rollout，跨迭代续接，解决长 rollout 导致的资源闲置；(3) **ASystem** 采用 SingleController + SPMD 架构，实现万亿参数分布式 RL 的高效执行。

Ring-1T 在多个 benchmark 上达到开源模型 SOTA：AIME 2025 93.40%（开源第一），CodeForces 2088（所有模型最高），ARC-AGI-1 55.94%（超 DeepSeek-V3.1 +15.32）。在 IMO 2025 评测中达到银牌水平，全程使用纯自然语言推理，不依赖代码生成或符号求解器。

**为什么重要**：Ling/Ring 系列证明了高稀疏度 MoE 的可行性和效率优势，Ling Scaling Laws 为万亿参数模型设计提供了系统性指导。IcePop 和 C3PO++ 解决了 MoE RL 训练的核心挑战，为后续模型提供了可复用的基础设施方案。LPO 的句子级优化粒度为 RL 算法设计提供了新思路。

### 1.9 GLM-5 (2026.02)

**背景与定位**：[[GLM-5 (2026)|GLM-5]] 是智谱 AI 在 2026 年 2 月发布的下一代旗舰模型，是 [[GLM-4.5 (2025)|GLM-4.5]] 的后继。核心主张是从 "vibe coding"（人类提示 AI 写代码）转向 "agentic engineering"（AI agent 自主规划、实现、迭代）。模型在 SWE-bench Verified、Terminal-Bench 2.0、BrowseComp 等 agentic benchmark 上达到开源 SOTA，整体性能接近 Claude Opus 4.5 和 GPT-5.2。

**核心技术选择**：GLM-5 采用 744B/40B 的 MoE 架构（256 experts，80 layers），相比 GLM-4.5 的 355B/32B 显著扩展。架构上有三个关键创新：(1) **MLA + Muon Split**——发现 MLA 在 [[Muon]] 优化器下性能不如 GQA-8，通过将 up-projection 矩阵按 head 拆分独立正交化解决；(2) **MTP 参数共享**——3 个 MTP layer 共享参数，保持 draft model 内存与 DeepSeek-V3 一致，同时提升 speculative decoding 接受率（2.76 vs DeepSeek-V3.2 的 2.55）；(3) **DeepSeek Sparse Attention (DSA)**——通过 continued pre-training 从 dense MLA 迁移到 content-aware 的稀疏 attention，在 128K 长上下文上与 MLA 持平甚至更优。

**Post-training 的核心创新**在于异步 Agentic RL。传统同步 RL 在 long-horizon agent 任务中会产生严重 GPU 空闲（等最慢的 rollout）。GLM-5 将推理引擎和训练引擎解耦到不同 GPU 上，通过 Multi-Task Rollout Orchestrator 支持 1k+ 并发 rollout。为解决异步带来的 off-policy bias，提出 Token-in-Token-out (TITO) Gateway 消除 re-tokenization 误差，以及 Direct Double-sided Importance Sampling 用双边 masking 过滤偏离过大的样本。此外，GLM-5 在 SWE、Terminal、Search、Slide Generation 四个领域构建了大规模可验证环境（SWE 环境覆盖 9 种语言、10K+ 可验证实例），为 agentic RL 提供 grounded feedback。

**关键数据**：总参数 744B，激活参数 40B，预训练 28.5T tokens。SWE-bench Verified 77.8%，Terminal-Bench 2.0 60.7%，BrowseComp（w/ Context Management）75.9%，HLE（w/ Tools）50.4%。在 Artificial Analysis Intelligence Index v4.0 上得分 50，是首个达到此分数的开源模型。

**为什么重要**：GLM-5 的核心贡献不在架构（MoE + MLA 已是共识），而在于**异步 Agentic RL 基础设施**和**大规模 Environment Scaling**。这两者共同解决了 agentic RL 训练的两大瓶颈：训练效率（异步解耦）和数据来源（可验证环境）。On-Policy Cross-Stage Distillation 为多阶段 RL 的能力退化问题提供了优雅的解决方案。GLM-5 标志着开源模型在 agentic engineering 上正式进入与闭源模型竞争的阶段。

### 1.10 Step 3.5 Flash (2026.02)

StepFun 在 2026 年 2 月发布的 [[Step 3.5 Flash (2025)|Step 3.5 Flash]] 是开源 LLM 发展的新里程碑，以 196B 总参数、11B 活跃参数的高效配置，在 agent、coding、math 任务上达到与 GPT-5.2 xHigh 和 Gemini 3.0 Pro 相当的前沿水平。这是首个以如此小的激活参数达到 frontier 级别的开源模型。

**核心技术选择**：Step 3.5 Flash 的架构创新包括 **3:1 Hybrid SWA**（三层 SWA + 一层 Full GQA-8 交替布局）和 **MTP-3**（三个轻量级 MTP heads，仅增加 0.81B 参数）。3:1 布局相比 1:1 布局在 prefill/decode FLOPs 上节省 37.5%，配合增加的 query heads 能以极低的额外成本接近 1:1 的质量。MTP-3 支持 [[Speculative Decoding]]，在 Hopper GPU 上达到 ~170 tokens/s。

最重要的创新是 **[[MIS-PO]]**（Metropolis Independence Sampling-Filtered Policy Optimization），这是本文的核心方法贡献。传统 [[Importance Sampling]] 通过 bounded ratio 缩放梯度，但方差很高。MIS-PO 改用**二值 mask** 过滤 off-distribution 样本，将保留的样本视为 on-policy，大幅降低梯度方差。双层过滤机制（token-level 和 trajectory-level）配合 Truncation-Aware Value Bootstrapping，解决了大规模 off-policy RL 训练的稳定性问题。

**关键数据**：总参数 196B，激活参数 11B，45 层 Transformer（3 dense + 42 MoE），每层 288 routed experts + 1 shared expert，top-8 routing。在 IMO-AnswerBench 上达到 85.4%（接近 GPT-5.2 xHigh 的 86.3%），LiveCodeBench-v6 86.4%（超越 Claude Opus 4.5 的 84.8%），τ²-Bench 88.2%（Agent 任务 SOTA），BrowseComp（w/ Context Management）69.0%（超越 GPT-5.2 xHigh 的 65.8%）。

**为什么重要**：Step 3.5 Flash 证明了三个关键趋势：(1) **极致效率的可能性**——11B 激活参数达到 frontier 水平，验证了高效架构 + 稳定 RL 训练的组合价值；(2) **MIS-PO 的突破**——用二值 mask 替代 importance sampling 的连续权重，为大规模 off-policy RL 提供了新范式；(3) **Hybrid Attention 的成熟**——3:1 SWA/Full 布局在效率和质量之间找到了新的平衡点。Step 3.5 Flash 标志着开源模型在 agentic intelligence 上正式进入 frontier 时代。

---

## Part II: 技术深度分析

### 2.1 架构篇

#### 2.1.1 MoE 成为绝对主流

2025-2026 年，**所有旗舰开源模型都采用 [[Mixture of Experts|MoE]] 架构**。这不是偶然——MoE 实现了"大模型容量 + 小模型推理成本"的解耦。

**稀疏度演进**：

| 模型 | 专家总数 | 激活专家 | 稀疏度 | 设计理由 |
|------|----------|----------|--------|----------|
| DeepSeek-V3 | 256 | 8 | 32x | 平衡性能与效率 |
| Kimi K2 | 384 | 8 | 48x | Scaling law 显示更高稀疏度持续降低 loss |
| Ling-1T | 256 | 8+1 shared | 28x | 3.5% 激活率实现 7x 效率杠杆 |
| MiMo-V2-Flash | 256 | 8 | 32x | 以最小激活参数达到 SOTA |
| Step 3.5 Flash | 288 | 8 | 18x | 高效配置，11B 激活参数 |
| GLM-5 | 256 | 8 | 19x | 744B/40B，80 层减少 EP 通信 |

> [!intuition] 稀疏度的权衡
> [[Kimi K2 (2025)|Kimi K2]] 的实验表明：在固定激活参数下，增加专家数（提高稀疏度）能持续降低 loss。稀疏度 48 相比稀疏度 8 可节省 1.69x FLOPs 达到相同 loss。但更高稀疏度也带来更大的通信开销和路由复杂性。

**负载均衡的共识**：[[Loss-Free Load Balancing]] 成为标准方案。传统辅助损失会引入干扰梯度，而 DeepSeek-V3 首创的动态 bias 调整方法被 Ling 2.0、GLM-4.5 等广泛采用。

#### 2.1.2 Attention 变体的分化

不同于 MoE 的高度共识，Attention 机制呈现**多路线并行**的格局：

| 方案 | 代表模型 | 核心思想 | 适用场景 |
|------|----------|----------|----------|
| [[Multi-head Latent Attention\|MLA]] | DeepSeek-V3, Kimi K2, LongCat | 低秩压缩 KV 到 latent vector | 通用场景，KV cache 敏感 |
| MLA → [[Sparse Attention\|DSA]] | GLM-5 | MLA 基础上迁移到 content-aware 稀疏 attention | 长上下文效率 |
| Lightning Attention | MiniMax-M1 | 7:1 hybrid（linear + softmax） | 超长上下文（1M tokens） |
| Hybrid SWA | MiMo-V2-Flash, Step 3.5 Flash | SWA + Global Attention 混合 | 长上下文 + 效率平衡 |
| 标准 GQA | GLM-4.5, Qwen3 | 分组共享 KV | 成熟稳定 |

> [!comparison] Attention 方案的效率对比
> | 方案 | 复杂度 | KV Cache | 长上下文支持 | 实现难度 |
> |------|--------|----------|--------------|----------|
> | MLA | $O(n^2)$ | 压缩到 latent | 128K | 中 |
> | Lightning | $O(n)$ | 标准 | 1M | 高 |
> | Hybrid SWA | $O(n^2)$ 但降低 | 标准 + sink | 128K | 中 |
>
> **MLA**：通过低秩分解压缩 KV cache，推理时只需缓存压缩后的 latent vector，适合需要精确 attention 的场景
>
> **Lightning Attention**：用 linear attention 替代大部分 softmax attention，复杂度从 $O(n^2)$ 降至 $O(n)$，在 100K token 生成时 FLOPs 仅为标准 attention 的 25%
>
> **Hybrid SWA**：MiMo-V2-Flash 采用 5:1 配置（W=128），Step 3.5 Flash 采用 3:1 配置（W=512），通过局部 attention + 稀疏全局 attention 平衡效率和质量

**MiMo-V2-Flash 的发现**：配合 [[Attention Sink|learnable attention sink bias]]，小窗口 (W=128) 的 Hybrid SWA 反而优于大窗口 (W=512)。Attention sink 让模型在不需要关注任何 token 时可以"丢弃"注意力。

**Step 3.5 Flash 的方案**：采用 3:1 配置（三层 SWA + 一层 Full Attention），使用 [[Gated Attention|Head-wise Gated Attention]] 替代传统 sink token。

#### 2.1.3 Multi-Token Prediction 成为标配

[[Multi-Token Prediction|MTP]] 从 DeepSeek-V3 的创新变成 2025 年的标配：

| 模型 | MTP 配置 | 用途 |
|------|----------|------|
| DeepSeek-V3 | 1 层 MTP | 训练目标 + Speculative Decoding |
| Ling 2.0 | 1 层，loss weight 0.1 | 提升 code/math 性能 |
| MiMo-V2-Flash | 1→3 层（post-training 复制） | 2.6x 推理加速 |
| Step 3.5 Flash | 3 层 MTP（MTP-3） | Speculative Decoding |
| LongCat-Flash | 有 MTP 层 | 训练 + 推理 |
| GLM-4.5 | 有 MTP 层 | Speculative Decoding |
| GLM-5 | 3 层共享参数 MTP | 内存不增，接受率提升（2.76 vs V3.2 的 2.55） |

> [!intuition] MTP 的双重价值
> 1. **训练信号增强**：预测多个未来 token 提供更丰富的梯度信号
> 2. **推理加速**：MTP 层可作为 draft model 用于 [[Speculative Decoding]]，低熵任务（如代码生成）可达 3.6 acceptance length
>
> **MiMo-V2-Flash 的方案**：训练时使用 1 层 MTP，post-training 阶段复制为 3 层，实现 2.6x 推理加速。
>
> **Step 3.5 Flash 的方案**：直接训练 3 层轻量级 MTP heads（MTP-3），每个包含 SWA + dense FFN，仅增加 0.81B 参数。
>
> **GLM-5 的方案**：3 个 MTP layer 共享参数，保持 draft model 内存与 DeepSeek-V3 一致，同时提升接受率。

---

### 2.2 训练篇

#### 2.2.1 RL 算法：从 PPO 到 GRPO 家族

[[GRPO]] 由 DeepSeekMath (2024) 首次提出，核心思想是用 **group relative advantage** 替代 value function：

$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

这一创新解决了 [[PPO]] 的两大痛点：
1. **内存开销**：不需要训练和存储 value network
2. **训练稳定性**：相对优势比绝对 value 更稳定

**GRPO 家族的演进**：

```
GRPO (2024)
    │
    ├─→ DAPO (2025): Clip-Higher + Dynamic Sampling
    │       解决 entropy collapse 和训练不稳定
    │
    ├─→ CISPO (MiniMax-M1): Clip importance weights 而非 token updates
    │       保留低概率重要 token 的梯度贡献
    │
    ├─→ GSPO (2025): 序列级重要性采样
    │       从根本上解决 MoE 训练-推理不一致
    │
    ├─→ LPO (Ling 2.0): 句子级策略优化
    │       在语义边界上进行更新，粒度介于 token 和 sequence 之间
    │
    ├─→ IcePop (Ring-1T): 双边校准 + Masking
    │       专门解决万亿参数 MoE 的训练稳定性
    │
    └─→ MIS-PO (Step 3.5 Flash): 双层二元掩码
            用 binary masking 替代 importance sampling，大幅降低梯度方差

GLM-5 的异步 RL:
    ├─→ Direct Double-sided IS: 双边 masking 过滤 off-policy 样本
    │       简化 IcePop，移除对 π_old 的依赖
    └─→ TITO Gateway: Token-in-Token-out 消除 re-tokenization 误差
```

> [!comparison] GRPO 家族算法对比
> | 算法 | 更新粒度 | 核心机制 | 优势 | 劣势 |
> |------|----------|----------|------|------|
> | GRPO | Token | Importance sampling + clip | 细粒度控制 | 噪声大，MoE 不稳定 |
> | GSPO | Sequence | 序列级 IS | 稳定，适合 MoE | 粒度过粗 |
> | LPO | Sentence | 句子级更新 | 与语义边界对齐 | 需要句子分割 |
> | CISPO | Token | Clip IS weights | 保留低概率 token | 实现复杂 |
> | IcePop | Token | 双边校准 + masking | 万亿参数稳定 | 需要额外校准 |
> | MIS-PO | Token + Trajectory | 双层二元掩码 | 低梯度方差 | 样本利用率降低 |

**各算法的核心创新**：

**DAPO**：通过 Clip-Higher 机制和 Dynamic Sampling 解决 entropy collapse，让模型保持探索能力。

**CISPO**（MiniMax-M1）：clip importance sampling weights 而非 token updates，保留低概率但重要的反思 token（如 "However", "Wait"）的梯度贡献，训练效率提升 2x。

**GSPO**：将重要性采样从 token 级提升到序列级，从根本上绕过 MoE 的 token 级概率不一致问题。

**LPO**（Ling 2.0）：将句子作为策略更新的基本单元，在语义边界上进行优化，粒度介于 token 和 sequence 之间。

**IcePop**（Ring-1T）：通过双边校准和高方差 token masking，专门解决万亿参数 MoE 的训练稳定性问题。

**MIS-PO**（Step 3.5 Flash）：用二元掩码替代连续的 importance sampling 权重，通过 token-level 和 trajectory-level 双层过滤，将通过过滤的样本视为 on-policy，大幅降低梯度方差。

> [!warning] 方法选择建议
> - **Dense 模型 + 短序列**：GRPO 足够，实现简单
> - **MoE 模型 + 中等序列**：GSPO 或 MIS-PO
> - **MoE 模型 + 长链推理**：MIS-PO 或 IcePop
> - **需要最大样本利用率**：GSPO，不丢弃样本
> - **万亿参数规模**：IcePop，专门针对超大规模设计

#### 2.2.2 MoE RL 训练的核心挑战：训练-推理不一致

[[Training-Inference Mismatch]] 是 MoE 模型 RL 训练的核心难题。问题根源：

1. **路由决策差异**：训练引擎（如 Megatron）和推理引擎（如 vLLM）的 MoE 路由实现不同
2. **数值精度差异**：FP8/BF16 混合精度导致概率计算偏差
3. **Long CoT 放大效应**：长序列中误差累积

**各家的解决方案**：

| 模型 | 方案 | 核心思想 |
|------|------|----------|
| Ring-1T | [[IcePop]] | 双边校准 + 高方差 token masking |
| MiMo-V2-Flash | [[Rollout Routing Replay]] | 记录推理时路由分布，训练时重放 |
| LongCat-Flash-Thinking | 改进 GRPO | 限制 KL 散度 + 动态 clip |
| Step 3.5 Flash | [[MIS-PO]] | 双层二元掩码过滤 off-policy 样本 |
| GLM-5 | Direct Double-sided IS | 双边 masking + TITO 消除 re-tokenization |
| GSPO | 序列级采样 | 绕过 token 级概率计算 |

#### 2.2.3 Post-Training 流程的标准化

2025 年的 post-training 流程趋于标准化：

```
Stage 1: SFT (Supervised Fine-Tuning)
    ↓ 高质量指令数据，建立基础能力

Stage 2: Reasoning RL
    ↓ 可验证任务（数学、代码），rule-based reward

Stage 3: Thinking Mode Fusion (可选)
    ↓ 融合 thinking + non-thinking 数据

Stage 4: General RL
    ↓ 通用任务，model-based reward
```

**Qwen3 的四阶段流程**是典型代表：
- Stage 1: Long-CoT SFT（QwQ-32B 生成数据）
- Stage 2: Reasoning RL（GRPO，AIME'24: 70.1 → 85.1）
- Stage 3: Thinking Mode Fusion（设计 /think 和 /no_think template）
- Stage 4: General RL（提升通用能力和模式切换准确性）

**GLM-5 的五阶段流程**进一步细化了 agentic 维度：
- Stage 1: SFT（扩展 Agent/Coding 数据，三种 thinking 模式）
- Stage 2: Reasoning RL（GRPO + IcePop，数学/科学/代码/TIR）
- Stage 3: Agentic RL（异步框架，SWE/Terminal/Search 环境）
- Stage 4: General RL（Hybrid Reward + Human-in-the-loop Style Alignment）
- Stage 5: On-Policy Cross-Stage Distillation（用前序阶段 checkpoint 作 teacher 防止能力退化）

#### 2.2.4 数据合成的重要性

高质量训练数据成为差异化竞争的关键。[[Kimi K2 (2025)|Kimi K2]] 的 **Agentic 数据合成管线**是典型案例：

```
┌─────────────────────────────────────────────────────┐
│  1. Seed Task Collection                            │
│     ├─ 真实用户 query                               │
│     └─ 合成任务（基于 seed 扩展）                    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  2. Trajectory Generation                           │
│     ├─ 多模型并行生成轨迹                           │
│     └─ 包含 tool use、multi-turn 交互               │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  3. Quality Filtering                               │
│     ├─ 执行验证（代码运行、工具调用）               │
│     └─ 模型评分 + 人工抽检                          │
└─────────────────────────────────────────────────────┘
```

**MiMo-V2-Flash 的 [[MOPD]]** 提供了另一种思路：多教师在线蒸馏，通过领域专家教师提供 token-level KL 奖励，解决 post-training 的能力失衡问题。

---

### 2.3 推理篇

#### 2.3.1 Thinking Mode 的实现方式

| 模型                     | 实现方式                    | 特点                             |
| ---------------------- | ----------------------- | ------------------------------ |
| Qwen3                  | `/think` `/no_think` 标志 | 首创双模式统一，用户可动态切换                |
| GLM-4.5                | Hybrid reasoning mode   | 可在 thinking 和 non-thinking 间切换 |
| Ring-1T                | 专门的 thinking model      | 基于 Ling-1T base 的 RL 训练        |
| GLM-5                  | 三种 thinking 模式          | Interleaved / Preserved / Turn-level |
| LongCat-Flash-Thinking | Heavy Thinking Mode     | 并行推理 + Summary 聚合              |

> [!intuition] Qwen3 的 Thinking Budget
> 一个意外发现：学会双模式后，模型自然获得了处理"中间状态"的能力。当 thinking 长度达到用户设定阈值时，可以手动插入停止指令，模型会基于已有推理生成最终答案。这种能力**未经显式训练**，是 Thinking Mode Fusion 的涌现结果。

#### 2.3.2 Test-Time Compute Scaling

**LongCat-Flash-Thinking 的 Heavy Thinking Mode** 代表了 test-time scaling 的新范式：

```
Stage 1: Parallel Reasoning
    ├─ Thinking Model 并行生成多个候选轨迹
    └─ 扩展探索宽度

Stage 2: Heavy Thinking
    ├─ Summary Model 对轨迹进行反思推理
    ├─ 综合中间推理和结果
    └─ 扩展推理深度
```

**效果**：AIME-25 上 token 消耗从 19,653 降至 6,965（-64.5%），准确率不降。

#### 2.3.3 Agentic Reasoning 的崛起

2025 年的一个重要趋势是 **Agentic 能力**成为核心竞争力：

| 模型 | SWE-Bench Verified | τ²-Bench | 核心方法 |
|------|-------------------|----------|----------|
| Step 3.5 Flash | 74.4% | 88.2% | MIS-PO + Hybrid SWA |
| GLM-5 | 77.8% | - | 异步 Agentic RL + Environment Scaling |
| MiMo-V2-Flash | 73.4% | - | MOPD + Agentic RL |
| LongCat-2601 | - | 67.1% | Environment Scaling + Noisy Training |
| Kimi K2 | 65.5% | 66.1% | 大规模 Agentic 数据合成 |
| GLM-4.5 | 64.2% | 70.1% | Expert Model Iteration |

**各模型的 Agentic 训练策略**：

**Kimi K2**：构建了覆盖 20,000+ 合成工具和 3,000+ 真实 MCP 工具的数据合成管线，通过大规模 tool use 数据提升 agent 能力。

**GLM-4.5**：采用 Expert Model Iteration，分别训练 Reasoning、Agent、General 三个专家模型，再通过 self-distillation 整合。

**MiMo-V2-Flash**：使用 [[MOPD]] 多教师在线蒸馏，通过领域专家教师（包括 Agentic 专家）提供 token-level KL 奖励。

**GLM-5**：构建了覆盖 SWE（9 种语言、10K+ 可验证实例）、Terminal、Search（Web Knowledge Graph + multi-hop QA）、Slide Generation 四个领域的大规模可验证环境，配合异步 RL 基础设施实现高效 agentic 训练。

**LongCat-2601**：通过 Environment Scaling（20+ 领域，每个领域 60+ tools）和 Noisy Environment Training 提升真实世界泛化能力。

> [!warning] Agentic 训练的挑战
> - **环境构建成本高**：需要大量工程投入构建可验证环境
> - **Noisy training 的泛化边界**：训练时注入的噪声类型可能无法覆盖所有真实世界的不完美
> - **长轨迹的 credit assignment**：如何将最终奖励分配到多轮交互的每一步

---

#### 2.3.4 效率优化

##### 训练效率

**DeepSeek-V3 的训练成本**树立了效率标杆：

| 阶段 | GPU 小时 | 成本 |
|------|----------|------|
| Pre-training | 2,664K | $5.328M |
| Context Extension | 119K | $0.238M |
| Post-training | 5K | $0.01M |
| **总计** | **2,788K H800** | **$5.576M** |

每 1T tokens 仅需 180K H800 GPU 小时，在 2048 卡集群上约 3.7 天。

**MiniMax-M1 的 RL 训练成本**：
- 512 H800 GPUs，3 周完成完整 RL 训练
- 约 $534,700 USD（租赁成本）
- 得益于 Lightning Attention 的低计算成本和 [[CISPO]] 的高训练效率

##### 推理效率

**动态计算分配**是 2025 年的重要趋势：

| 模型 | 方案 | 效果 |
|------|------|------|
| LongCat-Flash | Zero-Computation Experts | 激活参数 18.6B-31.3B 动态变化 |
| MiMo-V2-Flash | MTP Speculative Decoding | 2.6x 推理加速 |
| GLM-5 | MTP 参数共享 + DSA + DP-aware Routing | 接受率 2.76，KV cache 复用 |
| Step 3.5 Flash | MTP-3 + 3:1 Hybrid SWA | 高效推理 |
| MiniMax-M1 | Lightning Attention | 100K token 生成 FLOPs 仅为标准的 25% |

**LongCat-Flash 的 Zero-Computation Experts**：
- 部分专家的 FFN 输出直接设为零
- 模型学会根据 token 难度动态分配计算
- 简单 token 激活更少专家，复杂 token 激活更多

**MiMo-V2-Flash 的 MTP 方案**：
- 训练时使用 1 层 MTP，post-training 复制为 3 层
- 低熵任务（如代码生成）可达 3.6 acceptance length
- 实现 2.53x 推理加速

**MiniMax-M1 的 Lightning Attention**：
- 7:1 hybrid 配置（7 层 linear + 1 层 softmax）
- 在超长上下文场景下效率优势显著

##### 长上下文支持

| 模型 | 最大上下文 | 方案 |
|------|------------|------|
| MiniMax-M1 | 1M tokens | Lightning Attention (7:1 hybrid) |
| DeepSeek-V3 | 128K | MLA + Context Extension |
| Kimi K2 | 128K | MLA + 64 heads（优化长上下文效率） |
| MiMo-V2-Flash | 128K | Hybrid SWA (W=128 + sink) |
| Step 3.5 Flash | 128K | 3:1 Hybrid SWA (W=512) + Gated Attention |
| GLM-5 | 128K | MLA → DSA (content-aware sparse) + Hierarchical CM |

> [!comparison] 长上下文方案对比
> - **MLA**：通过压缩 KV cache 支持长上下文，但仍是 $O(n^2)$ 复杂度
> - **Lightning Attention**：$O(n)$ 复杂度，原生支持超长上下文，MiniMax-M1 达到 1M tokens
> - **Hybrid SWA**：局部 attention + 稀疏全局 attention，MiMo 和 Step 3.5 Flash 采用不同配置

---

## Part III: Benchmark 全景

### 3.1 数学推理（AIME 2025）

| 模型 | AIME 2025 | 备注 |
|------|-----------|------|
| LongCat-2601 | 99.6 (w/ tools) | Heavy Thinking Mode |
| Step 3.5 Flash | 97.3 | 11B 激活参数 |
| MiMo-V2-Flash | 94.1 | 15B 激活参数 |
| Ring-1T | 93.40 | 万亿参数 |
| Qwen3-235B | 81.5 | Thinking Mode |
| Kimi K2 | 49.5 (Avg@64) | Non-thinking |

### 3.2 代码能力（SWE-Bench）

| 模型 | SWE-Bench Verified | 备注 |
|------|-------------------|------|
| GLM-5 | 77.8% | 异步 Agentic RL + 10K+ 环境 |
| Step 3.5 Flash | 74.4% | 11B 激活参数 |
| MiMo-V2-Flash | 73.4% | 15B 激活参数 |
| Kimi K2 | 65.5% | 大规模 Agentic 数据 |
| GLM-4.5 | 64.2% | Expert Model Iteration |

### 3.3 Agent 能力

| 模型 | τ²-Bench | BrowseComp | 备注 |
|------|----------|------------|------|
| Step 3.5 Flash | 88.2% | 69.0% (w/ Ctx) | 11B 激活参数 |
| GLM-5 | - | 75.9% (w/ CM) | Hierarchical Context Management |
| LongCat-2601 | 67.1% | 73.1% (heavy) | Environment Scaling |
| GLM-4.5 | 70.1% | 26.4% | Expert Model Iteration |
| Kimi K2 | 66.1% | - | Agentic 数据合成 |

---

## Part IV: 洞察与趋势

### 4.1 技术共识

1. **MoE + Loss-Free Balancing**：所有旗舰模型的标配，DeepSeek-V3 首创的方案被广泛采用
2. **GRPO 家族**：替代 PPO 成为 RL 训练的主流，各家在此基础上发展出 DAPO、CISPO、GSPO、LPO、IcePop、MIS-PO 等变体
3. **MTP**：训练目标 + 推理加速的双重价值，从 DeepSeek-V3 的创新变成标配
4. **Thinking Mode**：从实验走向产品化，Qwen3 首创双模式统一
5. **Hybrid Attention**：SWA + Full Attention 的混合架构成为新趋势

### 4.2 差异化路线

| 维度 | 路线 A | 路线 B |
|------|--------|--------|
| Attention | MLA（DeepSeek, Kimi）/ MLA→DSA（GLM-5） | Hybrid SWA（MiMo, Step 3.5 Flash）/ Lightning（MiniMax） |
| RL 粒度 | Token 级（GRPO, DAPO, MIS-PO） | Sequence/Sentence 级（GSPO, LPO） |
| Post-training | 多阶段 RL（Qwen3） | Expert Model Iteration（GLM-4.5）/ MOPD（MiMo）/ 异步 Agentic RL（GLM-5） |
| 长上下文 | 压缩 KV（MLA） | 稀疏 Attention（SWA, Lightning） |
| 稀疏度 | 中等（18-32x） | 高稀疏度（48x，Kimi K2） |

### 4.3 未来方向

1. **更高稀疏度 MoE**：Kimi K2 的 48x 稀疏度证明了更激进稀疏化的可行性
2. **Agentic 能力**：从 benchmark 到真实世界应用，GLM-5 的 SWE-bench 77.8% 和 BrowseComp 75.9% 标志着开源模型在 agentic engineering 上正式进入与闭源模型竞争的阶段
3. **训练-推理一致性**：IcePop、MIS-PO、GSPO 等方案从不同角度解决 MoE RL 训练的核心挑战
4. **异步 RL 基础设施**：GLM-5 和 LongCat 的 DORA 证明异步训练是 agentic RL 的必经之路——long-horizon 任务的 rollout 时间极度不均衡，同步训练效率太低
5. **Test-time Compute Scaling**：Heavy Thinking Mode（LongCat）、Thinking Budget（Qwen3）等新范式
5. **效率极限**：MiMo-V2-Flash（15B）和 Step 3.5 Flash（11B）证明小激活参数可达 frontier 水平

---

## 延伸阅读

**核心方法笔记**：
- [[GRPO]] — 所有 2025 模型 RL 训练的基础
- [[Multi-head Latent Attention]] — DeepSeek 系架构的核心
- [[Loss-Free Load Balancing]] — MoE 负载均衡的标准方案
- [[Multi-Token Prediction]] — 训练 + 推理的双重价值
- [[IcePop]] — 双边校准解决 MoE 训练-推理不一致（Ring-1T, GLM-5）

**RL 算法演进**：
- [[DAPO]] — 解决 entropy collapse
- [[CISPO]] — 保留低概率 token 的梯度（MiniMax-M1）
- [[GSPO]] — 序列级重要性采样
- [[LPO]] — 句子级策略优化（Ling 2.0）
- [[IcePop]] — 万亿参数 MoE 的稳定训练（Ring-1T）
- [[MIS-PO]] — 双层二元掩码（Step 3.5 Flash）

**训练稳定性**：
- [[Training-Inference Mismatch]] — MoE RL 的核心挑战
- [[Rollout Routing Replay]] — 路由重放解决方案（MiMo-V2-Flash）
- [[Entropy Collapse]] — RL 训练的常见失败模式

**Post-training 范式**：
- [[MOPD]] — 多教师在线蒸馏（MiMo-V2-Flash）
- [[On-Policy Distillation]] — Strong-to-Weak 蒸馏的理论基础
- [[Agentic RL]] — Agent 能力的 RL 训练
