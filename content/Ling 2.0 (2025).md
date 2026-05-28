---
type: paper
description: Inclusion AI 的 1T 参数推理模型 technical report，采用高稀疏度 MoE 架构（3.5% 激活率）实现 7 倍计算效率，提出 Ling Scaling Laws 指导万亿参数设计，创新 LPO 句子级策略优化和 Evo-CoT 渐进推理训练
aliases:
  - Ling-1T
  - Ling 2.0
  - 灵犀 2.0
prerequisites:
  - "[[Mixture of Experts]]"
  - "[[Multi-Token Prediction]]"
  - "[[GRPO]]"
tags:
  - technical-report
  - moe
  - reasoning
  - post-training
created: 2025-02-04
updated: 2026-02-04T18:54
---

# Ling 2.0 (2025)

Ling 2.0 是 Inclusion AI 发布的推理导向基础模型系列，核心目标是在万亿参数规模下实现高效推理。通过高稀疏度 [[Mixture of Experts|MoE]] 架构（256 专家，8 激活，3.5% 激活率）和系统性的 Ling Scaling Laws，Ling-1T 以 51B 激活参数实现了与 dense 模型相当的性能，计算效率提升约 7 倍。

> [!paper] 论文信息
> - **标题**: Ling 2.0 Technical Report: Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation
> - **机构**: Inclusion AI
> - **发布**: 2025
> - **模型**: Ling-mini-2.0 (16B), Ling-flash-2.0 (103B), Ling-1T (1000B)

## 核心贡献

### 1. 高稀疏度 MoE 架构

Ling 2.0 采用统一的"高稀疏度、细粒度"设计：

| 配置 | Ling-mini-2.0 | Ling-flash-2.0 | Ling-1T |
|------|---------------|----------------|---------|
| 总参数 | 16B | 103B | 1000B |
| 激活参数 | 1.4B | 6.1B | 51B |
| 专家数 | 256 | 256 | 256 |
| 激活专家 | 8 + 1 shared | 8 + 1 shared | 8 + 1 shared |
| 激活率 | ~3.5% | ~3.5% | ~3.5% |

关键设计选择：
- **Aux-loss-free 负载均衡**：采用类似 [[DeepSeek-V3 (2024)|DeepSeek-V3]] 的 [[Loss-Free Load Balancing]] 策略，通过动态 bias 调整实现负载均衡
- **[[Multi-Token Prediction|MTP]]**：每个模型配置 1 个 MTP 层，loss weight 设为 0.1，在 code 和 math 任务上持续提升性能
- **First-K-Dense**：前 1/1/4 层（mini/flash/1T）设为 dense 层，减少参数量同时改善路由平衡

### 2. Ling Scaling Laws

基于超过 1000 个实验，Ling 建立了系统性的 MoE scaling laws：

> [!intuition] 核心发现
> 1. **激活率是效率的主要驱动因素**：效率增益随稀疏度增加呈幂律增长
> 2. **专家粒度是非线性调节器**：最优激活专家数在 8-12 范围
> 3. **计算预算有放大效应**：随着训练计算量增加，MoE 的效率优势更加显著

效率杠杆（Efficiency Leverage）公式：

$$EL(A, G, C) = \hat{A}^{\alpha + \gamma(\log G)^2 + \beta \log G}$$

其中 $A$ 是激活率，$G$ 是专家粒度，$C$ 是计算预算。Ling 2.0 的配置预测可实现 **7 倍效率杠杆**。

**Wind Tunnel 实验**：通过标准化的小规模实验流程，可以用不到 1% 的训练成本验证新技术在万亿参数规模的效果。

### 3. 预训练策略

**推理导向的数据配比**：
- 预训练 20T tokens，推理数据（Ling Math + Ling Code）占比从 32% 逐步提升到 46%
- 在 reasoning benchmarks 上平均提升 5-8%

**Mid-training 推理预激活**：
- 扩展上下文窗口
- 引入 Chain-of-Thought 数据"预激活"推理能力
- 为后续 SFT 和 RL 提供更稳定的基础

**WSM (Warmup-Stable-Merge) Scheduler**：
- 用 checkpoint merging 替代传统的 learning rate decay
- 在各 benchmark 上平均提升 1-2%，且优势延续到后训练阶段

### 4. 后训练创新

#### Decoupled Fine-Tuning (DFT)

通过差异化 system prompt 建立两种响应模式：
- **Instant Response** (`detailed think off`)：快速响应简单问题
- **In-Depth Reasoning** (`detailed think on`)：深度推理复杂问题

#### Evolutionary Chain-of-Thought (Evo-CoT)

渐进式深化推理能力的训练范式：

$$\pi_{t+1} = \arg \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}} \big[ \mathcal{J}(R(x, y), \theta) - \beta \cdot \text{KL}(\pi_{\theta} \| \pi_{\text{ref}}) \big]$$

奖励函数包含：
- **Accuracy** $R_{\text{correctness}}$：答案正确性
- **Dynamic Length** $R_{\text{length}}$：难度自适应的长度控制（简单问题鼓励简洁，复杂问题允许详细推理）
- **Format** $R_{\text{format}}$：惩罚显式 `<think>` 标记（-0.5）

#### Linguistic-unit Policy Optimization (LPO)

> [!intuition] 核心思想
> 将句子（而非 token 或整个序列）作为策略更新的基本单元，在语义边界上进行优化。

LPO 目标函数：

$$\mathcal{J}_{\text{LPO}} = \frac{1}{\sum_i |y_i|} \sum_i \sum_k |s_{i,k}| \cdot \min\left(r_{i,k}(\theta) \hat{A}_i, \text{clip}(r_{i,k}(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_i\right)$$

其中 $s_{i,k}$ 是第 $i$ 个响应的第 $k$ 个句子，$r_{i,k}(\theta)$ 是句子级重要性比率。

> [!comparison] 与 [[GRPO]]/[[GSPO]] 对比
> - **GRPO**：token 级更新，粒度过细
> - **GSPO**：sequence 级更新，粒度过粗
> - **LPO**：sentence 级更新，与自然语义边界对齐，在 AIME 2025 上显著优于 baseline

#### Group Arena Reward (GAR)

针对开放式主观任务的奖励机制：
- 将同一 policy 的多个响应放入"竞技场"
- 通过 round-robin 两两对比替代绝对评分
- 有效降低奖励噪声和方差

### 5. 基础设施

- **FP8 全流程训练**：最大规模的开源 FP8 训练模型，精度损失 <= 0.25%（900B tokens 后与 BF16 对比）
- **异构细粒度流水线**：交错 1F1B 调度 + 部分重计算，吞吐量提升约 40%
- **4C 原则**：Correct, Consistent, Complete, Co-Design

## 评估结果

Ling-1T 在推理任务上建立了新的 Pareto 前沿：
- **AIME 2025**: 43.75% Pass@1（显著领先同规模模型）
- **MATH**: 82.52%
- **HumanEval-Plus**: 76.22%

三个模型遵循 Ling Scaling Law 的预测，验证了"每个激活都有价值"的设计理念。

## 局限性

1. **长上下文效率**：GQA 架构在长上下文场景下效率受限，正在探索 linear/sparse attention
2. **推理深度**：有效推理长度和深度仍有提升空间
3. **复杂指令跟随和 Agentic 能力**：仍在开发中

## 与 Ring 系列的关系

Ling 2.0 是 Inclusion AI 的 instruct 模型系列（reflex-grade non-thinking），而 [[Ring-1T]] 是基于相同 Ling 2.0 base 构建的 deep thinking 模型系列。

## 延伸阅读

**相关 Technical Reports**：
- [[DeepSeek-V3 (2024)]] - 类似的高稀疏度 MoE 设计和 aux-loss-free 负载均衡
- [[Kimi K2 (2025)]] - 另一个 1T 参数 MoE 模型，侧重 Agentic 能力

**核心技术**：
- [[Mixture of Experts]] - MoE 架构基础
- [[Multi-Token Prediction]] - MTP 训练目标
- [[Loss-Free Load Balancing]] - 无辅助损失的负载均衡
- [[GSPO]] - 序列级策略优化（LPO 的对比方法）
