---
type: paper
description: 首个开源大规模 hybrid-attention 推理模型，结合 Lightning Attention 实现高效 test-time compute scaling，提出 CISPO 算法加速 RL 训练
aliases:
  - MiniMax-M1
authors:
  - MiniMax Team
affiliations:
  - MiniMax
year: 2025
arxiv: "2506.13585"
github: https://github.com/MiniMax-AI/MiniMax-M1
prerequisites:
  - "[[Mixture of Experts]]"
  - "[[GRPO]]"
  - "[[RLHF]]"
tags:
  - post-training
  - architecture
  - attention
  - reasoning
  - inference
created: 2026-02-04
updated: 2026-02-04T18:54
---

# MiniMax-M1 (2025)

MiniMax-M1 是首个开源的大规模 **hybrid-attention 推理模型**，通过 Lightning Attention 机制实现高效的 test-time compute scaling。模型基于 MiniMax-Text-01 开发，总参数量 456B，每 token 激活 45.9B 参数，原生支持 **1M token 上下文**（DeepSeek R1 的 8 倍）和 **80K token 生成长度**。

> [!intuition] 核心创新
> 1. **Lightning Attention**：hybrid 架构（7 层 linear attention + 1 层 softmax attention），在 100K token 生成时 FLOPs 仅为 DeepSeek R1 的 25%
> 2. **[[CISPO]]**：新型 RL 算法，clip importance sampling weights 而非 token updates，训练效率提升 2x
> 3. **多样化 RL 数据**：涵盖数学、逻辑推理、竞赛编程、软件工程等可验证任务，以及使用 GenRM 的通用任务

---

## 架构设计

### Hybrid Attention 机制

MiniMax-M1 采用 **hybrid MoE + Lightning Attention** 架构：

- **基础结构**：每 7 层 Transnormer block（使用 Lightning Attention）后接 1 层标准 Transformer block（使用 softmax attention）
- **参数规模**：456B 总参数，45.9B 激活参数，32 experts
- **上下文能力**：原生支持 1M token 输入，80K token 输出

> [!math] 计算效率对比
> 传统 softmax attention 的复杂度为 $O(n^2)$，而 Lightning Attention 为 $O(n)$。在长序列生成时：
> - 64K tokens：FLOPs < DeepSeek R1 的 50%
> - 100K tokens：FLOPs 约为 DeepSeek R1 的 25%

### 与其他模型的上下文对比

| 模型 | 最大输入 | 最大输出 |
|------|----------|----------|
| OpenAI o3 | 200K | 100K |
| Gemini 2.5 Pro | 1M | 64K |
| Claude 4 | 200K | 32K |
| DeepSeek R1 | 128K | 64K |
| **MiniMax-M1-80k** | **1M** | **80K** |

---

## 训练流程

### 1. Continual Pretraining

在 MiniMax-Text-01 基础上，使用 **7.5T tokens** 的 reasoning-intensive 语料继续预训练，增强模型的内在推理能力。

### 2. Supervised Fine-Tuning (SFT)

Cold-start SFT 阶段注入特定的 chain-of-thought 模式，为后续 RL 阶段建立基础。

### 3. Reinforcement Learning

RL 是 M1 开发的核心阶段，通过两个关键创新实现高效 scaling：

**算法创新 - [[CISPO]]**：
- 放弃 trust region 约束，改为 clip importance sampling weights
- 保留所有 token 的梯度贡献，包括低概率的反思 token（如 "However", "Wait", "Recheck"）
- 在 Qwen2.5-32B 实验中，用 50% 训练步数达到 [[DAPO]] 的性能

**架构适配 - Lightning Attention 的 RL 挑战**：

> [!warning] 精度不匹配问题
> 训练和推理 kernel 之间存在精度差异，导致 rolled-out tokens 的概率在两种模式下不一致，阻碍 reward 增长。
>
> **解决方案**：将 LM output head 的精度提升到 FP32，使概率相关性从 ~0.9x 提升到 ~0.99x。

其他工程优化：
- **优化器配置**：$\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-15}$（针对梯度范围 $10^{-18}$ 到 $10^{-5}$）
- **重复检测截断**：当连续 3000 个 token 概率都 > 0.99 时提前终止生成

---

## RL 数据与奖励

### 可验证任务（Rule-based Verification）

| 任务类型 | 数据量 | 数据来源 |
|----------|--------|----------|
| 数学推理 | ~50K | 竞赛级问题，pass@10 在 (0, 0.9) 之间 |
| 逻辑推理 | ~53K | SynLogic 框架合成，41 种任务（如 cipher、Sudoku） |
| 竞赛编程 | ~30K | 在线 OJ 平台，LLM 生成测试用例 |
| 软件工程 | 数千 | GitHub issues/PRs，容器化沙箱环境 |

### 通用任务（Model-based Feedback）

- **有标准答案的任务**：使用 Generative Reward Model (GenRM) 作为 verifier，五级评分
- **无标准答案的任务**：instruction-following、creative writing 等，使用 pairwise comparison（-1/0/1）

> [!intuition] GenRM 偏差处理
> 对于 long CoT 输出，GenRM 可能存在偏差。通过 multiple-blind consistent judgment、position-switched consistent judgment 等方法优化训练数据。

---

## Benchmark 表现

### 核心结果

MiniMax-M1 在多个领域与 DeepSeek R1、Qwen3-235B 等顶级开源模型竞争：

**数学推理**：
- AIME 2024: 86.0%（M1-80k），仅次于 DeepSeek R1-0528 的 91.4%
- MATH-500: 96.8%

**软件工程**：
- SWE-bench Verified: 56.0%（M1-80k），接近 DeepSeek R1-0528 的 57.6%

**长上下文理解**（M1 的优势领域）：
- OpenAI-MRCR (128k): 76.1%，超越 o3 (56.5%)、Claude 4 (48.9%)
- OpenAI-MRCR (1M): 58.6%，与 Gemini 2.5 Pro (58.8%) 持平
- LongBench-v2: 61.5%，超越所有开源模型

**Agentic Tool Use**：
- TAU-bench (airline): 62.0%，超越 Gemini 2.5 Pro (50.0%)

### RL Scaling 效果

训练过程中观察到性能与响应长度的强相关性：
- AIME 2024 准确率从 68% 提升到 80%
- 平均响应长度超过 20,000 tokens

---

## 训练成本

> [!example] 效率数据
> - **硬件**：512 H800 GPUs
> - **时间**：3 周完成完整 RL 训练
> - **成本**：约 $534,700 USD（租赁成本）

这一效率得益于 Lightning Attention 的低计算成本和 [[CISPO]] 的高训练效率。

---

## 局限性

> [!warning] 已知不足
> - **数学/编程竞赛**：落后于 DeepSeek R1-0528（AIME 2025: 76.9% vs 87.5%）
> - **事实性**：SimpleQA 得分 18.5%，低于 DeepSeek R1 的 30.1%
> - **GPQA Diamond**：70.0%，低于多数顶级模型

---

## 延伸阅读

**核心方法**：
- [[CISPO]] — 本文提出的 RL 算法，详细推导和对比

**相关架构**：
- [[Mixture of Experts]] — MiniMax-M1 的稀疏激活基础
- [[Flash Attention]] — 另一种高效 attention 实现
- [[Sparse Attention]] — 稀疏注意力机制

**RL 训练相关**：
- [[GRPO]] — CISPO 的基础算法
- [[DAPO]] — 另一种处理 token clipping 的方法
- [[Reward Model]] — 通用任务的反馈来源
