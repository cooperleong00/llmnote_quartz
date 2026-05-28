---
type: paper
title: "Mid-Training of Large Language Models: A Survey"
authors:
  - Kaixiang Mo
  - Yuxin Shi
  - Weiwei Weng
  - Zhiqiang Zhou
  - Shuman Liu
  - Haibo Zhang
  - Anxiang Zeng
affiliations:
  - Shopee
year: 2024
arxiv: "2510.06826"
core_method: "[[Mid-Training]]"
description: 首个 mid-training 综述，提出数据分布、学习率调度、长上下文扩展三维分类法，系统总结 mid-training 作为 LLM 训练流程中独立阶段的理论基础和实践方法
tags:
  - mid-training
  - survey
  - post-training
  - data
  - optimization
created: 2026-03-23
updated: 2026-03-23
---

# Mid-Training Survey (2024)

Shopee 团队发布的首个 [[Mid-Training]] 综述论文，系统性地将 mid-training 概念化为 LLM 训练流程中的独立阶段，提出三维分类法（数据分布、学习率调度、长上下文扩展），并总结理论基础和实践方法。

## 核心贡献

### 1. 首个 Mid-Training 分类法

> [!example] 三维分类框架
>
> **数据分布**（Data Distribution）：
> - 高质量过滤 web 数据（FineWeb-Edu、DCLM-baseline）
> - 代码和数学内容（Stack、OpenWebMath、FineMath）
> - 指令和 QA 风格数据（UltraChat、EvolInstruct）
> - 合成教科书和知识密集数据（Cosmopedia）
> - 长上下文数据
> - 推理和 CoT 数据
> - Fill-in-Middle (FIM) 数据
>
> **学习率调度**（Learning Rate Scheduling）：
> - Linear/Cosine decay
> - Multi-stage schedulers
> - Adaptive schemes
>
> **长上下文扩展**（Long-Context Extension）：
> - Position Interpolation
> - NTK-aware interpolation
> - YaRN
> - ABF

### 2. 理论基础

> [!intuition] Mid-Training 的三个理论视角
>
> **梯度噪声尺度**（Gradient Noise Scale, GNS）：
> - 高质量数据提升梯度方差，增大 GNS
> - 更大的 GNS 帮助模型逃离尖锐极小值，避免过拟合
> - 在训练后期改善优化，信号稀疏性可能导致收敛停滞
>
> **信息瓶颈**（Information Bottleneck, IB）：
> - 学习率退火阶段，模型逐步减少对噪声特征的依赖
> - 压缩内部状态，保留任务相关信息
> - 高质量监督信号提供更清晰、低熵的指导
> - 从大规模记忆转向抽象、可泛化的表征
>
> **课程学习**（Curriculum Learning）：
> - 早期预训练：多样化、噪声数据（泛化）
> - Mid-training：逐步转向更具挑战性、信息量更大的样本
> - 强化复杂技能（多步推理、代码生成）

### 3. 实践洞察

**数据混合策略**（来自 SOTA 模型）：

| 模型 | 高质量 Web | 代码 | 数学 | 指令/QA | 合成数据 |
|------|-----------|------|------|---------|---------|
| MiniCPM | Dolma, C4, Pile | Stack | OpenWebMath | SlimOrca, UltraChat | — |
| OLMo-2 | DCLM-Baseline | CodeSearchNet | Metamath, GSM8K | FLAN, StackExchange | TuleMath, SynthMath |
| SmolLM2 | DCLM, FineWeb-Edu | Stack-Edu | InfiWebMath-3+, FineMath 4+ | — | Cosmopedia v2 |

**训练阶段设计**（代表性模型）：

- **Qwen3**：3 阶段（30T → 5T 高质量 4K → 数百 B 长上下文）
- **Llama3 405B**：3 阶段（初始 → 800B 长上下文 → 40M annealing）
- **SmolLM2**：5 阶段（3 预训练 + 1 annealing + 1 长上下文扩展）
- **OLMo-2**：2 阶段（5T web → Dolmino Mix 1124，重复训练并平均）

### 4. 评估基准

**常用基准**：
- **通用能力**：MMLU、HellaSwag、ARC
- **推理**：GSM8K、MATH、BBH
- **代码**：HumanEval、MBPP、LiveCodeBench
- **长上下文**：RULER、LongBench
- **多语言**：MGSM、XLSum

## 关键发现

### Mid-Training 的必要性

> [!warning] 为什么需要 Mid-Training？
>
> **问题**：
> - 仅依赖 web 规模语料提供广泛但噪声监督
> - 模型在能力发展上停滞，尽管投入大量计算
> - 推理、编码、长上下文理解的边际收益有限
>
> **解决方案**：
> - 降低学习率 → 缓解梯度方差，稳定收敛
> - 转向精选/合成语料 → 提升每个 token 的边际效用
> - 从记忆转向抽象 → 强调结构化推理、事实性、指令遵循

### 三个维度的相互依赖

> [!intuition] 整体视角
>
> - **没有精炼数据** → 模型缺乏专业化
> - **没有适当调度** → 风险不稳定或低效利用高质量 tokens
> - **没有上下文扩展** → 下游应用范围受限
>
> **结论**：有效的 mid-training 不能通过孤立分析每个因素来理解，而是通过它们对模型性能和效率的综合影响。

### 数据类型的作用

**高质量过滤 Web 数据**：
- 基于内容质量、主题多样性、低毒性、教育价值、最小冗余
- 代表数据集：FineWeb-Edu、DCLM-baseline、peS2o

**代码和数学内容**：
- 显著增强符号推理和程序合成能力
- 代表数据集：Stack v2、OpenWebMath、FineMath 4+

**指令和 QA 风格数据**：
- 增强对人类意图的理解
- 提高响应一致性和推理能力
- 代表数据集：UltraChat、EvolInstruct、OpenOrca

**合成教科书数据**：
- 提供高质量教育内容
- 特别适用于低资源领域（数学、编码、多语言）
- 代表数据集：Cosmopedia v2、TuluMath

**长上下文数据**：
- 改善记忆、连贯性、跨文档推理
- 从现有语料过滤或通过数据合成获得

**推理和 CoT 数据**：
- 提供显式的问题分解演示
- 帮助模型学习结构化、可解释的推理模式
- 在数学或逻辑任务上表现强劲

**Fill-in-Middle (FIM) 数据**：
- 强化双向上下文推理
- 加强长程依赖建模
- 提高数据效率（单个样本多个预测机会）

## 未来研究方向

**开放挑战**：
1. **调度器设计**：最优持续时间和衰减形状因模型大小、架构、优化器而异
2. **数据混合优化**：如何系统地确定最优数据比例
3. **长上下文扩展**：如何在不重启预训练的情况下有效扩展
4. **理论理解**：需要更深入的理论分析支持实践

**潜在方向**：
- 自适应数据混合策略
- 动态学习率调度
- 多模态 mid-training
- 跨架构泛化研究

## 与 PRISM 的对比

| 维度 | Mid-Training Survey (2024) | PRISM (2026) |
|------|---------------------------|--------------|
| **类型** | 综述论文 | 实证研究 |
| **焦点** | 系统性总结现有方法 | 受控实验验证 |
| **理论** | 三个理论视角（GNS、IB、CL） | 机制分析（权重变化、表征相似度） |
| **分类** | 三维分类法（数据、调度、上下文） | 数据组合 + RL 交互 |
| **模型覆盖** | 多个 SOTA 模型（Qwen3、Llama3、OLMo-2 等） | 7 个基础模型（Granite、LLaMA、Mistral、Nemotron-H） |
| **贡献** | 首个分类法 + 实践总结 | 量化分析 + RL 协同效果 |

**互补性**：
- Survey 提供宏观视角和理论框架
- PRISM 提供微观机制和量化证据

## 相关概念

**核心方法**：
- [[Mid-Training]] — 本文系统研究的核心概念

**训练流程**：
- [[Pretraining]] — Mid-training 的前置阶段
- [[Post-training]] — Mid-training 的后续阶段
- [[SFT]] — Post-training 的第一步
- [[RLHF]] — Post-training 的核心

**数据相关**：
- [[Data Synthesis]] — 生成高质量训练数据
- [[Curriculum Learning]] — 渐进式数据分布调整

**优化相关**：
- [[Learning Rate Schedule]] — 学习率调度策略
- [[Gradient Noise Scale]] — 梯度噪声尺度

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/2510.06826/2510.06826|Mid-Training Survey 完整论文]] — 包含详细分类和案例研究

**相关研究**：
- [[PRISM (2026)]] — Mid-training 的实证研究
- MiniCPM (2024) — 早期 mid-training 实践
- OLMo-2 (2024) — 重复训练并平均的策略
- SmolLM2 (2024) — 小模型的 mid-training 实践
