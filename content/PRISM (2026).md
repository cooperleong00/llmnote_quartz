---
type: paper
title: "PRISM: Demystifying Retention and Interaction in Mid-Training"
authors:
  - Bharat Runwal
  - Ashish Agrawal
  - Anurag Roy
  - Rameswar Panda
affiliations:
  - IBM Research
  - MIT-IBM Watson AI Lab
year: 2026
arxiv: "2603.17074"
core_method: "[[Mid-Training]]"
description: 通过跨 7 个基础模型的系统实验，揭示 mid-training 如何通过 ~27B 高质量 tokens 提升推理能力，并为 RL 创造有效的权重配置
tags:
  - mid-training
  - post-training
  - rlhf
  - reasoning
  - data
created: 2026-03-23
updated: 2026-03-23
---

# PRISM (2026)

PRISM（Demystifying Retention and Interaction in Mid-Training）是 IBM Research 和 MIT-IBM Watson AI Lab 的系统性研究，通过跨 7 个基础模型（Granite、LLaMA、Mistral、Nemotron-H）、两种架构（dense Transformer 和 attention-Mamba hybrid）、3B-24B 参数规模的受控实验，全面揭示了 [[Mid-Training]] 的设计选择及其与 [[RLHF]] 的交互机制。

## 核心贡献

### 1. Mid-Training 显著提升推理能力

使用 ~27B 高质量 tokens 进行 mid-training：
- **数学**：+15 到 +40 分（AIME、MATH500）
- **代码**：+5 到 +12 分（LiveCodeBench、Codeforces）
- **科学**：+6 到 +13 分（GPQA-Diamond）
- **通用能力**：保持不变（Leaderboard-V1/V2）

### 2. Mid-Training 显著增强 RL 效果

> [!example] PRISM → RL 的协同效果
>
> **完整流程**（Base → Mid-Training → RL）：
> - 六个推理基准宏平均：从 <12 提升到 29-42（**3-4× 改进**）
>
> **直接 RL**（Base → RL）：
> - AIME 分数接近 0（几乎无效）
>
> **结论**：RL 需要 mid-training 创造的"可优化起点"才能有效工作。

### 3. 数据组合在 Mid-Training 阶段最关键

> [!warning] 关键发现
>
> **Mid-Training 阶段的数据决策**：
> - Math+Code → Math+Code+Science：AVG +3 到 +6 分
> - 在 mid-training 加入 Science 数据 → RL 阶段 GPQA-Diamond +17 到 +28 分
>
> **RL 阶段的数据决策**：
> - 改变 RL 数据混合：<2 分差异
>
> **启示**：应该在 mid-training 阶段优化数据组合，而非 RL 阶段。

### 4. 机制分析：密集重构 vs 稀疏调整

> [!math] 权重变化的量化分析
>
> **Mid-Training**：
> - 密集重构 >90% 的模型参数
> - 创造一个"可优化的权重配置"
>
> **RL**：
> - 稀疏调整 ~5% 的参数
> - 权重变化幅度比 mid-training 小 370-580×
> - 保持 mid-training 的表征几何结构（CKA >0.998）
>
> **起点不变性**（Starting-point Invariance）：
> - RL 对所有模型应用相同的权重更新（目标相同的 ~5% 参数）
> - 但只在 mid-trained 模型上成功
> - 说明 mid-training 创造的权重配置是 RL 成功的必要条件

## 实验设计

### 模型覆盖

| 模型家族 | 架构类型 | 参数规模 |
|---------|---------|---------|
| Granite | Dense Transformer | 3B, 8B |
| LLaMA | Dense Transformer | 3B, 8B |
| Mistral | Dense Transformer | 7B |
| Nemotron-H | Attention-Mamba Hybrid | 8B, 24B |

### 数据混合配置

**三层渐进配置**：
1. **Math-only**：数学推理 + 通用 web
2. **Math+Code**：添加代码推理和 web
3. **Math+Code+Science**：添加科学推理

**数据源**：
- **领域推理**：OpenR1 (MoT)、OpenCodeReasoning-2、OpenThoughts3
- **领域 web**：Megamath-Web-Pro、RefinCode、StarCoder2
- **通用数据**：DCLM-EDU（质量分数 ≥3）
- **指令数据**：WildChat-1M (GPT-4)、Tulu-3 SFT Personas

### 评估基准

| 类别 | 基准 | 目的 |
|------|------|------|
| 通用能力 | LB-V1, LB-V2 | 检测泛化退化 |
| 长上下文 | RULER | 确保长上下文能力保持 |
| 代码 | LiveCodeBench, Codeforces | 可执行程序合成 |
| 数学 | AIME, MATH500 | 数学推理 |
| 科学 | GPQA-Diamond | 专家级科学推理 |

## 关键洞察

### 1. RL 扩展可解性边界

对 Granite-3.3，RL 在 mid-trained 模型上逐步解决最初无法解决的问题，训练曲线在数百步后仍未饱和，表明进一步收益可期。

### 2. 长上下文能力的权衡

在 8K context 进行 mid-training 会降低长上下文能力，但可通过以下方式恢复：
- 短暂的 context extension phase
- Model merging

### 3. 架构泛化性

Dense Transformer 和 attention-Mamba hybrid 都从 PRISM 中一致受益，从 3B 到 24B 参数规模。

### 4. RL 优化动态

- **前置加载**：大部分权重变化发生在前 ~200-400 步
- **渐进扩展**：活跃参数集从 ~1.5% 逐步增长到 ~5%

## 与相关工作的对比

| 工作 | 焦点 | 局限 |
|------|------|------|
| OLMo (2025) | 高质量 annealing | 评估范围窄，未系统研究 RL 交互 |
| OctoThinker (2025) | Mid-training + RL | 主要聚焦数学，缺乏跨领域分析 |
| Liu et al. (2025) | 分布桥梁 | 小规模实验 |
| Zhang et al. (2025) | 受控框架 | 小规模，未覆盖多模型家族 |
| **PRISM** | **全面系统研究** | **3B-24B 规模，4 模型家族，2 架构，多阶段流程** |

## 局限与未来方向

**当前局限**：
- 主要聚焦推理任务（数学、代码、科学）
- 对其他能力（创意写作、多语言）的影响未充分探索
- 所有模型都在 pretraining 阶段包含长上下文训练

**未来方向**：
- 探索更大的 token 预算（>27B）
- 研究 mid-training 对非推理任务的影响
- 分析 mid-training 与不同 pretraining 配置的交互

## 相关概念

**核心方法**：
- [[Mid-Training]] — 本文系统研究的核心概念

**训练流程**：
- [[Pretraining]] — Mid-training 的前置阶段
- [[Post-training]] — Mid-training 的后续阶段
- [[RLHF]] — 与 mid-training 协同的 RL 训练

**数据相关**：
- [[Data Synthesis]] — 生成高质量训练数据
- [[Preference Data]] — RL 阶段使用的偏好数据

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/2603.17074/2603.17074|PRISM 完整论文]] — 包含详细实验和分析

**相关论文**：
- OLMo Team (2025) — Mid-training 作为 annealing 阶段
- Wang et al. (2025, OctoThinker) — Mid-training 与 RL 的交互
- Liu et al. (2025) — Mid-training 作为分布桥梁
- Zhang et al. (2025) — 受控实验框架
