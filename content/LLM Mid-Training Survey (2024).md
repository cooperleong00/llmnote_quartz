---
type: paper
title: "A Survey on LLM Mid-Training"
authors:
  - Chengying Tu
  - Xuemiao Zhang
  - Rongxiang Weng
  - Rumei Li
  - Chen Zhang
  - Yang Bai
  - Hongfei Yan
  - Jingang Wang
  - Xunliang Cai
affiliations:
  - Peking University
  - Meituan
year: 2024
arxiv: "2510.23081"
core_method: "[[Mid-Training]]"
description: 第二个 mid-training 综述，提供正式定义并建立优化框架（数据策划、训练策略、模型架构），分析主流模型的目标驱动实现
tags:
  - mid-training
  - survey
  - post-training
  - data
  - optimization
created: 2026-03-23
updated: 2026-03-23
---

# LLM Mid-Training Survey (2024)

北京大学和美团联合发布的第二个 [[Mid-Training]] 综述论文，提供了 mid-training 的正式定义，并建立了涵盖数据策划、训练策略、模型架构优化的完整优化框架，分析主流模型的目标驱动实现。

## 核心贡献

### 1. Mid-Training 的正式定义

> [!intuition] Mid-Training 的定位
>
> **定义**：Mid-training 是 pre-training 和 post-training 之间的关键桥梁，具有以下特征：
> - **中等计算需求**：介于 pre-training 和 post-training 之间
> - **目标数据利用**：使用大规模、有针对性的高质量数据
> - **双向能力平衡**：
>   - **前向传播**：通过课程引导的领域数据暴露，传播专业能力潜力
>   - **后向保留**：通过保留通用数据比例，保持通用能力
>
> **与 Continued Pretraining 的区别**：
> - **Mid-training**：有意识的发展阶段，具有过渡意图，混合 pre-training 数据比例，继承学习率动态，防止灾难性遗忘
> - **Continued Pretraining**：用领域特定数据扩展 pre-training，不考虑原始优化器状态或分布保留，可能导致通用能力丧失

### 2. 优化框架

本文建立了五个互联组件的统一优化框架：

#### 数据策划（Data Curation）

> [!example] 端到端数据工作流
>
> **1. 数据收集**（Data Collection）：
> - 通用来源：CommonCrawl、Wikipedia、C4、Pile、Dolma、RedPajama、Matrix
> - 专业格式：QA pairs（Stack Exchange QA、RedStone、MegaMath）
> - 目标：提升质量 + 保持分布连续性
>
> **2. 数据合成**（Data Synthesis）：
> - **蒸馏**（Distillation）：
>   - 风格改写：将低质量语料转换为高密度表达
>   - 扩散合成：设计 prompts 生成 QA pairs/指令数据
>   - 低资源语言翻译、多模态合成
> - **提取**（Extraction）：从语料中提取自然 QA pairs
> - **演化**（Evolution）：通过循环迭代生成增强问题和解决方案
>
> **3. 数据选择**（Data Selection）：
> - **目标采样**：下采样不相关领域，上采样领域特定内容
> - **评分过滤**：FastText、FineWeb-Edu、QuRater
>
> **4. 数据去污染**（Data Decontamination）：
> - N-gram 匹配（主流方法）
> - 基于嵌入的方法（语义敏感但计算昂贵）
> - 混合方法（n-gram + 最长公共子序列）
>
> **5. 数据混合**（Data Mixture）：
> - 平衡通用数据和领域数据
> - 动态调整混合比例

#### 训练策略（Training Strategies）

> [!example] 学习率调度和多阶段训练
>
> **学习率调度**：
> - Linear decay
> - Cosine decay
> - Multi-stage schedulers
> - Adaptive schemes
>
> **多阶段训练**：
> - 单阶段 annealing
> - 多阶段 annealing（渐进式能力提升）
> - Annealing + 长上下文扩展
> - 重复训练并平均（OLMo-2）

#### 模型架构优化（Model Architecture Optimization）

> [!example] 长上下文扩展的架构调整
>
> **位置编码调整**：
> - Position Interpolation
> - NTK-aware interpolation
> - YaRN
> - ABF
>
> **注意力机制优化**：
> - Hybrid attention（self-attention + SSM + lightning-attention）
> - Interleaving global and local attentions

#### Decay Scaling Laws

> [!math] 学习率衰减的缩放规律
>
> 研究学习率衰减阶段的最优持续时间和形状如何随模型大小、架构、优化器变化。

#### 评估（Evaluation）

**常用基准**：
- **通用能力**：MMLU、HellaSwag、ARC
- **推理**：GSM8K、MATH、BBH
- **代码**：HumanEval、MBPP、LiveCodeBench
- **长上下文**：RULER、LongBench
- **多语言**：MGSM、XLSum

### 3. 目标驱动的实现

本文分析了主流模型如何通过 mid-training 实现特定目标：

**核心认知技能**：
- 数学/STEM
- 推理

**任务执行**：
- 指令遵循
- 编码
- Agent 行为

**可扩展性**：
- 长上下文处理
- 多语言

## 关键洞察

### Mid-Training 的效率优势

> [!warning] 数据效率和性能提升
>
> **实证证据**：
> - Mid-training 比 pre-training 用更少的数据和计算实现更陡峭的性能提升
> - 高质量数据在 mid-training 阶段的边际效用更高
>
> **原因**：
> - 模型已经具备广泛的语言模式
> - 高质量领域数据直接针对目标能力
> - 学习率退火稳定优化

### 数据合成的重要性

> [!example] 合成数据的战略重要性
>
> **优势**：
> - 结构化和渐进式学习
> - 与推理上下文对齐
> - 解决数据稀缺、多样性不足、质量限制
>
> **代表性模型**：
> - **phi-4**：强调合成数据的优势
> - **OLMo-2**：mid-training 使用的领域特定高质量数据主要是合成的

### 多阶段训练的必要性

> [!intuition] 为什么需要多阶段？
>
> **单阶段的局限**：
> - 难以同时优化多个目标
> - 数据分布调整需要渐进式
>
> **多阶段的优势**：
> - 渐进式能力提升
> - 不同阶段针对不同目标
> - 更好的稳定性和泛化

## 与 Shopee Survey 的对比

| 维度 | Shopee Survey (2024) | PKU-Meituan Survey (2024) |
|------|---------------------|--------------------------|
| **定义** | 三维分类法（数据、调度、上下文） | 正式定义 + 与 Continued Pretraining 的区别 |
| **框架** | 三个维度 | 五个组件（数据、策略、架构、缩放、评估） |
| **数据** | 数据类型总结 | 完整的端到端数据策划流程 |
| **理论** | GNS、IB、CL | 更侧重实践框架 |
| **实现** | SOTA 模型的数据混合 | 目标驱动的实现分析 |
| **深度** | 宏观总结 | 更详细的方法论和工具 |

**互补性**：
- Shopee Survey 提供理论基础和宏观视角
- PKU-Meituan Survey 提供详细的实践框架和方法论

## 数据策划的详细方法

### 数据合成方法

**蒸馏方法**：
1. **风格改写**：
   - 将低质量、噪声语料转换为高密度表达
   - 与下游风格多样性对齐（QA、指令、对话）
   - 提升整体数据质量

2. **扩散合成**：
   - 设计特定 prompts 生成大规模 QA pairs/指令数据
   - 利用中介（如 personas）增强多样性和规模
   - **BoostQA**：识别 STEM 和高难度任务的不足，提出大规模多样化合成管道
   - **LinkQA**：利用知识点图上的路径采样构建多种子 QA 合成

3. **代码合成**：
   - 基于开源代码片段生成代码指令
   - 将代码抽象为概念进行指令构建

**提取方法**：
- **WebInstruct**：recall → extract → refine 管道，从 web 内容收集高质量 QA pairs
- **MegaMath-Synthetic**：结合 QA 提取和解决方案精炼，提升数学推理数据的多样性和质量

**演化方法**：
- **MathGenie**：从种子数据增强解决方案，反向翻译为新的数学问题，生成验证答案

### 数据选择瓶颈

**当前挑战**：
- 将 pre-training 数据过滤为高质量通用数据集
- 主要通过多个评分器的协同组合来挑选最优样本

**解决方案**：
- 合成或结构化专业数据受益于更有针对性的过滤策略
- 能够高效提取领域对齐的优质内容

## 实践指南

### 数据策划最佳实践

**数据收集**：
- 保持与 pre-training 数据分布的连续性
- 收集专业格式数据（QA pairs、指令数据）
- 初步清洗和质量过滤

**数据合成**：
- 优先使用 LLM 驱动的合成方法
- 结合蒸馏、提取、演化三种方法
- 严格的质量过滤或精炼

**数据选择**：
- 目标采样调整数据分布
- 使用多个评分器协同过滤
- 针对合成数据使用特定过滤策略

**数据去污染**：
- N-gram 匹配作为主流方法
- 考虑混合方法（n-gram + LCS）
- 平衡假阳性和假阴性

**数据混合**：
- 保留 30-40% 通用数据
- 动态调整领域数据比例
- 根据目标能力调整混合策略

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
- [[Data Decontamination]] — 数据去污染

**优化相关**：
- [[Learning Rate Schedule]] — 学习率调度策略
- [[Annealing]] — 学习率退火技术

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/2510.23081/2510.23081|LLM Mid-Training Survey 完整论文]] — 包含详细的方法论和案例研究

**相关综述**：
- [[Mid-Training Survey (2024)]] — Shopee 的 mid-training 综述（理论视角）

**实证研究**：
- [[PRISM (2026)]] — Mid-training 的系统实验研究

**代表性模型**：
- phi-3.5、phi-4 — 多语言和长上下文集成
- Yi-Lightning — 受控数据分布转移
- OLMo-2 — 课程学习和重复训练
