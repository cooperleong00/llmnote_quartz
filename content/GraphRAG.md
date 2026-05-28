---
type: method
description: 基于知识图谱的 RAG 方法，通过图社区结构和层次化摘要实现全局理解问题的回答
aliases:
  - Graph-based RAG
  - 图增强检索生成
prerequisites:
  - "[[RAG Survey (2023)]]"
tags:
  - rag
  - knowledge-graph
  - sensemaking
created: 2026-03-10
updated: 2026-03-10T01:01
---

# GraphRAG

GraphRAG（Graph-based Retrieval-Augmented Generation）是微软研究院提出的一种基于知识图谱的 RAG 方法，专门解决传统 RAG 无法回答的**全局理解问题**（global sensemaking questions）。它通过构建实体知识图谱、检测图社区、生成层次化摘要，将"整个语料库的主题是什么"这类问题转化为可扩展的 map-reduce 过程。

## 动机

> [!intuition] 为什么需要 GraphRAG？
>
> **传统 RAG 的失效场景**：
> - Vector RAG 擅长回答"局部问题"（local questions）：答案可以从少数几个文档片段中找到
> - 但面对"全局问题"（global questions）时失效：
>   - "这个数据集的主要主题是什么？"
>   - "过去十年跨学科研究如何影响科学发现的趋势？"
>
> **问题本质**：
> - 全局问题需要**整合整个语料库的信息**，而非检索几个相关片段
> - 这本质上是**查询导向的摘要任务**（Query-Focused Summarization, QFS），而非检索任务
> - 传统 QFS 方法无法扩展到 RAG 系统索引的文本量级（百万 token 级别）
>
> **GraphRAG 的核心洞察**：
> - 知识图谱的**社区结构**（community structure）天然提供了主题分区（thematic partitioning）
> - 通过预先生成社区摘要，可以将全局问题分解为并行的局部问题（map），再聚合答案（reduce）
> - 这种方法既保留了全局视角，又实现了计算的可扩展性


## 核心机制

GraphRAG 的工作流程分为两个阶段：**索引构建**（indexing）和**查询回答**（query answering）。

### 索引构建：从文本到层次化社区摘要

> [!intuition] 核心思想
> 
> 将文本转化为**带有语义层次的知识图谱**，每个层次的社区都有自己的摘要。这样，全局问题可以在不同粒度上回答。

**六步流程**：

```
文本文档 → 文本块 → 实体/关系 → 知识图谱 → 图社区 → 社区摘要
```

#### 1. 文本分块（Source Documents → Text Chunks）

- 将文档切分为固定大小的文本块
- 设计权衡：更长的块减少 LLM 调用次数（降低成本），但会降低信息召回率

#### 2. 实体与关系提取（Text Chunks → Entities & Relationships）

使用 LLM 从每个文本块中提取：
- **实体**（entities）：人物、地点、组织等命名实体
- **关系**（relationships）：实体之间的连接
- **声明**（claims）：关于实体的重要事实陈述

> [!example] 提取示例
> 
> **输入文本**：
> "NeoChip's shares surged in their first week of trading on the NewTech Exchange. The chipmaker was acquired by Quantum Systems in 2016."
> 
> **提取结果**：
> - 实体：NeoChip（描述：专注于低功耗处理器的上市公司）
> - 实体：Quantum Systems（描述：曾拥有 NeoChip 的公司）
> - 关系：Quantum Systems 在 2016 年收购了 NeoChip，直到 NeoChip 上市
> - 声明：NeoChip 的股票在 NewTech Exchange 首周交易中飙升

**关键技术**：
- 使用 few-shot learning 提供领域特定的示例（如科学、医学、法律领域）
- 这是一种**抽象摘要**（abstractive summarization）：提取的关系和声明可能在原文中并未明确表述

#### 3. 构建知识图谱（Entities & Relationships → Knowledge Graph）

- 将提取的实体作为**节点**（nodes）
- 将关系作为**边**（edges）
- 聚合重复的实体描述（通过 LLM 摘要）
- 关系的重复次数成为**边权重**（edge weights）

**实体匹配策略**：
- 论文使用精确字符串匹配
- 可以使用更软的匹配方法（如模糊匹配）
- GraphRAG 对重复实体有鲁棒性：重复实体通常会在后续步骤中被聚类到一起

#### 4. 社区检测（Knowledge Graph → Graph Communities）

使用 **Leiden 算法**进行**层次化社区检测**：
- 递归地检测社区内的子社区，直到叶子社区无法再分割
- 每一层提供一个**互斥且完备**的节点划分（mutually exclusive, collectively exhaustive）
- 这种层次结构支持**分治式全局摘要**（divide-and-conquer global summarization）

> [!intuition] 为什么社区结构有效？
> 
> 社区检测算法（如 Leiden）会将**紧密连接的实体**聚类在一起。在知识图谱中，紧密连接通常意味着**主题相关性**：
> - 同一主题的实体会频繁共现
> - 社区边界天然对应**主题边界**
> - 层次结构提供了从细粒度到粗粒度的主题视角

#### 5. 生成社区摘要（Graph Communities → Community Summaries）

为每个社区生成**报告式摘要**（report-like summaries），采用**自底向上**的方式：

**叶子社区**（leaf-level communities）：
- 按优先级将元素摘要（节点、边、声明）添加到 LLM 上下文窗口
- 优先级：按边的源节点和目标节点的度数之和排序（即整体显著性）

**高层社区**（higher-level communities）：
- 如果所有元素摘要能放入上下文窗口 → 直接摘要
- 否则 → 用子社区摘要（更短）替换元素摘要（更长），直到适配窗口大小

> [!intuition] 摘要的独立价值
> 
> 社区摘要本身就是理解语料库的有用工具：
> - 用户可以浏览某一层的摘要，寻找感兴趣的主题
> - 然后阅读下一层的详细摘要
> - 这是一种**探索式理解**（exploratory understanding）

### 查询回答：Map-Reduce 处理社区摘要

给定用户查询，使用**三步 map-reduce 流程**生成全局答案：

#### 1. 准备社区摘要（Prepare Community Summaries）

- 选择层次结构中的某一层（不同层提供不同的细节-范围平衡）
- 将社区摘要**随机打乱**并分成固定大小的块
- 目的：确保相关信息分布在多个块中，而非集中在单个上下文窗口中（避免信息丢失）

#### 2. Map：生成社区答案（Map Community Answers）

- **并行处理**：每个摘要块独立生成部分答案
- LLM 同时生成一个 0-100 的**有用性评分**（helpfulness score）
- 过滤掉评分为 0 的答案

#### 3. Reduce：聚合为全局答案（Reduce to Global Answer）

- 按有用性评分**降序排序**部分答案
- 迭代地将答案添加到新的上下文窗口，直到达到 token 限制
- 使用这个最终上下文生成返回给用户的全局答案

> [!math] Map-Reduce 的形式化
> 
> 设 $C = \{c_1, c_2, \ldots, c_n\}$ 为某一层的社区集合，$S(c_i)$ 为社区 $c_i$ 的摘要，$q$ 为用户查询。
> 
> **Map 阶段**：
> $$
> a_i = \text{LLM}(q, S(c_i)), \quad s_i = \text{Score}(a_i, q)
> $$
> 
> **Reduce 阶段**：
> $$
> A_{\text{global}} = \text{LLM}\left(q, \{a_i \mid s_i > 0\}_{\text{sorted by } s_i}\right)
> $$


## 与传统 RAG 的对比

> [!comparison] GraphRAG vs Vector RAG
> 
> | 维度 | Vector RAG | GraphRAG |
> |------|-----------|----------|
> | **适用问题** | 局部问题（答案在少数文档中） | 全局问题（需要整合整个语料库） |
> | **检索机制** | 向量相似度匹配 | 社区摘要 + map-reduce |
> | **索引结构** | 文本嵌入向量 | 知识图谱 + 层次化社区 |
> | **回答方式** | 直接从检索片段生成答案 | 并行生成部分答案 → 聚合 |
> | **可扩展性** | 受上下文窗口限制 | 通过分治实现扩展 |
> | **成本** | 查询时成本低 | 索引构建成本高，查询时成本取决于社区数量 |
> 
> **示例对比**：
> - **局部问题**："NeoChip 是什么时候被收购的？" → Vector RAG 更高效
> - **全局问题**："这个数据集中科技公司并购的主要趋势是什么？" → GraphRAG 更有效

## 局限性

> [!warning] 边界条件与失效场景
> 
> **评估局限**：
> - 当前评估仅针对两个约 100 万 token 的语料库
> - 需要更多领域和用例的验证
> - 缺乏对幻觉率（fabrication rate）的系统评估（如 SelfCheckGPT）
> 
> **方法局限**：
> - **索引构建成本高**：需要多次 LLM 调用（实体提取、摘要生成）
> - **实体匹配简单**：论文使用精确字符串匹配，可能导致重复实体
> - **社区检测依赖**：Leiden 算法的质量直接影响主题分区效果
> - **层次选择问题**：不同层次提供不同的细节-范围平衡，需要根据问题类型选择
> 
> **适用场景限制**：
> - 不适合需要精确事实检索的问题（如"某人的生日是哪天"）
> - 对于小规模语料库，传统 RAG 或直接摘要可能更高效
> - 需要多次查询同一数据集才能摊销索引构建成本

## 未来方向

论文提出的潜在改进方向：

1. **混合 RAG 方案**：
   - 结合基于嵌入的匹配和即时社区报告生成
   - "Roll-up" 方法：跨多层社区层次聚合
   - "Drill-down" 机制：从高层摘要探索到低层细节

2. **局部 GraphRAG**：
   - 使用图注释的嵌入匹配
   - 在局部和全局问题之间提供更好的平衡

3. **更好的评估**：
   - 系统评估幻觉率
   - 更多领域和用例的基准测试

## 相关概念

**前置知识**：
- [[RAG Survey (2023)]] — 理解传统 RAG 的局限性
- [[Query-Focused Summarization]] — GraphRAG 解决的核心任务
- [[Knowledge Graph]] — 图结构的语义表示

**相关方法**：
- [[Naive RAG]] — 传统的向量检索方法
- [[HippoRAG]] — 另一种基于图的 RAG 方法
- [[Agentic RAG]] — 使用 agent 进行多步推理的 RAG

**核心技术**：
- [[Leiden Algorithm]] — 社区检测算法
- [[Map-Reduce]] — 分布式计算范式
- [[LLM-as-a-Judge]] — 评估方法

## 速查

> [!example] 关键参数与实现
> 
> **索引构建**：
> - 文本块大小：需要权衡成本和召回率
> - 社区检测：Leiden 算法，递归检测子社区
> - 摘要生成：自底向上，优先级基于节点度数
> 
> **查询回答**：
> - 社区摘要随机打乱并分块
> - 并行生成部分答案，评分 0-100
> - 按评分降序聚合，直到达到 token 限制
> 
> **开源实现**：
> - 官方：https://github.com/microsoft/graphrag
> - 集成：LangChain, LlamaIndex, NebulaGraph, Neo4J

> [!interview] 面试视角
> 
> **Q: GraphRAG 的核心创新是什么？**
> A: 将全局理解问题转化为可扩展的 map-reduce 过程。通过知识图谱的社区结构提供主题分区，预先生成层次化摘要，使得全局问题可以并行处理并聚合答案。
> 
> **Q: 什么时候应该使用 GraphRAG 而不是传统 RAG？**
> A: 当问题需要整合整个语料库的信息时（如"主要主题是什么"），而非检索特定事实。前提是数据集足够大（百万 token 级别）且需要多次查询以摊销索引成本。
> 
> **Q: GraphRAG 的主要成本在哪里？**
> A: 索引构建阶段：需要对每个文本块调用 LLM 提取实体/关系，对每个社区调用 LLM 生成摘要。查询阶段的成本取决于社区数量和选择的层次。
> 
> **Q: 如何选择社区层次？**
> A: 不同层次提供不同的细节-范围平衡。根层社区提供最粗粒度的全局视角（成本最低），叶子社区提供最细粒度的细节（成本最高）。需要根据问题的抽象程度选择。

> [!paper] 原始论文
> 
> **From Local to Global: A Graph-RAG Approach to Query-Focused Summarization**
> - 作者：Microsoft Research
> - 年份：2024
> - 链接：[[Clippings/Paper/2404.16130/2404.16130|论文原文]]
> - 开源：https://github.com/microsoft/graphrag
