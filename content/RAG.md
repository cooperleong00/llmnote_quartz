---
type: concept
description: 通过从外部知识库检索相关信息来增强 LLM 生成能力，解决知识过时、幻觉和不可追溯等问题
aliases:
  - Retrieval-Augmented Generation
  - 检索增强生成
prerequisites:
  - "[[Embedding]]"
  - "[[Dense Retrieval]]"
tags:
  - rag
  - retrieval
  - knowledge-augmentation
created: 2026-03-23
updated: 2026-05-31T17:53
---

# RAG

检索增强生成（Retrieval-Augmented Generation, RAG）通过从外部知识库检索相关信息来增强 LLM 的生成能力。核心思想是将 LLM 的参数化知识与外部动态知识库结合，在生成前先检索相关文档作为上下文，从而解决知识过时、幻觉和不可追溯等问题。


## 动机

> [!intuition] 为什么需要 RAG？
> LLM 的知识来自训练数据，存在三大局限：
> 1. **知识过时**：训练后的新信息无法获取（如最新新闻、产品文档更新）
> 2. **幻觉问题**：LLM 可能生成看似合理但实际错误的内容，且无法追溯来源
> 3. **领域知识不足**：通用模型对特定领域（如企业内部文档）的知识有限
>
> RAG 的核心洞察：**不需要把所有知识都塞进模型参数**。将 LLM 视为"推理引擎"，外部知识库提供"事实材料"，在生成时动态检索相关信息作为上下文。

传统解决方案的不足：
- **Fine-tuning**：需要重新训练，成本高，知识更新慢，且容易过拟合
- **Long Context**：受限于 context window，无法处理大规模知识库，且成本随长度线性增长

RAG 提供了第三条路：保持模型不变，通过检索动态注入知识。

## 核心机制

RAG 的工作流程可以分为三个阶段：

> [!intuition] 检索-增强-生成流程
> 1. **Query Processing**：理解用户问题，可能进行查询重写或扩展
> 2. **Retrieval**：从知识库中检索相关文档/片段
> 3. **Generation**：将检索结果作为上下文，LLM 基于此生成回答

### 基本流程

```
用户问题 → [查询编码] → [检索 Top-K] → [整合上下文] → [LLM 生成]
   ↓                        ↓                    ↓
 Query                  Knowledge Base      Retrieved Docs
```

**关键组件**：
- **Embedding Model**：将查询和文档编码为向量，用于相似度计算
- **Vector Database**：存储文档 embeddings，支持高效的相似度检索
- **Retriever**：检索策略（如 [[Dense Retrieval]]、[[Sparse Retrieval]]、混合检索）
- **Generator**：LLM，基于检索到的上下文生成回答

### 检索策略

| 策略 | 原理 | 优势 | 局限 |
|------|------|------|------|
| **Dense Retrieval** | 语义向量相似度 | 理解语义，跨语言 | 计算成本高 |
| **Sparse Retrieval** | 关键词匹配（BM25） | 精确匹配，快速 | 无法理解语义 |
| **Hybrid Retrieval** | 结合 dense + sparse | 兼顾语义和精确性 | 需要权重调优 |

**后处理优化**：
- **[[Reranking]]**：用更精确的模型对候选集重新排序
- **Context Compression**：压缩检索结果，减少 token 消耗
- **Relevance Filtering**：过滤低相关性文档，减少噪声

## 范式演进

RAG 技术经历了三个主要范式的演进：

### Naive RAG

最早的 RAG 范式，采用简单的"检索 → 生成"流程：

```
Query → Embedding → Top-K Retrieval → Concatenate → LLM Generate
```

**特点**：
- 单次检索，直接拼接到 prompt
- 检索和生成解耦，无反馈机制
- 实现简单，但效果受限

**局限**：
- 检索质量不稳定（查询表达可能不精确）
- 上下文利用不充分（检索结果可能包含噪声）
- 无法处理复杂查询（需要多步推理）

### Advanced RAG

引入预检索和后检索优化，改进检索质量：

**预检索优化**：
- **[[Query Rewriting]]**：改写查询以提高检索效果
- **Query Expansion**：扩展查询关键词
- **HyDE**：生成假设性文档，用于检索

**后检索优化**：
- **[[Reranking]]**：用精确模型重新排序
- **Context Compression**：压缩冗余信息
- **Relevance Filtering**：过滤低相关性结果

### Modular RAG

模块化设计，灵活组合各组件，支持更复杂的检索策略：

**核心特性**：
- **迭代检索**：根据生成结果决定是否继续检索
- **自适应检索**：动态判断是否需要检索（如简单问题直接回答）
- **多源检索**：从多个知识库检索并融合
- **深度集成**：与 fine-tuning、推理等技术结合

**参考**：[[RAG Survey (2023)]]

## 对比分析

### RAG vs Fine-tuning

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| **知识更新** | 实时更新知识库即可 | 需要重新训练 |
| **成本** | 推理时增加检索成本 | 训练成本高 |
| **可追溯性** | 可引用来源文档 | 无法追溯 |
| **知识内化** | 知识在外部 | 知识融入参数 |
| **推理速度** | 检索增加延迟 | 无额外延迟 |
| **适用场景** | 动态知识、可追溯需求 | 领域适配、风格学习 |

> [!intuition] 何时选择 RAG？
> - 知识频繁更新（如新闻、产品文档）
> - 需要引用来源（如法律、医疗）
> - 知识库规模大（fine-tuning 难以覆盖）
> - 多租户场景（每个用户有独立知识库）

**混合方案**：RAG + Fine-tuning
- Fine-tuning 学习领域语言风格和推理模式
- RAG 提供最新的事实性知识
- 两者互补，效果最佳

### RAG vs Long Context

| 维度 | RAG | Long Context |
|------|-----|--------------|
| **知识规模** | 可扩展到 TB 级 | 受限于 context window |
| **检索精度** | 精准检索相关片段 | 需要模型自己定位 |
| **成本** | 检索 + 生成 | 随 context 长度线性增长 |
| **完整性** | 可能遗漏信息 | 完整上下文 |
| **适用场景** | 大规模知识库 | 单文档深度理解 |

> [!comparison] 实际应用中的选择
> - **RAG**：企业知识库、客服系统、研究助手（需要从海量文档中检索）
> - **Long Context**：合同分析、代码审查、长文档总结（需要完整上下文）
> - **混合**：先用 RAG 检索相关文档，再用 Long Context 深度理解

## 局限性

> [!warning] RAG 的边界条件
> 
> **检索质量依赖**：
> - 检索失败（相关文档未被检索到）会导致生成质量下降
> - 查询表达不精确时，检索结果可能不相关
> - 知识库质量直接影响最终效果（garbage in, garbage out）
> 
> **延迟增加**：
> - 检索过程增加推理延迟（embedding + 向量检索 + reranking）
> - 对实时性要求高的场景可能不适用
> 
> **成本上升**：
> - 需要维护向量数据库和检索系统
> - 检索到的文档增加 LLM 的 token 消耗
> - 大规模部署时成本显著
> 
> **上下文窗口限制**：
> - 检索结果受限于 LLM 的 context window
> - 需要在检索数量和相关性之间权衡
> 
> **复杂查询处理**：
> - 需要多步推理的问题，单次检索可能不足
> - 需要迭代检索或与推理系统集成（见 [[Agentic RAG]]）

## 相关概念

**核心技术**：
- [[Embedding]] — 将文本编码为向量的基础技术
- [[Dense Retrieval]] — 基于语义向量的检索方法
- [[Sparse Retrieval]] — 基于关键词的传统检索
- [[Reranking]] — 检索后的精排优化

**优化技术**：
- [[Query Rewriting]] — 改写查询以提高检索效果
- [[Context Compression]] — 压缩检索结果减少 token 消耗
- [[Hybrid Retrieval]] — 结合 dense 和 sparse 检索

**高级话题**：
- [[Agentic RAG]] — RAG 与推理系统的深度集成
- [[GraphRAG]] — 基于知识图谱的 RAG
- [[Adaptive RAG]] — 自适应决定是否检索

**领域导航**：
- [[05-MOC - RAG|MOC - RAG]] — RAG 技术全景图

## 延伸阅读

**综述论文**：
- [[RAG Survey (2023)]] — 系统性综述 RAG 三大范式演进
- [[Towards Agentic RAG with Deep Reasoning (2025)]] — RAG 与推理集成的最新进展

> [!interview] 面试要点
> 
> **Q: RAG 的核心优势是什么？**
> A: 三点核心优势：(1) 知识可实时更新，无需重训练；(2) 可追溯来源，减少幻觉；(3) 成本低于 fine-tuning，适合大规模知识库。
> 
> **Q: RAG 和 Fine-tuning 如何选择？**
> A: 看需求：RAG 适合动态知识和可追溯场景（如客服、研究助手），Fine-tuning 适合领域适配和风格学习（如代码生成、特定写作风格）。实际中常混合使用：Fine-tuning 学习推理模式，RAG 提供事实知识。
> 
> **Q: RAG 的主要挑战是什么？**
> A: 检索质量是核心挑战。包括：(1) 查询理解（用户问题可能表达不清）；(2) 检索精度（相关文档未被检索到）；(3) 噪声过滤（检索结果包含无关信息）。Advanced RAG 通过 query rewriting、reranking 等技术缓解这些问题。
> 
> **Q: 什么是 Naive/Advanced/Modular RAG？**
> A: 三代范式演进：Naive RAG 是简单的检索-生成流程；Advanced RAG 引入预检索和后检索优化（query rewriting、reranking）；Modular RAG 支持迭代检索、自适应检索，可与其他技术深度集成。