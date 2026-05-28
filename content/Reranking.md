---
type: concept
description: 两阶段检索的第二阶段，使用精确但昂贵的模型对候选集重新排序，在计算成本和检索质量间取得平衡
aliases:
  - 重排序
  - Re-ranking
prerequisites:
  - "[[BM25]]"
  - "[[Embedding]]"
tags:
  - rag
  - ranking
  - retrieval
created: 2026-03-07
updated: 2026-03-07T11:52
---

# Reranking

Reranking（重排序）是**两阶段检索**（two-stage retrieval）的第二阶段，通过精确但昂贵的模型对初筛候选集重新排序。核心思想是**分层优化**：第一阶段用快速检索（如 [[BM25]]、密集检索）从海量文档中召回 top-k 候选，第二阶段用高质量模型对小规模候选集精细排序。这种架构在计算成本和检索质量间取得平衡，是 [[RAG]] 系统提升准确率的关键技术。

## 动机

> [!intuition] 为什么需要两阶段检索？
>
> **第一阶段检索的困境**：
>
> - **速度优先**：需要在百万/亿级文档中快速筛选，必须使用轻量级模型
> - **质量受限**：[[BM25]] 只看词汇匹配，密集检索（bi-encoder）无法建模查询-文档的细粒度交互
> - **召回 vs 精度**：为了不漏掉相关文档，通常召回较多候选（top-50 ~ top-100），但排序质量不够
>
> **Reranking 的核心洞察**：
> - **候选集已经很小**：从百万文档缩减到 50-100 个，可以承受更高的计算成本
> - **精细交互建模**：使用 cross-encoder 或 LLM 对查询-文档对进行深度建模
> - **质量提升显著**：在 top-10 结果中，reranking 可以将准确率提升 10-30%

## 核心机制

### 两阶段检索架构

```
查询: "What is the capital of France?"
         ↓
┌────────────────────────────────────────┐
│  Stage 1: 快速检索（Retrieval）         │
│  - BM25 / Dense Retrieval              │
│  - 从 1M 文档中召回 top-100             │
│  - 速度: ~10ms                         │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  Stage 2: 精细排序（Reranking）        │
│  - Cross-Encoder / LLM                 │
│  - 对 100 个候选重新排序                │
│  - 速度: ~100ms                        │
└────────────────────────────────────────┘
         ↓
    返回 top-10 最相关文档
```

### Reranking 模型的类型

> [!comparison] Bi-Encoder vs Cross-Encoder
>
> **Bi-Encoder（第一阶段常用）**：
> - 查询和文档**独立编码**：`score = sim(encode(query), encode(doc))`
> - 优点：文档可以预先编码，检索时只需计算相似度（快）
> - 缺点：无法建模查询-文档的细粒度交互
>
> **Cross-Encoder（Reranking 常用）**：
> - 查询和文档**联合编码**：`score = model([query, doc])`
> - 优点：通过 attention 机制捕捉词级别的交互（准确）
> - 缺点：每个查询-文档对都需要前向传播（慢）

#### 1. Cross-Encoder

基于 BERT 等 Transformer 模型，将查询和文档拼接后输入：

```python
# 伪代码
input = "[CLS] query [SEP] document [SEP]"
score = cross_encoder(input)  # 输出相关性分数
```

**代表模型**：
- `ms-marco-MiniLM-L-12-v2`（轻量级）
- `ms-marco-electra-base`（平衡）
- `cross-encoder/ms-marco-TinyBERT-L-2-v2`（极速）

#### 2. LLM-based Reranking

使用大语言模型直接评分或生成排序：

**方法 1：Pointwise Scoring**
```python
prompt = f"Rate the relevance of this document to the query on a scale of 1-10.\nQuery: {query}\nDocument: {doc}"
score = llm.generate(prompt)
```

**方法 2：Listwise Ranking**
```python
prompt = f"Rank these documents by relevance to the query:\nQuery: {query}\nDocuments: {docs}"
ranking = llm.generate(prompt)  # 输出排序后的文档 ID
```

**优缺点**：
- ✅ 理解能力强，可以处理复杂语义
- ✅ 可以结合推理（如"这个文档虽然提到巴黎，但讨论的是巴黎圣母院，不是首都"）
- ❌ 延迟高（~1s per query）
- ❌ 成本高（API 调用费用）

## 与第一阶段检索的对比

| 维度 | 第一阶段检索 | Reranking |
|------|-------------|-----------|
| **目标** | 召回（Recall） | 精度（Precision） |
| **候选规模** | 百万/亿级 | 50-100 |
| **模型类型** | BM25 / Bi-Encoder | Cross-Encoder / LLM |
| **交互建模** | 无 / 浅层 | 深度交互 |
| **延迟** | ~10ms | ~100ms |
| **可预计算** | 是（文档向量） | 否（查询相关） |


## 在 RAG Pipeline 中的集成

Reranking 发生在**检索之后、生成之前**：

```
用户查询
    ↓
查询优化（Query Rewriting/Expansion）
    ↓
第一阶段检索（BM25 + Dense Retrieval）
    ↓ top-100 候选
Reranking（Cross-Encoder）
    ↓ top-10 精选
上下文构建（拼接文档）
    ↓
LLM 生成答案
```

### 实现示例

使用 `sentence-transformers` 库：

```python
from sentence_transformers import CrossEncoder

# 初始化 reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

# 第一阶段：快速检索（假设已完成）
candidates = retriever.search(query, top_k=100)

# 第二阶段：Reranking
pairs = [[query, doc.text] for doc in candidates]
scores = reranker.predict(pairs)

# 按分数重新排序
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
top_10 = [doc for doc, score in reranked[:10]]
```

### 混合检索 + Reranking

常见的最佳实践是结合多种检索策略：

```python
# 第一阶段：混合检索
bm25_results = bm25.search(query, top_k=50)
dense_results = dense_retriever.search(query, top_k=50)
candidates = merge_and_deduplicate(bm25_results, dense_results)  # ~100 个

# 第二阶段：Reranking
reranked = reranker.rank(query, candidates, top_k=10)
```

## 权衡与优化

> [!warning] 延迟 vs 质量的权衡
>
> **Reranking 的成本**：
> - Cross-Encoder：每个候选需要一次前向传播，100 个候选 ≈ 100ms
> - LLM-based：延迟可能达到 1-2 秒
>
> **优化策略**：
> 1. **减少候选数**：第一阶段只召回 top-50 而非 top-100
> 2. **使用轻量级模型**：TinyBERT（2 层）vs MiniLM（12 层）
> 3. **批处理**：并行处理多个查询-文档对
> 4. **混合策略**：只对 top-20 用 LLM rerank，其余用 cross-encoder

### 何时使用 Reranking

| 场景 | 是否需要 Reranking |
|------|-------------------|
| 简单事实查询（"巴黎是哪国首都"） | ❌ BM25 足够 |
| 复杂语义查询（"比较不同国家的首都规划理念"） | ✅ 需要深度理解 |
| 实时系统（延迟 < 50ms） | ⚠️ 使用轻量级 reranker |
| 离线批处理 | ✅ 可以用 LLM reranking |
| 高精度要求（如医疗、法律） | ✅ 必须 rerank |


## 局限性

> [!warning] Reranking 的边界
>
> 1. **依赖第一阶段召回**：如果相关文档在第一阶段就被过滤掉，reranking 无法挽回（"garbage in, garbage out"）
> 2. **计算成本**：候选数 × 模型复杂度，可能成为系统瓶颈
> 3. **长文档截断**：Cross-encoder 通常有 512 token 限制，长文档需要截断或分块
> 4. **训练数据偏差**：MS MARCO 等数据集训练的模型可能在特定领域（如医疗、法律）表现不佳
> 5. **无法修复查询问题**：如果查询本身表达不清，reranking 也无法理解用户真实意图

## 相关概念

**第一阶段检索**：
- [[BM25]] — 稀疏检索的核心算法
- [[Sparse Retrieval]] — 基于词项匹配的检索方法
- [[Embedding]] — 密集检索的基础

**查询优化**（发生在检索之前）：
- [[Query Rewriting]] — 改写查询以提升检索质量
- [[Query Expansion]] — 扩展查询词汇
- [[Multi-Query]] — 生成多个查询变体

**RAG 系统**：
- [[RAG]] — 检索增强生成的完整流程
- [[MOC - RAG]] — RAG 领域导航

## 速查

> [!example] 快速参考
>
> **常用 Cross-Encoder 模型**：
> - 轻量级：`cross-encoder/ms-marco-TinyBERT-L-2-v2`（2 层，~10ms/doc）
> - 平衡：`cross-encoder/ms-marco-MiniLM-L-12-v2`（12 层，~30ms/doc）
> - 高精度：`cross-encoder/ms-marco-electra-base`（~50ms/doc）
>
> **典型配置**：
> - 第一阶段召回：top-50 ~ top-100
> - Reranking 输出：top-10 ~ top-20
> - 延迟预算：100-200ms（包含两阶段）
>
> **实现库**：
> - `sentence-transformers`（Python，支持 cross-encoder）
> - `rank-bm25` + `sentence-transformers`（混合检索）
> - LangChain/LlamaIndex（集成 reranking 组件）

> [!interview] 面试要点
>
> **Q: 为什么需要两阶段检索，而不是直接用 cross-encoder 检索所有文档？**
> A: 计算成本。Cross-encoder 需要对每个查询-文档对做前向传播，无法预计算。如果有 100 万文档，每次查询需要 100 万次前向传播（~几分钟），无法实时响应。两阶段架构用快速检索缩小候选集，再用精确模型排序。
>
> **Q: Reranking 和 Hybrid Retrieval 有什么区别？**
> A: 
> - **Hybrid Retrieval**：在第一阶段结合多种检索方法（如 BM25 + dense retrieval），目标是提升召回率
> - **Reranking**：在第二阶段对候选集重新排序，目标是提升精度
> - 两者可以结合：Hybrid Retrieval → Reranking
>
> **Q: 什么时候不需要 Reranking？**
> A: 
> - 简单的词汇匹配查询（BM25 已经足够准确）
> - 延迟要求极高的场景（< 50ms）
> - 第一阶段检索质量已经很高（如精心调优的 dense retrieval）
> - 资源受限的环境（无法承受额外计算）
