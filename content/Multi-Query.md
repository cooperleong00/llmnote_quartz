---
type: concept
description: 从单个用户查询生成多个查询变体，通过覆盖不同检索角度提升召回率
aliases:
  - 多查询生成
  - Multi-Query Generation
prerequisites:
  - "[[RAG]]"
  - "[[Embedding]]"
tags:
  - rag
  - query-optimization
created: 2026-03-07
updated: 2026-03-07T11:39
---

# Multi-Query

Multi-Query（多查询生成）通过从单个用户查询生成多个语义相关的查询变体，从不同角度检索文档，提升 RAG 系统的召回率。核心思想是：单一查询可能无法覆盖所有相关文档，而多个变体可以捕捉不同的语义侧面。

## 动机

> [!intuition] 为什么需要多查询？
>
> **问题**：用户查询往往简短且表达单一视角，可能错过相关文档。
>
> 例如，查询"如何提升模型性能"可能错过标题为"优化训练效率"或"加速推理速度"的文档——它们都相关，但语义表达不同。
>
> **传统方案的不足**：
> - [[Query Expansion]]（查询扩展）：添加同义词或相关词，但仍是单一查询
> - [[Query Rewriting]]（查询重写）：改写为更好的单一查询，但无法同时覆盖多个角度
>
> **Multi-Query 的洞察**：与其猜测"最佳"查询，不如生成多个变体，让检索系统从不同角度搜索，再聚合结果。

## 核心机制

### 生成策略

> [!intuition] 如何生成查询变体？
>
> **基于 LLM 的生成**（主流方法）：
> 使用 LLM 生成查询变体，通过 prompt 引导生成不同角度的查询。
>
> **Prompt 示例**：
> ```
> 原始查询：{query}
> 
> 生成 3 个语义相关但表达不同的查询变体，覆盖：
> 1. 更具体的角度
> 2. 更广泛的角度
> 3. 不同的措辞方式
> ```
>
> **优势**：
> - 理解语义，生成高质量变体
> - 可控制生成数量和多样性
> - 适应不同领域（通过 few-shot 示例）

**基于规则的生成**（传统方法）：
- 同义词替换
- 词序变换
- 添加/删除修饰词

**劣势**：语义理解有限，变体质量不如 LLM。

### 聚合策略

生成多个查询后，需要聚合检索结果：

> [!math] 聚合方法
>
> 设 $Q = \{q_1, q_2, ..., q_n\}$ 为查询变体集合，$D_i$ 为查询 $q_i$ 的检索结果。
>
> **1. Union（并集）**：
> $$D_{\text{final}} = D_1 \cup D_2 \cup ... \cup D_n$$
> - 最大化召回率
> - 可能引入噪声
>
> **2. Intersection（交集）**：
> $$D_{\text{final}} = D_1 \cap D_2 \cap ... \cap D_n$$
> - 最大化精确率（只保留所有查询都检索到的文档）
> - 可能过于严格，召回率低
>
> **3. Ranking Fusion（排序融合）**：
> - **Reciprocal Rank Fusion (RRF)**：
>   $$\text{score}(d) = \sum_{i=1}^{n} \frac{1}{k + \text{rank}_i(d)}$$
>   其中 $k$ 是常数（通常为 60），$\text{rank}_i(d)$ 是文档 $d$ 在查询 $q_i$ 结果中的排名。
> - **优势**：平衡多个查询的排名信号，鲁棒性强

> [!example] RRF 示例
> 假设有 2 个查询变体，文档 A 在两个结果中的排名分别为 1 和 3：
> $$\text{score}(A) = \frac{1}{60+1} + \frac{1}{60+3} = 0.0164 + 0.0159 = 0.0323$$
>
> 文档 B 排名为 2 和 2：
> $$\text{score}(B) = \frac{1}{60+2} + \frac{1}{60+2} = 0.0161 + 0.0161 = 0.0322$$
>
> 尽管 B 在两个查询中都排名第 2，但 A 因为有一个第 1 名而得分略高。


## 何时使用

> [!intuition] 适用场景
>
> **适合使用 Multi-Query**：
> - **复杂查询**：用户查询包含多个方面或模糊意图
> - **低初始召回率**：单一查询检索结果不足
> - **领域术语多样**：同一概念有多种表达方式（如医学、法律领域）
> - **探索性搜索**：用户不确定如何精确表达需求
>
> **不适合使用**：
> - **简单事实查询**：如"Python 3.10 发布时间"（单一查询足够）
> - **实时性要求高**：多次检索增加延迟
> - **成本敏感**：LLM 生成查询变体有额外成本

## 局限性

> [!warning] 边界条件
>
> **1. 延迟增加**：
> - 生成查询变体需要 LLM 调用（~100-500ms）
> - 多次检索调用（如 3 个变体 = 3x 检索时间）
> - 可通过并行检索缓解，但仍有额外开销
>
> **2. 成本上升**：
> - LLM 生成成本
> - 多次 embedding 计算（如果变体需要重新编码）
> - 多次向量数据库查询
>
> **3. 噪声风险**：
> - 生成的变体可能偏离原始意图
> - Union 聚合可能引入不相关文档
> - 需要配合 [[Reranking]] 过滤噪声
>
> **4. 变体质量依赖 LLM**：
> - 小模型可能生成低质量变体
> - 需要精心设计 prompt 和 few-shot 示例
> - 不同领域可能需要不同的生成策略

## 实现要点

> [!example] 典型实现流程
>
> ```python
> # 1. 生成查询变体
> variants = llm.generate(
>     prompt=f"生成 3 个查询变体：{original_query}",
>     temperature=0.7  # 增加多样性
> )
>
> # 2. 并行检索
> results = []
> for query in [original_query] + variants:
>     docs = retriever.search(query, top_k=10)
>     results.append(docs)
>
> # 3. RRF 聚合
> final_docs = reciprocal_rank_fusion(results, k=60)
>
> # 4. 可选：Reranking
> final_docs = reranker.rerank(original_query, final_docs)
> ```

**关键参数**：
- **变体数量**：通常 2-5 个（更多不一定更好，增加噪声）
- **Temperature**：0.7-1.0（平衡多样性和质量）
- **聚合方法**：RRF 通常优于简单 Union
- **Top-k**：每个查询检索 5-20 个文档


## 与相关方法的对比

> [!comparison] Multi-Query vs 其他查询优化方法
>
> | 方法 | 核心思想 | 优势 | 劣势 |
> |------|----------|------|------|
> | **Multi-Query** | 生成多个查询变体 | 覆盖多个角度，召回率高 | 延迟和成本增加 |
> | [[Query Rewriting]] | 改写为更好的单一查询 | 低延迟，单次检索 | 只能优化单一视角 |
> | [[Query Expansion]] | 添加同义词/相关词 | 简单高效 | 语义理解有限 |
>
> **组合使用**：
> - Multi-Query + [[Reranking]]：先扩大召回，再精确排序
> - Query Rewriting → Multi-Query：先改写为更好的查询，再生成变体
> - Multi-Query + [[Hybrid Retrieval]]：不同变体使用不同检索策略（dense/sparse）

## 相关概念

**前置知识**：
- [[RAG]] — Multi-Query 是 RAG 检索优化的一种方法
- [[Embedding]] — 查询变体需要编码为向量

**同类方法**（查询优化）：
- [[Query Rewriting]] — 改写单一查询
- [[Query Expansion]] — 扩展查询词汇

**后续处理**：
- [[Reranking]] — 对聚合结果重新排序
- [[Hybrid Retrieval]] — 结合多种检索策略

**应用场景**：
- [[MOC - RAG]] — RAG 系统的整体架构

> [!interview] 面试要点
>
> **Q: Multi-Query 如何提升 RAG 性能？**
> A: 通过生成多个查询变体覆盖不同语义角度，提升召回率。单一查询可能因措辞限制错过相关文档，而多个变体可以从不同视角检索，再通过 RRF 等方法聚合结果。
>
> **Q: 如何平衡 Multi-Query 的成本和收益？**
> A: 
> - 控制变体数量（2-3 个通常足够）
> - 并行检索减少延迟
> - 只在复杂查询或低召回场景使用
> - 配合缓存机制（相似查询复用变体）
>
> **Q: Multi-Query 和 Query Expansion 有什么区别？**
> A: Query Expansion 是在原查询基础上添加同义词或相关词，仍是单一查询；Multi-Query 生成多个独立的查询变体，每个变体独立检索后聚合。Multi-Query 能覆盖更多样的语义角度，但成本更高。

## 延伸阅读

**原始论文与实现**：
- LangChain MultiQueryRetriever — 主流实现
- LlamaIndex Multi-Query Engine — 另一种实现

**相关技术**：
- [[RAG Survey (2023)]] — RAG 技术综述，包含查询优化策略
- [[Hybrid Retrieval]] — 结合多种检索方法
