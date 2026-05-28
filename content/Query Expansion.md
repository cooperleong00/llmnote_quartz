---
type: concept
description: 通过添加相关术语扩展原始查询，解决词汇不匹配问题以提升召回率
aliases:
  - 查询扩展
  - QE
prerequisites:
  - "[[Sparse Retrieval]]"
tags:
  - rag
  - query-optimization
created: 2026-03-07
updated: 2026-03-07T11:34
---

# Query Expansion

查询扩展（Query Expansion, QE）通过向原始查询添加相关术语来提升检索召回率。核心思想是：用户的查询往往过于简短或使用了与文档不同的词汇，通过扩展查询可以弥合这种**词汇不匹配**（vocabulary mismatch）问题。

## 动机

> [!intuition] 为什么需要查询扩展？
>
> **问题**：用户查询 "ML model training"，但文档中使用 "machine learning algorithm optimization"——语义相同，但词汇不匹配导致检索失败。
>
> **传统方案的局限**：
> - 精确匹配（BM25）：只能匹配相同词汇
> - 人工同义词库：维护成本高，覆盖不全
>
> **Query Expansion 的洞察**：自动发现相关术语并添加到查询中，让检索系统有更多机会匹配相关文档。

这与 [[Query Rewriting]] 的区别在于：Query Expansion **添加**术语（扩大搜索范围），而 Query Rewriting **替换**术语（改变搜索方向）。

## 核心方法

### 经典方法

#### 1. Pseudo-Relevance Feedback (PRF)

> [!intuition] 伪相关反馈
>
> 假设：初次检索的 top-k 结果大概率是相关的，从中提取高频术语扩展查询。

**流程**：
1. 用原始查询检索，获取 top-k 文档
2. 从这些文档中提取高频/高 TF-IDF 术语
3. 将这些术语添加到原始查询
4. 用扩展后的查询重新检索

**优点**：无需外部资源，自适应
**缺点**：如果初次检索质量差，会引入噪声（query drift）

#### 2. Thesaurus-Based Expansion

使用预定义的同义词库（如 WordNet）扩展查询。

**优点**：可控、可解释
**缺点**：覆盖有限，无法处理领域特定术语

### 现代方法

#### 1. Embedding-Based Expansion

> [!intuition] 语义相似度驱动
>
> 用词嵌入（word embeddings）或句嵌入找到语义相似的术语。

**流程**：
1. 将查询编码为向量
2. 在词汇表中找到最相似的 k 个词向量
3. 将这些词添加到查询

**优点**：捕捉语义相似性，覆盖广
**缺点**：可能引入语义漂移（如 "bank" → "river" vs "bank" → "finance"）

#### 2. LLM-Generated Expansion

> [!intuition] 生成式扩展
>
> 用 LLM 生成查询的同义词、相关术语或子问题。

**示例 Prompt**：
```
Original query: "ML model training"
Generate 5 related terms or synonyms that would help retrieve relevant documents.
```

**输出**：
- machine learning algorithm optimization
- neural network training
- model fine-tuning
- gradient descent
- backpropagation

**优点**：灵活、上下文感知、可生成多样化扩展
**缺点**：计算成本高，可能生成不相关术语

## 在 RAG 中的应用

Query Expansion 是 [[RAG]] 系统中 **Pre-Retrieval 优化** 的关键技术：

```
用户查询 → Query Expansion → 扩展查询 → 检索 → 重排序 → 生成
```

**典型场景**：
- **知识库检索**：用户查询简短，需要扩展以匹配文档的详细描述
- **多语言检索**：扩展为多语言同义词
- **领域特定检索**：扩展为领域术语（如医学、法律）

**与其他技术的配合**：
- **Query Expansion + [[Query Rewriting]]**：先重写纠正错误，再扩展增加覆盖
- **Query Expansion + Hybrid Retrieval**：扩展后的查询同时用于稀疏和密集检索

## 权衡：精确率 vs 召回率

> [!warning] 核心权衡
>
> Query Expansion 提升**召回率**（recall），但可能降低**精确率**（precision）。

**为什么**：
- 添加术语 → 匹配更多文档 → 召回率 ↑
- 但部分扩展术语可能不相关 → 引入噪声 → 精确率 ↓

**缓解策略**：
1. **限制扩展数量**：只添加 top-3 最相关术语
2. **加权扩展术语**：原始查询权重高，扩展术语权重低
3. **后处理过滤**：用 re-ranking 模型过滤不相关结果
4. **自适应扩展**：根据初次检索质量决定是否扩展

## 局限性

> [!warning] 边界条件
>
> - **Query Drift**：扩展术语偏离原始意图（如 "apple" → "fruit" 而非 "Apple Inc."）
> - **计算成本**：LLM-based 方法需要额外推理时间
> - **领域依赖**：通用扩展方法在专业领域效果有限
> - **短查询陷阱**：查询过短时，扩展可能完全改变语义

**何时不用 Query Expansion**：
- 查询已经很详细（如长问题）
- 需要精确匹配（如代码搜索、ID 查询）
- 初次检索质量已经很高

## 对比：Query Expansion vs Query Rewriting

| 维度 | Query Expansion | [[Query Rewriting]] |
|------|-----------------|---------------------|
| **操作** | 添加术语 | 替换/重构查询 |
| **目标** | 提升召回率 | 提升精确率 |
| **原始查询** | 保留 | 可能丢弃 |
| **风险** | 引入噪声 | 改变意图 |
| **适用场景** | 查询过短、词汇不匹配 | 查询有歧义、表达不清 |

**组合使用**：
```
原始查询: "ML training slow"
  ↓ Query Rewriting
重写查询: "Why is machine learning model training slow?"
  ↓ Query Expansion
扩展查询: "Why is machine learning model training slow? optimization performance bottleneck GPU"
```

## 相关概念

**前置知识**：
- [[Sparse Retrieval]] — Query Expansion 主要用于稀疏检索（BM25）

**相关技术**：
- [[Query Rewriting]] — 互补的查询优化技术
- [[Hybrid Retrieval]] — 结合稀疏和密集检索，扩展查询可用于两者
- [[Multi-Query Retrieval]] — 生成多个查询变体，可视为 Expansion 的变种

**应用场景**：
- [[RAG]] — Pre-Retrieval 优化的核心技术

## 速查

> [!example] 实现要点
>
> **PRF 参数**：
> - top-k 文档数：5-10
> - 扩展术语数：3-5
> - 术语权重：原始查询 1.0，扩展术语 0.3-0.5
>
> **LLM Prompt 模板**：
> ```
> Given the query: "{query}"
> Generate {k} related terms that would help retrieve relevant documents.
> Focus on synonyms, related concepts, and domain-specific terminology.
> ```
>
> **评估指标**：
> - Recall@k：扩展后召回率提升
> - Precision@k：精确率是否下降
> - MRR (Mean Reciprocal Rank)：相关文档排名

> [!interview] 面试视角
>
> **Q: Query Expansion 和 Query Rewriting 有什么区别？**
> A: Query Expansion **添加**相关术语以提升召回率，保留原始查询；Query Rewriting **替换**或重构查询以提升精确率，可能改变原始表达。两者互补，常组合使用。
>
> **Q: 什么时候不应该用 Query Expansion？**
> A: (1) 查询已经很详细；(2) 需要精确匹配（如代码搜索）；(3) 初次检索质量已经很高。盲目扩展会引入噪声，降低精确率。
>
> **Q: 如何缓解 Query Drift 问题？**
> A: (1) 限制扩展术语数量；(2) 给扩展术语较低权重；(3) 用 re-ranking 过滤不相关结果；(4) 根据初次检索质量自适应决定是否扩展。
