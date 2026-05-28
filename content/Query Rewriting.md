---
type: concept
description: 在检索前转换用户查询以提升检索质量，通过扩展、改写或分解查询来弥合语义鸿沟
aliases:
  - 查询重写
  - Query Transformation
prerequisites:
  - "[[Embedding]]"
  - "[[Dense Retrieval]]"
tags:
  - rag
  - query-optimization
created: 2026-03-07
updated: 2026-03-07T12:00
---

# Query Rewriting

查询重写（Query Rewriting）是 [[MOC - RAG|RAG]] 系统中的预检索优化技术，通过转换用户原始查询来提升检索质量。核心思想是：用户的自然语言查询往往不是最优的检索表达——可能过于简短、含糊或使用了与文档不同的词汇。查询重写弥合这种**语义鸿沟**（semantic gap），将用户意图转化为更适合检索的形式。

## 动机

> [!intuition] 为什么需要查询重写？
>
> **问题**：用户查询与文档表达的不匹配
> - 用户问"DPO 为什么稳定"，文档可能写"DPO 避免了 RL 训练的不稳定性"
> - 用户问"怎么加速推理"，文档可能分散在"量化"、"剪枝"、"KV Cache 优化"等章节
> - 用户问"transformer 注意力机制"，可能需要同时检索"self-attention"、"multi-head attention"等相关概念
>
> **传统方法的不足**：
> - 直接用原始查询检索 → 召回率低，遗漏相关文档
> - 依赖用户手动改写 → 用户不知道如何优化查询
>
> **查询重写的洞察**：
> 自动将查询转换为更适合检索的形式——扩展关键词、改写表达、分解复杂问题。

## 核心技术

### 1. Query Expansion（查询扩展）

**目标**：增加查询的覆盖面，提升召回率

**方法**：
- **同义词扩展**：添加同义词、相关术语
  - "transformer" → "transformer + self-attention + encoder-decoder"

- **伪相关反馈**（Pseudo-Relevance Feedback）：从初次检索结果中提取关键词
  - 经典 IR 技术，现代 RAG 中较少使用

**示例**：


### 2. Query Reformulation（查询改写）

**目标**：用不同表达重述查询，提升语义匹配

**方法**：
- **同义改写**：用不同词汇表达相同意图
  - "如何加速推理" → "推理优化方法"
- **视角转换**：从不同角度提问
  - "DPO 为什么稳定" → "DPO 相比 PPO 的优势"
- **LLM 改写**：用大模型生成多个改写版本


### 3. Query Decomposition（查询分解）

**目标**：将复杂查询拆解为多个子查询，分别检索后合并

**方法**：
- **逻辑分解**：识别查询中的多个子问题
  - "对比 DPO 和 PPO 的优缺点" → ["DPO 的优点", "DPO 的缺点", "PPO 的优点", "PPO 的缺点"]
- **时序分解**：将多步骤问题拆解
  - "如何训练并部署一个 LLM" → ["LLM 训练流程", "LLM 部署方法"]

**实现**：
- 经典方法：规则 + NLP 解析
- 现代方法：LLM 生成子查询（如 [[Multi-Query]] 技术）

## 经典 vs LLM 方法

| 维度 | 经典方法 | LLM 方法 |
|------|----------|----------|
| **技术** | 同义词词典、TF-IDF、伪相关反馈 | Prompt LLM 生成改写/扩展 |
| **优势** | 快速、可控、无额外成本 | 语义理解强、灵活、效果好 |
| **劣势** | 语义理解弱、依赖词典 | 延迟高、成本高、可能过度改写 |
| **适用场景** | 低延迟要求、领域词典完善 | 复杂查询、开放域检索 |

**混合策略**：
- 简单查询 → 经典方法（同义词扩展）
- 复杂查询 → LLM 方法（分解 + 改写）

## 在 RAG Pipeline 中的位置

查询重写发生在**检索之前**（Pre-Retrieval），是 RAG 优化的第一道防线：

```
用户查询 → [查询重写] → 检索 → 重排序 → 生成
```

**与其他优化的配合**：
- **Query Rewriting + [[Dense Retrieval]]**：改写后的查询生成更好的 embedding
- **Query Rewriting + [[Reranking]]**：扩展查询提升召回，重排序保证精度
- **Query Rewriting + [[Context Compression]]**：多查询检索后压缩冗余内容

## 权衡与选择

> [!warning] 关键权衡
>
> **延迟 vs 质量**：
> - 查询重写增加一次 LLM 调用（~100-500ms）
> - 对于实时应用（如聊天），延迟敏感
> - 对于离线分析（如研究助手），质量优先
>
> **召回 vs 精度**：
> - 查询扩展提升召回，但可能引入噪声
> - 需要配合重排序过滤无关结果
>
> **复杂度 vs 收益**：
> - 简单查询可能不需要重写（如"什么是 DPO"）
> - 复杂查询收益明显（如"对比 DPO、PPO、TRPO 的优缺点"）

**何时使用查询重写**：
- ✅ 用户查询简短、模糊
- ✅ 领域术语多样（需要同义词扩展）
- ✅ 复杂问题需要分解
- ❌ 查询已经很明确、具体
- ❌ 延迟要求极高（<100ms）
- ❌ 文档库很小，直接检索已足够

## 实现示例

### 经典方法：同义词扩展

```python
from nltk.corpus import wordnet

def expand_query(query: str) -> list[str]:
    """用 WordNet 扩展查询"""
    words = query.split()
    expanded = [query]  # 保留原始查询
    
    for word in words:
        synonyms = wordnet.synsets(word)
        for syn in synonyms[:2]:  # 取前 2 个同义词
            expanded.append(query.replace(word, syn.lemmas()[0].name()))
    
    return expanded
```

### LLM 方法：多角度改写

```python
def rewrite_query_llm(query: str, num_rewrites: int = 3) -> list[str]:
    """用 LLM 生成多个改写版本"""
    prompt = f"""
    Rewrite the following query in {num_rewrites} different ways, 
    each focusing on a different aspect or using different terminology:
    
    Query: {query}
    
    Return only the rewritten queries, one per line.
    """
    
    response = llm.generate(prompt)
    return response.strip().split('
')
```

### 查询分解

```python
def decompose_query(query: str) -> list[str]:
    """用 LLM 分解复杂查询"""
    prompt = f"""
    Break down the following complex query into simpler sub-queries:
    
    Query: {query}
    
    Return sub-queries as a numbered list.
    """
    
    response = llm.generate(prompt)
    # 解析返回的子查询列表
    return parse_subqueries(response)
```

## 相关概念

**查询优化技术**：
- [[Query Expansion]] — 查询扩展的详细技术
- [[Multi-Query]] — 生成多个查询并行检索（与查询分解相关）

**RAG Pipeline**：
- [[MOC - RAG]] — RAG 系统全景
- [[Dense Retrieval]] — 查询重写后的检索方法
- [[Reranking]] — 检索后的精度优化

**对比**：
- **Query Rewriting vs [[Query Expansion]]**：Query Expansion 是 Query Rewriting 的一种技术，专注于扩展关键词；Query Rewriting 是更广泛的概念，包括改写、分解等
- **Query Rewriting vs [[Reranking]]**：前者在检索前优化查询，后者在检索后优化结果；两者互补

## 延伸阅读

**原始论文与深入材料**：
- [[RAG Survey (2023)]] — 系统性综述 RAG 优化技术，包括查询重写的各种方法

> [!interview] 面试要点
>
> **Q: 查询重写的核心目标是什么？**
> A: 弥合用户查询与文档表达之间的语义鸿沟，将用户意图转化为更适合检索的形式。
>
> **Q: 查询重写有哪些主要技术？**
> A: 三类：(1) Query Expansion（扩展关键词）、(2) Query Reformulation（改写表达）、(3) Query Decomposition（分解复杂查询）。
>
> **Q: 何时应该使用查询重写？**
> A: 当用户查询简短、模糊，或需要分解复杂问题时。但要权衡延迟成本——如果查询已经明确或延迟要求极高，可能不需要。
>
> **Q: LLM 方法相比经典方法的优劣？**
> A: LLM 方法语义理解强、效果好，但延迟高、成本高；经典方法快速、可控，但语义理解弱。实践中常用混合策略：简单查询用经典方法，复杂查询用 LLM。
>
> **Q: 查询重写可能带来什么问题？**
> A: (1) 过度扩展引入噪声，降低精度；(2) 增加延迟；(3) LLM 改写可能偏离原始意图。需要配合重排序和人工评估。