---
type: concept
description: 在 RAG 生成前压缩检索上下文，通过过滤、提取或重写减少 token 数量，平衡效率与信息保留
aliases:
  - 上下文压缩
  - Context Curation
prerequisites:
  - "[[RAG]]"
tags:
  - rag
  - optimization
created: 2026-03-07
updated: 2026-03-07T12:11
---

# Context Compression

上下文压缩（Context Compression）在 RAG 系统中，于检索后、生成前对检索到的文档进行压缩，通过移除冗余信息、提取关键片段或重写内容，减少输入 LLM 的 token 数量，同时尽可能保留相关信息。

## 动机

> [!intuition] 为什么需要压缩上下文？
> 检索系统倾向于"宁可错召回，不可漏召回"——返回 10 篇文档比漏掉 1 篇关键信息更安全。但这导致三个问题：
>
> 1. **Context window 限制**：检索到的文档可能超过 LLM 的输入长度限制
> 2. **成本问题**：更长的输入 = 更高的 API 费用（按 token 计费）
> 3. **噪声干扰**：无关信息会稀释注意力，降低生成质量（"Lost in the Middle" 现象）
>
> Context Compression 的核心权衡：**用少量信息损失换取显著的效率提升**。

### 检索的召回-精度权衡

```
检索阶段：高召回（Recall）优先
    ↓
  返回 10-20 篇文档（可能包含冗余/噪声）
    ↓
压缩阶段：提升精度（Precision）
    ↓
  保留 2-3 篇核心内容 → 输入 LLM
```

这种两阶段策略比"直接检索少量文档"更安全——先广撒网，再精筛选。

## 核心技术

### 1. Extractive Compression（提取式压缩）

**思路**：从原文中选择关键句子/段落，丢弃其余部分。

**方法**：
- **基于相关性评分**：计算每个句子与查询的相似度，保留 top-k
- **基于位置启发式**：保留文档开头/结尾（通常包含摘要信息）
- **基于关键词匹配**：保留包含查询关键词的句子

**优势**：
- 保留原文表述，不引入新错误
- 实现简单，速度快

**劣势**：
- 可能丢失上下文连贯性（句子之间的逻辑关系）
- 压缩率有限（句子是最小单位）

> [!example] 示例
> **原文**（200 tokens）：
> "深度学习是机器学习的一个分支。它使用多层神经网络。神经网络由神经元组成。每个神经元执行简单计算。深度学习在图像识别中表现出色..."
>
> **提取式压缩**（50 tokens）：
> "深度学习是机器学习的一个分支。它使用多层神经网络。深度学习在图像识别中表现出色..."

### 2. Abstractive Compression（生成式压缩）

**思路**：用 LLM 重写文档，生成更简洁的摘要。

**方法**：
- **Prompt-based summarization**：
  ```
  Summarize the following document in 100 tokens, 
  focusing on information relevant to: [query]
  ```
- **Fine-tuned compression model**：训练专门的压缩模型（如 LongLLMLingua）

**优势**：
- 更高的压缩率（可以改写、合并信息）
- 保持语义连贯性

**劣势**：
- 可能引入幻觉（生成不存在的信息）
- 额外的 LLM 调用成本和延迟

### 3. LLM-based Filtering（基于 LLM 的过滤）

**思路**：让 LLM 判断每个文档/段落是否与查询相关，过滤无关内容。

**方法**：
- **Binary relevance判断**：
  ```
  Is the following passage relevant to answering: [query]?
  Answer: Yes/No
  ```
- **Relevance scoring**：让 LLM 给每个段落打分（1-10），保留高分内容

**优势**：
- 语义理解能力强（比简单的相似度计算更准确）
- 可以处理复杂的相关性判断（如"这段话虽然不直接回答问题，但提供了重要背景"）

**劣势**：
- 成本高（每个段落都需要 LLM 调用）
- 可能过度过滤（LLM 的判断不一定准确）

### 4. Hybrid Approaches（混合方法）

实践中常用的组合策略：

```
检索 → [快速过滤] → [Reranking] → [LLM 压缩] → 生成
         ↑              ↑              ↑
      移除明显无关   精排序前 k 篇   提取/改写关键内容
```

**示例流程**：
1. 检索返回 20 篇文档
2. 用 BM25 快速过滤，保留 10 篇
3. 用 [[Reranking]] 模型重排序，保留 top-3
4. 对每篇文档做提取式压缩，从 500 tokens 压缩到 150 tokens
5. 总输入从 10,000 tokens 降至 450 tokens

## 在 RAG Pipeline 中的位置

Context Compression 发生在**检索后、生成前**（Post-Retrieval）：

```
用户查询
  ↓
[Query Rewriting]  ← Pre-Retrieval 优化
  ↓
检索
  ↓
[Reranking]        ← 重排序（调整顺序）
  ↓
[Context Compression]  ← 压缩（减少长度）
  ↓
LLM 生成
```

> [!comparison] Reranking vs Context Compression
> 两者都是 Post-Retrieval 优化，但目标不同：
>
> | 维度 | [[Reranking]] | Context Compression |
> |------|---------------|---------------------|
> | **目标** | 调整文档顺序，将最相关内容放在关键位置 | 减少 token 数量 |
> | **输出** | 重新排序的文档列表（长度不变） | 压缩后的文档（长度减少） |
> | **何时使用** | Context window 足够，但需要优化相关性 | Context window 不足，或需要降低成本 |
> | **可组合性** | 先 Rerank 再 Compress（常见组合） | 可单独使用或与 Rerank 结合 |

**组合使用**：
- [[Query Rewriting]] + Context Compression：多查询检索后压缩冗余内容
- [[Reranking]] + Context Compression：先精排序，再压缩 top-k 文档

## 权衡与选择

### 核心权衡：信息损失 vs 效率提升

| 压缩率 | 信息保留 | 适用场景 |
|--------|----------|----------|
| **低（50-70%）** | 高 | 查询复杂，需要完整上下文 |
| **中（30-50%）** | 中 | 平衡成本与质量 |
| **高（10-30%）** | 低 | 简单查询，或成本敏感场景 |

### 何时使用 Context Compression？

**适合的场景**：
- 检索返回大量文档，超过 context window
- 成本敏感（API 按 token 计费）
- 检索结果包含大量冗余信息（如多篇文档重复相同内容）

**不适合的场景**：
- 查询需要完整上下文（如"总结这篇文章的所有论点"）
- 检索结果已经很精简（如只返回 2-3 篇短文档）
- 对生成质量要求极高，不能容忍信息损失

### 方法选择指南

| 场景          | 推荐方法                | 原因          |
| ----------- | ------------------- | ----------- |
| **低延迟要求**   | 提取式压缩               | 无需额外 LLM 调用 |
| **高压缩率需求**  | 生成式压缩               | 可以改写、合并信息   |
| **复杂相关性判断** | LLM-based Filtering | 语义理解能力强     |
| **成本敏感**    | 提取式 + 启发式过滤         | 避免额外 LLM 成本 |

## 局限性

> [!warning] 边界条件
> 1. **不可逆的信息损失**：压缩后无法恢复原文，可能丢失关键细节
> 2. **压缩质量难以评估**：很难事先知道压缩后是否保留了足够信息
> 3. **额外延迟**：生成式压缩需要额外的 LLM 调用
> 4. **可能引入错误**：生成式方法可能产生幻觉
> 5. **与长文本能力的权衡**：随着 LLM context window 增大（如 Gemini 1M tokens），压缩的必要性降低

### 常见误区

- **误区 1**："压缩率越高越好" → 过度压缩会丢失关键信息，导致生成质量下降
- **误区 2**："压缩可以替代 Reranking" → 两者目标不同，通常需要组合使用
- **误区 3**："生成式压缩一定比提取式好" → 生成式可能引入幻觉，提取式更安全

## 实现示例

### 提取式压缩：基于相似度

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def extractive_compress(documents, query, top_k=5):
    """保留与查询最相关的 top-k 个句子"""
    query_emb = model.encode(query)
    
    # 将所有文档拆分为句子
    sentences = []
    for doc in documents:
        sentences.extend(doc.split('. '))
    
    # 计算每个句子与查询的相似度
    sentence_embs = model.encode(sentences)
    scores = util.cos_sim(query_emb, sentence_embs)[0]
    
    # 保留 top-k 句子
    top_indices = scores.argsort(descending=True)[:top_k]
    compressed = '. '.join([sentences[i] for i in sorted(top_indices)])
    
    return compressed
```

### 生成式压缩：Prompt-based

```python
def abstractive_compress(documents, query, max_tokens=200):
    """用 LLM 生成压缩摘要"""
    context = "\n\n".join(documents)
    
    prompt = f"""
    Summarize the following documents in {max_tokens} tokens or less,
    focusing ONLY on information relevant to answering: "{query}"
    
    Documents:
    {context}
    
    Compressed summary:
    """
    
    return llm.generate(prompt, max_tokens=max_tokens)
```

### LLM-based Filtering

```python
def llm_filter(documents, query):
    """让 LLM 判断每个文档是否相关"""
    relevant_docs = []
    
    for doc in documents:
        prompt = f"""
        Is the following passage relevant to answering: "{query}"?
        Answer only "Yes" or "No".
        
        Passage: {doc}
        """
        
        response = llm.generate(prompt, max_tokens=5)
        if "yes" in response.lower():
            relevant_docs.append(doc)
    
    return relevant_docs
```

## 相关概念

**RAG Pipeline 组件**：
- [[Reranking]] — 重排序检索结果，优化相关性（与 Context Compression 常组合使用）
- [[Query Rewriting]] — Pre-Retrieval 优化，改写查询提升检索质量

**长文本处理的替代方案**：
- [[MemAgent (2025)]] — 通过 memory overwrite 机制处理长文本，避免压缩带来的信息损失

**基础概念**：
- [[RAG]] — 检索增强生成的基本框架
- [[Dense Retrieval]] — 检索阶段的核心技术

## 延伸阅读

**Survey 与综述**：
- [[RAG Survey (2023)]] — 将 Context Compression 归类为 Post-Retrieval 优化的核心技术

**实际应用**：
- [[Query Rewriting#相关概念]] — 讨论了 Query Rewriting + Context Compression 的组合策略
