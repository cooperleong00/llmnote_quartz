---
type: concept
description: 将文档切分为检索单元的策略，决定 RAG 系统的检索粒度与上下文完整性的平衡
aliases:
  - 分块策略
  - Chunking Strategy
  - Document Chunking
prerequisites:
  - "[[RAG]]"
  - "[[Embedding]]"
tags:
  - rag
  - data-preparation
created: 2026-03-07
updated: 2026-03-07T12:08
---

# Chunk Strategy

分块策略（Chunk Strategy）决定了如何将长文档切分为可检索的单元，是 [[RAG]] 系统中最关键的设计决策之一。它直接影响检索的精确度（能否找到相关信息）和上下文完整性（检索到的片段是否包含足够的语义）。

## 动机

> [!intuition] 为什么需要分块？
>
> **问题**：LLM 的上下文窗口有限，向量检索需要将文档表示为固定维度的向量。
>
> - 整篇文档作为检索单元 → 向量表示过于粗糙，无法精确定位相关段落
> - 逐句检索 → 缺乏上下文，语义不完整（如"它解决了这个问题"中的"它"和"这个问题"指代不明）
>
> **核心权衡**：
> - **小块**：检索精确，但上下文碎片化
> - **大块**：上下文完整，但检索粗糙，噪声多
>
> 分块策略的目标是在这两者之间找到最优平衡点。

## 常见分块策略

### 1. Fixed-Size Chunking（固定大小分块）

**机制**：按固定 token/字符数切分，可选重叠窗口。

```python
# 示例：512 tokens per chunk, 50 tokens overlap
chunks = split_text(document, chunk_size=512, overlap=50)
```

**优点**：
- 实现简单，计算高效
- 可预测的 chunk 数量和大小

**缺点**：
- 可能在句子中间切断，破坏语义
- 忽略文档的自然结构（段落、章节）

**适用场景**：文档结构不明显，或需要快速原型验证。

### 2. Semantic Chunking（语义分块）

**机制**：基于语义相似度切分，保持语义连贯的片段在同一 chunk。

**实现方式**：
- 计算相邻句子的 embedding 相似度
- 当相似度低于阈值时切分（表示主题转换）

```python
# 伪代码
embeddings = [embed(sentence) for sentence in sentences]
chunks = []
current_chunk = [sentences[0]]

for i in range(1, len(sentences)):
    similarity = cosine_similarity(embeddings[i-1], embeddings[i])
    if similarity < threshold:
        chunks.append(current_chunk)
        current_chunk = [sentences[i]]
    else:
        current_chunk.append(sentences[i])
```

**优点**：
- 保持语义完整性
- 自适应文档内容

**缺点**：
- 计算成本高（需要对每句话做 embedding）
- chunk 大小不可控，可能过大或过小

**适用场景**：高质量检索场景，对精度要求高于效率。

### 3. Recursive Chunking（递归分块）

**机制**：按文档结构层次递归切分（章节 → 段落 → 句子），直到满足大小限制。

```python
# LangChain 的 RecursiveCharacterTextSplitter
separators = ["\n\n", "\n", ". ", " ", ""]  # 优先级递减
# 先尝试用双换行符（段落）切分，如果仍超过限制，再用单换行符，依此类推
```

**优点**：
- 尊重文档自然结构
- 平衡语义完整性和大小控制

**缺点**：
- 依赖文档格式质量（如果格式混乱，效果差）
- 仍可能在不理想的位置切分

**适用场景**：结构化文档（技术文档、学术论文、书籍）。

### 4. Document-Structure-Aware Chunking（文档结构感知分块）

**机制**：解析文档结构（Markdown 标题、HTML 标签、PDF 章节），按逻辑单元切分。

**示例**：
- Markdown：按 `#` 标题层级切分
- HTML：按 `<section>`, `<article>`, `<div>` 切分
- PDF：按页码、章节标题切分

**优点**：
- 最大化语义完整性
- 保留文档层次信息（可作为 metadata）

**缺点**：
- 需要针对不同格式定制解析器
- 某些章节可能过长或过短

**适用场景**：高质量结构化文档，需要保留层次信息的场景。

## Chunk Size 的权衡

> [!warning] 没有"最佳" chunk size
> 最优大小取决于：
> - **查询类型**：事实性问题（小块）vs 需要上下文的问题（大块）
> - **文档类型**：技术文档（中等）vs 对话记录（小块）
> - **Embedding 模型**：模型的最大输入长度限制
> - **LLM 上下文窗口**：检索到的 chunks 总和不能超过 LLM 限制

**经验法则**：
- **小块**（128-256 tokens）：精确检索，适合事实查询
- **中块**（512-1024 tokens）：平衡精度和上下文，最常用
- **大块**（2048+ tokens）：保留完整上下文，适合需要理解长篇论述的场景

**实验建议**：
```python
# 对比不同 chunk size 的检索质量
for chunk_size in [256, 512, 1024, 2048]:
    evaluate_retrieval_quality(chunk_size)
```

## Overlap 策略


**问题**：固定边界切分可能导致关键信息跨越 chunk 边界，检索时丢失。

**解决方案**：相邻 chunks 之间保留重叠区域。

```
Chunk 1: [tokens 0-512]
Chunk 2: [tokens 462-974]  ← 50 tokens overlap with Chunk 1
Chunk 3: [tokens 924-1436] ← 50 tokens overlap with Chunk 2
```

**Overlap 大小的选择**：
- **小 overlap**（10-20%）：减少冗余，适合存储成本敏感的场景
- **中 overlap**（20-30%）：平衡冗余和召回，最常用
- **大 overlap**（50%+）：最大化召回，但存储和计算成本高

> [!warning] Overlap 的代价
> - 存储成本增加（更多 chunks）
> - 检索时可能返回重复内容
> - Embedding 计算成本增加

**何时使用 overlap**：
- ✅ Fixed-size chunking（必须）
- ✅ Recursive chunking（推荐）
- ⚠️ Semantic chunking（可选，语义边界已较清晰）
- ❌ Document-structure-aware（通常不需要，结构边界已明确）

## Metadata 增强

**核心思想**：为每个 chunk 附加元数据，支持混合检索（向量 + 过滤）。

**常见 metadata**：

| 类型      | 示例                     | 用途         |
| ------- | ---------------------- | ---------- |
| **文档级** | 文件名、作者、日期、来源           | 过滤检索范围     |
| **结构级** | 章节标题、页码、层级             | 定位原文、理解上下文 |
| **内容级** | 关键词、实体、主题标签            | 提升检索精度     |
| **统计级** | chunk 长度、位置（第几个 chunk） | 排序、去重      |

**实现示例**：
```python
chunk = {
    "text": "...",
    "metadata": {
        "source": "paper.pdf",
        "page": 5,
        "section": "3.2 Methodology",
        "keywords": ["RAG", "chunking", "retrieval"],
        "chunk_id": 12,
        "total_chunks": 50
    }
}
```

**检索时的应用**：
```python
# 混合检索：向量相似度 + metadata 过滤
results = vector_db.search(
    query_embedding,
    filter={"source": "paper.pdf", "section": "Methodology"}
)
```


> [!example] Metadata 的威力
> **场景**：用户问"论文的实验部分用了什么数据集？"
>
> - **无 metadata**：检索到整篇论文的相关 chunks，包括引言、相关工作等噪声
> - **有 metadata**：直接过滤 `section="Experiments"`，只检索实验部分，精度大幅提升

## 对检索质量的影响

| 维度 | 影响因素 | 优化方向 |
|------|----------|----------|
| **召回率** | chunk 太大 → 粗糙匹配，漏掉精确信息 | 减小 chunk size，增加 overlap |
| **精确率** | chunk 太小 → 上下文不足，误匹配 | 增大 chunk size，使用语义分块 |
| **上下文完整性** | 固定切分 → 破坏语义 | 使用结构感知或语义分块 |
| **检索速度** | chunk 太多 → 检索慢 | 增大 chunk size，减少 overlap |
| **可解释性** | 缺少 metadata → 无法溯源 | 添加文档结构 metadata |

**实验评估指标**：
- **Retrieval Accuracy**: 检索到的 chunks 是否包含答案
- **Context Sufficiency**: 检索到的 chunks 是否有足够上下文供 LLM 生成答案
- **Redundancy**: 检索到的 chunks 之间的重复度
- **Latency**: 检索 + 生成的总时间

## 实践指南

### 快速开始（推荐配置）

```python
# 通用场景的保守配置
chunk_size = 512       # tokens
overlap = 128          # 25% overlap
strategy = "recursive" # 尊重文档结构
metadata = ["source", "section", "page"]
```

### 针对不同场景的调优

| 场景 | Chunk Size | Strategy | Overlap | Metadata |
|------|------------|----------|---------|----------|
| **FAQ / 事实查询** | 256 | Fixed | 50 | 关键词、类别 |
| **技术文档** | 512-1024 | Recursive | 128 | 章节、页码 |
| **学术论文** | 1024 | Structure-aware | 0-50 | 章节、引用 |
| **对话记录** | 256-512 | Semantic | 50-100 | 时间、说话人 |
| **代码库** | 函数/类级别 | AST-based | 0 | 文件路径、函数名 |

### 迭代优化流程

1. **Baseline**：从 512 tokens + recursive chunking 开始
2. **评估**：在真实查询上测试召回率和精确率
3. **诊断**：
   - 召回率低 → 减小 chunk size 或增加 overlap
   - 精确率低 → 增大 chunk size 或改用语义分块
   - 上下文不足 → 增大 chunk size 或添加相邻 chunks
4. **A/B 测试**：对比不同配置的端到端效果（不只是检索指标，还要看生成质量）


## 局限性

> [!warning] 边界条件
>
> **分块无法解决的问题**：
> - **跨 chunk 推理**：答案需要综合多个不相邻的 chunks（如"对比第 2 章和第 5 章的观点"）
>   - 解决方案：使用 [[Query Rewriting]] 分解查询，或采用 multi-hop retrieval
> - **全局理解**：需要理解整篇文档的主旨（如"这篇论文的核心贡献是什么"）
>   - 解决方案：为整篇文档生成摘要作为额外的检索单元
> - **动态上下文需求**：不同查询需要不同粒度的 chunks
>   - 解决方案：多粒度索引（同时索引句子级、段落级、章节级）
>
> **常见误区**：
> - ❌ "更小的 chunk 总是更好"（会丢失上下文）
> - ❌ "overlap 越大越好"（冗余和成本增加）
> - ❌ "一套配置适用所有场景"（需要针对数据和查询类型调优）

## 相关概念

**核心依赖**：
- [[RAG]] — 分块是 RAG 索引阶段的核心步骤
- [[Embedding]] — chunk 需要被向量化才能检索

**检索优化**：
- [[Query Rewriting]] — 改写查询以匹配 chunk 粒度
- [[Query Expansion]] — 扩展查询以覆盖更多相关 chunks
- [[Sparse Retrieval]] — 基于关键词的检索，对 chunk 边界不敏感
- [[BM25]] — 稀疏检索算法，可与分块策略结合

**后续发展**：
- [[Context Compression]] — 检索后压缩 chunks，减少 LLM 输入
- [[Multi-Query]] — 生成多个查询变体，提升召回率
- [[Reranking]] — 对检索到的 chunks 重新排序

## 延伸阅读

**原始论文与深入材料**：
- [[RAG Survey (2023)]] — 系统性综述 RAG 技术，包括索引优化章节

> [!interview] 面试视角
>
> **Q: 如何选择 chunk size？**
> A: 没有通用最优值，需要权衡：
> - 查询类型（事实查询 → 小块，需要上下文 → 大块）
> - 文档类型（结构化 → 中等，对话 → 小块）
> - Embedding 模型限制（通常 512-1024 tokens）
> - 实验评估召回率和精确率，迭代调优
>
> **Q: Overlap 的作用是什么？**
> A: 防止关键信息跨越 chunk 边界导致检索失败。通常设置 20-30% overlap，但会增加存储和计算成本。语义分块和结构感知分块通常不需要 overlap。
>
> **Q: 如何处理跨 chunk 的问题？**
> A: 三种方案：
> 1. 增大 chunk size（简单但可能引入噪声）
> 2. Multi-hop retrieval（检索多轮，逐步定位）
> 3. 多粒度索引（同时索引不同粒度，根据查询选择）
