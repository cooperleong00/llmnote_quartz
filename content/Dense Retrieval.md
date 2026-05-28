---
type: concept
description: 基于神经网络学习的语义向量进行检索的方法，通过 embedding 将查询和文档映射到同一向量空间，计算相似度来衡量相关性
aliases:
  - 密集检索
  - Semantic Retrieval
  - Neural Retrieval
prerequisites:
  - "[[Embedding]]"
  - "[[Information Retrieval]]"
tags:
  - rag
  - information-retrieval
  - embedding
created: 2026-03-23
updated: 2026-03-23T00:53
---

# Dense Retrieval

密集检索（Dense Retrieval）是基于**神经网络学习的语义向量**进行检索的方法，通过 embedding 将查询和文档映射到同一低维密集向量空间，计算向量相似度来衡量相关性。与 [[Sparse Retrieval|稀疏检索]] 相比，Dense Retrieval 能够理解**语义**而非仅匹配字面词汇，解决了词汇不匹配（vocabulary mismatch）问题。

## 动机

> [!intuition] 为什么需要 Dense Retrieval？
> 
> [[Sparse Retrieval|稀疏检索]] 的核心问题：**词汇不匹配**（Vocabulary Mismatch）
> 
> **示例**：
> - Query: "如何训练神经网络"
> - Document: "深度学习模型的优化方法"
> 
> 虽然语义高度相关，但没有共同词项 → [[BM25]] 无法检索到。
> 
> **其他词汇不匹配场景**：
> - 同义词："汽车" vs "车辆"
> - 释义："提升性能" vs "优化效果"
> - 跨语言：英文查询 vs 中文文档
> 
> **Dense Retrieval 的核心洞察**：
> - 用神经网络学习**语义表示**，而非依赖字面词汇
> - 将查询和文档映射到同一向量空间，语义相似 → 向量接近
> - 通过大规模训练数据学习"什么是相关"

---

## 核心机制

### Bi-Encoder 架构

Dense Retrieval 的标准架构是 **Bi-Encoder**（双编码器）：

```
Query: "如何训练神经网络"
    ↓
Query Encoder (BERT)
    ↓
q = [0.2, -0.5, 0.8, ...]  (768-dim)

Document: "深度学习模型的优化方法"
    ↓
Document Encoder (BERT)
    ↓
d = [0.3, -0.4, 0.7, ...]  (768-dim)

相似度 = cosine(q, d) = 0.92  (高相关性)
```

**关键特性**：
- 查询和文档**独立编码**（可以预先计算文档向量）
- 输出低维密集向量（通常 768 或 1024 维）
- 每个维度都有值（密集），不像稀疏向量大部分为 0

> [!comparison] Bi-Encoder vs Cross-Encoder
> 
> **Bi-Encoder（Dense Retrieval 常用）**：
> - 查询和文档独立编码：`score = sim(encode(q), encode(d))`
> - 优点：文档可预先编码，检索时只需计算相似度（快）
> - 缺点：无法建模查询-文档的细粒度交互
> 
> **Cross-Encoder（[[Reranking]] 常用）**：
> - 查询和文档联合编码：`score = model([q, d])`
> - 优点：通过 attention 捕捉词级别交互（准确）
> - 缺点：每个查询-文档对都需要前向传播（慢）
> 
> **为什么 Dense Retrieval 用 Bi-Encoder**：
> - 需要在百万/亿级文档中检索，必须预先计算文档向量
> - Cross-Encoder 无法预计算，每次查询需要对所有文档做前向传播（不可行）

### 向量相似度计算

最常用的相似度度量是 **余弦相似度**（Cosine Similarity）：

$$
\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} = \cos(\theta)
$$

其中 $\theta$ 是两个向量的夹角。

> [!intuition] 为什么用余弦相似度？
> 
> - **归一化**：消除向量长度的影响，只关注方向
> - **语义直觉**：方向相近 → 语义相似
> - **高效计算**：如果向量已归一化，余弦相似度 = 点积

**其他相似度度量**：
- **点积**（Dot Product）：$\mathbf{q} \cdot \mathbf{d}$（更快，但受向量长度影响）
- **欧氏距离**（Euclidean Distance）：$\|\mathbf{q} - \mathbf{d}\|$（较少使用）

### 近似最近邻搜索（ANN）

在百万级文档中找到最相似的 top-k 文档，暴力计算所有相似度不可行。

**问题规模**：
- 100 万文档 × 768 维 × 每次查询 → ~7 亿次浮点运算
- 暴力搜索延迟：~1 秒（不可接受）

**解决方案：近似最近邻（Approximate Nearest Neighbor, ANN）**

```
精确搜索（Brute Force）
  ↓
计算所有文档的相似度 → 排序 → 返回 top-k
延迟: ~1s，准确率: 100%

近似搜索（ANN）
  ↓
用索引结构快速定位候选 → 返回 top-k
延迟: ~10ms，准确率: ~95%
```

**常用 ANN 算法**：

| 算法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **FAISS** | 聚类 + 倒排索引 | 极快，支持 GPU | 需要调参 |
| **HNSW** | 分层图结构 | 高召回率，易用 | 内存占用大 |
| **ScaNN** | 量化 + 树结构 | 平衡速度和准确率 | 实现复杂 |

> [!example] FAISS 示例
> 
> ```python
> import faiss
> import numpy as np
> 
> # 假设有 100 万个 768 维文档向量
> doc_vectors = np.random.randn(1000000, 768).astype('float32')
> 
> # 构建索引（IVF: Inverted File Index）
> quantizer = faiss.IndexFlatIP(768)  # 内积（点积）
> index = faiss.IndexIVFFlat(quantizer, 768, 1000)  # 1000 个聚类
> index.train(doc_vectors)
> index.add(doc_vectors)
> 
> # 查询
> query = np.random.randn(1, 768).astype('float32')
> distances, indices = index.search(query, k=10)  # 返回 top-10
> ```

---

## 典型模型

### 早期模型

#### DPR (Dense Passage Retrieval, 2020)

第一个广泛使用的 dense retrieval 模型：

- **架构**：双 BERT 编码器（query encoder + passage encoder）
- **训练**：对比学习（contrastive learning）
  - 正样本：相关的查询-文档对
  - 负样本：不相关的文档（in-batch negatives）
- **数据集**：MS MARCO, Natural Questions

#### ANCE (2020)

**Approximate Nearest Neighbor Negative Contrastive Learning**：

- **核心改进**：困难负样本挖掘（hard negative mining）
- **方法**：用当前模型检索最相似但不相关的文档作为负样本
- **效果**：比 DPR 提升 ~5% 准确率

### 现代模型

#### ColBERT (2020)

**Late Interaction** 架构：

```
Query: "深度学习"
  ↓
Token-level embeddings: [e_深, e_度, e_学, e_习]

Document: "神经网络训练"
  ↓
Token-level embeddings: [e_神, e_经, e_网, e_络, e_训, e_练]

相似度 = MaxSim(query_tokens, doc_tokens)
```

**优势**：
- 比 bi-encoder 更精确（token 级别交互）
- 比 cross-encoder 更快（可预计算文档 token embeddings）

#### 工业级模型

| 模型 | 维度 | 特点 |
|------|------|------|
| **BGE (BAAI)** | 768/1024 | 中英双语，开源，性能优异 |
| **E5 (Microsoft)** | 768/1024 | 多语言，指令微调 |
| **text-embedding-ada-002 (OpenAI)** | 1536 | 闭源 API，质量高但成本高 |
| **Cohere Embed** | 1024/4096 | 支持长文本（4096 tokens） |

---

## 训练方法

### 对比学习（Contrastive Learning）

Dense Retrieval 的核心训练范式：

**目标**：让相关的查询-文档对向量接近，不相关的远离。

**损失函数**（InfoNCE Loss）：

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+))}{\exp(\text{sim}(q, d^+)) + \sum_{i=1}^{N} \exp(\text{sim}(q, d_i^-))}
$$

其中：
- $d^+$：正样本（相关文档）
- $d_i^-$：负样本（不相关文档）
- $N$：负样本数量

> [!intuition] 对比学习的直觉
> 
> 想象一个"语义空间"：
> - 训练前：查询和文档随机分布
> - 训练后：相关的查询-文档聚在一起，不相关的分散开
> 
> 类比：学习"什么是猫"
> - 正样本：各种猫的图片（拉近）
> - 负样本：狗、椅子、汽车（推远）

### 负样本策略

负样本的质量直接影响模型性能：

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| **Random** | 随机采样文档 | 简单 | 太容易，学不到东西 |
| **In-batch** | 同 batch 内其他样本 | 高效，无额外计算 | 负样本质量一般 |
| **Hard Negative** | 用 BM25/模型检索相似但不相关的文档 | 质量高，提升显著 | 需要额外检索 |
| **Cross-batch** | 跨 batch 共享负样本 | 增加负样本多样性 | 实现复杂 |

> [!example] Hard Negative Mining 示例
> 
> Query: "Python 教程"
> - ✅ 正样本："Python 入门指南"
> - ❌ 简单负样本："Java 教程"（太容易区分）
> - ❌ **困难负样本**："Python 库文档"（相似但不相关，更有挑战）

---

## 与 Sparse Retrieval 的系统性对比

> [!comparison] Dense vs Sparse Retrieval
> 
> | 维度 | [[Sparse Retrieval]] | Dense Retrieval |
> |------|---------------------|-----------------|
> | **表示** | 高维稀疏向量（10k-100k 维） | 低维密集向量（768-1024 维） |
> | **匹配方式** | 词项重叠 | 语义相似度 |
> | **训练需求** | 无需训练 | 需要大量标注数据 |
> | **语义理解** | 无（字面匹配） | 强（学习语义） |
> | **精确匹配** | 优秀（专有名词、代码） | 较弱 |
> | **可解释性** | 高（能看到匹配词） | 低（黑盒向量） |
> | **计算成本** | 低（倒排索引） | 高（需要编码器 + ANN） |
> | **存储成本** | 低（稀疏向量） | 高（密集向量） |
> | **冷启动** | 优秀（开箱即用） | 差（需要训练） |
> | **跨语言** | 不支持 | 支持（多语言模型） |
> 
> **互补性**：两者各有优势，实际系统常用 [[Hybrid Search|混合检索]] 结合两者。

---

## 局限性

> [!warning] Dense Retrieval 的边界
> 
> 1. **训练数据依赖**
>    - 需要大量高质量的查询-文档对（通常 10 万+）
>    - 领域迁移能力弱（在通用数据上训练的模型在医疗/法律领域可能表现不佳）
> 
> 2. **精确匹配能力弱**
>    - 专有名词："GPT-4" vs "GPT-3.5"（可能被视为相似）
>    - 代码片段：`list.append()` vs `list.extend()`
>    - ID/编号：订单号、产品型号
> 
> 3. **计算成本高**
>    - 需要 GPU 进行编码（每次查询 ~10ms）
>    - 向量存储占用大（100 万文档 × 768 维 × 4 字节 ≈ 3GB）
> 
> 4. **黑盒性**
>    - 难以解释"为什么检索到这个文档"
>    - 调试困难（不像 BM25 能看到匹配词）
> 
> 5. **对抗样本脆弱**
>    - 语义相似但不相关的文档可能被错误检索
>    - 示例：Query "如何学习 Python"，检索到 "如何学习 Java"

---

## 混合检索策略

实际系统中，Dense 和 Sparse Retrieval 通常结合使用：

### 并行混合

```
Query: "深度学习优化方法"
    ↓
┌─────────────────┬─────────────────┐
│  BM25           │  Dense          │
│  (精确匹配)      │  (语义理解)      │
└─────────────────┴─────────────────┘
    ↓                   ↓
top-50 候选        top-50 候选
    ↓                   ↓
        合并 + 去重 + 加权
              ↓
         top-100 候选
              ↓
         [[Reranking]]
              ↓
         top-10 结果
```

### 权重分配策略

| 场景 | Sparse 权重 | Dense 权重 | 原因 |
|------|------------|-----------|------|
| 专有名词查询 | 70% | 30% | 精确匹配更重要 |
| 语义问答 | 30% | 70% | 语义理解更重要 |
| 代码检索 | 80% | 20% | 字面匹配关键 |
| 跨语言检索 | 10% | 90% | 只有 dense 能跨语言 |

> [!example] 混合检索实现
> 
> ```python
> # 并行检索
> bm25_results = bm25.search(query, top_k=50)
> dense_results = dense_retriever.search(query, top_k=50)
> 
> # 加权融合（Reciprocal Rank Fusion）
> def rrf_score(rank, k=60):
>     return 1 / (k + rank)
> 
> scores = {}
> for rank, doc in enumerate(bm25_results):
>     scores[doc.id] = scores.get(doc.id, 0) + 0.3 * rrf_score(rank)
> 
> for rank, doc in enumerate(dense_results):
>     scores[doc.id] = scores.get(doc.id, 0) + 0.7 * rrf_score(rank)
> 
> # 按分数排序
> final_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
> ```

---

## 在 RAG Pipeline 中的应用

Dense Retrieval 是 [[RAG]] 系统的核心组件：

```
用户查询："解释一下 Transformer 的注意力机制"
    ↓
查询优化（[[Query Rewriting]]）
    ↓
Dense Retrieval（检索相关文档）
    ↓ top-100 候选
[[Reranking]]（精细排序）
    ↓ top-10 精选
上下文构建
    ↓
LLM 生成答案
```

### 实现示例

使用 `sentence-transformers` 库：

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 加载模型
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# 2. 编码文档（离线预计算）
documents = ["doc1 text", "doc2 text", ...]
doc_embeddings = model.encode(documents, normalize_embeddings=True)

# 3. 构建 FAISS 索引
index = faiss.IndexFlatIP(768)  # Inner Product (余弦相似度)
index.add(doc_embeddings)

# 4. 查询时编码 + 检索
query = "user query"
query_embedding = model.encode([query], normalize_embeddings=True)
distances, indices = index.search(query_embedding, k=10)

# 5. 返回 top-10 文档
results = [documents[i] for i in indices[0]]
```

---

## 速查

> [!example] 快速参考
> 
> **常用开源模型**：
> - 英文：`BAAI/bge-large-en-v1.5`（1024 维）
> - 中文：`BAAI/bge-large-zh-v1.5`（1024 维）
> - 多语言：`intfloat/multilingual-e5-large`（1024 维）
> 
> **典型配置**：
> - 向量维度：768 或 1024
> - 相似度度量：余弦相似度（Cosine Similarity）
> - ANN 算法：FAISS (IVFFlat) 或 HNSW
> - 召回数量：top-50 ~ top-100
> 
> **性能指标**：
> - 编码延迟：~10ms/query（GPU）
> - 检索延迟：~10ms（FAISS，100 万文档）
> - 存储：~3GB（100 万文档 × 768 维）

> [!interview] 面试要点
> 
> **Q: Dense Retrieval 和 Sparse Retrieval 的核心区别是什么？**
> A: 
> - **表示**：Dense 用低维密集向量（768 维），Sparse 用高维稀疏向量（10k+ 维）
> - **匹配**：Dense 基于语义相似度，Sparse 基于词项重叠
> - **训练**：Dense 需要大量标注数据训练，Sparse 无需训练
> - **优势**：Dense 能理解语义（解决词汇不匹配），Sparse 精确匹配能力强
> 
> **Q: 为什么 Dense Retrieval 用 Bi-Encoder 而不是 Cross-Encoder？**
> A: 
> - **可预计算**：Bi-encoder 可以预先计算文档向量，检索时只需计算相似度
> - **效率**：Cross-encoder 需要对每个查询-文档对做前向传播，无法在百万级文档中实时检索
> - **权衡**：Bi-encoder 用于第一阶段快速召回，Cross-encoder 用于第二阶段 [[Reranking]]
> 
> **Q: 什么时候应该用混合检索而不是纯 Dense Retrieval？**
> A: 
> - 需要精确匹配（专有名词、代码、ID）
> - 资源受限（无法承受 Dense 的计算成本）
> - 需要可解释性（能看到匹配词）
> - 冷启动阶段（没有训练数据）
> 
> **Q: Dense Retrieval 的训练需要什么数据？**
> A: 
> - **查询-文档对**：(query, relevant_doc, irrelevant_docs)
> - **数量**：通常需要 10 万+ 对
> - **质量**：困难负样本（hard negatives）比随机负样本更重要
> - **来源**：人工标注、点击日志、合成数据（用 LLM 生成查询）

---

## 延伸阅读

**核心概念**：
- [[Embedding]] — 向量表示的基础
- [[Sparse Retrieval]] — 基于词项匹配的检索方法
- [[Hybrid Search]] — 结合 Dense 和 Sparse 的混合策略
- [[Reranking]] — 两阶段检索的第二阶段

**相关技术**：
- [[BM25]] — 最广泛使用的稀疏检索算法
- [[Query Rewriting]] — 查询优化技术
- [[Context Compression]] — 压缩检索到的上下文

**RAG 应用**：
- [[RAG]] — 检索增强生成的完整流程
- [[MOC - RAG]] — RAG 领域导航
