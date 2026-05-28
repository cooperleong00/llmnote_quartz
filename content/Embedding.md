---
type: concept
description: 将离散符号映射到连续向量空间的技术，使语义相似的对象在向量空间中距离接近，是 Dense Retrieval 和语义搜索的基础
aliases:
  - 嵌入
  - 向量表示
  - Vector Representation
prerequisites:
  - "[[Tokenization]]"
  - "[[Transformer]]"
tags:
  - rag
  - foundations
  - retrieval
created: 2026-03-23
updated: 2026-03-23T00:52
---

# Embedding

Embedding（嵌入）将离散符号（文本、图像等）映射到连续向量空间，使得**语义相似的对象在向量空间中距离接近**。这是从符号到几何的转换：原本无法比较的文本片段，通过 embedding 变成可以计算距离的向量。这项技术是 [[Dense Retrieval]]、[[Reranking]]、语义搜索的基础——没有 embedding，就无法用"相似度"来检索信息。

## 动机：为什么需要向量空间？

> [!intuition] 从符号到几何的跨越
>
> **离散符号的困境**：
> - "dog" 和 "puppy" 语义相近，但作为符号它们没有任何关系
> - "king" - "man" + "woman" = ? 在符号世界中无法计算
> - 无法回答"找出与查询最相似的文档"——因为符号之间没有"距离"
>
> **向量空间的优势**：
> - **可计算相似度**：通过向量距离（cosine similarity, L2 distance）量化语义相似性
> - **支持代数运算**：向量可以加减，捕捉语义关系（如 king - man + woman ≈ queen）
> - **支持高效检索**：通过近似最近邻搜索（ANN）在百万级向量中快速找到相似项

**核心问题**：如何将符号映射到向量，使得语义相似性反映为几何距离？

## 核心机制

### 向量空间的性质

> [!math] 数学形式
> 
> Embedding 是一个映射函数：
> $$f: \mathcal{V} \rightarrow \mathbb{R}^d$$
> 
> - $\mathcal{V}$：符号空间（如词汇表、句子集合）
> - $\mathbb{R}^d$：$d$ 维连续向量空间（通常 $d = 128 \sim 1536$）
> - 目标：$\text{sim}(x, y) \approx \text{dist}(f(x), f(y))$
>   - 语义相似的符号 → 向量距离近
>   - 语义不同的符号 → 向量距离远

**关键设计选择**：

1. **维度 $d$**：
   - 太小（如 50）：无法捕捉复杂语义
   - 太大（如 4096）：计算成本高，易过拟合
   - 常见选择：768（BERT）、1536（OpenAI ada-002）、1024（BGE）

2. **距离度量**：
   | 度量 | 公式 | 特点 |
   |------|------|------|
   | **Cosine Similarity** | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ | 只关注方向，忽略长度；最常用 |
   | **Euclidean (L2)** | $\|\mathbf{u} - \mathbf{v}\|_2$ | 考虑绝对距离 |
   | **Dot Product** | $\mathbf{u} \cdot \mathbf{v}$ | 快速但受向量长度影响 |

> [!intuition] 为什么 Cosine Similarity 最常用？
> 
> 文本长度不应影响相似度判断："I love dogs" 和 "I really really love dogs" 语义几乎相同，但后者可能产生更大的向量。Cosine similarity 归一化了长度，只看方向（语义），这更符合直觉。

### 从词到句子：Embedding 的演进

#### 1. 静态词向量（2013-2018）

**Word2Vec / GloVe**：每个词有固定的向量表示

```
"bank" → [0.2, -0.5, 0.8, ...]  # 无论上下文
```

**局限**：无法区分多义词
- "bank" 在 "river bank" 和 "bank account" 中应该有不同的表示

#### 2. 上下文相关词向量（2018-2020）

**BERT / RoBERTa**：同一个词在不同上下文中有不同的向量

```
"bank" in "river bank"    → [0.3, 0.1, -0.2, ...]
"bank" in "bank account"  → [-0.1, 0.8, 0.5, ...]
```

**问题**：BERT 是为 token-level 任务设计的，直接用于句子检索效果不佳
- 简单的 pooling（如取 [CLS] token）丢失了大量信息

#### 3. 句子级 Embedding（2019-现在）

**Sentence-BERT (SBERT)**：专门为句子相似度优化

- 使用 **Siamese 网络**训练：让相似句子的向量接近，不相似的远离
- 训练数据：自然语言推理（NLI）、问答对、检索数据

**现代 Embedding 模型**：
- **OpenAI text-embedding-ada-002**：1536 维，通用性强
- **BGE (BAAI General Embedding)**：中英文双语，开源
- **E5 (Text Embeddings by Weakly-Supervised Contrastive Pre-training)**：多任务训练
- **Cohere embed-v3**：支持多语言，可调节检索/聚类模式

> [!example] 训练目标示例
> 
> **Contrastive Learning**（对比学习）：
> - 正样本对：(query, relevant_doc) → 拉近距离
> - 负样本对：(query, irrelevant_doc) → 推远距离
> 
> 损失函数（InfoNCE）：
> $$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{d^-} \exp(\text{sim}(q, d^-) / \tau)}$$
> 
> - $q$：查询
> - $d^+$：相关文档（正样本）
> - $d^-$：不相关文档（负样本）
> - $\tau$：温度参数

## 应用场景

### 1. Dense Retrieval（密集检索）

将查询和文档都转换为 embedding，通过向量相似度检索：

```
Query: "What is machine learning?"
  ↓ Embedding
[0.2, -0.5, 0.8, ...]
  ↓ ANN Search (Approximate Nearest Neighbor)
Top-K 最相似的文档向量
```

这是 [[RAG]] 系统中 [[Dense Retrieval]] 的核心技术。

### 2. Clustering（聚类）

将相似的文本自动分组：
- 新闻聚类：将相同主题的新闻归为一类
- 用户画像：根据行为文本聚类用户群体

### 3. Classification（分类）

用 embedding 作为特征，训练分类器：
- 情感分析：正面/负面
- 意图识别：查询、投诉、咨询

### 4. Recommendation（推荐）

计算用户历史与候选项的相似度：
- 文章推荐：推荐与用户阅读历史相似的文章
- 商品推荐：基于描述文本的相似度

## 局限性

> [!warning] Embedding 的边界

### 1. 维度诅咒

高维空间中，所有点之间的距离趋于相等：
- 当维度很高时，"最近邻"和"最远邻"的距离差异变小
- 需要大量数据才能有效学习高维空间的结构

### 2. 训练数据偏差

Embedding 模型继承训练数据的偏见：
- 如果训练数据中"医生"多与"男性"共现，模型会学到这种偏差
- 领域外数据（out-of-domain）效果下降

### 3. 无法捕捉精确匹配

Embedding 关注语义相似性，但有时需要精确匹配：
- 查询 "COVID-19" 应该匹配包含 "COVID-19" 的文档，而非只是"疫情"相关
- 专有名词、代码、ID 等需要精确匹配的场景

**解决方案**：[[Hybrid Retrieval]]
- 结合 [[Sparse Retrieval]]（如 [[BM25]]）捕捉精确匹配
- 结合 Dense Retrieval 捕捉语义相似性

### 4. 计算与存储成本

- **存储**：百万文档 × 1536 维 × 4 bytes ≈ 6GB
- **检索**：需要 ANN 索引（如 FAISS、Annoy）加速，否则暴力搜索太慢

## 与相关概念的关系

**前置步骤**：
- [[Tokenization]] — 将文本转换为 token 序列，是 embedding 的输入
- [[Transformer]] — 现代 embedding 模型的基础架构

**应用技术**：
- [[Dense Retrieval]] — 使用 embedding 进行语义检索
- [[Reranking]] — 在 embedding 检索后精细排序
- [[Hybrid Retrieval]] — 结合 embedding 和稀疏检索

**对比技术**：
- [[Sparse Retrieval]] — 基于词项匹配，不需要 embedding，互补关系

> [!interview] 面试要点

**Q: Embedding 和 One-hot Encoding 有什么区别？**

A: One-hot 是稀疏的、高维的、无语义的（每个词是独立的维度）；Embedding 是稠密的、低维的、有语义的（相似词在向量空间中接近）。One-hot 维度 = 词表大小（如 50k），Embedding 维度通常 128-1536。

**Q: 为什么 BERT 的 [CLS] token 不适合直接用于句子检索？**

A: BERT 是为 token-level 任务（如 MLM）预训练的，[CLS] token 没有专门优化句子级相似度。Sentence-BERT 通过 Siamese 网络和对比学习，让句子向量直接反映语义相似性。

**Q: Dense Retrieval 什么时候不如 BM25？**

A: 当需要精确匹配时（专有名词、代码、ID），或者查询和文档的词汇重叠很重要时。Dense retrieval 可能因为"过度泛化"而召回不相关但语义相似的文档。实践中常用 Hybrid Retrieval 结合两者优势。

## 延伸阅读

**基础论文**：
- Word2Vec: Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
- GloVe: Pennington et al., "GloVe: Global Vectors for Word Representation" (2014)
- Sentence-BERT: Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)

**现代模型**：
- [[BGE]] — 开源中英文双语 embedding 模型
- [[E5]] — 多任务对比学习的 embedding 模型
