---
type: paper
description: 系统性综述 RAG 技术演进的三大范式（Naive/Advanced/Modular）及其核心组件（检索/生成/增强）
aliases:
  - RAG 综述
  - Retrieval-Augmented Generation Survey
tags:
  - rag
  - survey
  - retrieval
  - llm
created: 2026-03-06
updated: 2026-03-06T18:05
---

# Retrieval-Augmented Generation for Large Language Models: A Survey (2023)

> [!paper] 论文信息
> **作者**: Yunfan Gao et al. (Tongji University, Fudan University)
> **发表**: arXiv 2312.10997 (2023)
> **类型**: Comprehensive Survey

这篇综述系统性地梳理了 RAG（Retrieval-Augmented Generation）技术的演进路径，将其发展划分为三个范式（Naive RAG → Advanced RAG → Modular RAG），并从检索（Retrieval）、生成（Generation）、增强（Augmentation）三个核心维度分析了 RAG 系统的技术栈。覆盖 26 个任务、约 50 个数据集，提供了完整的评估框架。

## 核心贡献

### 1. RAG 的三大演进范式


#### Naive RAG（基础范式）

**核心流程**: Indexing → Retrieval → Generation（"Retrieve-Read" 框架）

- **Indexing**: 文档清洗 → 分块（chunking）→ 向量化 → 存储到向量数据库
- **Retrieval**: 查询向量化 → 相似度计算 → 返回 Top-K 最相关文档块
- **Generation**: 将查询 + 检索到的文档合成 prompt → LLM 生成答案

**局限性**:
- 检索质量受限于简单的语义相似度
- 缺乏对检索结果的后处理
- 固定的线性流程，缺乏灵活性

#### Advanced RAG（优化范式）

**核心改进**: 引入 Pre-Retrieval 和 Post-Retrieval 优化策略

**Pre-Retrieval 优化**（检索前）:
- **索引优化**: 滑动窗口、细粒度分块、元数据附加
- **查询优化**: Query Rewriting、Query Expansion、Query Transformation

**Post-Retrieval 优化**（检索后）:
- **Re-ranking**: 重新排序检索结果，将最相关内容放在 prompt 边缘
- **Context Compression**: 压缩上下文，避免信息过载

**特点**: 仍然遵循链式结构，但在每个环节引入优化模块

#### Modular RAG（模块化范式）

**核心特征**: 高度灵活的模块化架构，支持模块替换和流程重组

**新增模块**:
- **Search Module**: 支持多数据源（搜索引擎、数据库、知识图谱）
- **Memory Module**: 利用 LLM 记忆引导检索
- **Routing Module**: 根据查询类型选择最优路径
- **Predict Module**: 直接生成上下文，减少冗余
- **Task Adapter**: 针对下游任务定制 RAG 流程

**新模式**:
- **Rewrite-Retrieve-Read**: 先重写查询，再检索
- **Generate-Read**: 用 LLM 生成内容替代检索
- **Recursive Retrieval**: 迭代式检索，逐步细化
- **Adaptive Retrieval**: LLM 主动判断何时检索（如 Self-RAG、Flare）

### 2. 三大核心组件的技术栈

#### Retrieval（检索）

**数据源优化**:
- 非结构化数据（文档、网页）
- 结构化数据（数据库、知识图谱）
- 混合检索（关键词 + 语义 + 向量）

**索引优化**:
- Chunking Strategy（分块策略）
- Metadata Attachments（元数据附加）
- Structural Index（结构化索引）

**查询优化**:
- Query Expansion（查询扩展）
- Query Transformation（查询转换）
- Query Routing（查询路由）

**Embedding 优化**:
- 混合/多模态检索
- Fine-tuning Embedding Model

#### Generation（生成）

**Context Curation**（上下文管理）:
- Re-ranking（重排序）
- Context Compression（上下文压缩）

**LLM Fine-tuning**:
- 结合检索结果微调 LLM
- 端到端联合训练

#### Augmentation（增强）

**三个增强阶段**:
1. **Pre-training Stage**: 通过检索增强预训练（如 REALM）
2. **Fine-tuning Stage**: 检索增强的微调
3. **Inference Stage**: 推理时动态检索（最常见）

### 3. 评估体系

**覆盖范围**:
- 26 个下游任务
- 约 50 个数据集
- 多个评估框架（RGB、RAGAS、ARES、TruLens、CRUD）

**评估维度**:
- **Retrieval Quality**: 检索相关性、噪声鲁棒性
- **Generation Quality**: 答案准确性、忠实度（Faithfulness）
- **Robustness**: 对抗性输入、反事实鲁棒性

## 关键洞察

### RAG vs Fine-tuning

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| 知识更新 | 动态更新外部知识库 | 需要重新训练 |
| 可解释性 | 可追溯到原始文档 | 黑盒 |
| 成本 | 低（无需重训练） | 高（需要大量数据和计算） |
| 适用场景 | 知识密集型任务、需要时效性 | 特定领域深度定制 |

**趋势**: 混合方法（RAG + Fine-tuning）成为主流

### RAG vs Long Context

随着 LLM 上下文窗口扩展（>200K tokens），RAG 的价值：
- **效率**: 分块检索比一次性加载全文更快
- **可追溯性**: 可定位原始引用，验证答案
- **可观测性**: 检索和推理过程透明，而长上下文仍是黑盒

**未来方向**: 在超长上下文场景下开发新的 RAG 方法

### 鲁棒性挑战

- **噪声问题**: 检索到的无关或矛盾信息会降低输出质量
- **反直觉发现**: 某些情况下，包含无关文档反而提升准确性 >30%
- **研究方向**: 开发专门策略整合检索与生成，提升对抗性输入的鲁棒性

## 技术演进的逻辑

```
Naive RAG（基础流程）
    ↓ 问题：检索质量低、缺乏优化
Advanced RAG（优化检索前后）
    ↓ 问题：固定流程、缺乏灵活性
Modular RAG（模块化 + 自适应）
    ↓ 趋势：与 Fine-tuning 结合、适应长上下文
```

## 未来方向

1. **混合方法**: RAG + Fine-tuning 的最优整合方式（顺序、交替、端到端联合训练）
2. **超长上下文**: 在 200K+ tokens 场景下的 RAG 新方法
3. **鲁棒性**: 提升对噪声和对抗性输入的抵抗力
4. **自适应检索**: LLM 主动判断何时检索、检索什么（如 Self-RAG）
5. **多模态 RAG**: 扩展到图像、视频等多模态数据

## 相关概念

本文是 RAG 领域的基础性综述，以下概念在文中被系统性讨论：

**核心技术**（文中详细分析）:
- [[RAG]] — 检索增强生成的核心概念
- [[Retrieval]] — 检索技术的优化方法
- [[Embedding]] — 向量化表示与相似度计算
- [[LLM]] — 大语言模型作为生成器

**相关方法**（文中提及的具体实现）:
- [[Self-RAG]] — 自适应检索的代表方法
- [[REALM]] — 预训练阶段的检索增强
- [[Flare]] — 基于置信度的动态检索

## 延伸阅读

**原始论文**:
- [[Clippings/Paper/2312.10997/2312.10997|RAG Survey Paper Clipping]] — 完整论文提取（约 500 行）

**后续发展**:
- 本综述发表于 2023 年，RAG 技术仍在快速演进
- 关注 Modular RAG 的最新实现和混合方法的实践
