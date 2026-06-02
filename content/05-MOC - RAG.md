---
description: RAG 领域导航：检索增强生成的组件、查询优化、上下文整合、GraphRAG 与推理集成
type: moc
aliases:
  - MOC - RAG
tags:
  - rag
  - retrieval
  - information-retrieval
created: 2026-03-06
updated: 2026-05-31T17:54
---

# MOC - RAG

Retrieval-Augmented Generation (RAG) 通过外部知识检索增强 LLM 的生成能力。它把模型能力、检索系统、上下文组织和引用归因连接起来，适合解决知识更新、可追溯、企业私有知识和长尾事实问题。

---

## 概览

```
RAG
├── 总体范式
│   ├── [[RAG]] — 检索增强生成基础框架
│   └── [[RAG Survey (2023)]] — Naive / Advanced / Modular RAG
│
├── 检索层
│   ├── [[Embedding]] — 向量表示
│   ├── [[Dense Retrieval]] — 语义向量检索
│   ├── [[Sparse Retrieval]] / [[BM25]] — 词项匹配检索
│   └── [[GraphRAG]] — 知识图谱与社区摘要
│
├── 查询优化
│   ├── [[Query Rewriting]] — 改写或分解原始查询
│   ├── [[Query Expansion]] — 扩展召回词项
│   └── [[Multi-Query]] — 多角度查询生成
│
├── 上下文整合
│   ├── [[Reranking]] — 精排候选文档
│   ├── [[Chunk Strategy]] — 控制检索粒度
│   ├── [[Context Compression]] — 压缩证据上下文
│   └── [[Citation Generation]] — 输出归因与可追溯
│
└── 推理集成
    ├── [[Chain-of-Thought]] — 推理驱动检索规划
    └── [[Towards Agentic RAG with Deep Reasoning (2025)]]
```

---

## RAG 范式演进

[[RAG Survey (2023)]] 将 RAG 演进分为 Naive、Advanced、Modular 三个阶段：

| 阶段 | 核心流程 | 主要改进点 |
|------|----------|------------|
| Naive RAG | 检索 → 拼接上下文 → 生成 | 建立基础闭环 |
| Advanced RAG | 预检索优化 + 后检索优化 | 查询改写、重排序、上下文过滤 |
| Modular RAG | 组件化和多轮控制 | 迭代检索、路由、工具化和 agentic 工作流 |

[[RAG]] 作为方法入口，适合先建立完整 pipeline：indexing、retrieval、reranking、generation、citation。

---

## 检索层

### Sparse Retrieval

[[Sparse Retrieval]] 基于词项匹配，优点是可解释、无需训练、对关键词和专有名词敏感。[[BM25]] 是常用排序函数，通过词频饱和和文档长度归一化改进 TF-IDF。

### Dense Retrieval

[[Dense Retrieval]] 依赖 [[Embedding]] 将 query 和 document 映射到同一向量空间，适合语义匹配和同义表达。它的核心风险是召回难以解释，对 embedding 模型和语料分布敏感。

### GraphRAG

[[GraphRAG]] 将实体、关系和社区摘要引入 RAG，适合全局理解类问题，例如“这个知识库中有哪些主要主题及其关系”。它把 retrieval 从片段匹配扩展到图结构上的聚合与推理。

---

## 查询优化

很多 RAG 失败来自 query 和文档表达之间的语义鸿沟。查询优化在检索前提升召回质量。

| 方法 | 作用 | 适合场景 |
|------|------|----------|
| [[Query Rewriting]] | 改写、分解或规范化用户查询 | 口语化、含糊、多意图问题 |
| [[Query Expansion]] | 添加相关词、别名、上下位概念 | 关键词缺失、召回不足 |
| [[Multi-Query]] | 生成多个查询角度并合并结果 | 复杂问题、多方面问题 |

[[Chain-of-Thought]] 可以用于生成检索计划：先拆解问题，再为每个子问题生成检索 query。

---

## 上下文整合

检索召回后，系统需要决定哪些证据进入 prompt，以及如何组织它们。

| 组件 | 核心问题 | 常见权衡 |
|------|----------|----------|
| [[Reranking]] | 从候选集中选出最相关证据 | 质量更高，计算更贵 |
| [[Chunk Strategy]] | 文档切成多大、是否重叠 | 粒度越细召回越准，上下文越容易碎 |
| [[Context Compression]] | 如何减少 token 同时保留证据 | 压缩过度会丢失关键条件 |
| [[Citation Generation]] | 如何把回答归因到来源 | 可追溯性提升，生成约束更强 |

上下文整合的目标是让模型看到足够、相关、互相不冲突的证据。对于企业知识库，citation 和 chunk strategy 通常比模型本身更影响可信度。

---

## RAG 与推理集成

[[Towards Agentic RAG with Deep Reasoning (2025)]] 把 RAG 与 reasoning 的关系分成三类：

1. Reasoning-Enhanced RAG：用推理能力改进查询生成、检索路径和证据筛选。
2. RAG-Enhanced Reasoning：用外部知识补足推理前提，减少事实错误。
3. Synergized RAG-Reasoning：检索和推理交替进行，形成多步工作流。

这一方向自然连接 [[Agentic RL]]：当检索变成可交互环境中的动作，系统需要学习何时检索、检索什么、如何用证据更新计划。

---

## 应用场景

| 场景 | RAG 的价值 | 关键组件 |
|------|------------|----------|
| 企业知识库问答 | 使用私有文档并保留来源 | [[Chunk Strategy]], [[Reranking]], [[Citation Generation]] |
| 开放域问答 | 补充模型参数中缺失或过时的事实 | [[Dense Retrieval]], [[Sparse Retrieval]], [[Query Rewriting]] |
| Deep Research | 多步检索、阅读、综合和归因 | [[Multi-Query]], [[Context Compression]], [[GraphRAG]] |
| 代码/技术文档助手 | 精确检索 API、错误信息和版本差异 | [[BM25]], [[Reranking]], citation |

---

## 对比分析

### RAG vs Fine-tuning

RAG 侧重外部知识的动态注入和可追溯来源，fine-tuning 侧重把行为模式或领域风格内化到模型参数中。事实更新频繁、来源必须可查时，RAG 更合适；稳定任务格式、固定输出风格或特定工具使用习惯，可以考虑 fine-tuning。

### RAG vs Long Context

RAG 通过检索减少上下文规模，long context 直接把更多材料交给模型。大规模知识库和低成本查询适合 RAG，单文档深度阅读和强跨段关联适合 long context。实际系统常把两者结合：先检索，再把少量高价值文档放入长上下文窗口。

---

## 学习路径建议

**基础路线**：
1. [[RAG]] → [[RAG Survey (2023)]] — 建立 pipeline 和范式演进。
2. [[Embedding]] → [[Dense Retrieval]] / [[Sparse Retrieval]] → [[BM25]] — 理解检索基础。
3. [[Chunk Strategy]] → [[Reranking]] → [[Citation Generation]] — 理解证据进入 prompt 的过程。

**系统优化路线**：
1. [[Query Rewriting]] → [[Query Expansion]] → [[Multi-Query]] — 提升召回。
2. [[Context Compression]] — 控制 token 成本。
3. [[GraphRAG]] — 处理全局理解和关系型知识。

**推理集成路线**：
1. [[Chain-of-Thought]] — 用推理规划检索。
2. [[Towards Agentic RAG with Deep Reasoning (2025)]] — 理解 RAG-reasoning 协同框架。
3. [[Agentic RL]] — 将检索动作放入交互式 agent 训练。

---

## 相关 MOC

- [[01-MOC - Foundations|MOC - Foundations]] — Embedding、信息检索和 Transformer 基础
- [[04-MOC - Inference|MOC - Inference]] — context compression、serving 与长上下文成本
- [[07-MOC - Post-training|MOC - Post-training]] — agentic workflow、tool use 和 post-training 数据
- [[06-MOC - Reinforcement Learning|MOC - Reinforcement Learning]] — Agentic RL 与交互式环境训练
