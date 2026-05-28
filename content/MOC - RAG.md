---
description: RAG 领域导航：检索增强生成的演进、核心组件与推理集成
type: moc
created: 2026-03-06
updated: 2026-03-06
---

# MOC - RAG

Retrieval-Augmented Generation (RAG) 通过从外部知识库检索相关信息来增强 LLM 的生成能力，解决知识过时、幻觉和不可追溯等问题。RAG 已成为知识密集型任务的关键技术。

---

## 概览

```
Query
  ↓
Retrieval (检索)
  ├── Indexing (索引构建)
  ├── Query Optimization (查询优化)
  └── Embedding (向量化)
  ↓
Integration (整合)
  ├── Reranking (重排序)
  ├── Deduplication (去重)
  └── Conflict Resolution (冲突解决)
  ↓
Generation (生成)
  ├── Context Integration (上下文整合)
  ├── Reasoning (推理)
  └── Answer Synthesis (答案合成)
```

---

## RAG 范式演进

RAG 技术经历了三个主要发展阶段：

### Naive RAG
- 最早的 RAG 范式
- 简单的"检索 → 生成"流程
- 局限：检索质量不稳定、上下文利用不充分

### Advanced RAG
- 引入预检索和后检索优化
- 查询重写、混合检索、重排序
- 改进检索质量和相关性

### Modular RAG
- 模块化设计，灵活组合各组件
- 支持迭代检索、自适应检索
- 可与其他技术（如 fine-tuning）深度集成

**参考**：[[RAG Survey (2023)]]

---

## 核心组件

### 检索 (Retrieval)

**索引与向量化**：
- [[Embedding]] — 文本向量化表示
- [[Dense Retrieval]] — 密集检索
- [[Sparse Retrieval]] — 稀疏检索（BM25 等）
- [[Hybrid Retrieval]] — 混合检索策略

**查询优化**：
- [[Query Rewriting]] — 查询重写
- [[Query Expansion]] — 查询扩展
- [[Multi-Query]] — 多查询生成

### 整合 (Integration)

- [[Reranking]] — 重排序模型
- [[Context Compression]] — 上下文压缩
- [[Chunk Strategy]] — 分块策略

### 生成 (Generation)

- [[In-Context Learning]] — 上下文学习
- [[Prompt Engineering]] — 提示工程
- [[Citation Generation]] — 引用生成

---

## RAG 与推理的集成

从静态检索到动态推理的范式转变：

### Reasoning-Enhanced RAG
使用推理能力优化 RAG 各阶段：
- [[Chain-of-Thought]] 引导查询生成
- 推理驱动的检索策略
- 多步推理规划

### RAG-Enhanced Reasoning
检索为推理提供知识支撑：
- 外部知识补充推理前提
- 事实验证与纠错
- 领域知识注入

### Synergized RAG-Reasoning
检索与推理的迭代交互：
- [[ReAct]] — 推理与行动交替
- [[MCTS]] 驱动的搜索策略
- Agentic 决策框架

**参考**：[[Towards Agentic RAG with Deep Reasoning (2025)]]

---

## 高级话题

### 自适应 RAG
- [[Self-RAG]] — 自我反思的 RAG
- [[Adaptive Retrieval]] — 自适应检索触发
- [[Retrieval Necessity]] — 判断是否需要检索

### 多模态 RAG
- [[Multimodal Retrieval]] — 多模态检索
- [[Vision-Language RAG]] — 视觉-语言 RAG
- [[Cross-Modal Alignment]] — 跨模态对齐

### RAG 优化
- [[RAG Fine-tuning]] — RAG 系统的微调
- [[Retrieval-Augmented Pretraining]] — 检索增强预训练
- [[REALM]] — 端到端可训练的 RAG

### 评估与基准
- [[RAG Evaluation]] — RAG 评估方法
- [[Faithfulness]] — 忠实度评估
- [[Relevance]] — 相关性评估
- [[Answer Quality]] — 答案质量评估

---

## 应用场景

### 知识密集型任务
- 开放域问答
- 事实核查
- 知识库问答

### 企业应用
- 文档问答系统
- 客服机器人
- 知识管理

### 研究与发现
- 科学文献检索
- 专利分析
- Deep Research 系统

---

## 对比分析

### RAG vs Fine-tuning
- **RAG 优势**：知识可更新、无需重训练、可追溯来源
- **Fine-tuning 优势**：知识内化、推理速度快、无需外部系统
- **混合方案**：RAG + Fine-tuning 结合

### RAG vs Long Context
- **RAG 优势**：精准检索、成本低、可扩展
- **Long Context 优势**：完整上下文、无检索误差
- **适用场景**：RAG 适合大规模知识库，Long Context 适合单文档深度理解

---

## 延伸阅读

**综述论文**：
- [[RAG Survey (2023)]] — RAG 技术全面综述
- [[Towards Agentic RAG with Deep Reasoning (2025)]] — RAG 与推理集成的前沿

**原始论文**：
- [[Clippings/Paper/2312.10997/2312.10997|RAG Survey 原文]]
- [[Clippings/Paper/2507.09477/2507.09477|Agentic RAG Survey 原文]]

**相关领域**：
- [[MOC - Post-training]] — Post-training 方法
- [[MOC - Inference]] — 推理优化
- [[MOC - Foundations]] — LLM 基础
