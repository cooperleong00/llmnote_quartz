---
type: moc
description: LLM 全栈知识库的总入口，连接基础、训练、后训练、推理、RAG 与强化学习导航
aliases:
  - Index
created: 2025-01-25
updated: 2026-05-31T17:54
---

# LLM 全栈知识库

这是一个面向 LLM 研究、工程实践和面试准备的知识库。当前结构围绕基础理论、模型架构、训练系统、post-training、inference、RAG 和 reinforcement learning 组织。

---

## 知识地图

### 核心 MOC

| 入口 | 覆盖范围 | 适合什么时候进入 |
|------|----------|------------------|
| [[01-MOC - Foundations|MOC - Foundations]] | 数学基础、优化器、Transformer、RL 基础 | 需要补齐前置概念时 |
| [[02-MOC - Attention|MOC - Attention]] | Attention、KV Cache、位置编码、高效注意力 | 需要理解 LLM 架构和长上下文机制时 |
| [[03-MOC - Distributed Training|MOC - Distributed Training]] | DP/ZeRO/FSDP、TP/PP/EP、显存与训练效率 | 需要理解大模型训练工程时 |
| [[04-MOC - Inference|MOC - Inference]] | KV Cache、Paged/Radix Attention、Speculative Decoding、量化 | 需要理解部署与 serving 优化时 |
| [[05-MOC - RAG|MOC - RAG]] | 检索、查询优化、重排序、上下文压缩、GraphRAG | 需要构建知识增强系统时 |
| [[06-MOC - Reinforcement Learning|MOC - Reinforcement Learning]] | 经典 RL、policy gradient、PPO/GRPO/VAPO、Agentic RL | 需要系统理解 RL 算法演进时 |
| [[07-MOC - Post-training|MOC - Post-training]] | SFT、RLHF、Direct Alignment、Online RL、OPD、Agentic RL | 需要理解模型对齐和 reasoning model 训练时 |

### 关键主题线

**从基础到模型结构**：[[Tokenization]] → [[Embedding]] → [[Transformer]] → [[Attention]] → [[Multi-Head Attention]] → [[RoPE]] / [[ALiBi]]

**从训练到系统扩展**：[[AdamW]] → [[Mixed Precision Training]] → [[Data Parallelism]] → [[ZeRO]] / [[FSDP]] → [[Tensor Parallelism]] / [[Pipeline Parallelism]] → [[3D Parallelism]]

**从 SFT 到 reasoning model**：[[SFT]] → [[SFT-then-RL]] → [[RLHF]] → [[PPO]] → [[GRPO]] → [[DAPO]] / [[GSPO]] / [[VAPO (2025)]]

**从偏好数据到直接对齐**：[[Preference Data]] → [[Bradley-Terry Model]] → [[Reward Model]] → [[DPO]] → [[IPO]] / [[KTO]] / [[ORPO]] / [[SimPO]]

**从检索到知识增强推理**：[[RAG]] → [[Embedding]] → [[Dense Retrieval]] / [[Sparse Retrieval]] → [[Query Rewriting]] → [[Reranking]] → [[Context Compression]] → [[GraphRAG]]

---

## 当前重点

### Post-training 与 RL

当前增长最快的区域是 LLM RL 和 reasoning model 训练。优先串联这些笔记：

1. [[LLM RL Algorithm Evolution]] — 用一条主线理解 REINFORCE、PPO、GRPO、DAPO、GSPO、VAPO 的演进。
2. [[Post-Training Phase Transition]] — 从 SFT、RL、On-Policy Distillation 的阶段切换理解训练信号变化。
3. [[On-Policy Distillation]] — 用自生成轨迹承接 teacher 或自反馈信号，连接 distillation 与 RL。
4. [[Training-Inference Mismatch]] — 理解 MoE 与 long-CoT RL 训练中的稳定性问题。
5. [[Agentic RL]] — 将 RL 扩展到多轮交互、工具调用和环境反馈。

### 工程与部署

训练和推理系统的主线已经形成：

1. [[GPU Memory Calculation]] — 先建立显存账本。
2. [[Activation Checkpointing]] / [[Gradient Accumulation]] — 单机显存与 batch size 的基础技巧。
3. [[ZeRO]] / [[FSDP]] — 数据并行中的状态切分。
4. [[Tensor Parallelism]] / [[Pipeline Parallelism]] / [[Expert Parallelism]] — 模型并行的三个维度。
5. [[KV Cache]] / [[Paged Attention]] / [[Radix Attention]] / [[Continuous Batching]] — serving 端的核心效率路径。

---

## 快速入口

- **面试高频**：[[PPO]]、[[DPO]]、[[GRPO]]、[[KL Divergence]]、[[Bradley-Terry Model]]、[[KV Cache]]、[[Flash Attention]]
- **前沿 RL**：[[DAPO]]、[[Dr. GRPO]]、[[GSPO]]、[[CISPO]]、[[SAPO]]、[[MIS-PO]]、[[VAPO (2025)]]
- **Agentic RL**：[[Agentic RL]]、[[ASTRA]]、[[GiGPO]]、[[SAMPO]]、[[EMPG]]、[[ARLArena (2026)]]
- **RAG 系统**：[[RAG]]、[[BM25]]、[[Dense Retrieval]]、[[Sparse Retrieval]]、[[Query Rewriting]]、[[Reranking]]、[[GraphRAG]]
- **模型报告**：[[DeepSeek-V3 (2024)]], [[Kimi K2 (2025)]], [[Qwen3 (2025)]], [[DeepSeek-V4 (2026)]], [[GLM-5 (2026)]]

---

## 使用指南

知识库规范见 [[AGENTS]]。需要按 Obsidian 语义搜索、管理 properties 或检查链接时，优先参考 [[Obsidian CLI Guide]]。
