---
type: paper
description: 综合 RAG 与推理集成的演进，从单向增强到协同框架的范式转变，连接 Deep Research 产品趋势
aliases:
  - Agentic RAG Survey
  - RAG-Reasoning Survey
tags:
  - rag
  - reasoning
  - survey
  - agentic
created: 2026-03-06
updated: 2026-03-06T18:05
---

# Towards Agentic RAG with Deep Reasoning (2025)

这篇由清华、UIUC、东京大学等机构合作的综述论文系统梳理了 RAG 与推理能力的集成演进，揭示了从静态的 Retrieval-Then-Reasoning 到动态协同框架的范式转变。核心贡献在于统一视角下的三层分类：Reasoning-Enhanced RAG（推理优化检索）、RAG-Enhanced Reasoning（检索支持推理）、以及最前沿的 Synergized RAG-Reasoning（检索与推理的迭代交互）。

这一演进直接对应了 OpenAI Deep Research、Gemini、Perplexity 等产品的技术路线——通过 agentic 能力编排多步搜索与推理的深度耦合。


## 核心问题

> [!intuition] 为什么需要 RAG-Reasoning 集成？
> 
> LLM 面临两个根本性限制：
> 1. **知识幻觉**：参数化知识静态且有限
> 2. **推理困难**：复杂多步推理能力不足
> 
> 这两个问题本质上相互纠缠：**缺失的知识会阻碍推理，错误的推理会妨碍知识利用**。
> 
> 早期方案采用单向增强（要么用推理改进 RAG，要么用 RAG 支持推理），但都局限于静态的 Retrieval-Then-Reasoning 框架，存在三大问题：
> - **检索充分性无法保证**：预检索的知识可能无法对齐推理过程中动态出现的知识需求
> - **推理深度受限**：检索到的错误或冲突信息会干扰模型的固有推理能力
> - **系统适应性不足**：缺乏迭代反馈和动态检索机制

## 三种增强范式

### 1. Reasoning-Enhanced RAG (Reasoning → RAG)

**核心思想**：利用推理能力优化 RAG 流程的各个阶段。

**三个优化方向**：

| 阶段 | 优化方法 | 代表工作 |
|------|----------|----------|
| **Retrieval** | 推理驱动的查询重写、检索策略规划 | Query decomposition, multi-hop planning |
| **Integration** | 相关性评估、信息过滤与融合 | Relevance filtering, conflict resolution |
| **Generation** | 上下文感知的生成策略、grounded 生成控制 | Context-aware synthesis |

> [!warning] 局限性
> - 仍然是单向流程：推理 → 检索 → 生成
> - 无法根据推理过程动态调整检索策略
> - 检索质量的上限限制了推理深度

### 2. RAG-Enhanced Reasoning (RAG → Reasoning)

**核心思想**：通过检索外部知识为推理提供事实基础和上下文线索。

**两类检索增强**：

**External Knowledge Retrieval**：
- Knowledge Base：结构化知识图谱
- Tool Using：调用外部工具（计算器、搜索引擎）

**In-context Retrieval**：
- Prior Experience：检索历史经验和案例
- Example/Training Data：检索相关示例作为 few-shot prompt

> [!warning] 局限性
> - 检索发生在推理之前，无法根据推理中间状态调整
> - 噪声或错误的检索结果会误导推理
> - 缺乏对检索充分性的动态评估

### 3. Synergized RAG-Reasoning (RAG ↔ Reasoning)

**核心思想**：检索与推理的**迭代交互**——推理主动引导检索，新检索的知识持续优化推理过程。

这是范式转变的关键：从静态的 Retrieval-Then-Reasoning 到动态的 **Reasoning-Retrieval Loop**。

#### 3.1 Reasoning Workflow（推理结构）

| 结构类型 | 特点 | 代表方法 |
|----------|------|----------|
| **Chain-based** | 线性推理链，每步后可触发检索 | IRCoT, RAT, CoV-RAG, Chain-of-Note |
| **Tree-based** | 分支探索多条推理路径 | RATT (ToT), AirRAG (MCTS), MCTS-RAG |
| **Graph-based** | 在知识图谱上游走推理 | Think-on-Graph, GraphReader, Graph-CoT |

**演进逻辑**：
- Chain：最简单，但容易错误传播
- Tree：通过分支避免早期错误假设锁定
- Graph：利用结构化知识的关系进行全局推理

#### 3.2 Agent Orchestration（智能体编排）

**Single-Agent 方法**：

| 范式 | 机制 | 代表工作 |
|------|------|----------|
| **Prompting** | 通过 prompt 引导交替推理与检索 | ReAct, Self-Ask, IR-CoT, Self-RAG |
| **SFT** | 在合成数据上微调交错搜索与推理 | Toolformer, INTERS |
| **RL** | 通过奖励信号优化搜索策略 | WebGPT, Search-R1, Deep-Researcher |

**Multi-Agent 方法**：
- 多个专门化 agent 协作（如 planner + retriever + reasoner）
- 适合复杂的 deep research 任务

> [!example] Deep Research 产品
> OpenAI Deep Research、Gemini、Perplexity 的核心技术栈：
> - Agentic 能力编排多步 web 搜索
> - 推理引导检索策略（何时搜索、搜索什么）
> - 检索结果持续优化推理路径
> - 迭代验证与自我修正

## 关键洞察

### 为什么 Synergized 框架更强？

1. **动态适应性**：推理过程中发现知识缺口时，可以立即触发针对性检索
2. **错误修正**：检索到的新证据可以纠正推理中的错误假设
3. **深度探索**：支持多轮迭代，逐步深化对问题的理解

### 从 RTR 到 Synergized 的范式转变

```
静态 RTR：
Query → Retrieve → Reason → Answer
         ↑_____一次性_____↑

动态 Synergized：
Query → Reason₁ → Retrieve₁ → Reason₂ → Retrieve₂ → ... → Answer
        ↑_________迭代反馈循环_________↑
```

### 与 [[Agentic RL]] 的连接

Synergized RAG-Reasoning 本质上是一个 **agentic 决策问题**：
- **Action space**：何时检索、检索什么、如何整合
- **State**：当前推理状态 + 已检索知识
- **Reward**：最终答案的正确性 + 推理的可解释性

这解释了为什么 RL 方法（WebGPT, Search-R1, Deep-Researcher）在 deep research 任务上表现突出——它们学会了**动态决策**而非静态流程。

## 开放挑战

1. **多模态适应**：如何将 RAG-Reasoning 扩展到图像、视频等多模态场景
2. **可信度保障**：如何验证检索内容的真实性和推理过程的可靠性
3. **效率优化**：迭代检索与推理的计算成本如何控制
4. **人机协作**：如何让人类有效介入和引导 agentic RAG-Reasoning 系统

## 延伸阅读

**原始论文与资源**：
- [[Clippings/Paper/2507.09477/2507.09477|论文 PDF 提取]] — 完整技术细节
- [GitHub: Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning) — 论文集合与资源

**相关概念**（需要创建的笔记）：
- [[RAG]] — Retrieval-Augmented Generation 基础
- [[Reasoning]] — LLM 推理能力
- [[Chain-of-Thought]] — 思维链推理
- [[ReAct]] — Reasoning + Acting 范式
- [[MCTS]] — Monte Carlo Tree Search 在推理中的应用

**Agentic 系统**：
- [[Agentic RL]] — 智能体强化学习
- [[ASTRA]] — Agentic 轨迹合成
- [[SAMPO]] — 稳定的 Agentic 多轮策略优化
