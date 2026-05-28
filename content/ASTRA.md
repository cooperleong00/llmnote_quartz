---
type: method
description: 自动化合成 agentic 轨迹和强化学习环境的端到端框架，结合 SFT 和 online RL 训练 tool-augmented agents
aliases:
  - Automated Synthesis of agentic Trajectories and Reinforcement Arenas
prerequisites:
  - "[[SFT]]"
  - "[[Agentic RL]]"
  - "[[Data Synthesis]]"
  - "[[PPO]]"
tags:
  - post-training
  - agent
  - reinforcement-learning
  - data-synthesis
created: 2026-02-04
updated: 2026-02-04T18:53
---

# ASTRA

ASTRA（Automated Synthesis of agentic Trajectories and Reinforcement Arenas）是一个**全自动化的端到端框架**，用于训练 tool-augmented language model agents。它通过两个核心创新解决了 [[Agentic RL]] 的数据和环境瓶颈：(1) 利用 tool-call graph 的静态拓扑合成多轮 [[SFT]] 轨迹；(2) 将 Q&A traces 转换为 code-executable、rule-verifiable 的 RL 环境。ASTRA 采用 SFT + online RL 两阶段训练，使用 trajectory-level rewards 平衡任务完成和交互效率。

> [!paper] 论文出处
> Beike Language and Intelligence (BLI), LianjiaTech, 2025
> GitHub: https://github.com/LianjiaTech/astra

---

## 动机

> [!intuition] 为什么需要 ASTRA？

现有 [[Agentic RL]] 方法面临三大瓶颈：

1. **环境不可验证**：依赖 LLM-simulated environments，无法自动判断 agent 行为的正确性
2. **数据获取困难**：高质量的多轮 agent 轨迹需要人工标注，成本高、规模有限
3. **训练范式单一**：要么只用 [[SFT]]（缺乏环境反馈），要么只用 RL（受限于初始策略能力）

**ASTRA 的核心洞察**：

- **SFT 数据合成**：tool-call graph 的**静态拓扑**（哪些工具可以组合）可以指导轨迹生成
- **RL 环境合成**：人类推理的**组合拓扑**（如何分解问题）可以转换为可执行环境
- **统一训练**：SFT 提供强初始策略，online RL 通过环境交互进一步优化

| 方法 | 数据来源 | 环境验证 | 训练范式 |
|------|----------|----------|----------|
| 传统方法 | 人工标注 | LLM 模拟（不可验证） | SFT 或 RL 单一 |
| **ASTRA** | 自动合成 | Code-executable（可验证） | SFT + online RL |

---

## 核心机制

### 1. Trajectory Synthesis Pipeline（用于 SFT）

> [!intuition] 核心思想
> 利用 tool-call graph 的**静态拓扑**——工具之间的依赖关系——来指导多轮轨迹的生成。

**流程**：

```
Tool Documents → Tool-chain Graph → Task Generation → Multi-turn Rollout → Quality Scoring
```

**关键步骤**：

1. **Tool-chain Construction**：
   - 对每个 MCP server，用 LLM 合成 (task, tool-chain) pairs
   - 构建 transition graph $\widehat{G}=(V, \widehat{E}, w)$，节点是工具，边是调用顺序
   - 通过 random walk 采样候选 tool-chains

2. **Task Construction**：
   - Chain-conditioned：给定 tool-chain 生成任务（保证可执行性）
   - Server-only：只给 server spec 生成任务（保证多样性）
   - 三维度评分过滤：Question Quality、Scenario Realism、Tool-use Necessity

3. **Trajectory Collection**：
   - 用 LLM 与真实 MCP server 交互，收集多轮轨迹
   - 七维度 reward modeling（无需人工标注）

> [!math] Trajectory Reward
> 七个维度的平均：
> - Query Understanding (QU)
> - Query Planning (QP)
> - Tool-response Context Understanding (TCU)
> - Tool-response Context-conditioned Planning (TCP)
> - Tool Call Status (TCS)
> - Tool Conciseness (TC)
> - Final Answer Quality (FA)

### 2. Environment Synthesis Framework（用于 RL）

> [!intuition] 核心思想
> 将人类推理的**组合拓扑**——问题分解的依赖结构——转换为 code-executable、rule-verifiable 的 RL 环境。

**流程**：

```
Q&A Decomposition → Quality Validation → Environment Synthesis → Sub-Environment Merging
```

**关键创新**：

1. **Q-A Instance Synthesis**：
   - 主问题 $q_0$ 分解为子问题集合 $S = \{(q_i, a_i)\}_{i=1}^m$
   - 子问题形成依赖图 $\mathcal{G}$（DAG 结构）
   - 最终答案 $a_0 = \Phi(\{a_i\}, \mathcal{G})$

2. **Quality Validation**（四维度）：
   - Dependency Consistency：依赖关系是否正确
   - Sub-Question Atomicity：子问题是否不可再分
   - Answer Correctness：答案是否正确
   - Answerability：子问题是否可回答

3. **Environment Synthesis**：
   - 每个子问题 → 一个可执行的 tool call
   - 验证规则：code-based，非 LLM 判断
   - 支持 deterministic multi-turn RL

### 3. 两阶段训练

**Stage 1: SFT**
- 使用 trajectory synthesis pipeline 生成的数据
- 学习 multi-turn tool interaction 的基本能力
- 产出：stronger initial policy

**Stage 2: Online RL**
- 使用 environment synthesis framework 生成的环境
- 基于 [[GRPO]] 的 policy optimization
- **Irrelevant-tool mixing**：混入无关工具增加难度
- **F1-style trajectory-level reward**：平衡 task completion 和 interaction efficiency

> [!warning] 训练稳定性
> ASTRA 使用 **Adaptive Batch Filling** 解决 GRPO 中 reward variance 为零导致的梯度消失问题：只有 reward variance 非零的 rollout 才参与训练。

---

## 局限性

1. **领域依赖**：数据质量依赖 tool documentation 的质量
2. **可验证性边界**：只能处理 code-verifiable 的任务
3. **训练成本**：online RL 需要大量环境交互
4. **泛化性**：合成环境与真实场景可能存在 gap

---

## 面试要点

> [!interview] 常见问题

**Q: ASTRA 的两个 pipeline 分别解决什么问题？**

A: Trajectory Synthesis 解决 SFT 数据获取问题，利用 tool-call graph 的静态拓扑自动生成多轮轨迹；Environment Synthesis 解决 RL 环境不可验证问题，将 Q&A 分解转换为 code-executable 环境。

**Q: 为什么 ASTRA 比纯 SFT 或纯 RL 更好？**

A: SFT 缺乏环境反馈，无法学习 error recovery；纯 RL 受限于初始策略能力。ASTRA 用 SFT 建立强初始策略，再用 RL 通过环境交互优化。

**Q: ASTRA 如何保证 RL 环境的可验证性？**

A: 将 Q&A 分解为子问题 DAG，每个子问题对应一个 tool call，验证规则是 code-based 而非 LLM 判断，保证 deterministic rewards。

---

## 相关概念

- [[Agentic RL]] — ASTRA 的应用领域
- [[Data Synthesis]] — 核心技术之一
- [[GRPO]] — RL 阶段使用的优化算法
- [[MCP]] — Model Context Protocol，ASTRA 使用的 tool server 标准
