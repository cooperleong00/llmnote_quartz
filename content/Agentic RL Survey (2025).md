---
type: paper
description: 系统化 Agentic RL 领域的综合性调研，通过 POMDP 框架形式化 LLM Agent 训练，提出能力维度和应用领域的双重分类法
aliases:
  - The Landscape of Agentic Reinforcement Learning for LLMs
  - Agentic RL Survey
prerequisites:
  - "[[Agentic RL]]"
  - "[[Markov Decision Process]]"
  - "[[RLHF]]"
tags:
  - survey
  - agentic
  - reinforcement-learning
  - agent
created: 2026-03-08
updated: 2026-03-31T12:34
---
	
# Agentic RL Survey (2025)

这是一篇系统化 Agentic RL 领域的综合性调研（95 页，26 位作者），通过 POMDP 框架形式化了 LLM Agent 训练范式，提出了能力维度和应用领域的双重分类法，综合了超过 500 篇相关工作。

> [!paper] 论文信息
> - **标题**: The Landscape of Agentic Reinforcement Learning for LLMs: A Survey
> - **作者**: 26 位作者（多机构合作）
> - **发表**: Transactions on Machine Learning Research (2026-01)
> - **arXiv**: 2509.02547
> - **页数**: 95 页
> - **原文**: [[Clippings/Paper/2509.02547/2509.02547|Paper Clipping]]

---

## 核心贡献

### 1. 形式化范式转变：从 PBRFT 到 Agentic RL

论文通过 MDP/POMDP 框架清晰区分了两种 RL 范式：

**Preference-Based RFT (PBRFT)**：
- 退化的单步 MDP：$\langle\mathcal{S}_{\text{trad}}, \mathcal{A}_{\text{trad}}, \mathcal{P}_{\text{trad}}, \mathcal{R}_{\text{trad}}, T=1, \gamma=1\rangle$
- 状态空间：单一 prompt
- 动作空间：纯文本序列
- 转移函数：确定性终止
- 奖励：单一标量 $r(a)$
- 目标：$\mathbb{E}_{a \sim \pi_\theta}[r(a)]$

**Agentic RL**：
- 时序扩展的 POMDP：$\langle\mathcal{S}_{\text{agent}}, \mathcal{A}_{\text{agent}}, \mathcal{P}_{\text{agent}}, \mathcal{R}_{\text{agent}}, \gamma, \mathcal{O}\rangle$
- 状态空间：$s_t \in \mathcal{S}_{\text{agent}}$，观测 $o_t = \mathcal{O}(s_t)$
- 动作空间：$\mathcal{A}_{\text{text}} \cup \mathcal{A}_{\text{action}}$（文本 + 工具调用）
- 转移函数：动态 $P(s_{t+1} | s_t, a_t)$
- 奖励：逐步 $R(s_t, a_t)$，结合稀疏任务奖励和密集子奖励
- 目标：$\mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t R(s_t, a_t)]$

> [!intuition] 核心区别
> PBRFT 优化的是**固定数据集上的单轮输出**，Agentic RL 优化的是**动态环境中的多步行为**。
> 
> 这不是量的差异（单步 vs 多步），而是质的转变：从"对齐文本生成器"到"训练自主决策者"。

---

### 2. 双重分类法

论文提出两个互补的组织维度：

#### 能力维度（Capability Perspective）

将 Agentic RL 分解为 6 个核心能力模块：

| 能力 | 定义 | RL 的作用 |
|------|------|-----------|
| **Planning** | 分解任务、制定执行计划 | 从启发式规划器演进为端到端学习的规划策略 |
| **Tool Use** | 调用外部工具（API、搜索、代码执行） | 学习工具选择、参数生成、错误恢复 |
| **Memory** | 维护和检索长期信息 | 优化记忆写入/读取策略，平衡容量和相关性 |
| **Reasoning** | 逐步推理、验证、自我纠正 | 学习推理链生成、验证器调用时机 |
| **Self-Improvement** | 反思、自我批评、迭代优化 | 学习何时反思、如何从失败中学习 |
| **Perception** | 处理多模态输入（视觉、音频） | 学习注意力分配、跨模态融合 |

> [!warning] 能力的涌现性
> 这些能力不是独立的模块，而是通过 RL 训练**涌现**的行为模式。
> 
> 例如：Planning 不是预定义的算法（如 A*），而是模型学会的"何时分解任务、如何排序子目标"的策略。

#### 应用领域（Application Perspective）

按任务类型组织：

| 领域 | 典型任务 | 代表性工作 |
|------|----------|------------|
| **Search & Research** | 多步信息检索、综合 | WebGPT, Perplexity |
| **Code Agent** | 代码生成、调试、软件工程 | [[Kimi K2 (2025)]], MiMo-V2 |
| **Mathematical Agent** | 形式化证明、符号推理 | DeepSeek-Prover, Lean |
| **GUI Agent** | 网页/应用操作 | WebArena, OSWorld |
| **Vision Agent** | 视觉推理、机器人控制 | RT-2, PaLM-E |
| **Multi-Agent Systems** | 协作、竞争、通信 | MARL for LLMs |

---

### 3. 开源生态整合

论文系统整理了 Agentic RL 的基础设施：

**环境模拟器**：
- Web 环境：WebArena, WebShop
- GUI 环境：OSWorld, AndroidEnv
- 代码环境：SWE-bench, CodeContests
- 通用环境：AgentGym, BALM

**RL 框架**：
- 在线 RL：[[ASTRA]], [[Let It Flow (2025)|ROLL]]
- 离线 RL：Trajectory filtering, Advantage reweighting
- 混合方法：SFT + RL, Curriculum learning

**评估基准**：
- 代码：HumanEval, MBPP, SWE-bench
- 推理：MATH, GSM8K, AIME
- 工具使用：ToolBench, API-Bank
- 多模态：VQA, Embodied QA

---

## 关键洞察

### 1. RL 是能力涌现的关键机制

> [!intuition] 为什么需要 RL？
> 
> **SFT 的局限**：只能模仿演示，无法超越演示质量，难以处理：
> - 长 horizon 任务（需要信用分配）
> - 稀疏奖励（需要探索）
> - 动态环境（需要适应）
> 
> **RL 的优势**：
> - 通过试错学习最优策略（而非模仿次优演示）
> - 自动发现新策略（如 AlphaGo 的创新走法）
> - 适应环境变化（通过在线交互）

### 2. POMDP 框架的必要性

Agentic RL 本质上是 POMDP 而非 MDP：

- **部分可观测性**：Agent 无法看到完整状态（如网页的后端状态、代码库的全部文件）
- **观测函数** $o_t = \mathcal{O}(s_t)$：Agent 只能通过工具调用获取局部信息
- **记忆的必要性**：需要维护历史信息来推断隐藏状态

> [!example] POMDP 的实例
> 在 SWE-bench 任务中：
> - **完整状态** $s_t$：整个代码库 + 测试结果 + issue 描述
> - **观测** $o_t$：当前打开的文件 + 最近的搜索结果
> - **动作**：搜索、读取文件、编辑、运行测试
> - **转移**：环境根据编辑更新代码库状态

### 3. 从启发式到端到端学习

论文强调 Agentic RL 的演进路径：

**阶段 1：RL-free 启发式**
- 固定的 plan-execute-reflect 循环
- 预定义的工具调用规则
- 人工设计的记忆管理

**阶段 2：RL 作为外部引导**
- RL 优化规划器的输出
- 学习工具选择策略
- 但核心架构仍是固定的

**阶段 3：端到端 RL**
- 整个 Agent 行为由 RL 学习
- 能力作为涌现行为而非预定义模块
- 更强的泛化和适应能力

> [!warning] 当前的局限
> 大多数工作仍在阶段 2，端到端 Agentic RL 面临：
> - 训练不稳定（见 [[ARLArena (2026)]]）
> - 环境多样性不足
> - 评估标准不统一

---

## 开放挑战

论文在第 6 节讨论了 6 个关键挑战：

### 1. 可信性（Trustworthiness）

- **安全性**：防止有害工具调用（如删除文件、泄露隐私）
- **可解释性**：理解 Agent 的决策过程
- **鲁棒性**：对抗对抗性输入和环境扰动

### 2. 训练规模化（Scaling up Training）

- **计算成本**：在线 RL 需要大量环境交互
- **样本效率**：如何减少所需轨迹数量
- **并行化**：异步训练架构（见 [[GLM-5 (2026)]]）

### 3. 环境规模化（Scaling up Environments）

- **多样性**：覆盖更多任务类型和领域
- **真实性**：缩小模拟环境与真实世界的差距
- **自动生成**：程序化生成训练环境

### 4. RL 的机制性辩论

- **RL 是否必要**：某些能力是否可以通过 SFT 获得？
- **何时使用 RL**：哪些任务受益最大？
- **RL 的作用机制**：是学习新能力还是激活已有能力？

### 5. 架构模式

- **模块化 vs 端到端**：预定义模块还是完全学习？
- **记忆架构**：短期 vs 长期，检索 vs 压缩
- **工具接口**：如何设计可学习的工具调用接口？

### 6. 评估标准

- **统一基准**：跨领域的标准化评估
- **泛化能力**：训练任务 vs 测试任务的分布差异
- **人类评估**：自动指标 vs 人类偏好

---

## 与知识库的连接

### 已有概念的深化

- [[Agentic RL]]：本 survey 提供了形式化定义和系统分类
- [[GRPO]]：作为 Agentic RL 的常用算法被讨论
- [[SAMPO]]：作为解决训练不稳定性的方法被引用
- [[ARLArena (2026)]]：作为分析训练挑战的代表性工作

### 需要补充的概念

论文揭示了以下空缺（作为未来笔记的方向）：

**核心能力**：
- Planning（规划能力的 RL 训练）
- Tool Use（工具使用的学习）
- Memory Management（记忆管理策略）
- Self-Improvement（自我改进机制）

**环境与基准**：
- WebArena（Web 交互环境）
- SWE-bench（软件工程基准）
- AgentGym（通用 Agent 环境）

**训练方法**：
- Curriculum Learning for Agents（课程学习）
- Environment Scaling（环境扩展）
- Asynchronous Agentic RL（异步训练）

---

## 面试视角

> [!interview] 高频问题
> 
> **Q1: Agentic RL 和传统 RLHF 的本质区别是什么？**
> 
> A: 不是单步 vs 多步的表面差异，而是优化目标的根本不同：
> - RLHF 优化**固定数据集上的偏好对齐**（静态、单轮）
> - Agentic RL 优化**动态环境中的任务完成**（交互、多步）
> 
> 形式化地说，RLHF 是退化的 MDP（$T=1$），Agentic RL 是 POMDP（$T>1$，部分可观测）。
> 
> **Q2: 为什么 Agentic RL 需要 POMDP 而不是 MDP？**
> 
> A: 因为 Agent 无法观测完整状态：
> - 代码 Agent 看不到整个代码库，只能通过搜索/读取获取局部信息
> - Web Agent 看不到网页的后端状态，只能通过交互观察反馈
> - 这要求 Agent 维护记忆、推断隐藏状态，这正是 POMDP 的核心特征
> 
> **Q3: RL 在 Agentic 能力中的作用是什么？**
> 
> A: RL 将静态的启发式模块转化为可学习的策略：
> - Planning：从固定的分解规则 → 学习何时分解、如何排序
> - Tool Use：从预定义的调用逻辑 → 学习工具选择、参数生成
> - Memory：从固定的存储策略 → 学习何时写入、何时检索
> 
> 关键是**端到端优化**：让 Agent 自己发现最优行为，而非人工设计。
> 
> **Q4: 当前 Agentic RL 的最大挑战是什么？**
> 
> A: 训练不稳定性（见 [[ARLArena (2026)]]）：
> - 长 horizon 导致高方差梯度
> - 稀疏奖励导致探索困难
> - 环境多样性不足导致过拟合
> 
> 解决方向：
> - 算法层面：[[SAMPO]]、advantage normalization
> - 环境层面：Environment Scaling（[[GLM-5 (2026)]]）
> - 架构层面：异步训练（[[LongCat-Flash-Thinking (2025)|DORA]]）

---

## 延伸阅读

**原始论文与深入材料**：
- [[Clippings/Paper/2509.02547/2509.02547|完整 Paper Clipping]] — 95 页详细内容

**相关 Survey**：
- [[Towards Agentic RAG with Deep Reasoning (2025)]] — RAG 与推理的协同
- [[LLM RL Algorithm Evolution]] — RL 算法在 LLM 中的演进

**代表性系统**：
- [[Kimi K2 (2025)]] — Agentic 数据合成与训练
- [[GLM-5 (2026)]] — 异步 Agentic RL 与环境扩展
- [[LongCat-Flash-Thinking (2025)]] — DORA 异步训练框架

**核心方法**：
- [[ASTRA]] — 自动化轨迹合成
- [[SAMPO]] — 稳定 Agentic RL 训练
- [[Let It Flow (2025)]] — 完整的 Agentic Learning Ecosystem
