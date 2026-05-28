---
type: paper
title: "MemEvolve: Meta-Evolution of Agent Memory Systems"
authors:
  - OPPO AI Agent Team
  - LV-NUS lab
affiliations:
  - OPPO
  - National University of Singapore
year: 2025
arxiv: "2512.18746"
description: 提出 MemEvolve 元进化框架，通过双层优化同时演进 Agent 的经验知识和记忆架构，实现跨任务的自适应记忆系统
aliases:
  - MemEvolve 论文
  - 记忆系统元进化
tags:
  - agent
  - meta-learning
  - self-evolution
github: https://github.com/bingreeky/MemEvolve
created: 2026-03-06
updated: 2026-03-06T18:48
---

# MemEvolve (2025)

MemEvolve 提出了一个**元进化框架**（meta-evolutionary framework），能够同时演进 Agent 的经验知识和记忆架构本身。与传统的固定记忆系统不同，MemEvolve 通过双层优化让 Agent 不仅能积累经验，还能逐步优化"如何从经验中学习"的方式。

论文同时引入 **EvolveLab**，一个统一的代码库，将 12 个代表性记忆系统抽象为模块化设计空间（Encode, Store, Retrieve, Manage），为记忆系统的研究提供标准化实现和公平的实验平台。

> [!paper] 论文信息
> **MemEvolve: Meta-Evolution of Agent Memory Systems**
> OPPO AI Agent Team, LV-NUS lab
> arXiv:2512.18746, December 2025
> [GitHub](https://github.com/bingreeky/MemEvolve)

---

## 核心动机

### 问题：静态记忆架构的局限

现有的 Agent 记忆系统（如 Voyager, ExpeL, Agent Workflow Memory）采用**固定的记忆架构**：
- 记忆系统的设计（如何编码、存储、检索经验）在部署前就已确定
- Agent 只能在固定架构下积累经验，无法调整记忆策略
- 不同任务需要不同的记忆方式（如 API 库适合 web 任务，self-critique 适合推理任务），但静态系统无法适应

**类比人类学习**：
- **低效学习者**：不从错误中学习（无记忆的 Agent）
- **熟练学习者**：能从经验中提取可复用技能，但使用固定的抽象方式（现有记忆系统）
- **自适应学习者**：根据学科动态调整学习策略（文学靠记忆，数学靠模板）—— **这是 MemEvolve 的目标**

### 核心问题

> 如何让记忆系统不仅促进 Agent 的演进，还能**元适应（meta-adapt）自身架构**，在特定任务领域获得更好性能的同时保持泛化能力？

---

## MemEvolve 框架

### 双层优化过程

MemEvolve 通过**双层优化**（bilevel optimization）实现记忆系统的元进化：

**内循环（Inner Loop）— 一阶演进**：
- Agent 在**固定的记忆系统** $\Omega_j^{(k)}$ 下执行任务
- 通过与环境交互积累经验，更新记忆状态 $M_t$
- 生成轨迹 $\tau$ 并评估性能 $R(\tau)$

**外循环（Outer Loop）— 二阶演进**：
- 基于内循环的性能反馈，**元学习更有效的记忆架构**
- 选择高性能候选，生成新的变体
- 产生下一轮迭代的候选记忆系统集合

这形成了一个**良性循环**：
1. 更好的记忆架构 → 提升 Agent 学习效率
2. 更强的 Agent → 生成更高质量的轨迹
3. 更精确的适应度信号 → 驱动下一轮架构演进

### 模块化设计空间

为了让记忆架构的优化可控，MemEvolve 将任何记忆系统 $\Omega$ 分解为四个模块：

$$
\Omega = (\mathcal{E}, \mathcal{U}, \mathcal{R}, \mathcal{G})
$$

| 模块 | 功能 | 示例 |
|------|------|------|
| **Encode** $\mathcal{E}$ | 将原始经验转换为结构化表示 | 压缩轨迹、提取可泛化的教训 |
| **Store** $\mathcal{U}$ | 将编码后的经验整合到持久化记忆 | 向量数据库、知识图谱 |
| **Retrieve** $\mathcal{R}$ | 根据任务上下文召回相关记忆 | 语义搜索、对比比较 |
| **Manage** $\mathcal{G}$ | 离线维护记忆质量 | 去重、剪枝、巩固 |

每个记忆系统可以表示为这四个模块的具体实现组合，形成可演化的"基因型"。

---

## EvolveLab 代码库

### 统一实现

EvolveLab 将 **12 个代表性记忆系统**重新实现为统一的模块化框架：

| 系统 | 年份 | Encode | Store | Retrieve | Manage |
|------|------|--------|-------|----------|--------|
| Voyager | 2023.5 | Traj. & Tips | Vector DB | Semantic Search | N/A |
| ExpeL | 2023.8 | Traj. & Insights | Vector DB | Contrastive Comparison | N/A |
| AWM | 2024.9 | Workflows | Vector DB | Semantic Search | N/A |
| SkillWeaver | 2025.4 | APIs | Tool Library | Function Matching | Skill Pruning |
| G-Memory | 2025.6 | Tips & Workflow | Graph | Graph/Semantic Search | Episodic Consolidation |
| Agent-KB | 2025.7 | Tips & Workflow | Hybrid DB | Hybrid Search | Deduplication |

所有系统继承自统一的抽象基类 `BaseMemoryProvider`，强制实现四个模块接口。

### 标准化评估

EvolveLab 提供开箱即用的基准测试支持：
- **基准**：GAIA, xBench, WebWalkerQA, TaskCraft
- **评估模式**：
  - **Online**：边执行任务边更新记忆
  - **Offline**：先从静态轨迹积累经验，再在新任务上评估
- **评估协议**：精确匹配、LLM-as-a-Judge

---

## 实验结果

### 主要发现

**1. 记忆系统对 Agent 性能至关重要**
- SmolAgent + MemEvolve：在 xBench 上从 51% → 57%（pass@1），68%（pass@3）
- Flash-Searcher + MemEvolve：在 WebWalkerQA 上从 71.18% → 74.71%（pass@1）

**2. 跨任务泛化**
- 在 TaskCraft 上演进的记忆系统，直接迁移到 WebWalkerQA 和 xBench 仍有显著提升
- 说明 MemEvolve 学到的是**任务无关的记忆设计原则**，而非过拟合特定数据集

**3. 跨 LLM 泛化**
- 用 GPT-5-mini 演进的记忆系统，迁移到 Kimi K2 和 DeepSeek V3.2 仍有效
- Kimi K2 在 WebWalkerQA 上提升 17.06%，TaskCraft 上提升 10.0%

**4. 跨框架泛化**
- 在 TaskCraft 上演进的记忆系统，迁移到不同的 Agent 框架（Cognitive Kernel-Pro, OWL）仍能带来一致的性能提升
- 说明 MemEvolve 学到的是**框架无关的记忆抽象**

### 性能对比

在 GAIA 基准上（pass@3）：
- Flash-Searcher（无记忆）：69.09%
- Flash-Searcher + MemEvolve：**80.61%**（+11.52%）
- 超越多个强多智能体系统（OWL-Workforce, CK-Pro）

---

## 核心贡献

1. **EvolveLab 代码库**：统一实现 12 个记忆系统的模块化框架，提供标准化基准支持
2. **MemEvolve 框架**：通过双层优化同时演进经验知识和记忆架构
3. **实验验证**：
   - 显著性能提升（最高 +17.06%）
   - 强跨任务、跨 LLM、跨框架泛化能力

---

## 技术细节

### 双层优化形式化

**内循环（经验演进）**：
对于每个候选记忆系统 $\Omega_j^{(k)}$，在轨迹 $\tau \in \mathcal{T}_j^{(k)}$ 上更新记忆状态：

$$
M_{t+1,j}^{(k)} = \Omega_j^{(k)}(M_{t,j}^{(k)}, \epsilon_\tau), \quad \epsilon_\tau \in \mathcal{E}_j^{(k)}(\tau)
$$

评估性能并汇总为适应度 $\mathbf{F}_j^{(k)}$。

**外循环（架构演进）**：
基于性能反馈，元进化算子 $\mathcal{F}$ 选择高性能候选并生成新变体：

$$
\{\Omega_{j'}^{(k+1)}\}_{j' \in \mathcal{J}^{(k+1)}} = \mathcal{F}(\{\Omega_j^{(k)}\}_{j \in \mathcal{J}^{(k)}}, \{\mathbf{F}_j^{(k)}\}_{j \in \mathcal{J}^{(k)}})
$$

### 实验配置

- **迭代次数**：$K_{\max} = 3$
- **幸存者预算**：每轮保留 top-1，扩展为 3 个后代
- **内循环批次**：60 个任务（40 新任务 + 20 复用任务）
- **Agent 框架**：SmolAgent, Flash-Searcher（训练），CK-Pro, OWL（泛化测试）
- **LLM 骨干**：GPT-5-mini（主要），Kimi K2, DeepSeek V3.2（泛化测试）

---

## 局限与未来方向

### 当前局限

1. **计算成本**：双层优化需要多轮迭代，每轮评估多个候选系统
2. **演进轮数有限**：实验只进行 3 轮迭代，更长期的演进效果未知
3. **模块粒度**：四模块分解可能不够细粒度，限制了搜索空间

### 未来方向

1. **更高效的元学习**：减少内循环评估成本，加速架构搜索
2. **更细粒度的模块化**：探索更细致的记忆组件分解
3. **持续演进**：研究长期演进中的稳定性和收敛性
4. **多任务联合演进**：同时在多个任务上演进，学习更通用的记忆架构

---

## 相关概念

**前置知识**：
- Agent 系统的基本架构（状态、动作、策略）
- 强化学习中的经验回放和记忆机制

**相关方法**：
- **Voyager**：基于技能库的 Agent 记忆系统
- **ExpeL**：通过对比学习提取经验洞察
- **Agent Workflow Memory**：存储可复用的工作流模板

**元学习视角**：
- 内循环 = 任务级学习（学习解决具体任务）
- 外循环 = 元级学习（学习如何学习）

---

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/2512.18746/2512.18746|MemEvolve 论文全文]]

**相关工作**：
- Voyager (Wang et al., 2023) — 早期技能库记忆系统
- ExpeL (Zhao et al., 2024) — 基于洞察的经验学习
- Agent Workflow Memory (Wang et al., 2024) — 工作流记忆
- SkillWeaver (Zheng et al., 2025) — API 级技能演进
