---
description: 将强化学习应用于训练 LLM Agent 的方法，让模型学会在交互式环境中通过多轮对话、工具调用完成复杂任务
type: method
aliases:
  - Agentic Reinforcement Learning
  - Agent RL
  - 智能体强化学习
prerequisites:
  - "[[RLHF]]"
  - "[[PPO]]"
  - "[[GRPO]]"
tags:
  - post-training
  - reinforcement-learning
  - agent
created: 2026-01-27
updated: 2026-02-01T00:59
---

# Agentic RL (Agentic Reinforcement Learning)

智能体强化学习（Agentic RL）是将强化学习应用于训练 LLM Agent 的方法。与传统 [[RLHF]] 的单轮生成不同，Agentic RL 让模型学会在**交互式环境**中通过多轮对话、工具调用来完成复杂任务。

> [!paper] 代表性工作
> - MiMo-V2-Flash (2025) — Code Agent RL scaling
> - LongCat-Flash-Thinking (2025) — Environment scaling, DORA 框架
> - Kimi-K2 (2025) — Agentic capabilities

---

## 动机

> [!intuition] 为什么需要 Agentic RL？

传统 RLHF 的局限：

1. **单轮交互**：RLHF 优化的是单次 prompt → response，无法处理需要多轮交互的复杂任务
2. **无环境反馈**：传统 RL 只有 [[Reward Model]] 的打分，没有真实环境的执行反馈
3. **短 horizon**：轨迹长度固定且较短，无法学习长程规划

现实世界的 Agent 任务需要：
- 在代码仓库中定位和修复 bug（需要多次搜索、阅读、编辑）
- 通过多步搜索整合信息（需要判断何时停止、如何组合结果）
- 调用多种工具完成复杂任务（需要工具选择、错误恢复）

---

## 与传统 RL 的核心区别

> [!comparison] Non-Agentic RL vs Agentic RL

| 维度 | Non-Agentic RL (RLHF) | Agentic RL |
|------|----------------------|------------|
| **交互模式** | 单轮 prompt → response | 多轮 agent ↔ environment |
| **环境** | 无真实环境 | 需要可执行环境 |
| **奖励** | 即时（outcome-based） | 延迟/稀疏（任务完成时） |
| **轨迹长度** | 短且固定 | 长且变化大 |
| **状态** | 无状态 | 需要追踪环境状态 |
| **动作空间** | 文本生成 | 文本 + 工具调用 |

> [!intuition] 本质区别
> 传统 RLHF 是 **generation alignment**——让模型生成人类喜欢的回答。
> Agentic RL 是 **behavior alignment**——让模型学会在环境中采取正确的行动序列。

---

## 核心挑战

### 1. 长尾分布

> [!warning] 轨迹长度差异巨大

不同任务的轨迹长度差异可达 10 倍以上：
- 简单任务：2-3 轮交互
- 复杂任务：20+ 轮交互

这导致：
- **批处理困难**：短轨迹等待长轨迹完成
- **梯度估计偏差**：长轨迹的 reward 信号被稀释
- **内存管理复杂**：需要处理变长序列

### 2. 环境多样性

不同任务需要不同的环境和工具：
- Code Agent 需要代码执行环境、文件系统
- Search Agent 需要搜索 API、网页解析
- Tool-Use Agent 需要各种外部 API

### 3. 稀疏奖励

> [!warning] 只有任务完成时才有明确反馈

与传统 RLHF 的 outcome reward 不同，Agentic RL 的稀疏性更严重：
- 中间步骤没有 reward 信号
- 错误可能在很多步之后才暴露
- 难以判断哪一步导致了失败

### 4. 探索效率

环境交互成本高昂：
- 每次 rollout 需要真实执行
- 工具调用有延迟和配额限制
- 错误的探索可能破坏环境状态

---

## 典型 Agent 类型

> [!example] 主要 Agent 类型及其评估基准

### Code Agent
- **任务**：在代码仓库中定位和修复 bug
- **环境**：代码执行环境、文件系统、测试框架
- **基准**：SWE-Bench Verified/Multilingual
- **挑战**：需要理解大型代码库、多文件修改

### Search Agent
- **任务**：多步搜索和信息整合
- **环境**：搜索 API、网页内容
- **基准**：BrowseComp, RWSearch
- **挑战**：判断何时停止搜索、如何整合矛盾信息

### Tool-Use Agent
- **任务**：通用工具调用完成复杂任务
- **环境**：多种外部 API
- **基准**：τ²-Bench
- **挑战**：工具选择、参数构造、错误处理

### Terminal Agent
- **任务**：命令行问题解决
- **环境**：Shell 环境
- **基准**：Terminal-Bench
- **挑战**：命令组合、状态追踪

---

## 环境设计

> [!intuition] 好的训练环境需要什么？

### 核心要求

1. **可执行性**：环境必须能真实执行 agent 的动作
2. **可验证性**：需要自动化的正确性判断（不能依赖人工）
3. **多样性**：覆盖不同领域和难度
4. **可扩展性**：能够自动化生成新的训练任务

### LongCat 的环境扩展方法

> [!example] 自动化环境扩展 Pipeline

LongCat 提出了一个自动化的环境扩展方法：
1. **领域识别**：识别 20+ 个可训练的 agent 领域
2. **任务生成**：自动生成该领域的训练任务
3. **验证器构建**：为每个任务构建自动化验证器
4. **难度分级**：根据任务复杂度分级

覆盖领域包括：代码、搜索、数学、数据分析、文件操作等。

---

## 训练策略

### Cold Start

> [!definition] 用 [[SFT]] 数据初始化基本 agent 行为

在 RL 训练之前，需要用监督数据让模型学会：
- 基本的工具调用格式
- 环境交互的 turn-taking
- 常见任务的解决模式

没有 cold start，模型可能完全无法与环境交互。

### Curriculum Learning

> [!intuition] 从简单任务逐步到复杂任务

直接在困难任务上训练效率很低：
- 随机探索难以找到正确解
- 稀疏 reward 导致梯度信号弱

Curriculum 策略：
1. 先在简单任务上训练，建立基本能力
2. 逐步增加任务难度
3. 难度可以是：轨迹长度、工具数量、环境复杂度

### Noisy Environment Training

> [!warning] 真实世界的环境是不完美的

训练时引入噪声：
- 工具调用偶尔失败
- 环境返回不完整信息
- 网络延迟和超时

这让模型学会：
- 错误检测和恢复
- 重试策略
- 降级方案

---

## 基础设施需求

> [!example] 大规模 Agentic RL 的基础设施

### 大规模环境并行

- 需要 10,000+ 并发环境实例
- 每个环境独立运行，互不干扰
- 环境状态需要持久化和恢复

### 异步训练架构

传统同步训练的问题：
- 短轨迹等待长轨迹完成
- GPU 利用率低
- 无法处理长尾分布

**DORA 框架**（LongCat）：
- Rollout 和训练解耦
- 异步收集轨迹
- 支持 partial rollout（长轨迹分段处理）

### 工具调用管理

- **Toolbox**：统一的工具接口抽象
- **Tool Manager**：工具调用的调度和限流
- **Error Handler**：工具失败的统一处理

---

## 算法选择

> [!comparison] Agentic RL 中的算法选择

| 算法 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| [[PPO]] | 稳定、成熟 | 需要 Critic、内存大 | 小规模实验 |
| [[GRPO]] | 无需 Critic、高效 | 组大小敏感 | 中等规模 |
| [[DAPO]] | 解决 entropy collapse | 更复杂 | 大规模训练 |

> [!intuition] 为什么 GRPO 系列更适合 Agentic RL？
>
> Agentic RL 的轨迹长度变化大，传统 value function 难以准确估计。GRPO 的 group relative advantage 天然适应这种变化——同一任务的不同 rollout 形成自然的比较组。

---

## 评估基准

> [!example] 主要评估基准

### 代码能力
- **SWE-Bench Verified**：真实 GitHub issue 修复
- **SWE-Bench Multilingual**：多语言代码修复

### 搜索能力
- **BrowseComp**：复杂信息检索
- **RWSearch**：真实世界搜索任务

### 工具使用
- **τ²-Bench**：通用工具调用评估

### 终端操作
- **Terminal-Bench**：命令行任务

---

## 局限性

> [!warning] Agentic RL 的当前局限

1. **环境构建成本高**：每个领域都需要专门的环境和验证器
2. **训练效率低**：环境交互比纯文本生成慢得多
3. **泛化困难**：在一个环境训练的能力难以迁移到新环境
4. **安全风险**：Agent 在真实环境中执行可能造成不可逆影响
5. **评估困难**：复杂任务的成功标准难以自动化判断

---

## 面试要点

> [!interview] 常见问题

> **Q1: Agentic RL 与传统 RLHF 的核心区别是什么？**
> A: 传统 RLHF 是单轮生成优化，Agentic RL 是多轮交互优化。Agentic RL 需要处理环境状态、工具调用、长 horizon 轨迹，面临稀疏奖励和长尾分布等挑战。

> **Q2: Agentic RL 的主要挑战有哪些？**
> A: 四个核心挑战：(1) 长尾分布——轨迹长度差异大；(2) 环境多样性——不同任务需要不同环境；(3) 稀疏奖励——只有任务完成时才有反馈；(4) 探索效率——环境交互成本高。

> **Q3: 为什么需要 Cold Start？**
> A: 没有 SFT 初始化，模型无法与环境正常交互。Cold Start 让模型学会基本的工具调用格式和交互模式，为后续 RL 训练提供起点。

> **Q4: DORA 框架解决了什么问题？**
> A: 解决了长尾分布导致的训练效率问题。通过异步 rollout 和训练解耦，避免短轨迹等待长轨迹，支持 partial rollout 处理超长轨迹。

> **Q5: 如何设计好的训练环境？**
> A: 四个要求：可执行性（能真实执行）、可验证性（自动判断正确性）、多样性（覆盖不同领域）、可扩展性（能自动生成新任务）。

---

## 相关概念

- [[RLHF]] — Agentic RL 的基础，单轮版本
- [[PPO]] — 常用的 RL 算法
- [[GRPO]] — 更适合 Agentic RL 的高效算法
- [[DAPO]] — 解决大规模训练稳定性问题

---

## 延伸阅读

- [[Tool Use]] — Agent 的核心能力之一
- [[ReAct]] — 推理与行动交替的 Agent 框架
- [[Function Calling]] — 工具调用的实现方式
