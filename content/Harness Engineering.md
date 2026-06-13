---
type: concept
description: Harness Engineering 将 LLM agent 的可靠性拆成模型外部运行时基础设施的设计、治理与评估问题，关注执行、工具、上下文、状态、权限、观测和验证如何共同约束长程行动
aliases:
  - Harness 工程
  - Agent Harness
  - Agent Harness Engineering
  - 智能体 Harness
  - 智能体 Harness 工程
prerequisites:
  - "[[Agentic RL]]"
  - "[[SWE-Bench]]"
  - "[[Context Compression]]"
tags:
  - agent
  - infrastructure
  - evaluation
  - architecture
  - software-engineering
created: 2026-06-13
updated: 2026-06-13
---

# Harness Engineering

Harness Engineering（Harness 工程）研究如何设计、优化和治理围绕 LLM agent 的运行时基础设施。它的核心判断是：长程智能体的表现来自 **模型、harness 与环境** 的耦合；当模型要持续观察、行动、记忆、验证和恢复时，harness 决定模型看到什么、能做什么、如何执行、失败如何反馈、结果如何被评估。

一个实用的理解方式是：模型负责局部判断和语言推理，harness 负责把这些判断接入可执行世界。没有高质量 harness，强模型也可能因为工具接口模糊、上下文腐化、状态丢失、权限过宽、验证器薄弱而在长任务中失败。

> [!definition] 定义
> Harness Engineering 是面向 LLM agent 的系统工程方法：围绕模型构造执行环境、工具接口、上下文管理、状态存储、生命周期控制、可观测性、验证评估和治理安全机制，使模型输出能转化为可执行、可恢复、可审计、可评估的长程行为。

## 为什么需要这个概念

早期 LLM 应用的主要工程对象是一次调用：给模型什么 prompt，如何约束输出格式，如何把答案展示给用户。Agent 系统把任务形态变成多步过程：模型需要读文件、查网页、调用 API、执行 shell、修改状态、等待反馈、重试失败路径，甚至把任务委托给其他 agent。

这里出现了一个结构性错配：

| 模型原生接口 | 长程任务要求 |
| --- | --- |
| 单次输入输出 | 多轮状态转移 |
| 自由文本生成 | 结构化行动与工具调用 |
| 上下文窗口内记忆 | 跨轮次、跨会话、跨环境的持久状态 |
| 输出质量评估 | 轨迹、成本、安全和结果联合评估 |
| 局部推理能力 | 目标维持、失败恢复、权限治理和回滚 |

[[Agentic RL]] 关注如何训练模型在交互环境中采取更好的行动序列；Harness Engineering 关注这些行动序列运行在什么基础设施里。两者是同一问题的两面：训练让模型更会行动，harness 让行动更可控、更可观测、更容易验证。

> [!intuition] 直觉理解
> 如果把模型输出看成“行动意图”，harness 就是把意图变成现实动作的运行层。它会检查这个意图是否合法，选择合适工具，准备上下文，执行动作，记录轨迹，把结果反馈给模型，并在失败时决定重试、回滚、升级审批或终止。

## Agent Harness 与 Harness Engineering

这两个词经常一起出现，但粒度不同。

| 概念 | 关注点 | 例子 |
| --- | --- | --- |
| Agent harness | 具体运行时系统 | Claude Code / Codex / OpenHands 的执行循环、工具、沙箱、权限、日志和评估接口 |
| Harness Engineering | 设计和优化这类系统的方法 | 选择工具 schema、设计上下文策略、加入 trace、构造回归集、调整权限和重试策略 |

Agent harness 是被设计的对象，Harness Engineering 是围绕这个对象做系统设计、评估、优化和治理的学科化视角。

## 核心机制：把一次生成变成长程闭环

一个典型 harness 可以被看成下面的闭环：

```text
用户目标
  -> 任务状态建模
  -> 上下文组装
  -> 模型产生计划或行动
  -> 生命周期策略检查
  -> 工具 / 环境执行
  -> 观察结果、日志、成本、错误回流
  -> 状态更新、记忆写入、验证评估
  -> 下一轮行动或终止
```

这个闭环里，模型每一步只看到 harness 投影出来的任务状态。真实状态分散在文件系统、浏览器、数据库、工具返回、轨迹日志、审批记录、用户反馈和长期记忆中；harness 的上下文策略决定哪些信息进入当前窗口，状态策略决定哪些信息跨轮次保留，验证策略决定哪些行为算成功。

> [!math] 形式化视角
> 可以把 harness 看成一个带治理的状态转移系统：
>
> $$
> o_t = C(q_t, h_t), \quad a_t \sim M(o_t), \quad e_t = T(a_t), \quad q_{t+1} = L(q_t, a_t, e_t)
> $$
>
> 其中 $q_t$ 是 harness 维护的运行状态，$h_t$ 是历史轨迹，$C$ 是上下文管理器，$M$ 是模型，$T$ 是工具或环境执行层，$L$ 是生命周期与策略控制器。评估接口 $V$ 再把轨迹 $\tau=(q_0,o_0,a_0,e_0,\dots)$ 映射成成功率、成本、风险、恢复能力和可复现性等指标。
>
> 这个表达式强调一点：模型动作 $a_t$ 的质量只是一部分，$C,T,L,V$ 的设计会改变模型实际面对的问题。

## 两个常用框架

现有相关笔记里有两个互补框架：[[Agent Harness for Large Language Model Agents (2026)]] 的 ETCSLV 六组件框架，以及 [[Agent Harness Engineering (2026)]] 的 ETCLOVG 七层工程分类。前者适合定义最小运行时接口，后者适合排查生产系统责任边界。

### ETCSLV：运行时治理接口

ETCSLV 把 harness 写成 $\mathcal{H}=(E,T,C,S,L,V)$：

| 组件 | 负责的问题 | 常见失败 |
| --- | --- | --- |
| $E$ Execution loop | 模型怎样反复观察、行动、终止或恢复 | 死循环、终止条件不清、环境 reset 不可靠 |
| $T$ Tool registry | 工具有哪些、怎样发现、校验参数和路由调用 | 工具选择错误、权限过宽、工具描述污染 |
| $C$ Context manager | 每一步模型看到什么信息 | 上下文爆炸、关键状态丢失、旧注入滞留 |
| $S$ State store | 什么状态跨轮次、跨会话、跨 agent 保留 | 中断后丢状态、记忆污染、并发写冲突 |
| $L$ Lifecycle hooks | 调用前后如何鉴权、审计、审批、回滚 | 越权行动、缺少审批、侧效应不可追踪 |
| $V$ Verification interface | 轨迹和结果如何进入评估系统 | 只能看最终结果，无法归因失败 |

这组框架的价值在于判断一个系统的完整性。很多 agent framework 提供链、图、工具调用封装，却把状态恢复、权限、轨迹格式和评估接口留给使用者；完整 harness 会把这些运行时治理功能纳入同一个系统。

### ETCLOVG：生产工程分层

ETCLOVG 把生产级 harness 拆成七层：

| 层 | 关键问题 | 设计对象 |
| --- | --- | --- |
| Execution Environment | 智能体在哪里行动 | 沙箱、工作区、依赖镜像、浏览器、shell、reset/checkpoint |
| Tool Interface | 行动空间怎样暴露 | function calling、OpenAPI、MCP、工具检索、参数校验、工具版本 |
| Context Management | 模型每轮看到什么 | prompt、检索、摘要、上下文压缩、状态投影、来源标记 |
| Lifecycle / Orchestration | 多步流程怎样推进 | 计划、任务队列、重试、回滚、子 agent、预算和终止条件 |
| Observability | 执行过程怎样被诊断 | trace tree、span、token/cost、工具参数、错误、文件 diff、回放 |
| Verification / Evaluation | 行为怎样被判断 | 结果评估、轨迹评估、grader 评估、回归集、oracle 充分性 |
| Governance / Security | 能力怎样落进责任边界 | 身份、权限、凭证隔离、审批门、策略 DSL、审计、合规 |

ETCSLV 更像运行时 API，ETCLOVG 更像系统设计清单。实际设计时可以先用 ETCSLV 确认组件覆盖面，再用 ETCLOVG 排查每个组件背后的工程责任。

## 设计原则

### 1. 状态要外化、版本化、可恢复

长程任务的状态不能完全依赖模型上下文。计划、已读文件、失败命令、用户决策、工具返回、审批记录、当前假设和中间产物都需要以可检索、可审计、可回滚的形式保存在 harness 中。

这里和 [[Context Compression]] 的关系很直接：压缩可以降低上下文成本，但压缩后的摘要只是一种状态视图。可靠 harness 还要保存原始轨迹或可追溯 handle，让失败诊断和回放评估能回到证据源。

### 2. 行动空间要按任务裁剪

工具越多，智能体覆盖的任务越广，同时选择噪声、token 开销和攻击面也会上升。可靠 harness 会维护完整工具注册表，但每一步只给模型暴露当前任务需要的候选工具，并记录工具定义版本、权限范围、调用参数、执行结果和错误类型。

[[Model Context Protocol]] 这类协议可以成为工具层标准化接口，但协议只解决一部分问题。harness 还要处理身份、授权传播、工具输出清洗、异常重试、凭证隔离和审计。

### 3. 上下文是状态投影

模型看到的是 harness 根据目标、预算、风险和历史轨迹构造出的任务状态投影。这个投影需要回答四个问题：

- 当前决策真正需要哪些事实？
- 哪些信息可以只放引用或摘要？
- 哪些内容来自不可信工具输出或用户输入？
- 哪些旧信息已经过期，需要降权或移出活跃窗口？

当投影质量下降时，模型可能表现得像推理失败；实际根因可能是上下文层隐藏了关键证据、保留了过期假设，或把恶意工具输出当成可信系统指令。

### 4. 验证要覆盖结果、过程和评估器

长程 agent 的成功率需要结合最终结果、执行过程和评估器质量。成熟 harness 至少要区分三层证据：

| 评估层 | 问题 | 例子 |
| --- | --- | --- |
| 结果级 | 任务最终完成了吗？ | 测试通过、页面状态正确、订单未出错 |
| 过程级 | 完成路径可靠吗？ | 工具调用合理、无越权、成本可接受、能从失败恢复 |
| 评估器级 | 判断成功的 oracle 可靠吗？ | 测试覆盖是否足够、grader 是否稳定、人工审核是否一致 |

[[SWE-Bench]] 的价值在于把真实 GitHub issue 转成可执行验证任务；它也暴露了 harness 评估的典型问题：同一个 patch 成功与否取决于仓库快照、依赖环境、测试命令、fail-to-pass 测试、patch 应用方式和日志处理。

### 5. 任何 side effect 都要绑定身份、权限和审计

当 agent 能写文件、发邮件、调用企业 API、开 PR、下单或操作浏览器时，harness 必须记录谁授权、以什么身份执行、使用了哪个工具版本、读写了哪些资源、触发了哪个策略、是否经过审批、失败后如何回滚。

这也是 Harness Engineering 相比普通 LLM 应用工程更接近系统安全的地方。风险会沿着工具结果、记忆写入、子 agent 委托和长期状态传播；治理层需要跟执行层、状态层和评估层一起设计。

## 设计一个 harness 的问题清单

下面这组问题适合用来检查一个代码 agent、浏览器 agent、企业自动化 agent 或研究助理 agent。

### 任务与环境

- 任务的成功标准能否自动验证？需要哪些人工 gate？
- 环境是否可 reset、checkpoint、diff 和 replay？
- 执行是否发生在隔离沙箱？依赖、网络、文件权限如何限定？
- 外部 API、数据库或真实用户状态是否会产生不可逆 side effect？

### 工具与行动

- 工具注册表如何维护？工具 schema 是否版本化？
- 每一步暴露给模型的工具集合怎样裁剪？
- 工具参数怎样校验？工具返回怎样过滤、压缩和标注来源？
- 工具失败时，错误属于参数问题、权限问题、环境问题还是工具实现问题？

### 上下文与状态

- 当前上下文由哪些来源组成？每个来源的可信度和新鲜度如何记录？
- 哪些信息留在活跃窗口，哪些进入外部状态，哪些可以丢弃？
- 计划、假设、失败原因、中间产物是否能跨轮次恢复？
- 长期记忆写入是否需要验证、去重、过期和撤销机制？

### 生命周期与编排

- agent 何时规划、执行、验证、反思、重试、终止？
- 失败后是重试、回滚、换工具、请求用户、委托子 agent，还是停止？
- 多 agent 协作时，身份、消息格式、共享状态和权限怎样传播？
- 预算约束如何施加到 token、时间、工具次数、成本和高风险动作上？

### 可观测性与评估

- trace 是否能重放一次运行的关键决策？
- 每次模型调用、工具调用、上下文组装和状态写入是否有 span？
- 评估是否同时记录成功率、成本、延迟、工具调用次数、重试次数、权限事件和人工介入？
- 回归集是否覆盖历史失败、边界任务、安全事件和高成本路径？

## 典型案例：代码智能体

代码 agent 是理解 Harness Engineering 的最佳入口之一。一个能在 [[SWE-Bench]] 上工作的代码 agent，至少需要以下 harness 能力：

1. 接收 issue，并把仓库状态、文件结构、测试线索和历史轨迹组织成上下文。
2. 暴露搜索、读取、编辑、shell、测试运行和版本控制工具，同时限制路径、命令和网络权限。
3. 保存计划、已检查文件、失败测试、补丁 diff、依赖状态和当前假设。
4. 在每次编辑后运行验证器，把编译错误、测试失败、lint 结果和安全扫描反馈给下一轮。
5. 记录完整轨迹，支持事后判断失败来自模型误解、检索不足、工具错误、环境配置、弱测试或权限限制。

[[Code as Agent Harness (2026)]] 把这类系统进一步概括成以代码为中心的 harness：代码、测试、日志、配置、仓库和执行环境共同构成可执行、可检查、可共享的状态层。这里的 Plan-Execute-Verify loop 很关键：Plan 把目标写成状态变更契约，Execute 在沙箱里执行变更，Verify 用测试、静态分析、人工 review 或其他 oracle 判断新状态是否可接受。

## 与相邻概念的关系

| 概念 | 侧重点 | 与 Harness Engineering 的关系 |
| --- | --- | --- |
| Prompt Engineering | 单次调用的指令、示例和输出格式 | 是 harness 中上下文设计的一小部分 |
| [[Context Engineering]] | 多信息源如何进入模型上下文 | 是 harness 的 $C$ 层核心能力 |
| [[RAG]] | 从外部知识库检索相关信息 | 可作为上下文和工具层组件，服务于状态投影 |
| Function Calling / Tool Use | 模型如何产生结构化工具调用 | 是 $T$ 层的基础接口，仍需权限、路由、审计和恢复 |
| [[Agentic RL]] | 在交互环境中训练多步行动策略 | 训练模型利用 harness；harness 也提供 rollout 环境和反馈信号 |
| Agent Framework | 提供构建 agent 的开发原语 | 可能只覆盖链、图、工具封装；生产 harness 还要覆盖状态、权限、观测和评估 |
| RL Environment | 提供 observation/action/reward/reset 接口 | 为 harness 提供可复现交互接口的历史来源之一 |
| Software Testing Harness | 控制程序输入、执行和结果记录 | 为 agent harness 提供隔离、setup/teardown 和结果证据的历史来源之一 |

## 常见失败模式

> [!warning] Harness 失败并不总是表现为系统崩溃
> 许多 harness 问题会伪装成“模型不够聪明”：模型选错工具，可能来自工具描述过宽；模型忘记约束，可能来自上下文投影丢失；模型反复试错，可能来自验证器太弱；模型越权行动，可能来自生命周期钩子缺失。

| 失败模式 | 现象 | 根因 |
| --- | --- | --- |
| Context rot | 长任务后模型越来越依赖旧假设 | 旧工具结果、过期摘要和注入内容长期滞留 |
| Tool overload | 模型在大量工具中频繁选错 | 工具候选没有按任务裁剪，schema 描述不清 |
| State drift | 模型以为任务处于某状态，真实环境已经变化 | 状态更新和上下文投影脱节 |
| Weak oracle | 指标显示成功，真实语义仍有错误 | 测试、grader 或人工 rubric 覆盖不足 |
| Reward / evaluator hacking | agent 学会满足评估器表面规则 | 验证器太窄，轨迹级检查不足 |
| Retry amplification | 重试提高少数任务成功率，同时成本暴涨 | 缺少预算控制和失败类型归因 |
| Permission creep | 为了完成更多任务不断放宽工具权限 | 权限与任务风险没有绑定，审计和审批滞后 |
| Memory poisoning | 不可信内容进入长期记忆并影响未来任务 | 写入缺少来源、置信度、验证和撤销机制 |
| Multi-agent coordination debt | 多 agent 系统比单 agent 更慢更脆 | 共享状态、身份、消息协议和收敛标准不清 |
| Harness overfitting | 在某个 benchmark 上提升，换环境后退化 | harness 策略贴合特定测试或环境噪声 |

## 如何评估 Harness Engineering 的效果

Harness 的评估单位应是 **model-harness-environment pair**。报告“某模型在某基准上的分数”时，需要同时说明模型版本、工具集合、沙箱镜像、上下文策略、重试预算、权限规则、验证器、成本限制和评估接口。

一个较完整的评估画像包括：

| 指标族 | 衡量内容 |
| --- | --- |
| Task success | 最终成功率、pass@k、部分完成度 |
| Cost / latency | token、工具调用次数、wall-clock、并发环境成本 |
| Trajectory quality | 冗余步骤、错误恢复、计划更新、工具选择质量 |
| State quality | 状态一致性、checkpoint 恢复率、记忆污染率 |
| Safety / governance | 越权尝试、审批触发、凭证暴露、策略违规 |
| Observability | trace 完整度、可重放性、失败归因覆盖率 |
| Oracle adequacy | 测试覆盖、grader 稳定性、人工审核一致性 |
| Robustness | 环境扰动、工具失败、上下文缺失、长任务压力下的表现 |

这个评估方式会改变系统优化的目标。单纯提高成功率可能让系统通过更多重试、更多 token 或更宽权限获得分数；生产系统还要看单位成本成功率、风险调整后的成功率，以及失败时是否能留下足够证据。

## 自动化 Harness Engineering

随着 agent 系统留下越来越多结构化轨迹，harness 本身也可以成为优化对象。自动化 Harness Engineering 的基本循环是：

1. 从运行轨迹中定位失败模式：缺文件、工具误用、验证器盲区、权限阻塞、上下文超载、重试无效。
2. 生成候选 harness 修改：改工具 schema、调整上下文打包、加入验证器、改变重试策略、增加审批 gate、分离子 agent。
3. 在回归集和安全用例上回放评估。
4. 只发布通过不变量检查的 harness 变更，并保留回滚路径。

这条路线把 harness 从固定包装层推进到可测量、可搜索、可演化的系统对象。真正困难的部分在于防止自我优化破坏安全边界、隐藏失败、过拟合 benchmark、增加不可接受的成本，或让稀有任务回归。

## 边界与局限

> [!warning] 使用边界
> - 短任务、低风险、无工具调用的 LLM 应用通常只需要轻量上下文和输出校验。完整 harness 会增加工程复杂度和延迟。
> - 模型基础能力不足时，harness 只能改善任务呈现、执行控制和错误恢复，无法凭空补齐核心推理或领域知识。
> - 任务 oracle 很弱时，harness 会更擅长产生“看起来可验证”的行为，真实语义质量仍取决于验证器覆盖面。
> - 高权限环境里，能力扩展必须和权限、审计、审批、回滚成对设计。
> - 多 agent、长期记忆、自演化 harness 都会引入持久状态污染和责任归因问题，收益需要用优化过的单 agent 基线对照。

## 论文来源与知识库连接

> [!paper] 主要来源
> - [[Agent Harness Engineering (2026)]] — 提出 ETCLOVG 七层分类，适合做生产系统设计和故障定位清单。
> - [[Agent Harness for Large Language Model Agents (2026)]] — 提出 $\mathcal{H}=(E,T,C,S,L,V)$ 六组件框架，适合理解 harness 的运行时治理接口。
> - [[Agent Systems with Harness Engineering (2026)]] — 把 scaffold 侧设计、模型侧适配和 harness capacity 评测放进同一个联合优化视角。
> - [[Externalization in LLM Agents (2026)]] — 从认知外化角度解释记忆、技能、协议和 harness 为什么会共同成为 agent 基础设施。
> - [[Code as Agent Harness (2026)]] — 解释代码、测试、日志、仓库和执行环境如何成为可执行、可验证、可共享的 harness 介质。

Harness Engineering 可以作为 [[Agentic RL]] 的系统侧补充：Agentic RL 训练模型利用交互环境，Harness Engineering 设计这个环境的接口、状态、权限、观测和验证。它也可以作为 [[SWE-Bench]] 的解释层：SWE-Bench 分数测量的是模型、代码仓库、工具、测试、沙箱和评估流程组成的整体。

## 面试视角

> [!interview] Q: 一句话解释 Harness Engineering？
> A: Harness Engineering 是围绕 LLM agent 构建运行时基础设施的方法，把模型的局部判断接入工具、状态、权限、观测和验证闭环，使长程任务变成可执行、可恢复、可审计、可评估的系统行为。

> [!interview] Q: 为什么同一个模型在不同 agent 系统里表现差异很大？
> A: Harness 决定模型看到的上下文、可用工具、错误格式、状态恢复、重试策略、权限边界和评估接口。长程任务会把这些差异逐步放大，所以最终表现通常是模型、harness 和环境的组合结果。

> [!interview] Q: Harness Engineering 和 Context Engineering 的关系是什么？
> A: Context Engineering 管理模型每一步看到的信息，是 harness 的上下文层。Harness Engineering 覆盖更大的运行时系统，还包括执行环境、工具、状态、生命周期、可观测性、验证和治理。

> [!interview] Q: 如何判断一个 harness 设计是否成熟？
> A: 看它能否稳定执行、限制行动空间、维护状态、恢复失败、记录可重放轨迹、提供多层验证、绑定权限责任，并在报告评估结果时披露模型、工具、环境和策略配置。

> [!interview] Q: Harness Engineering 最大的风险是什么？
> A: 风险来自复杂性和耦合。更复杂的 harness 可能提升成功率，也可能增加调试面、隐藏失败、扩大权限、污染长期状态，或让评估分数依赖某个特定环境。成熟做法是把 harness 变更当成系统变更，用回归集、安全不变量、成本指标和审计记录共同约束。

## 速查

> [!example] 记忆锚点
> - 一句话：Harness Engineering 设计模型外部的 agent 运行时治理系统。
> - 核心单位：model-harness-environment pair。
> - 最小框架：$\mathcal{H}=(E,T,C,S,L,V)$。
> - 生产清单：Execution、Tool、Context、Lifecycle、Observability、Verification、Governance。
> - 关键问题：模型看到什么、能做什么、状态如何保留、失败如何恢复、行为如何验证、责任如何追踪。
> - 典型指标：成功率、成本、轨迹质量、状态一致性、权限事件、oracle 充分性、可重放性。
