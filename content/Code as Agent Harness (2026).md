---
type: paper
title: Code as Agent Harness
authors:
  - Xuying Ning
  - Katherine Tieu
  - Dongqi Fu
  - Tianxin Wei
  - Zihao Li
  - Yuanchen Bei
  - Jiaru Zou
  - Mengting Ai
  - Zhining Liu
  - Ting-Wei Li
  - Lingjie Chen
  - Yanjun Zhao
  - Ke Yang
  - Bingxuan Li
  - Cheng Qian
  - Gaotang Li
  - Xiao Lin
  - Zhichen Zeng
  - Ruizhong Qiu
  - Sirui Chen
  - Yifan Sun
  - Xiyuan Yang
  - Ruida Wang
  - Rui Pan
  - Chenyuan Yang
  - Dylan Zhang
  - Liri Fang
  - Zikun Cui
  - Yang Cao
  - Pan Chen
  - Dorothy Sun
  - Ren Chen
  - Mahesh Srinivasan
  - Nipun Mathur
  - Yinglong Xia
  - Hong Li
  - Hong Yan
  - Pan Lu
  - Lingming Zhang
  - Tong Zhang
  - Hanghang Tong
  - Jingrui He
year: 2026
arxiv: "2605.18747"
github: https://github.com/YennNing/Awesome-Code-as-Agent-Harness-Papers
description: 提出 Code as Agent Harness 视角，将代码定位为智能体体系中的可执行、可验证、可共享的基础载体，并系统梳理其接口、机制与多智能体扩展。
aliases:
  - Code as Agent Harness
  - "Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems"
  - 代码作为智能体 harness
  - 代码作为 Agent Harness
prerequisites:
  - "[[Agent Harness for Large Language Model Agents (2026)]]"
  - "[[Agent Harness Engineering (2026)]]"
  - "[[Externalization in LLM Agents (2026)]]"
  - "[[SWE-Bench]]"
tags:
  - evaluation
  - optimization
  - scaling
  - inference
created: 2026-05-31
updated: 2026-05-31T20:26
---

# Code as Agent Harness (2026)

Code as Agent Harness 指一种以代码为中心理解智能体系统的视角：代码从模型的生成结果扩展为智能体推理、行动、建模环境、接收反馈和协调多智能体的运行介质。它把代码智能体、GUI/OS 智能体、具身智能体、科学发现智能体和企业自动化智能体中反复出现的模式放进同一张图里：智能体通过可执行、可检查、可持久化的工件完成长程闭环。

这篇综述适合作为 [[Agent Harness for Large Language Model Agents (2026)|智能体 harness]] 方向的代码中心版本来读。[[Agent Harness Engineering (2026)]] 讨论模型外部 harness 的系统工程层次，[[Externalization in LLM Agents (2026)]] 讨论智能体能力如何外化为记忆、技能、协议和 harness；本文进一步追问：当这些外化对象以代码、测试、仓库、日志、执行环境、工具结构定义或可运行工作流的形式存在时，智能体系统会获得什么新的可靠性结构？

> [!paper] 论文出处
> Ning et al., 2026, *Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems*。本笔记吸收自 [[Clippings/Paper/2605.18747/2605.18747|Code as Agent Harness]]，配套论文列表见 [Awesome-Code-as-Agent-Harness-Papers](https://github.com/YennNing/Awesome-Code-as-Agent-Harness-Papers)。

## 为什么需要这个视角

传统代码生成任务主要关心“模型能否写出正确程序”。智能体系统里的问题更大：模型需要连续地读环境、改文件、跑命令、查日志、更新计划、处理失败、保留状态，并在必要时把工作交给其他智能体或人类。此时，代码承担的角色已经覆盖整个闭环。

> [!intuition] 直觉理解
> 如果把 LLM 看成一个会提出意图的决策器，harness 就是把意图接到现实环境的运行层。Code as Agent Harness 的核心洞察是：代码天然适合成为这层运行介质，因为代码能被执行、被检查、被版本化、被共享，也能把失败暴露成可诊断的信号。

这个视角把智能体的可靠性问题从“下一次生成是否正确”推进到“整个执行系统是否能持续保持可验证状态”。一个代码智能体在 [[SWE-Bench]] 上修 issue 时，真正参与决策的对象包括仓库结构、issue 描述、测试、补丁、命令输出、失败日志、上下文摘要、权限边界和回滚点。模型给出动作，代码化的 harness 让这些动作进入可观察、可约束、可复现的轨道。

## 什么算 Code

本文里的代码是广义的可执行或机器可检查工件，包括程序、脚本、形式化规格、证明脚本、API 结构定义、工具定义、测试、仓库、模拟器、配置文件，以及由可执行系统产生或消费的轨迹和日志。

这个边界很重要。原始视觉输入、物理世界状态、人类意图、模型内部隐状态属于代码之外的对象；harness 可以把它们序列化为文件、状态图、工具调用、传感器读数、测试结果或可验证契约，从而让智能体在闭环里使用它们。

## 为什么代码适合作为 Harness 介质

代码在智能体 harness 中同时提供五种能力。

1. **推理基底**：模型可以把中间计算、逻辑步骤、证明尝试或搜索过程写成程序，再交给解释器、符号求解器、证明器或测试器检查。Program-of-Thoughts、Lean 证明、执行轨迹反馈都属于这一类。
2. **行动接口**：模型可以把意图转成工具调用、GUI 命令、机器人控制程序、工作流、行为树或临时脚本。此时代码是从语言意图到环境动作的桥。
3. **环境表示**：仓库、测试、模拟器、配置、日志和运行时状态能表示世界的结构、动态和反馈。智能体通过读写这些对象来理解环境变化。
4. **反馈与验证表面**：编译器、linter、类型检查、单元测试、fuzzer、安全扫描、CI 和人工 review 把行为结果变成证据。harness 可以据此继续、修复、回滚或升级审批。
5. **共享协作工件**：多智能体系统可以围绕同一个仓库、黑板、测试集、diff、轨迹或执行状态协作。代码让规划器、编码器、测试器、审阅器和安全检查智能体拥有共同工作面。

## 三层框架

论文把 Code as Agent Harness 组织成三层。理解这三层时，可以把它们看成从“单次动作如何接入环境”逐步走向“长程系统如何保持一致”的认知路径。

| 层次 | 问题 | 代码承担的角色 |
|---|---|---|
| Harness 接口 | 模型怎样接触任务环境？ | 推理基底、行动接口、环境表示 |
| Harness 机制 | 单个智能体怎样维持长程可靠性？ | 规划、记忆、工具、PEV 控制、harness 优化 |
| Harness 扩展 | 多个智能体怎样围绕同一任务协作？ | 共享代码状态、角色分工、执行反馈、收敛判据 |

### 第一层：Harness 接口

Harness 接口关心智能体输出如何变成可执行、可检查、可持久化的结构。

**代码用于推理**：把内部推理外化成程序、证明脚本或执行轨迹。模型负责提出过程，harness 负责运行、检查、存储中间状态，并把结果反馈给下一轮推理。这样一来，算术、符号推理、定理证明和复杂搜索可以由外部执行系统承担低层计算，模型把注意力放在问题分解与策略选择上。

**代码用于行动**：把高层意图转成环境中的可执行动作。具身智能体可以生成机器人控制策略，GUI 智能体可以调用 DOM、可访问性树或坐标动作，软件智能体可以编辑文件、运行测试、调用 API。行动代码的系统价值在于连接感知、可行动性（affordance）、控制器、安全层和验证器。

**代码用于环境建模**：把环境状态写成智能体可读写的对象。对软件工程来说，仓库、依赖图、测试输出、栈轨迹和补丁就是环境；对科学发现来说，Jupyter kernel、仿真器、实验脚本和数据分析流水线就是环境；对 GUI/OS 智能体来说，DOM、截图、可访问性树、UI 模拟器和评估器共同构成一个程序化世界。

### 第二层：Harness 机制

当代码进入智能体循环，系统就需要控制长程状态。论文把机制层拆成规划、记忆、工具使用、Plan-Execute-Verify loop 和 agentic harness engineering。

**Planning** 在这里是 harness 控制的一部分。一个好的计划会明确要读哪些文件、改哪些位置、保持哪些 invariant、运行哪些验证命令、哪些动作需要回滚点，以及哪些步骤有风险。计划因此成为可检查的 contract，后续执行和验证都能引用这个外化工件。

**记忆与上下文工程** 管理任务状态在不同存储层之间的流动。工作记忆保留下一步决策所需线索；语义记忆从仓库和文档中检索证据；经验记忆保存可复用的失败和修复轨迹；长期记忆保留经过验证的项目知识；多智能体记忆同步不同角色的状态。上下文压缩和状态卸载则把完整日志、轨迹、diff、测试报告等高容量证据放到外部存储，只把摘要和可追溯 handle 放进上下文。

**工具使用** 是模型意图和外部系统之间的治理接口。可靠 harness 需要决定工具结构定义如何暴露、哪些工具可用、执行发生在哪里、结果怎样清洗和压缩、权限如何分层，以及高风险动作何时进入人工审批。

> [!definition] PEV Loop
> [[Plan-Execute-Verify Loop|Plan-Execute-Verify（PEV）loop]] 是代码智能体的核心控制循环：Plan 把用户目标外化为状态变更契约；Execute 在沙箱和权限边界内执行变更；Verify 用确定性传感器和人工 gate 判断新状态是否可接受。失败信号会回到计划、检索、编辑、测试或升级审批。

PEV 的价值在于把调试、测试、反思和人工 review 放进同一个状态转移机制。编译错误、运行时异常、覆盖率缺口、安全告警、资源超限、CI 结果和人工意见都成为 harness 调节下一步动作的证据。

**Agentic Harness Engineering** 把 harness 本身也变成可测量、可修改的系统对象。深度遥测会记录提示词、检索内容、工具参数、权限请求、编辑文件、沙箱快照、命令输出、测试结果、失败路径、人工干预和成本。Evolution Agent 可以基于这些轨迹提出 harness 改动，例如调整工具描述、加入验证器、改变上下文打包、修改重试策略或新增人工 gate。由于 harness 改动会改变未来智能体行为，它需要回放评估、回归测试、安全不变量和可审计的发布流程。

### 第三层：Scaling Harness

多智能体代码系统缓解三类单智能体限制：上下文窗口装不下完整仓库和轨迹，一个通用智能体同时承担计划、编码、测试、review 的效率较低，自我验证缺少独立通道。ChatDev、MetaGPT、AgentCoder 等系统把架构师、程序员、测试员、审阅员、执行器等角色分开，让多个智能体围绕共享代码工件协作。

真正困难的部分是共享状态。论文提出 [[Shared Code-Centric Harness Substrate|代码中心共享 harness 基底]]：多个智能体共同居住、查询、更新和验证的代码中心状态层。现有系统大致有四种表示方式。

| 共享状态表示 | 含义 | 主要风险 |
|---|---|---|
| 隐式 / 仅文件 | 共享状态由当前文件和对话历史临时重建 | 智能体信念容易偏离真实程序状态 |
| repository-based | 仓库结构、调用关系、依赖图和版本历史可导航 | 静态结构难以覆盖运行时行为 |
| 执行结果驱动 | 用编译、测试、fuzzing、性能和仿真结果表示状态 | oracle 覆盖不足会制造虚假收敛 |
| 黑板 / 共享状态 | 所有角色读写持久化共享数据结构 | 权威性、冲突合并和访问粒度更复杂 |

共享 harness 的目标是让多智能体协作拥有可追踪的收敛标准。测试通过、安全扫描通过、性能指标达标、仿真 mismatch 降低、review consensus 达成，都可以作为不同形式的 convergence signal。论文特别强调，仓库结构视角和执行行为视角需要结合：前者回答“哪些组件相关”，后者回答“程序运行时发生了什么”。

## 应用图谱

这篇综述覆盖的应用领域很宽，但可以用同一个问题串起来：某个领域的“世界状态”能否被写成可执行、可检查、可复用的工件？

在代码助手中，世界状态就是仓库、issue、依赖、测试和补丁。[[SWE-Bench]] 的影响力来自它把真实 GitHub issue 变成可执行验证任务，使智能体的工作从补全文本升级为仓库级修复流程。

在 GUI/OS 智能体中，世界状态包含截图、DOM、可访问性树、浏览器或桌面动作、UI 模拟器和评估器。智能体的关键能力是把用户目标映射到可执行界面动作，并用渲染状态或程序化检查器确认结果。

在具身智能体中，代码是语义计划和物理控制之间的边界。技能库、控制原语、运动规划器、传感器状态和安全约束共同构成可执行动作空间。可复用技能同时是行动接口和长期记忆。

在科学发现中，假设、实验设计、仿真、数据分析和论文撰写可以被组织成程序图。Jupyter notebook、实验协议、化学或生物仿真器、Lean 证明器、LaTeX 报告都能成为智能体循环中的状态和证据。

在个性化系统中，偏好状态具有噪声、隐含性和长期漂移。结构化偏好记忆、用户可编辑记录、反馈日志和治理策略可以把 personalization 从一次性推荐扩展为持续更新的 harness。

## 边界与开放问题

> [!warning] 可执行反馈的边界
> 可执行性提供强证据，也会带来“绿色测试即正确”的错觉。测试、静态分析、GUI 评估器、仿真器和人工 review 都只覆盖一部分语义。可靠 harness 需要记录每个验证器的范围、假设、盲区和剩余风险。

论文提出的开放问题可以浓缩成六个设计挑战。

1. **Harness 级评估**：评估对象需要覆盖完整运行层，包括轨迹效率、验证强度、恢复能力、状态一致性、安全合规和可重放性。
2. **[[Oracle Adequacy|Oracle 充分性]]**：执行反馈的可信度取决于评估器是否覆盖真实任务语义。单元测试、GUI 检查器、科学脚本和机器人模拟器都可能漏掉关键失败。
3. **语义验证**：未来 harness 需要组合单元测试、集成测试、property-based testing、fuzzing、形式化规格、覆盖率、模型 critique 和人工 review，并让每个证据声明自己的适用范围。
4. **无回归的自演化 harness**：自动改 harness 要像修改安全关键运行时一样管理。每个变更都应包含目标失败模式、预期收益、必须保持的不变量、可证伪评估和回滚方式。
5. **[[Transactional Shared Program State|事务化共享程序状态]]**：多智能体同时读写仓库、计划、测试、记忆和权限时，需要读集合、写集合、假设、版本依赖、验证义务和冲突策略。
6. **[[Multimodal Code-Harness System|多模态代码 harness 系统]]**：GUI、具身和科学场景需要把图像、UI 元素、物体位姿、传感器读数、图表和显微图像纳入持久、可查询、可验证的状态层。

## 和现有 Harness 笔记的关系

读 [[Agent Harness for Large Language Model Agents (2026)]] 可以先建立智能体 harness 的通用组件视角：模型、工具、执行环境、状态、控制流和验证层如何组成一个智能体。读 [[Agent Harness Engineering (2026)]] 可以看到更工程化的 ETCLOVG 分类：执行、工具、上下文、生命周期、可观测性、验证和治理。

Code as Agent Harness 在这两条线之间补上一层更具体的媒介分析：当 harness 的关键状态以代码和执行工件存在时，推理、行动、记忆、验证、多智能体协作会被重新组织。它也和 [[Externalization in LLM Agents (2026)]] 的外化视角自然相连：代码是外化能力中最容易执行、审计、复用和版本化的一类载体。

## 面试视角

> [!interview] Q: Code as Agent Harness 的一句话解释是什么？
> A: 它把代码视为智能体系统的运行介质：智能体通过代码化工件进行推理、行动、环境建模、反馈验证和多智能体协作，从而把长程任务放进可执行、可检查、可持久化的闭环。

> [!interview] Q: 这个视角和普通代码生成有什么区别？
> A: 普通代码生成关注最终程序是否正确；Code as Agent Harness 关注完整执行系统如何保持可靠，包括计划、工具、沙箱、记忆、测试、日志、权限、回滚、多智能体共享状态和人工 gate。

> [!interview] Q: 为什么仅有执行反馈还不够？
> A: 执行反馈只和 oracle 一样可靠。测试覆盖不足、静态分析过近似、GUI 评估器漏掉危险中间动作、仿真器偏离物理现实时，智能体可能优化到错误信号。成熟 harness 需要多层验证器，并显式记录每个证据的覆盖范围和不确定性。

> [!interview] Q: 多智能体场景里最大的新增问题是什么？
> A: 共享程序状态的一致性。多个角色会读写同一仓库、计划、测试、记忆和权限策略，系统需要显式管理版本、假设、冲突、验证义务和收敛标准。

## 速查

- 核心概念：代码作为智能体的可执行、可检查、可持久化 harness 介质。
- 三层框架：Harness Interface → Harness Mechanisms → Scaling Harness。
- 五种代码角色：推理基底、行动接口、环境表示、反馈验证表面、共享协作工件。
- 核心控制环：Plan-Execute-Verify。
- 关键系统问题：oracle adequacy、语义验证、harness 自演化回归、多智能体共享状态、HITL 安全、多模态状态管理。
