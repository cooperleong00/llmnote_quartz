---
type: concept
description: 从真实 GitHub PR 构建的软件工程基准，通过 issue + patch + test 三元组实现可执行验证，是评估 coding agent 的核心标准
aliases:
  - SWE-bench
  - Software Engineering Benchmark
prerequisites:
  - "[[Agentic RL]]"
tags:
  - benchmark
  - code
  - agent
created: 2026-02-04
updated: 2026-02-04T00:15
---

# SWE-Bench

SWE-Bench（Software Engineering Benchmark）是评估 LLM 在真实软件工程任务上能力的核心基准。其核心创新在于：从 GitHub 的 Pull Request 中提取 **issue（问题描述）+ patch（参考解）+ test（验证器）** 三元组，构建可执行验证的评估环境。这使得模型生成的代码可以通过运行真实测试来客观评判，而非依赖人工评估。

## 动机

> [!intuition] 为什么需要 SWE-Bench？
> 传统代码基准（如 HumanEval）的问题是：
> 1. **任务过于简单**：自包含的函数级问题，几行代码就能解决
> 2. **与真实开发脱节**：真实软件工程需要理解大型代码库、跨文件修改、处理复杂依赖
> 3. **评估不够鲁棒**：简单的输入输出匹配无法捕捉代码质量的多个维度
>
> SWE-Bench 的洞察：GitHub 的 PR 流程天然包含了 **问题定义 + 解决方案 + 验证机制**，可以直接转化为可验证的评估任务。

## 核心设计：PR 到 Verifiable Environment

SWE-Bench 将每个 GitHub PR 转化为一个 "gym"：

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Pull Request                       │
├─────────────────────────────────────────────────────────────┤
│  Issue Description  →  问题描述（自然语言）                    │
│  Code Changes       →  参考解（ground truth patch）           │
│  Test Changes       →  验证器（fail-to-pass tests）           │
│  Base Commit        →  代码库快照（评估环境）                   │
└─────────────────────────────────────────────────────────────┘
```

### 构建流程（三阶段过滤）

**Stage I: Repo Selection**
- 从 12 个流行的 Python 开源仓库收集约 90,000 个 PR
- 选择维护良好、有清晰贡献指南、测试覆盖率高的仓库

**Stage II: Attribute-based Filtering**
- 筛选已合并的 PR
- 要求 PR 关联一个 GitHub issue
- 要求 PR 修改了测试文件（表明贡献者添加了验证测试）

**Stage III: Execution-based Filtering**
- 运行测试，记录 PR 应用前后的测试结果
- 保留至少有一个 **fail-to-pass** 测试的实例（测试状态从失败变为通过）
- 过滤掉安装或运行时错误的实例

最终产出约 **2,294 个任务实例**。

### 任务形式

给定：
- Issue 文本描述
- 代码库快照（base commit）

要求模型生成：
- 一个 patch（代码修改）

评估方式：
- 应用 patch 后运行测试
- 检查 fail-to-pass 测试是否通过
- 同时运行其他测试确保不破坏已有功能

## 为什么这个设计有效

> [!intuition] 核心洞察
> SWE-Bench 的设计巧妙之处在于：**利用开源社区已有的质量控制机制**。
>
> - Issue 是真实用户提出的问题，不是人工构造的
> - Patch 是经过 code review 并被合并的解决方案
> - Test 是开发者为验证修复而编写的，具有明确的语义

**优势**：

1. **真实性**：任务来自真实的软件开发场景，不是人工构造的玩具问题
2. **可验证性**：通过执行测试客观评判，无需人工评估
3. **多样性**：覆盖 bug 修复、新功能实现、性能优化等多种任务类型
4. **可扩展性**：可以持续从新的 PR 中提取任务，保持基准的新鲜度
5. **解决方案自由度**：模型可以生成与参考解不同的有效方案

## 主要变体

| 变体 | 规模 | 特点 | 用途 |
|------|------|------|------|
| **SWE-Bench** | 2,294 | 原始完整集，仅 Python | 全面评估 |
| **SWE-Bench Lite** | 300 | 精选子集，难度适中 | 快速迭代 |
| **SWE-Bench Verified** | 500 | 人工验证，确保质量 | 可靠评估 |
| **SWE-Bench Multilingual** | 多语言 | 扩展到 Java、JS、Go 等 | 跨语言能力 |

### SWE-Bench Verified

原始 SWE-Bench 存在一些噪声（如测试不够严格、issue 描述模糊）。SWE-Bench Verified 通过人工审核筛选出约 500 个高质量实例，确保：
- Issue 描述清晰完整
- 测试能够准确验证修复
- 不存在 "hacked" 解决方案（如通过字符串匹配绕过测试）

**当前 SOTA**：Qwen3-Max-Thinking 达到 75.3%（2026 年初）

## 在 Agentic RL 中的作用

SWE-Bench 及其衍生数据集在 [[Agentic RL]] 训练中扮演关键角色：

### 作为 Mid-training 数据源

大规模 SWE 环境（如 [[SWE-Universe]] 的 80 万+ 实例）可用于 mid-training：
- 在 pre-training 和 post-training 之间注入软件工程能力
- 通过 rejection sampling 生成高质量 agentic trajectories
- 帮助模型学习代码库导航、多文件编辑、测试驱动开发

### 作为 RL Reward Signal

SWE 环境的 **pass/fail 测试结果** 天然提供二元奖励信号：
- 无需训练 reward model
- 奖励信号客观、可靠、无偏
- 支持大规模 on-policy RL 训练

```
Reward = 1  if all fail-to-pass tests pass
         0  otherwise
```

> [!example] 训练效果示例
> SWE-Universe 论文显示：
> - Qwen3-30B-A3B 通过 agentic RL 在 SWE-Bench Multilingual 上从 32% 提升到 42%
> - 10 个百分点的绝对提升证明了 RL 在跨语言任务上的泛化能力

## 局限性

> [!warning] 边界条件

**数据层面**：
- **语言偏向**：原始版本仅支持 Python，多语言扩展仍在进行中
- **仓库偏向**：集中在少数流行仓库，可能不代表长尾场景
- **时间偏向**：基于历史 PR，可能与当前代码库状态不一致

**验证层面**：
- **弱验证器**：部分测试可能被 "hacked"（如通过字符串匹配绕过）
- **不完整验证**：通过测试不等于代码质量高（可读性、效率、安全性）
- **测试覆盖不足**：部分实例的 fail-to-pass 测试数量有限

**构建层面**：
- **低产出率**：从原始 PR 到可用实例的转化率约 10-15%
- **环境复杂性**：真实仓库的依赖、配置、构建工具链高度异构
- **成本高昂**：大规模环境构建需要大量计算资源

## 生态系统

SWE-Bench 催生了一系列相关工作：

**数据扩展**：
- [[SWE-Universe]] — 百万级多语言环境，自动化构建
- SWE-Gym — 训练专用环境
- SWE-rebench — 2.1 万+ Python 实例

**评估改进**：
- SWE-Bench++ — 更严格的验证
- SWE-bench-Live — 持续更新的实时基准

**多语言扩展**：
- Multi-SWE-bench — 多语言支持
- SWE-PolyBench — 跨语言基准

> [!interview] 面试视角
> **Q: SWE-Bench 和 HumanEval 有什么区别？**
> A: HumanEval 是函数级的自包含问题，几行代码就能解决；SWE-Bench 是仓库级的真实问题，需要理解大型代码库、定位问题、跨文件修改。SWE-Bench 更接近真实软件工程场景。
>
> **Q: SWE-Bench 的验证机制是什么？**
> A: 通过运行 fail-to-pass 测试。这些测试在应用 patch 前失败、应用后通过，证明 patch 确实解决了问题。同时运行其他测试确保不破坏已有功能。
>
> **Q: 为什么 SWE-Bench 对 Agentic RL 很重要？**
> A: 它提供了可执行验证的环境，测试结果可以直接作为 RL 的 reward signal，无需训练 reward model。这使得大规模 agentic RL 训练成为可能。

## 相关概念

- [[Agentic RL]] — SWE-Bench 是 code agent 的核心评估基准
- [[SWE-Universe]] — 百万级 SWE 环境生成框架
- [[Verifiable Reward]] — SWE-Bench 的测试结果是典型的 verifiable reward

> [!paper] 原始论文
> Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", ICLR 2024
> - 提出从 GitHub PR 构建可验证评估环境的方法
> - 建立了 2,294 个 Python 任务实例
> - 发现当时最强模型（Claude 2）仅能解决 1.96% 的问题
