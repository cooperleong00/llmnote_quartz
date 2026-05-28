---
type: concept
description: 评估语言模型从复杂上下文中学习新知识并应用于推理的能力，揭示当前模型在 Context Learning 上的严重不足
aliases:
  - CL-Bench
  - Context Learning Benchmark
prerequisites:
  - "[[In-Context Learning]]"
  - "[[Long-Context Modeling]]"
tags:
  - benchmark
  - evaluation
  - long-context
  - reasoning
created: 2026-02-04
updated: 2026-02-04T18:20
---

# CL-Bench

CL-Bench（Context Learning Benchmark）是评估语言模型 **Context Learning** 能力的基准测试。与传统的 [[In-Context Learning]]（ICL）不同，Context Learning 强调模型从复杂上下文中**获取新知识**并**应用于推理**的能力，而非仅仅学习任务格式或浅层启发式。CL-Bench 的核心发现是：即使是最强的前沿模型（GPT-5.1）也只能解决 23.7% 的任务，揭示了当前模型在这一关键能力上的严重不足。

## 动机

> [!intuition] 为什么需要 CL-Bench？
> 当前语言模型的评估存在一个根本性的 mismatch：
> - **模型被优化的方向**：利用预训练知识对 prompt 进行推理
> - **真实世界的需求**：从任务特定的上下文中学习新知识，并应用于解决问题
>
> 现有的 long-context benchmark 主要测试**检索**和**阅读理解**能力，而 ICL benchmark 主要测试从少量示例中学习**任务格式**的能力。两者都没有真正评估模型能否从复杂上下文中**学习并应用新知识**。

### Context Learning vs In-Context Learning

> [!comparison] 关键区别
> | 维度 | In-Context Learning | Context Learning |
> |------|---------------------|------------------|
> | **学习内容** | 任务格式、输出模式 | 新知识、新规则、新程序 |
> | **上下文复杂度** | 几个示例（few-shot） | 复杂文档（平均 10K tokens） |
> | **知识来源** | 主要依赖预训练知识 | 必须从上下文获取新知识 |
> | **推理类型** | 模式匹配、格式遵循 | 知识获取 + 应用推理 |

## 基准规模

CL-Bench 由领域专家精心构建，包含：

| 统计项 | 数值 |
|--------|------|
| 复杂上下文 | 500 个 |
| 任务 | 1,899 个 |
| 验证 rubrics | 31,607 个 |
| 平均输入长度 | 10.4K tokens |
| 最大输入长度 | 65K tokens |
| 平均每上下文任务数 | 3.8 |
| 多轮依赖任务比例 | 51.1% |

## 四大类别

CL-Bench 将上下文分为四个类别，反映人类在真实世界中学习和应用知识的不同方式：

### Category 1: Domain Knowledge Reasoning（190 contexts）

上下文提供专业领域知识（如虚构的法律体系、新创建的金融工具、小众专业知识），模型需要学习并应用这些知识来解决任务。

**子类别**：金融、医疗、人文、法律咨询、生活方式、管理、科学

**示例任务**：理解虚构国家的完整法律体系，包括判例和法律原则，并应用于案件裁决

### Category 2: Rule System Application（140 contexts）

上下文提供具有明确规则的形式系统（如新游戏机制、数学形式主义、编程语言语法），模型需要理解并正确应用这些规则。

**子类别**：游戏机制、数学形式主义、编程语法、法规、技术标准

**示例任务**：学习新的编程语言语法，解决代码相关任务

### Category 3: Procedural Task Execution（100 contexts）

上下文提供复杂的程序、工作流或操作指南（如产品手册、软件文档），模型需要学习并正确执行这些程序。

**子类别**：指导性程序、操作性程序、工作流编排

**示例任务**：理解无人机物流系统的 API 文档，将自然语言指令转换为伪代码

### Category 4: Empirical Discovery & Simulation（70 contexts）

上下文提供实验数据、观察记录或模拟环境，模型需要从数据中**归纳**出规律或在模拟环境中进行推理。这是最具挑战性的类别，因为它需要**归纳推理**而非前三类的**演绎推理**。

**子类别**：实验数据、观察数据、模拟环境

**示例任务**：分析电子在磁场中螺旋运动的实验数据，发现并应用物理规律

> [!warning] 归纳 vs 演绎
> 前三类主要是**演绎推理**：给定规则/知识，应用于具体情况。
> 第四类是**归纳推理**：从数据中发现规律，这对当前模型更具挑战性。

## 评估结果

> [!intuition] 核心发现
> 10 个前沿模型的平均任务解决率仅为 **17.2%**，最好的 GPT-5.1 也只有 **23.7%**。这表明 Context Learning 是当前模型的关键瓶颈。

### 模型表现

| 模型 | 整体解决率 |
|------|-----------|
| GPT-5.1 (High) | 23.7% |
| Claude-Opus-4.5 Thinking | ~18% |
| GPT-5.2 (High) | ~18% |
| o3 (High) | ~17% |
| Kimi-K2 Thinking | 17.6% |
| HY-2.0 Thinking | 17.2% |
| **平均** | **17.2%** |

### 类别难度差异

| 类别 | 平均解决率 | 特点 |
|------|-----------|------|
| Domain Knowledge Reasoning | ~20% | 相对最容易 |
| Rule System Application | ~17% | 法规类较易，数学形式主义最难 |
| Procedural Task Execution | ~17% | 工作流编排较易 |
| Empirical Discovery & Simulation | **11.8%** | 最难，需要归纳推理 |

### 错误类型分析

| 错误类型 | 比例 | 说明 |
|----------|------|------|
| Context Ignored | 55-66% | 忽略上下文中的知识 |
| Context Misused | 60-66% | 错误应用上下文知识 |
| Format Error | 33-46% | 输出格式不符合要求 |
| Refusal | 0.3-3.3% | 拒绝回答 |

> [!warning] 关键洞察
> 大多数失败源于模型**忽略**或**错误应用**上下文中的知识，而非无法处理长上下文本身。这表明 long-context 能力和 instruction-following 能力是 Context Learning 的**必要但不充分**条件。

## 设计特点

CL-Bench 的设计确保了评估的有效性：

1. **真实且高质量**：每个上下文和任务由领域专家精心构建，经过多轮质量审核
2. **无污染**：上下文包含预训练中不存在的新知识（虚构创作、已有知识修改、小众新兴知识）
3. **具有挑战性**：平均每个上下文需要 20 小时专家工作量，51.1% 的任务有多轮依赖
4. **严格可验证**：平均每个上下文有 63.2 个 rubrics，从多个维度评估正确性和完整性

## 局限性

> [!warning] 边界条件

**评估层面**：
- 当前仅评估了 10 个前沿模型，未覆盖开源模型的完整生态
- 高推理努力设置可能不反映实际部署场景

**构建层面**：
- 专家构建成本高昂，难以大规模扩展
- 部分新知识可能与预训练知识冲突，增加了任务难度

**泛化层面**：
- 四个类别的划分可能不完全覆盖所有真实世界场景
- 英文为主，多语言覆盖有限

## 对未来研究的启示

CL-Bench 揭示了几个重要的研究方向：

1. **Context Learning 作为独立能力**：不能简单等同于 long-context 或 ICL，需要专门的训练和评估
2. **归纳推理的瓶颈**：从数据中发现规律的能力远弱于应用已知规则
3. **知识获取 vs 知识应用**：模型需要同时具备从上下文获取新知识和正确应用的能力

> [!interview] 面试视角
> **Q: CL-Bench 和传统 long-context benchmark 有什么区别？**
> A: 传统 long-context benchmark（如 Needle-in-a-Haystack）主要测试检索和阅读理解能力，模型只需找到并复述信息。CL-Bench 要求模型从上下文中**学习新知识**并**应用于推理**，这是更高层次的能力。
>
> **Q: 为什么 Empirical Discovery 类别最难？**
> A: 因为它需要**归纳推理**——从数据中发现规律，而其他三类主要是**演绎推理**——应用已知规则。当前模型在归纳推理上的能力明显不足。
>
> **Q: Context Learning 和 In-Context Learning 的核心区别是什么？**
> A: ICL 主要学习任务格式和浅层启发式，依赖预训练知识；Context Learning 强调从复杂上下文中获取**新知识**并应用，这些知识在预训练中不存在。

> [!paper] 原始论文
> Dou et al., "CL-Bench: A Benchmark for Context Learning", arXiv:2602.03587, 2026
> - Hunyuan Team (Tencent) + Fudan University
> - 网站：[clbench.com](https://clbench.com)
> - 提出 Context Learning 概念，区别于 In-Context Learning
> - 构建 500 上下文、1,899 任务、31,607 rubrics 的评估基准
> - 发现前沿模型平均只能解决 17.2% 的任务

## 相关概念

- [[In-Context Learning]] — Context Learning 的前身，主要学习任务格式
- [[Long-Context Modeling]] — Context Learning 的必要但不充分条件
- [[SWE-Bench]] — 另一个评估复杂推理能力的基准，聚焦软件工程
