---
type: paper
title: "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models"
authors:
  - GLM-4.5 Team (Zhipu AI, Tsinghua University)
year: 2025
arxiv: "2508.06471"
description: 开源 MoE 模型系列，355B 总参数 32B 激活参数，通过 Expert Model Iteration 和多阶段 RL 训练，在 Agentic、Reasoning、Coding 三大能力上达到开源 SOTA
aliases:
  - GLM-4.5
  - GLM-4.5-Air
prerequisites:
  - "[[MoE]]"
  - "[[GRPO]]"
  - "[[RLHF]]"
tags:
  - architecture
  - post-training
  - rlhf
  - reasoning
  - code
  - agentic
created: 2026-02-04
updated: 2026-02-04T00:28
---

# GLM-4.5

GLM-4.5 是智谱 AI 发布的开源 MoE 大语言模型系列，核心目标是在单一模型中统一 **Agentic**（智能体）、**Reasoning**（推理）、**Coding**（编程）三大能力。GLM-4.5 采用 355B 总参数 / 32B 激活参数的 MoE 架构，通过 Expert Model Iteration 的 post-training 策略，在 ARC 任务上达到开源模型 SOTA，整体排名第 3（仅次于 o3 和 Claude Sonnet 4）。

## 核心贡献

### 1. 架构设计：更深而非更宽

与 [[DeepSeek-V3]] 和 Kimi K2 不同，GLM-4.5 选择**减少宽度、增加深度**：

| 特点 | GLM-4.5 | DeepSeek-V3 | Kimi K2 |
|------|---------|-------------|---------|
| 总参数 | 355B | 671B | 1043B |
| 激活参数 | 32B | 37B | 32B |
| MoE 层数 | 89 | 58 | 60 |
| Hidden Dim | 5120 | 7168 | 7168 |

关键设计选择：
- **更多 attention heads**（96 heads for 5120 hidden dim，是常规的 2.5 倍）：虽然不改善 training loss，但在 MMLU、BBH 等推理 benchmark 上表现更好
- **QK-Norm**：稳定 attention logits 范围
- **Loss-free balance routing** + **Sigmoid gates**：MoE 路由策略
- **MTP 层**：支持 speculative decoding

### 2. 训练流程：Pre-training → Mid-training → Post-training

```
Pre-training (23T tokens, 4K context)
    ↓
Mid-training (扩展到 128K context)
├── Repo-level Code Training (32K)
├── Synthetic Reasoning Data
└── Long-context & Agent Training (128K)
    ↓
Post-training: Expert Model Iteration
├── Stage 1: Expert Training (Reasoning / Agent / General)
└── Stage 2: Unified Training (Self-distillation → Hybrid Model)
```

**Mid-training** 是一个独特的阶段，介于 pre-training 和 post-training 之间：
- 使用中等规模的领域特定数据（包括 instruction data）
- 逐步扩展 context length：4K → 32K → 128K
- 包含 repo-level code、synthetic reasoning、agent trajectories

### 3. Post-training：Expert Model Iteration

这是 GLM-4.5 最核心的训练创新。分为两个阶段：

**Stage 1: Expert Training** — 分别训练三个专家模型：
- **Reasoning Expert**：数学、代码、科学推理
- **Agent Expert**：工具调用、web search、coding agent
- **General Expert**：通用对话

**Stage 2: Unified Training** — 通过 self-distillation 整合专家能力：
- 从各专家模型收集百万级样本
- 训练出支持 **hybrid reasoning** 的统一模型
- 可在 thinking mode（深度推理）和 non-thinking mode（即时响应）之间切换

### 4. RL 训练策略

GLM-4.5 的 RL 训练基于 [[GRPO]]（去掉 KL loss），针对不同任务设计了专门策略：

#### Reasoning RL

- **Difficulty-based Curriculum Learning**：两阶段难度递进，避免 reward 全 0 或全 1 导致的梯度消失
- **Single-Stage RL at 64K**：直接在最大长度训练，避免多阶段长度递增导致的能力退化
- **Dynamic Sampling Temperature**：根据 reward 收敛情况动态调整，平衡 exploration 和 exploitation
- **Token-weighted Mean Loss**（for Code RL）：比 sequence-mean loss 收敛更快

#### Agentic RL

- **Outcome Supervision + Process Format Penalty**：最终答案正确性作为 reward，格式错误直接 0 分
- **Iterative Self-distillation**：RL → 生成新 SFT 数据 → 更强 SFT 模型 → 继续 RL
- **Test-time Scaling via Interaction Turns**：通过增加与环境的交互轮数提升性能

#### General RL

- **Holistic RL**：覆盖 7 大类 139 小类的平衡数据集，结合 human + AI feedback
- **Instruction Following RL**：151 种约束类型的细粒度分类，混合 rule-based + reward model + critique model
- **Function Calling RL**：step-wise rule-based + end-to-end multi-turn 两种模式
- **Pathology RL**：针对低频病态行为（语言混杂、过度重复）的定向修复

## 性能表现

| 任务类型 | Benchmark | GLM-4.5 | 对比 |
|----------|-----------|---------|------|
| Agentic | TAU-Bench | 70.1% | ≈ Claude Sonnet 4 |
| Agentic | BrowseComp | 26.4% | > Claude Opus 4 (18.8%) |
| Reasoning | AIME 24 | 91.0% | — |
| Reasoning | GPQA | 79.1% | — |
| Coding | SWE-bench Verified | 64.2% | > GPT-4.1, ≈ Claude Sonnet 4 |

GLM-4.5 在参数效率上表现突出：仅用 DeepSeek-R1 一半、Kimi K2 三分之一的参数，达到相近或更好的性能。

## 技术细节

### Function Call Template 优化

传统 JSON 格式的 function call 在包含代码时需要大量转义字符。GLM-4.5 提出 XML-like 模板：

```xml
<tool_call>{function-name}
<arg_key>{arg-key}</arg_key>
<arg_value>{arg-value}</arg_value>
</tool_call>
```

减少转义负担，降低模型学习难度。

### 训练基础设施

基于开源框架 [Slime](https://github.com/THUDM/slime) 构建，支持灵活的混合训练和数据生成架构。

## 相关资源

- 模型权重：[HuggingFace](https://huggingface.co/zai-org/GLM-4.5)
- 评估工具：[glm-simple-evals](https://github.com/zai-org/glm-simple-evals)
- 原始论文：[[Clippings/Paper/250806471v1/250806471v1|Paper Clipping]]
