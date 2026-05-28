---
type: paper
description: 美团 560B MoE 推理模型，通过 DORA 异步 RL 框架实现 3x 训练加速，采用 Domain-Parallel RL 分域训练后融合，在 agentic reasoning 上实现 64.5% token 消耗降低
aliases:
  - LongCat-Flash-Thinking
  - LongCat-Flash-Thinking-2601
prerequisites:
  - "[[LongCat-Flash (2025)]]"
  - "[[GRPO]]"
  - "[[Mixture of Experts]]"
tags:
  - reasoning-model
  - rl-infrastructure
  - agentic
  - post-training
created: 2025-02-04
updated: 2025-02-04
---

# LongCat-Flash-Thinking (2025)

LongCat-Flash-Thinking 是美团 LongCat 团队基于 [[LongCat-Flash (2025)]] 构建的 560B 参数 [[Mixture of Experts|MoE]] 推理模型（27B 平均激活参数）。核心贡献在于 **DORA 异步 RL 框架**（3x 训练加速）和 **Domain-Parallel RL Training**（分域训练后融合），在复杂推理任务上达到开源模型 SOTA，并在 agentic reasoning 上实现 64.5% 的 token 消耗降低。

> [!paper] 论文信息
> - **标题**: Introducing LongCat-Flash-Thinking: A Technical Report
> - **机构**: Meituan LongCat Team
> - **时间**: 2025
> - **链接**: [HuggingFace](https://huggingface.co/meituan-longcat/LongCat-Flash-Thinking) | [GitHub](https://github.com/meituan-longcat/LongCat-Flash-Thinking)
> - **2601 更新**: [LongCat-Flash-Thinking-2601](https://huggingface.co/meituan-longcat/LongCat-Flash-Thinking-2601)

## 动机

大规模 RL 训练面临两个核心挑战：

1. **RL 调度效率**：disaggregated 架构导致设备空闲（阶段间依赖），colocated 架构则因异构负载（生成是 memory-bound，训练是 compute-bound）导致次优性能
2. **Skewed Generation 问题**：同步训练中整个 batch 被最长输出阻塞，在长上下文推理场景尤为严重

此外，传统 mixed-domain RL 训练在异步场景下常出现**负迁移**，因为不同领域的响应特征差异巨大（如 STEM vs Code vs Agentic 的长度分布完全不同）。

## 训练流程

```
LongCat-Flash-Base
       │
       ▼
┌──────────────────────────────────────┐
│  Phase 1: Long CoT Cold-Start        │
│  ├─ Mid-training (reasoning data)    │
│  └─ Reasoning-oriented SFT           │
│      ├─ General Reasoning            │
│      ├─ Formal Reasoning (Lean4)     │
│      └─ Agentic Reasoning            │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Phase 2: Large-Scale RL (DORA)      │
│  ├─ Domain-Parallel Training         │
│  │   ├─ STEM Expert                  │
│  │   ├─ Code Expert                  │
│  │   └─ Agentic Expert               │
│  ├─ Expert Fusion                    │
│  └─ General RL Alignment             │
└──────────────────────────────────────┘
```

## DORA: Dynamic ORchestration for Asynchronous Rollout

DORA 是 LongCat-Flash-Thinking 的核心基础设施创新，通过**多版本 Actor 流式生成**和**弹性角色调度**实现 3x 训练加速。

> [!intuition] 核心思想
> 传统异步训练（如 partial rollout）将长响应切分为多段，用最新 Actor 生成每段，但这导致：
> 1. 中断样本需要重新 prefill（开销大）
> 2. 同一响应的不同段由不同策略版本生成（理论上影响收敛）
>
> DORA 的解决方案：**保留多个旧版本 Actor**，让每个响应由同一版本完整生成，完成后立即流式传递到下一阶段。

### 架构设计

DORA 采用 disaggregated 架构，将加速器集群分为两组：

| 组别 | 角色 | 特点 |
|------|------|------|
| **Standalone Generator Group** | 专用于 Generator | 优化 rollout 吞吐 |
| **Elastic Role Group** | 动态切换角色 | Generator / Reference & Actor / Reward & Critic |

### 工作流程

**Generation Phase**:
- 两组都激活推理引擎进行 rollout
- 推理实例维护最多 N 个策略版本（staleness 控制）
- Load Balancing Controller 在版本间重新分配资源，复用 KV-cache
- 完成的样本**立即流式传递**，不阻塞后续阶段

**Experience-Maker Phase**:
- Elastic Group 缩减 Generator 角色，激活其他 RL 角色
- Reference & Actor 和 Reward & Critic 并行执行推理
- Standalone Group 临时切换到训练引擎重算 log probabilities（消除推理/训练引擎数值差异）

**Model Training Phase**:
- Actor 和 Critic 在收集的经验上训练
- Standalone Group 继续生成（不阻塞），同时重新平衡负载
- 训练完成后，最新权重通过 layer-wise P2P 通信同步回 Generator

> [!comparison] DORA vs 传统方法
> | 特性 | 同步训练 | Partial Rollout | DORA |
> |------|----------|-----------------|------|
> | 长尾阻塞 | 严重 | 缓解 | 消除 |
> | 策略一致性 | 完美 | 不一致 | 完美 |
> | KV-cache 复用 | N/A | 需重新 prefill | 高效复用 |
> | 设备利用率 | 有 bubble | 有 bubble | 近零 bubble |

## RL 算法：改进的 GRPO

基于 [[GRPO]] 进行异步训练适配，解决 **distribution drift** 问题：

### Distribution Drift 的两个来源

1. **Engine Numerical Gap**: 推理引擎（如 vLLM）的优化（kernel fusion）与训练引擎（如 Megatron）不保证 bitwise 一致
2. **Policy Staleness**: 异步训练中样本可能来自多个旧版本策略

### 改进措施

```math
\mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{G \cdot T_{\max}} \sum_{i=1}^{G} \sum_{t=1}^{|y_i|} \min(r_{i,t}(\mu), C) \cdot \max\left(\min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(\cdot)\hat{A}_{i,t}\right), \varepsilon_{\text{neg}_{\text{high}}}\hat{A}_{i,t} \right) \right]
```

关键改进：
- **移除 KL 散度项**：默认 $k_3$ 估计器的梯度有偏，移除后允许更大的策略更新
- **Token-level loss + 全局长度归一化**：用 $T_{\max}$ 作为分母，消除长度偏差
- **Triplet clipping**：$\varepsilon_{\text{neg}_{\text{low}}}$, $\varepsilon_{\text{neg}_{\text{high}}}$, $\varepsilon_{\text{pos}_{\text{high}}}$ 分别约束负/正 advantage 的 importance ratio
- **Truncated importance sampling**：缓解推理/训练引擎的数值差异

## Domain-Parallel RL Training

> [!intuition] 为什么不能 mixed-domain 训练？
> 不同领域的响应长度分布差异巨大（STEM 偏长、Code 中等、Agentic 变化大），在异步训练中导致 batch 间的分布剧烈变化，引发负迁移。

### 分域训练配置

| 领域 | Context Length | 特殊策略 |
|------|----------------|----------|
| **STEM** | 固定 64K | Curriculum learning（逐步降低 pass-rate 阈值）+ 动态调整 clipping bound |
| **Code** | 48K → 56K → 64K | 多阶段 curriculum，90th percentile 接近上限时扩展 |
| **Agentic** | 固定 48K | 结构化对话模板（`<longcat_think>`, `<longcat_tool_call>`）+ tool-call format reward |

### Expert Fusion

训练完成后，将三个 domain expert 融合为单一模型：

1. **Normalization**: 归一化 task vector $\tau_i = \theta_{RL}^i - \theta_{SFT}$ 的幅度
2. **Dropout**: 类似 DARE，剪枝冗余 delta 参数
3. **Erase**: 类似 SCE，擦除少数方向更新的参数元素

融合后的模型在 STEM、Code、Agentic 三个领域都接近 Pareto 最优。

## Reward System

| 任务类型 | Reward 方法 | 说明 |
|----------|-------------|------|
| **Non-Verifiable** | Discriminative RM | 基于 LongCat-Flash SFT checkpoint，在偏好数据上训练 |
| **STEM** | Generative RM (GenRM) | 带推理过程，能处理等价表达式（如 $a^2-b^2$ vs $(a+b)(a-b)$），准确率 98.8% |
| **Code** | 分布式 Sandbox | 支持 20+ 编程语言，异步接口处理大批量代码执行 |

## 2601 版本更新

LongCat-Flash-Thinking-2601 在原版基础上进一步强化 **agentic reasoning** 能力。

### Environment Scaling

> [!intuition] 核心洞察
> 高级 agentic 能力的核心是**泛化**——在已知环境中学习有效行为，并可靠迁移到未见过的设置。这需要在**足够多样的环境**中训练。

**自动化环境构建流程**：
1. 从高层领域定义出发，合成 domain-specific tool set
2. 抽象为统一的 database schema + tool-database mapping
3. 自动生成 database 实现和 tool 代码
4. 通过 unit testing + debugging agent 验证（成功率 >95%）
5. 构建 tool dependency graph

**可验证性保持的环境扩展**：
- 从 seed executable tool chain $s_1$ 出发
- BFS 式扩展：只添加依赖已满足的新 tool node
- 保持 database 一致性，避免级联依赖问题
- 每个环境至少包含 20 个 tools

最终构建了覆盖 **20+ 领域**、每个领域 **60+ tools** 的环境集合。

### Noisy Environment Training

> [!warning] 现实世界的不完美
> 训练环境通常是理想化的，但真实环境存在：
> - **Instruction noise**: 用户交互模式的模糊性和变化性
> - **Tool noise**: 执行失败、不一致响应、部分结果

**Curriculum-based Robust RL**：
- 从轻微扰动开始，逐步增加噪声难度和多样性
- 当模型在当前级别展示足够鲁棒性后才升级
- 确保注入的不完美不会使任务无法解决

| 数据集 | ColdStart | Training w/o Noise | Training w/ Noise |
|--------|-----------|--------------------|--------------------|
| VitaBench | 10.0 | 28.6 | **29.3** |
| VitaBench-Noise | 6.3 | 13.3 | **20.5** |
| Tau2Bench | 78.8 | 87.1 | **88.2** |
| Tau2Bench-Noise | 58.8 | 62.2 | **67.1** |

### Heavy Thinking Mode

Test-time scaling 的新范式：**同时扩展推理深度和宽度**。

```
┌─────────────────────────────────────────────────────┐
│  Stage 1: Parallel Reasoning                        │
│  ├─ Thinking Model 并行生成多个候选轨迹              │
│  └─ 扩展探索宽度                                     │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: Heavy Thinking                            │
│  ├─ Summary Model 对轨迹进行反思推理                 │
│  ├─ 综合中间推理和结果                               │
│  └─ 得出最终决策                                     │
└─────────────────────────────────────────────────────┘
```

- **Context Memory Module**: 存储消息历史，支持 tool use 和多轮对话
- **额外 RL 阶段**: 专门强化 summary phase 的聚合和精炼能力
- Thinking 和 Summary 模块可以共享参数或使用不同模型

### 训练策略改进

**GSPO (Group Sequence Policy Optimization)**：
- 相比 GRPO，在 MoE 模型上更有效
- 提供更稳定的 sequence-level 优化

**Dynamic Budget Allocation**：
- 根据模型实时训练状态动态分配 rollout 预算
- 难度匹配当前能力的任务获得更多资源

**Self-Verification**：
- 将模型同时作为 actor 和 verifier
- 当生成器停滞时触发验证阶段
- 作为辅助任务加速收敛

**Context Management**（针对 agentic 场景）：
- Summary-based: 超过 80K token 时压缩历史 tool call 结果
- Discard-based: 超过最大轮数时 discard-all 重启
- Hybrid: 结合两者，在 BrowseComp 上从 55.8% 提升到 73.1%

## 性能表现

### 原版 (LongCat-Flash-Thinking)

| Benchmark | LongCat-Flash-Thinking | DeepSeek-V3.1-Thinking | Qwen3-235B |
|-----------|------------------------|------------------------|------------|
| MATH-500 | **99.2** | 98.8 | 99.6 |
| AIME-24 (Mean@32) | 93.3 | 93.9 | 93.9 |
| AIME-25 (Mean@32) | 90.6 | 87.9 | **92.5** |
| HMMT-25 (Mean@32) | 83.7 | 80.4 | **83.8** |

**Agentic 效率**: AIME-25 上 token 消耗从 19,653 降至 6,965（**-64.5%**），准确率不降。

### 2601 版本

| Benchmark | LongCat-2601 | DeepSeek-V3.2 | Kimi-K2 | Claude-Opus-4.5 |
|-----------|--------------|---------------|---------|-----------------|
| AIME-25 (w/ tools) | **99.6** / 100.0 (heavy) | 93.5 | 99.1 | 100.0 |
| BrowseComp | 56.6 / **73.1** | 51.4 / 67.6 | - / 60.2 | - |
| RWSearch | **79.5** | 74.0 | 63.0 | 75.5 |
| $\tau^2$-Avg | **88.2** | 80.6 | 74.3 | 82.4 |
| VitaBench | **29.3** | 24.0 | 12.8 | 28.5 |

## 局限性

> [!warning] 边界条件
> - **General QA 相对较弱**: MMLU-Pro 82.6%（vs DeepSeek 84.4%），可能因为 RL 资源集中在推理任务
> - **依赖高质量环境**: Environment scaling 需要大量工程投入构建可验证环境
> - **Heavy Thinking 成本**: 并行推理 + summary 显著增加推理成本
> - **Noisy training 的泛化边界**: 训练时注入的噪声类型可能无法覆盖所有真实世界的不完美

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/250918883v2/250918883v2|LongCat-Flash-Thinking Technical Report]]
- [[Clippings/Paper/260116725v1/260116725v1|LongCat-Flash-Thinking-2601 Technical Report]]

**相关工作**：
- [[DeepSeek-R1]] — 另一个大规模 RL 训练的推理模型
- [[Qwen3]] — 竞争对手的推理模型系列
