---
type: paper
description: 提出完整的 Agentic Learning Ecosystem (ALE) 和 ROME 模型，包含 ROLL 训练框架、ROCK 环境引擎、iFlow CLI 和 IPA 算法
aliases:
  - Let It Flow
  - ROME
  - ALE
  - Agentic Learning Ecosystem
prerequisites:
  - "[[REINFORCE]]"
  - "[[PPO]]"
  - "[[Importance Sampling]]"
tags:
  - agentic-rl
  - post-training
  - rl-infrastructure
created: 2026-03-08
updated: 2026-05-01T12:32
---

# Let It Flow: Agentic Crafting on Rock and Roll (2025)

这篇论文提出了完整的 **Agentic Learning Ecosystem (ALE)**——一个端到端的 agent LLM 训练基础设施，以及基于此训练的 **ROME** 模型。核心贡献是将数据生成、环境执行、策略优化整合为统一的生产级系统，并提出 **IPA (Interaction-Perceptive Agentic Policy Optimization)** 算法，通过 chunk-level 优化解决 long-horizon agentic RL 的稳定性问题。

> [!paper] 论文信息
> - 标题：Let It Flow: Agentic Crafting on Rock and Roll
> - 团队：ROCK & ROLL & iFlow & DT Joint Team
> - arXiv：2512.24873
> - 发布时间：2025 年 1 月

---

## 动机：为什么需要完整的 Agentic 生态系统

**Agentic crafting 的挑战**：

传统 LLM 应用是 one-shot 生成——给定 prompt，输出 response。但真实的软件工程任务需要：
- **多轮交互**：执行 → 观察结果 → 调整策略 → 再执行
- **环境反馈**：与 terminal、代码仓库、工具 API 交互
- **长时域决策**：一个任务可能需要数十轮交互才能完成

**现有方案的不足**：

1. **SFT 方法**（如 [[InstructGPT]]）：依赖有限的人类演示数据，难以覆盖复杂的多轮交互模式
2. **Ad-hoc RL 方法**：缺乏统一的基础设施，训练不稳定，难以扩展到生产环境
3. **系统割裂**：数据生成、训练、部署使用不同的系统，导致 train-deployment mismatch

**ALE 的解决方案**：

构建一个 **闭环的端到端生态系统**，统一处理：
- 数据生成（在真实环境中生成轨迹）
- 策略优化（稳定的 RL 训练）
- 生产部署（与训练环境一致的 context 管理）

---

## ALE 生态系统架构

ALE 由三个核心组件构成，形成完整的训练-部署闭环：

### 1. ROLL：[[Agentic RL]] 训练框架

**核心功能**：

- **分布式 RL 训练**：将 rollout、reward、training 解耦为独立的 worker 角色
- **Fine-grained Rollout**：在 sample 级别并行化 LLM 生成、环境交互、reward 计算
- **Asynchronous Training**：rollout 和 training 异步执行，用 sample buffer 解耦
- **Train-Rollout Multiplexing**：动态分配 GPU 资源，在 rollout 和 training 之间切换

**关键优化**：

> [!intuition] 为什么需要 Fine-grained Rollout
> 
> Agentic RL 的 rollout 阶段占总时间的 ~70%，而环境交互本身又占 rollout 的 ~15%。
> 
> 传统做法是串行执行：生成完所有 token → 等待环境返回 → 计算 reward。
> 
> ROLL 在 **sample 级别** 并行化这三个阶段，让 LLM 生成和环境交互流水线执行。

**Asynchronous Ratio**：

ROLL 引入 **asynchronous ratio** 控制 off-policy 的 staleness：
- 定义为当前 policy 和生成该 sample 的 policy 之间的版本差距
- 超过阈值的 sample 被丢弃，保证训练稳定性

### 2. ROCK：环境执行引擎

**核心功能**：

- **Sandbox 管理**：为每个任务提供隔离的沙箱环境
- **GYM API 兼容**：提供标准的 `reset()`, `step()`, `close()` 接口
- **多 Agent 支持**：支持多个 agent 在共享或隔离的沙箱中协作
- **大规模调度**：支持数万个并发环境

**关键特性**：

1. **Fault Isolation**：每个 sandbox 的崩溃不影响其他任务
2. **Network Policy**：限制每个 sandbox 的网络访问，防止恶意代码
3. **Agent Native Mode**：解决 train-deployment mismatch（见下文）

> [!intuition] Agent Native Mode 的设计
> 
> **问题**：训练框架（ROLL）和部署系统（iFlow CLI）对 context 的管理方式不同，导致训练时表现好的 agent 部署后性能下降。
> 
> **Naive 方案**：让 ROLL 完全模仿 iFlow CLI 的 context 管理逻辑 → 维护负担巨大，每次更新 agent 逻辑都要同步修改 ROLL。
> 
> **ROCK 的方案**：在 sandbox 内运行 **ModelProxyService**，拦截所有 LLM 请求。这些请求已经包含了 iFlow CLI 构建的完整 context，proxy 只需转发给推理服务。
> 
> 结果：ROLL 简化为纯粹的生成引擎，iFlow CLI 保留完整的 context 控制权，训练和部署完全一致。

### 3. iFlow CLI：Agent 框架

**核心功能**：

- **Context Engineering**：管理多轮交互的 prompt 构建和历史管理
- **Tool Integration**：封装工具调用的接口
- **User Interface**：提供面向用户的 agent 交互界面

**与 ROCK 的配合**：

iFlow CLI 在 ROCK 的 sandbox 中运行，负责：
1. 接收环境观察
2. 构建完整的 prompt（包含历史 context）
3. 调用 LLM（通过 ModelProxyService）
4. 解析 LLM 输出，提取工具调用
5. 返回 action 给环境

---

## IPA 算法：Chunk-Level Policy Optimization

### 动机：为什么 Token-Level 优化不适合 Agentic RL

**Agentic RL 的特点**：

- **Sparse Reward**：只有任务完成时才有 reward，中间步骤没有反馈
- **Long Horizon**：一个任务可能需要数百个 token、数十轮交互
- **Tool-Mediated**：大部分 token 只是推理过程，只有工具调用才改变环境状态

**Token-Level 优化的问题**：

1. **Granularity Mismatch**：绝大多数 token 对环境没有影响，但都参与梯度计算
2. **Credit Assignment 困难**：如何将最终的 reward 分配给数百个 token？
3. **训练不稳定**：长序列的 importance sampling ratio 方差极大

### Chunked MDP：重新建模 Agentic 任务

**核心思想**：

将轨迹划分为 **interaction chunks**，每个 chunk 是从一次环境交互到下一次交互的完整推理过程。

**形式化定义**：

给定 token 序列 $\tau_{[1:T]}$，划分为 chunks $\{c_1, c_2, \ldots, c_K\}$，其中 $K \ll T$。

每个 chunk $c_k$ 包含：
- **推理过程**：思考、规划、参数构建
- **工具调用**：触发环境状态变化的 action

**Chunked MDP** 定义为 $(S, \mathcal{C}, \mathcal{P}, \mathcal{R}, \gamma)$：
- $S$：状态空间（包含完整的交互历史）
- $\mathcal{C}$：chunk-action 空间（变长 token 序列）
- $\mathcal{P}$：转移动力学（由 LLM 生成和工具响应决定）
- $\mathcal{R}$：稀疏 reward（只在任务完成时给出）
- $\gamma$：chunk-level 折扣因子

> [!comparison] Chunk vs Token vs Sentence
> 
> - **Token-level**：粒度太细，绝大多数 token 对环境无影响
> - **Sentence-level**：仍然不对齐，一次工具调用可能需要多个句子（意图 → 参数 → 调用）
> - **Chunk-level**：对齐环境转移的语义单元，每个 chunk 对应一次有意义的交互

### IPA 的训练目标

**Baseline：Off-Policy REINFORCE**

IPA 从一个增强的 [[REINFORCE]] 变体出发（详见论文 §3.2.4.1）：

$$
\nabla J_{\text{RL}}(\pi) = \sum_{\tau \in \mathcal{T}^+} \mu_{\theta_{\text{old}}}(\tau) R(\tau) \nabla \log \pi_\theta(\tau) + \sum_{\tau \in \mathcal{T}^-} \mu_{\theta_{\text{old}}}(\tau) [\rho(\tau)]_0^1 R(\tau) \nabla \log \pi_\theta(\tau)
$$

关键设计：
- **正样本**：直接用 weighted SL update（类似 [[GRPO]]）
- **负样本**：用 truncated importance sampling 防止分布偏移
- **Mismatch Masking**：过滤 inference-training engine 不一致导致的高方差 token

**Chunk-Level Return**：

对于 chunk $c_k$，其 return 定义为：

$$
G_k = \gamma^{\Delta(j,k)} \times R_{\text{final}}
$$

其中：
- $\Delta(j,k)$：chunk $c_k$ 和最终 chunk $c_j$ 之间的 chunk 数量
- $R_{\text{final}}$：任务的最终 reward
- 同一 chunk 内的所有 token 共享相同的 $G_k$

**Chunk-Level Importance Sampling**：

$$
\rho_{\text{chunk}}(c_k) = \left( \prod_{t \in c_k} \frac{\pi_\theta(\tau_t | \tau_{<t})}{\pi_{\theta_{\text{old}}}(\tau_t | \tau_{<t})} \right)^{1/|c_k|}
$$

使用 geometric mean 而非连乘，降低对低概率 token 的敏感度。

**Chunk-Level Mismatch Masking**：

定义 chunk-level mask：

$$
m_k = \mathbb{I}\left( \frac{\pi_{\theta_{\text{old}}}^{\text{megatron}}(c_k)}{\mu_{\theta_{\text{old}}}^{\text{SGLang}}(c_k)} \leq H \right)
$$

只有 $m_k = 1$ 的 chunk 参与梯度更新。

### Chunk-Level Initialized Sampling

**动机**：

传统 rollout 从初始状态开始采样，但大部分轨迹在早期步骤就失败了，浪费计算资源。

**方法**：

1. 从 **中间 chunk** 开始采样（而非总是从头开始）
2. 使用之前成功轨迹的 checkpoint 作为初始状态
3. 动态调整采样起点，平衡探索和效率

**效果**：

- 减少低质量轨迹的生成
- 提高 sample efficiency
- 加速收敛

---

## ROME 模型

### 训练流程

**Stage I：Continuous Pre-training**

- 在 code-centric 数据上继续预训练
- 目标：建立基础的编程能力和工具使用模式

**Stage II：Supervised Fine-Tuning**

- 在高质量的 agentic 轨迹上 SFT
- 引入 **Error-Masked Training**：只对正确的 action 计算 loss
- 引入 **Task-Aware Context Masking**：根据任务类型动态调整 context 长度

**Stage III：Reinforcement Learning**

- 使用 IPA 算法进行 RL 训练
- 在 ROCK 环境中生成轨迹
- 动态过滤噪声轨迹（API 失败、非确定性响应、非法工具调用）

### 数据组成

**Code-Centric Basic Data**：

- 编程基础能力（语法、算法、数据结构）
- 多语言代码生成

**Agentic Data**：

- **General Tool-Use**：通用工具调用（搜索、计算、数据处理）
- **Programming-Centric**：软件工程任务（debug、测试、重构）

**Safety-Aligned Data**：

- 拒绝恶意请求
- 安全的工具使用

### 性能表现

**Terminal-Based Benchmarks**（与同规模模型对比）：

| Benchmark | ROME (30B-A3B) | Qwen3-Coder 30B-A3B | Devstral Small 2 (24B) |
|-----------|----------------|---------------------|------------------------|
| Terminal-Bench 1.0 | **41.50%** | 28.50% | 28.33% |
| Terminal-Bench 2.0 | **24.72%** | 13.48% | 18.20% |
| SWE-bench Verified | **57.40%** | 46.33% | 51.87% |
| SWE-bench Multilingual | **40.00%** | 30.00% | 27.00% |

**与大模型对比**：

- ROME (3B activated) 在 Terminal-Bench 1.0 上超越 Qwen3-Coder 480B-A35B (37.92%) 和 DeepSeek-V3.1 (38.75%)
- 在 SWE-bench Verified 上接近 Claude-Haiku-4.5 (69.60%)

**Tool-Use Benchmarks**：

- MTU-Bench (Single-Turn): **62.45%**（超越 DeepSeek-V3.1 的 61.71%）
- 平均表现：**49.46%**（与 DeepSeek-V3.1 的 49.94% 相当）

---

## 关键洞察

### 1. 生态系统优于单点优化

ALE 的价值不在于某个算法的创新，而在于：
- **端到端一致性**：训练和部署使用相同的 context 管理
- **闭环优化**：数据生成 → 训练 → 评估 → 数据生成
- **生产级稳定性**：fault isolation、dynamic filtering、resource multiplexing

### 2. Chunk 是 Agentic RL 的自然粒度

Token-level 和 sentence-level 都不对齐 agent-environment 交互的语义：
- **Token**：太细，大部分 token 对环境无影响
- **Sentence**：仍然不对齐，一次工具调用可能跨越多个句子
- **Chunk**：对齐环境转移，每个 chunk 是一个完整的推理-执行单元

### 3. Off-Policy 需要多层保护

Industrial-scale agentic RL 的不稳定性来自多个源头：
- **Distributional Shift**：old policy 和 current policy 的差距
- **Inference-Training Mismatch**：不同 engine 的数值差异
- **Environment Noise**：API 失败、非确定性响应

IPA 通过三层机制应对：
1. **Truncated IS**：clip importance ratio
2. **Mismatch Masking**：过滤高方差 token/chunk
3. **Dynamic Filtering**：丢弃噪声轨迹并 on-the-fly resample

### 4. 小模型可以接近大模型性能

ROME (3B activated) 在多个 benchmark 上超越 100B+ 模型，说明：
- **数据质量 > 模型规模**：高质量的 agentic 轨迹 + 稳定的 RL 训练
- **系统设计 > 算法创新**：train-deployment 一致性、环境隔离、资源调度

---

## 局限性

1. **Terminal Bench Pro 性能仍然有限**：所有模型（包括 ROME 和大模型）在更严格的 benchmark 上表现都不理想，说明 long-horizon agentic 任务仍然是开放问题

2. **Chunk 划分依赖工具调用**：IPA 假设 chunk 以工具调用结束，对于没有明确工具调用的任务（如纯推理）可能不适用

3. **系统复杂度高**：ALE 需要协调三个大型系统（ROLL、ROCK、iFlow CLI），部署和维护成本较高

---

## 相关工作

**Agentic RL 方法**：
- [[SAMPO]]：通过 sequence-level clipping 和 environment-level advantage 稳定 agentic RL
- [[ARLArena (2026)]]：系统分析 agentic RL 不稳定性的框架
- [[ASTRA]]：自动化合成 agentic 轨迹和强化学习环境

**RL 算法基础**：
- [[REINFORCE]]：IPA 的起点
- [[PPO]]：IPA 对比的 baseline
- [[GRPO]]：用 group relative advantage 替代 value function
- [[Importance Sampling]]：off-policy 学习的理论基础

**RL 基础设施**：
- veRL：分布式 RL 训练框架
- OpenRLHF：开源 RLHF 框架

---

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/2512.24873/2512.24873|Let It Flow 论文全文]]

**相关 Benchmark**：
- Terminal Bench Pro：论文提出的新 benchmark，更严格的 contamination control
- SWE-bench Verified：软件工程 agent 评估

**系统设计**：
- Agent Native Mode：解决 train-deployment mismatch 的关键设计
- Train-Rollout Multiplexing：动态 GPU 资源分配
