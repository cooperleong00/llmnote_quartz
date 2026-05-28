---
type: paper
description: 提出 GiGPO 算法，通过两层分组结构解决 multi-turn LLM agent 训练中的细粒度信用分配问题
aliases:
  - Group-in-Group Policy Optimization for LLM Agent Training
tags:
  - post-training
  - rlhf
  - agent
  - policy-optimization
created: 2026-03-06
updated: 2026-03-06
---

# GiGPO (2025)

**Group-in-Group Policy Optimization for LLM Agent Training**

**作者**：Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An (Nanyang Technological University & Skywork AI)

**arXiv**：2505.10978

**GitHub**：https://github.com/langfengQ/verl-agent

## 核心贡献

这篇论文提出 [[GiGPO]]（Group-in-Group Policy Optimization），一种专为 multi-turn LLM agent 训练设计的 group-based RL 算法。核心创新是通过**两层分组结构**实现细粒度的 [[Credit Assignment|信用分配]]，在保持 [[GRPO]] 的 critic-free、低内存、稳定收敛优势的同时，解决了长序列任务中的 per-step credit assignment 问题。

> [!intuition] 核心洞察
>
> 在相同任务和初始状态下，多条轨迹会**自然地重复访问相同的环境状态**（如重复访问的网页、房间、游戏场景）。这些重复状态提供了"免费的"step-level groups——不需要额外 rollout，只需 retroactively 识别和分组。

## 主要方法

### 两层分组结构

1. **Episode-level grouping**：计算 macro relative advantages，基于完整轨迹的总回报（与 [[GRPO]] 相同）

2. **Step-level grouping**：通过 **anchor state grouping** 机制，retroactively 识别跨轨迹的重复状态，构建 step-level groups 来计算 micro relative advantages

**关键机制**：
- 对每个唯一状态 $\tilde{s}$，收集所有"从该状态出发的动作"
- 使用 discounted return（而非即时奖励）计算 step relative advantage
- 最终 advantage = episode advantage + $\omega$ × step advantage

### 与现有方案的对比

| 方案 | Episode-level | Step-level | 额外成本 |
|------|---------------|------------|----------|
| [[GRPO]] | ✅ | ❌ | 无 |
| Per-step rollout | ✅ | ✅ | 极高（额外 LLM forward） |
| RAGEN | ✅ | ❌ | 无（但长 horizon 扩展性差） |
| [[PPO]] | ✅ | ✅ | 中（value network） |
| **[[GiGPO]]** | ✅ | ✅ | 极低（< 0.002%） |

## 实验结果

### Benchmarks

**ALFWorld**（embodied household tasks）：
- 3,827 个任务实例，6 个类别（Pick & Place, Examine in Light, Clean & Place, Heat & Place, Cool & Place, Pick Two & Place）
- GiGPO 相比 GRPO 提升 > 12%

**WebShop**（web-based shopping）：
- 1.1M 产品，12K 用户指令
- GiGPO 相比 GRPO 提升 > 9%

**Search-augmented QA**：
- Single-hop：NQ, TriviaQA, PopQA
- Multi-hop：HotpotQA, 2Wiki, MuSiQue, Bamboogle
- 结果：42.1% (Qwen2.5-3B), 47.2% (Qwen2.5-7B)

### 计算成本

- **时间开销**：< 0.002%（几乎可忽略）
- **GPU 内存**：与 GRPO 相同
- **LLM rollout 次数**：与 GRPO 相同

### Baselines

**Closed-source LLMs**：
- GPT-4o
- Gemini-2.5-Pro

**Prompting agents**：
- ReAct
- Reflexion

**RL training methods**：
- [[PPO]]（actor-critic，需要 value network）
- [[RLOO]]（group-based, critic-free）
- [[GRPO]]（group-based, critic-free）

## 局限性与未来方向

> [!warning] 主要局限
>
> **依赖状态匹配**：GiGPO 的 step-level grouping 依赖于识别重复状态。在高度复杂或噪声环境中，相同状态可能难以检测。

**缓解方案**：
- **Similarity-based grouping**：使用 longest matching subsequence，相似度 > 0.9 时分组
- **优雅降级**：极端情况下（无重复状态），自动退化为 [[GRPO]]

**未来方向**：
- Embedding-based state representations
- Domain-specific structural equivalence

## 与相关工作的关系

### Group-based RL

GiGPO 属于 group-based RL 家族，与以下方法相关：
- [[RLOO]]：REINFORCE Leave-One-Out
- [[GRPO]]：Group Relative Policy Optimization
- Dr. GRPO, DAPO, CPPO

这些方法的共同特点：
- Critic-free（无需 value function）
- 通过 group 内的相对比较估计 advantage
- 适合大规模 LLM 训练

### LLM Agent 训练

**早期工作**：
- DQN 用于 text-based games
- PPO/AWR 用于 Android device control, ALFWorld, card games

**近期工作**：
- **ArCHer, AgentQ**：针对 WebShop，但需要额外 value network 或 MCTS
- **CoSo**：entropy-based RL
- **LOOP**：混合 RLOO 和 PPO
- **RAGEN**：trajectory-level GRPO，将整个 episode 拼接为单个 response（长 horizon 扩展性差）

### RLHF 与 LLM RL

GiGPO 的技术路线与 [[RLHF]] 相关，但专注于 agent 训练：
- RLHF 主要用于 alignment（对齐人类偏好）
- 近期 group-based RL 方法在 mathematical reasoning, search, tool use 等任务中表现优异
- GiGPO 将这些方法扩展到 multi-turn agent 场景

## 技术细节

**模型**：Qwen2.5-1.5B/3B/7B-Instruct

**超参数**：
- 平衡系数 $\omega = 1$（无需调优）
- Rollout group size $N = 8$（ALFWorld, WebShop）
- Rollout group size $N = 5$（Search-augmented QA）
- Max turn = 4（Search-augmented QA）
- Similarity threshold = 0.9（similarity-based grouping）

**Retriever**：E5（用于 search-augmented QA）

## 延伸阅读

**核心方法**：
- [[GiGPO]] — 详细的算法机制和数学推导

**相关方法**：
- [[GRPO]] — GiGPO 的 episode-level 基础
- [[RLOO]] — 另一种 critic-free group-based 方法
- [[PPO]] — 传统 actor-critic 方法

**相关概念**：
- [[Credit Assignment]] — GiGPO 要解决的核心问题
- [[Value Function]] — GiGPO 通过 group-based advantage 避免显式建模

**应用领域**：
- [[MOC - Post-training]] — LLM agent 训练是 post-training 的重要方向
