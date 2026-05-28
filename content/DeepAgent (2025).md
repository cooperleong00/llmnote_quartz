---
type: paper
description: 端到端深度推理智能体，通过统一推理流程实现自主工具发现、调用和记忆管理，用 ToolPO 训练支持万级工具集的通用任务求解
aliases:
  - DeepAgent
  - 深度智能体
prerequisites:
  - "[[Agentic RL]]"
  - "[[REINFORCE]]"
tags:
  - agent
  - tool-use
  - reinforcement-learning
  - reasoning
created: 2026-03-06
updated: 2026-03-06T18:48
---

# DeepAgent (2025)

DeepAgent 是一个端到端的深度推理智能体框架，将工具发现、调用和记忆管理统一到单一连贯的推理流程中。与传统的预定义工作流（如 ReAct）不同，DeepAgent 让大型推理模型（LRM）在推理过程中自主决定何时搜索工具、调用工具或压缩记忆，支持从数十到上万个工具的可扩展工具集。

> [!paper] 论文信息
> **标题**: DeepAgent: A General Reasoning Agent with Scalable Toolsets
> **作者**: Xiaoxi Li et al. (Renmin University + Xiaohongshu Inc.)
> **会议**: WWW 2026
> **链接**: arXiv:2510.21618
> **代码**: https://github.com/RUC-NLPIR/DeepAgent

## 动机

传统智能体框架（如 ReAct、Plan-and-Solve）存在四个核心局限：

1. **缺乏自主性**：执行步骤和整体流程被预定义的工作流限制
2. **工具发现受限**：无法在任务执行过程中动态发现新工具
3. **记忆管理被动**：缺乏自主的交互历史管理机制
4. **推理深度不足**：在整个任务上缺乏连贯的全局推理

虽然大型推理模型（LRM）展示了通过"慢思考"解决复杂问题的能力，但现有的工具集成方案（如 Deep Research）通常局限于固定的小工具集（搜索、浏览、编码），限制了通用性。

> [!intuition] 核心洞察
> 将工具使用嵌入到推理流程中，而非作为外部工作流步骤。让 LRM 在单一连贯的思考过程中自主决定何时需要工具、搜索什么工具、如何调用，以及何时压缩记忆以保持全局视角。

## 核心贡献

### 1. 统一的自主推理范式

DeepAgent 将传统的"Reason-Act-Observe"循环替换为单一的连贯推理流程：

```
传统 Agent:
[Reason] → [Act] → [Observe] → [Reason] → [Act] → ...
  ↑                                ↑
  预定义工作流                    固定工具集

DeepAgent:
[Continuous Reasoning with Embedded Actions]
  ├─ <think> 内部推理
  ├─ <tool_search> 动态工具发现
  ├─ <tool_call> 工具调用
  └─ <fold_thought> 自主记忆压缩
```

**关键特性**：
- **全局视角**：LRM 对整个任务保持连贯理解，而非局限于单步操作
- **按需工具发现**：工具不是预先检索，而是在需要时动态搜索
- **自主记忆管理**：模型决定何时"喘口气"，压缩历史以避免错误路径

### 2. 自主记忆折叠（Autonomous Memory Folding）

**问题**：长时程交互导致上下文膨胀和错误累积。

**解决方案**：让 Agent 在推理过程中生成 `<fold_thought>` 触发记忆压缩，将原始交互历史转换为结构化记忆：

$$
(M_E, M_W, M_T) = f_{\text{compress}}(s_t; \theta_{\text{aux}})
$$

**脑启发的三层记忆架构**：

| 记忆类型 | 作用 | 内容 |
|---------|------|------|
| **Episodic Memory** ($M_E$) | 长期上下文 | 关键事件、决策点、子任务完成记录 |
| **Working Memory** ($M_W$) | 短期连续性 | 当前子目标、遇到的障碍、近期计划 |
| **Tool Memory** ($M_T$) | 工具经验 | 已用工具、调用方式、效果评估 |

**设计要点**：
- 使用 JSON schema 而非自然语言，确保结构稳定和细节保留
- 由辅助 LLM 并行生成三种记忆，减少主推理模型负担
- Agent 可在完成子任务或发现错误路径时主动触发

> [!comparison] vs 传统记忆管理
> - **传统方法**：固定窗口截断或外部记忆检索系统
> - **DeepAgent**：Agent 自主决定何时压缩，保留任务相关信息而非机械截断

### 3. 可扩展工具管理

**工具搜索**：
- Agent 生成 `<tool_search> query </tool_search>`
- 密集检索：$\mathcal{T}_{\text{retrieved}} = \text{top-k}(\text{sim}(E(q_s), E(d_i)))$
- 辅助 LLM 过滤和总结工具文档（如果过长）

**工具调用**：
- 结构化调用：`<tool_call> {"name": "tool_name", "arguments": ...} </tool_call>`
- 辅助 LLM 去噪和压缩工具返回结果
- 反馈到推理上下文：`<tool_call_result> helpful information </tool_call_result>`

**架构分工**：
- **主推理模型**：专注高层战略推理
- **辅助 LLM**：处理工具文档过滤、结果总结、记忆压缩

### 4. ToolPO：端到端 RL 训练

**挑战**：
1. 与数千个真实 API 交互训练不稳定、慢且昂贵
2. 仅用最终结果的稀疏奖励无法保证中间工具调用的正确性

**解决方案**：

#### 工具模拟器（Tool Simulator）
- 用辅助 LLM 模拟真实 API 响应（如 RapidAPI）
- 提供稳定、高效、低成本的训练环境

#### 双层优势归因（Global + Tool-Call Advantage Attribution）

**奖励设计**：
- **任务成功奖励**：$R_{\text{succ}}(\tau)$ — 最终结果质量
- **动作级奖励**：$R_{\text{action}}(\tau) = \lambda_1 \sum_{i=1}^{I} C(a_i^{\text{call}}) + \lambda_2 S_{\text{pref}}(\tau)$
  - $C(a_i^{\text{call}})$：工具调用是否正确（0/1）
  - $S_{\text{pref}}(\tau)$：记忆折叠效率偏好分数

**优势计算**：
- **全局优势**（归因到所有 token）：
  $$A_{\text{succ}}(\tau_k) = R_{\text{succ}}(\tau_k) - \frac{1}{K}\sum_{j=1}^{K} R_{\text{succ}}(\tau_j)$$

- **动作级优势**（仅归因到工具调用和记忆折叠 token）：
  $$A_{\text{action}}(\tau_k) = R_{\text{action}}(\tau_k) - \frac{1}{K}\sum_{j=1}^{K} R_{\text{action}}(\tau_j)$$

**总优势**：
$$A(y_i) = A_{\text{succ}}(\tau_k) + M(y_i) \cdot A_{\text{action}}(\tau_k)$$

其中 $M(y_i)$ 是掩码：如果 $y_i$ 是工具调用或记忆折叠 token 则为 1，否则为 0。

**优化目标**（类似 PPO 的 clipped objective）：
$$\mathcal{L}_{\text{ToolPO}}(\theta) = \mathbb{E}_{\tau_k}\left[\sum_{i=1}^{|\tau_k|} \min\left(\rho_i(\theta) A(y_i), \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) A(y_i)\right)\right]$$

> [!intuition] 为什么需要细粒度归因？
> 稀疏奖励只告诉模型"最终结果好/坏"，但无法指导中间的工具调用。通过将动作级奖励精确归因到工具调用 token，模型能学到"这个工具调用是对的"，而非仅仅"这条轨迹最终成功了"。

## 实验结果

### 通用工具使用任务

在 ToolBench、API-Bank、TMDB、Spotify、ToolHop 上的表现（Pass@1）：

**场景 1：已知工具集**
- DeepAgent-32B-RL 在所有任务上超越 QwQ-32B 基线（+14-26%）
- 在 TMDB 和 Spotify 上超越 GPT-4o 和 DeepSeek-R1

**场景 2：开放集工具检索**（从大规模工具库动态检索）
- DeepAgent-32B-RL 相比 QwQ-32B 基线提升 **2-3 倍**
- 在 TMDB 上达到 55% 成功率（vs ReAct 18%）

### 下游应用

在 ALFWorld、WebShop、GAIA、HLE 上的表现：
- ALFWorld（交互式环境）：**82.0%** vs QwQ-32B 的 68.0%
- WebShop（电商任务）：**68.5%** vs QwQ-32B 的 52.0%
- GAIA（通用 AI 助手）：**48.2%** vs QwQ-32B 的 38.5%

### 消融实验

| 组件 | 平均性能 | 影响 |
|------|---------|------|
| 完整 DeepAgent | 48.1 | - |
| 去掉 ToolPO 训练 | 44.3 | **-3.8** |
| 去掉记忆折叠 | 46.7 | -1.4 |
| 去掉工具调用奖励 | 46.9 | -1.2 |

**关键发现**：
- ToolPO 训练是最关键的组件
- 记忆折叠在长时程任务（如 WebShop）上尤为重要
- 工具调用级奖励显著提升中间步骤的准确性

## 局限性与未来方向

> [!warning] 当前局限
> 1. **工具模拟器的真实性**：LLM 模拟的 API 可能与真实 API 存在分布差异
> 2. **记忆压缩的信息损失**：结构化记忆可能丢失某些细微但关键的信息
> 3. **计算成本**：端到端 RL 训练需要大量采样和模拟
> 4. **工具检索质量**：密集检索可能在语义模糊的查询上失效

**未来方向**：
- 混合真实 API 和模拟 API 的训练策略
- 更精细的记忆压缩机制（如分层压缩）
- 与工具提供商的 API schema 标准化集成
- 多模态工具使用（视觉、音频工具）

## 相关概念

**前置知识**：
- [[Agentic RL]] — DeepAgent 的训练范式
- [[REINFORCE]] — ToolPO 的理论基础
- [[PPO]] — ToolPO 的 clipped objective 借鉴自 PPO

**对比方法**：
- [[ReAct]] — 传统的 Reason-Act-Observe 工作流
- [[Plan-and-Solve]] — 预定义规划的智能体框架

**相关工作**：
- [[ARLArena (2026)]] — 分析 Agentic RL 训练不稳定性的框架
- [[Kimi K2 (2025)]] — 另一个强调 Agentic 数据合成的模型

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/2510.21618/2510.21618|DeepAgent 论文全文]]

**实现细节**：
- 训练数据：ToolBench、ALFWorld、WebShop、WebDancer、WebShaperQA、DeepMath
- 工具检索：bge-large-en-v1.5 embedding
- VQA 工具：Qwen2.5-VL-32B-Instruct
- 训练：100 steps ToolPO，batch size 64，rollout size K=8，max seq len 32,768
- 硬件：64 × NVIDIA H20-141GB GPUs
