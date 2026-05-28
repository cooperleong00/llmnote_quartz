---
type: method
description: 整合 sequence-level clipping、环境级优势和 dynamic filtering 的稳定 Agentic RL 方法，相比 GRPO 平均提升 25.2%
aliases:
  - SAMPO
  - Stable Agentic Multi-turn Policy Optimization
  - 稳定智能体多轮策略优化
prerequisites:
  - "[[GRPO]]"
  - "[[Importance Sampling]]"
  - "[[Agentic RL]]"
tags:
  - post-training
  - reinforcement-learning
  - agent
created: 2026-03-05
updated: 2026-03-06T02:51
---

# SAMPO

稳定智能体多轮策略优化（SAMPO, Stable Agentic Multi-turn Policy Optimization）是 [[ARLArena (2026)]] 提出的 [[Agentic RL]] 训练方法。它通过**系统性地整合三个已验证的设计选择**——sequence-level clipping、环境级优势设计和 dynamic filtering——实现稳定且高效的多轮交互训练。

> [!paper] 论文出处
> **ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning**
> Wang et al., UCLA, 2026.02

## 核心思想

SAMPO 不是"发明新算法"，而是**基于系统分析的原则性整合**：

1. **Sequence-level clipping**（来自 Finding 1）：用序列级 IS ratio 裁剪，抑制有害轨迹
2. **环境级优势设计**（来自 Finding 2）：融入 state-level 信息，缓解奖励稀疏性
3. **Dynamic filtering**（来自 Finding 3）：过滤无信息样本，提高梯度质量

每个组件都有明确的稳定性保证，组合后形成鲁棒的整体。

## 数学形式

### 损失函数

$$
\mathcal{L}(\theta) = \frac{1}{\sum_{i=1}^N T_i} \sum_{i=1}^N \sum_{t=0}^{T_i-1} \min\Big( s_i(\theta) A_i', \text{clip}(s_i(\theta), 1 \pm \varepsilon) A_i' \Big)
$$

$$
\text{s.t.} \quad 0 < |\{y \mid \text{is\_equivalent}(a, y)\}| < G
$$

### 关键组件

**1. Sequence-level IS ratio**：

$$
s_i(\theta) = \exp\left( \frac{1}{|T_i|} \sum_{t=0}^{|T_i|-1} \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\theta_\text{old}}(y_t | x, y_{<t})} \right)
$$

- 用序列级别的平均 log-ratio 作为裁剪标准
- 相比 token-level clipping（[[GRPO]]），更能抑制整体偏离的轨迹

**2. 环境级优势**：

$$
A_i' = A_i + \omega \cdot A_{\text{step}}(\hat{y}_{i,k})
$$

其中：
- $A_i$：序列级优势（如 group relative advantage）
- $A_{\text{step}}(\hat{y}_{i,k})$：同一环境状态 $s_k$ 下动作的相对优势
- $\omega$：权重超参数

**3. Dynamic filtering 约束**：

$$
0 < |\{y \mid \text{is\_equivalent}(a, y)\}| < G
$$

- 过滤掉所有样本都成功或都失败的 group
- 重新采样，确保每个 group 有信息梯度

> [!intuition] 为什么这三个组件协同工作？
> - **Sequence-level clipping** 防止策略崩溃（稳定性）
> - **环境级优势** 提供更细粒度的学习信号（性能）
> - **Dynamic filtering** 提高样本效率（收敛速度）
>
> 三者分别解决不同问题，互不冲突，组合后形成"稳定 + 高效 + 高性能"的完整方案。

## 与基线方法的对比

### SAMPO vs GRPO

| 维度 | [[GRPO]] | SAMPO |
|------|----------|-------|
| **IS Clipping** | Token-level | Sequence-level |
| **Advantage** | Group relative | Group relative + state-level |
| **Filtering** | 无 | Dynamic filtering |
| **稳定性** | 中等（易受负优势样本影响） | 高（sequence masking 抑制有害轨迹） |
| **性能** | 基线 | 平均 +25.2% |

### SAMPO vs GSPO

| 维度 | GSPO | SAMPO |
|------|------|-------|
| **IS Clipping** | Sequence-level | Sequence-level |
| **Advantage** | Group relative | Group relative + state-level |
| **Filtering** | 无 | Dynamic filtering |
| **性能** | +13.3% vs GRPO | +25.2% vs GRPO |

SAMPO 在 GSPO 的 sequence-level clipping 基础上，进一步整合了环境级优势和 dynamic filtering。

### SAMPO vs DAPO$_{\text{[[GiGPO]]}}$

| 维度 | DAPO$_{\text{[[GiGPO]]}}$ | SAMPO |
|------|----------------------|-------|
| **IS Clipping** | Token-level | Sequence-level |
| **Advantage** | State-level | Group + state-level |
| **Filtering** | Dynamic filtering | Dynamic filtering |
| **性能** | +11.0% vs GRPO | +25.2% vs GRPO |

SAMPO 的关键优势是 **sequence-level clipping**，这是稳定性的核心保证。

## 实验结果

### 主要任务性能

| 任务 | GRPO | SAMPO | 提升 |
|------|------|-------|------|
| **ALFWorld** | 62.36% | 92.72% | +48.7% |
| **WebShop** | 57.71% | 77.73% | +34.7% |
| **Sokoban** | 83.90% | 88.86% | +5.9% |
| **平均** | 67.99% | 86.44% | +27.1% |

### 训练稳定性

SAMPO 在所有任务上都展现出**单调稳定的改进曲线**：
- 无训练崩溃
- 梯度范数稳定
- KL 散度平稳增长
- Valid format ratio 快速收敛

相比之下：
- [[CISPO]] 和 [[SAPO]] 在 ~130 步崩溃
- [[GRPO]] 有波动但最终稳定
- GSPO 稳定但性能不如 SAMPO

### 与闭源模型对比

在 ALFWorld 上：
- **SAMPO (Qwen3-4B)**：92.72%
- GPT-5.2 (SLA)：51.56%
- o3 (Multi-Agent Debate)：56.25%

> [!intuition] 核心洞察
> 小模型 + 稳定 RL 训练 > 大模型 + 复杂推理工程。这验证了**环境对齐训练**的重要性。

## 实现细节

### 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| $\varepsilon$ | 0.001 | Clipping 范围 |
| $\omega$ | 0.5 | 环境级优势权重 |
| $G$ | 4 | Dynamic filtering 的 group size |
| KL coef | 0.02 | KL 正则化系数 |
| Learning rate | 1e-6 | 学习率 |

### 训练配置

- **模型**：Qwen3-4B（SFT 初始化）
- **初始化**：Behavior cloning on self-generated trajectories
- **Format penalty**：强制 `<think>...</think>` 和 `<action>...</action>` 结构
- **KL regularization**：用 $k_3$ 估计器约束与参考模型的距离

### 关键实现技巧

1. **Sequence-level IS ratio 计算**：
   ```python
   log_ratio = log_prob_new - log_prob_old  # shape: (batch, seq_len)
   seq_is_ratio = torch.exp(log_ratio.mean(dim=1))  # shape: (batch,)
   ```

2. **环境级优势计算**：
   ```python
   # Group actions by environment state
   state_groups = group_by_state(trajectories)
   for group in state_groups:
       # Compute relative advantage within group
       group_advantages = group.rewards - group.rewards.mean()
       A_step = group_advantages / (group_advantages.std() + 1e-8)
   ```

3. **Dynamic filtering**：
   ```python
   # Filter groups where all samples have same outcome
   valid_groups = [g for g in groups if 0 < g.success_count < len(g)]
   # Resample if needed
   while len(valid_groups) < target_size:
       new_samples = rollout_policy(env)
       valid_groups.extend(filter_groups(new_samples))
   ```

## 局限性

### 1. 计算开销

- Dynamic filtering 需要额外的 rollout
- Sequence-level IS ratio 计算需要完整序列
- 环境级优势需要 state grouping

**缓解**：可以用 mini-batch 近似，或只在早期训练使用 dynamic filtering。

### 2. 超参数敏感性

- $\varepsilon$ 需要针对任务调整（论文用网格搜索）
- $\omega$ 影响环境级优势的权重
- Dynamic filtering 的 $G$ 影响样本效率

**缓解**：论文提供了标准化的超参数搜索流程。

### 3. 依赖高质量初始化

SAMPO 假设策略已通过 behavior cloning 初始化到合理的行为流形。如果初始化很差，可能仍会崩溃。

**缓解**：使用 self-bootstrapped SFT（用强模型生成轨迹，训练弱模型）。

## 适用场景

### ✅ 适合使用 SAMPO 的场景

- **长时域多轮交互**：ALFWorld（平均 10+ 轮）、WebShop（平均 5+ 轮）
- **稀疏奖励**：只有完成任务才有奖励
- **需要稳定训练**：不能容忍训练崩溃
- **有环境状态信息**：可以利用 state-level advantage

### ❌ 不适合使用 SAMPO 的场景

- **单轮生成任务**：如 summarization、translation（用 [[DPO]] 或 [[GRPO]] 即可）
- **密集奖励**：如 token-level reward（环境级优势收益有限）
- **极短序列**：sequence-level clipping 的优势不明显

## 面试要点

> [!interview] 常见问题
>
> **Q1: SAMPO 和 GRPO 的核心区别是什么？**
>
> A: 三个关键区别：
> 1. **IS clipping**：SAMPO 用 sequence-level ratio，GRPO 用 token-level ratio
> 2. **Advantage**：SAMPO 融入环境级信息，GRPO 只用 group relative
> 3. **Filtering**：SAMPO 有 dynamic filtering，GRPO 没有
>
> 其中 sequence-level clipping 是稳定性的核心保证。
>
> **Q2: 为什么 sequence-level clipping 比 token-level clipping 更稳定？**
>
> A: Token-level clipping 允许单个 token 偏离，但整体轨迹可能严重偏离。Sequence-level clipping 用整个序列的平均 log-ratio 作为标准，能更好地抑制**整体有害的轨迹**。
>
> 实验发现，训练崩溃主要由"负优势 + 低 sequence-level IS ratio"的样本驱动。Sequence-level clipping 天然屏蔽了这些样本。
>
> **Q3: SAMPO 的三个组件能否单独使用？**
>
> A: 可以，但效果不如组合：
> - 只用 sequence-level clipping（GSPO）：+13.3%
> - 只用环境级优势（[[GiGPO]]）：+3.4%
> - 只用 dynamic filtering（DAPO$_{\text{GRPO}}$）：-7.6%（甚至有害！）
>
> SAMPO 的关键是**原则性整合**：每个组件解决不同问题，互不冲突。
>
> **Q4: SAMPO 为什么能超越 GPT-5.2 和 o3？**
>
> A: 核心原因是**环境对齐训练 vs 通用推理能力**：
> - GPT-5.2/o3 是通用模型，没有针对特定环境优化
> - SAMPO 训练的 Qwen3-4B 通过 RL 学会了环境的动态和最优策略
>
> 这说明在 agentic 任务上，**小模型 + 环境对齐 RL > 大模型 + 复杂推理**。

## 相关概念

**前置知识**：
- [[GRPO]] — SAMPO 的基线
- [[Importance Sampling]] — IS clipping 的理论基础
- [[Agentic RL]] — SAMPO 的应用场景

**相关方法**：
- [[CISPO]] — Tolerant clipping（不稳定）
- [[SAPO]] — Soft clipping（不稳定）
- GSPO — Sequence-level clipping（稳定但性能不如 SAMPO）
- [[GiGPO]] — 环境级优势（性能提升有限）

**论文**：
- [[ARLArena (2026)]] — SAMPO 的出处

**应用**：
- [[GLM-5 (2026)]] — 可能受益于 SAMPO 的稳定性
- [[ASTRA]] — 自动化 agentic 轨迹合成

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/2602.21534/2602.21534|ARLArena 论文全文]]

**实现资源**：
- GitHub: https://github.com/WillDreamer/ARL-Arena
- HuggingFace: https://huggingface.co/UCLA-SCAI/models
