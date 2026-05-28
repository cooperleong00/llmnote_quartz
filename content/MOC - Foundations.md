---
description: LLM 基础知识导航：数学基础、RL 基础、架构基础
type: moc
tags:
  - foundations
  - mathematics
  - reinforcement-learning
created: 2026-01-27
updated: 2026-01-31T22:42
---

# MOC - Foundations

Foundations 是理解 LLM 全栈的基石。本 MOC 组织所有基础知识，包括数学基础、强化学习基础和架构基础。这些概念是理解 Post-training、Inference 等进阶主题的前置知识。

---

## 概览

```
Foundations
├── 数学基础
│   ├── [[KL Divergence]] — KL 散度，分布差异度量
│   ├── [[Bradley-Terry Model]] — 偏好建模基础
│   ├── [[Cross-Entropy Loss]] — 交叉熵损失（待创建）
│   ├── [[Entropy]] — 熵，信息论基础（待创建）
│   └── [[Importance Sampling]] — 重要性采样
│
├── RL 基础
│   ├── 核心概念
│   │   ├── [[Policy Gradient]] — 策略梯度定理
│   │   ├── [[Actor-Critic]] — Actor/Critic 架构
│   │   └── [[GAE]] — 广义优势估计
│   │
│   ├── 算法
│   │   ├── [[TRPO]] — 信任域策略优化
│   │   ├── [[PPO]] — 近端策略优化
│   │   └── [[A2C]] — Advantage Actor-Critic（待创建）
│   │
│   └── 学习方法
│       └── [[TD Learning]] — 时序差分学习（待创建）
│
└── 架构基础
    ├── [[Transformer]] — Transformer 架构
    └── [[Mixture of Experts]] — MoE 稀疏激活架构
```

---

## 数学基础

### 信息论

| 概念 | 描述 | 状态 |
|------|------|------|
| [[Entropy]] | 随机变量的不确定性度量，$H(X) = -\sum p(x) \log p(x)$ | 待创建 |
| [[Cross-Entropy Loss]] | 预测分布与真实分布的差异，LLM 预训练的核心损失函数 | 待创建 |
| [[KL Divergence]] | 两个分布的差异度量，在 RLHF 中用于约束策略更新 | 已完成 |

**学习顺序**：Entropy → Cross-Entropy Loss → KL Divergence

### 概率与统计

| 概念 | 描述 | 状态 |
|------|------|------|
| [[Bradley-Terry Model]] | 成对比较的概率模型，Reward Model 和 DPO 的理论基础 | 已完成 |
| [[Importance Sampling]] | 用一个分布估计另一个分布的期望，PPO 的核心技术 | 已完成 |

---

## RL 基础

强化学习是 RLHF 的理论基础。理解这些概念对于深入理解 Post-training 至关重要。

### 核心概念

#### Policy Gradient

[[Policy Gradient]] 是策略优化的基础方法：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a) \right]
$$

**关键点**：
- 直接优化策略参数
- REINFORCE 是最基础的实现
- Baseline 减少方差

#### Actor-Critic

[[Actor-Critic]] 结合了策略梯度和价值函数：

| 组件 | 角色 | 输出 |
|------|------|------|
| Actor | 策略网络 | $\pi(a|s)$ |
| Critic | 价值网络 | $V(s)$ 或 $Q(s,a)$ |

**优势**：比纯 Policy Gradient 方差更低，比纯 Value-based 更稳定。

#### GAE

[[GAE]]（Generalized Advantage Estimation）平衡 bias 和 variance：

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

**关键参数**：$\lambda$ 控制 bias-variance 权衡。

### 算法

#### 信任域方法

| 算法 | 约束方式 | 特点 | 状态 |
|------|----------|------|------|
| [[TRPO]] | KL 散度硬约束 | 理论保证，计算复杂 | 已完成 |
| [[PPO]] | Clip 软约束 | 简单高效，工业标准 | 已完成 |

**PPO 的核心**：
$$
L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

#### 其他算法

| 算法 | 描述 | 状态 |
|------|------|------|
| [[A2C]] | 同步版 Actor-Critic，A3C 的简化版本 | 待创建 |
| [[TD Learning]] | 时序差分学习，Critic 训练的基础 | 待创建 |

---

## 架构基础

### Transformer

[[Transformer]] 是现代 LLM 的基础架构：

```
Transformer
├── Encoder-Decoder（原始架构）
├── Decoder-only（GPT 系列，当前主流）
└── Encoder-only（BERT 系列）
```

**核心组件**：
- [[Attention]] — 注意力机制
- [[Multi-Head Attention]] — 多头注意力
- Feed-Forward Network
- Layer Normalization
- Residual Connection

详见 [[MOC - Attention]] 了解 Attention 相关内容。

### Mixture of Experts

[[Mixture of Experts]] 是一种稀疏激活架构，实现参数量与计算量的解耦：

| 方面 | Dense 模型 | MoE 模型 |
|------|------------|----------|
| 参数效率 | 参数量 ≈ 计算量 | 参数量 >> 计算量 |
| 代表模型 | GPT、Llama | Mixtral、DeepSeek-V3 |

**核心组件**：
- Experts（专家网络）— 多个并行的 FFN
- Router/Gate — 决定激活哪些专家
- Top-K Selection — 选择 K 个专家

### Model Merging

[[Model Merging]] 是一种无需额外训练即可组合多个模型能力的技术：

- 将多个微调模型的权重合并
- 常见方法：Linear、SLERP、TIES、DARE
- 与 [[LoRA]] 结合使用效果显著

---

## 学习路径建议

### 入门路线

1. **数学基础**：
   - [[Entropy]] → [[Cross-Entropy Loss]] → [[KL Divergence]]
   - 理解信息论基础

2. **架构基础**：
   - [[Transformer]] → [[MOC - Attention]]
   - 理解 LLM 的基础架构

### 进阶路线（面向 RLHF）

1. **RL 基础**：
   - [[Policy Gradient]] → [[Actor-Critic]] → [[GAE]]
   - 理解策略优化的核心思想

2. **信任域方法**：
   - [[TRPO]] → [[PPO]]
   - 理解 RLHF 中使用的优化算法

3. **偏好建模**：
   - [[Bradley-Terry Model]] → [[Reward Model]]
   - 理解人类偏好如何转化为奖励信号

### 面试重点

- KL 散度的定义、性质、非对称性
- Policy Gradient 的推导和 baseline 的作用
- PPO 的 Clip 机制为什么有效
- Transformer 的核心组件和计算复杂度

---

## 相关 MOC

- [[MOC - Attention]] — Attention 机制详解
- [[MOC - Post-training]] — 后训练方法（SFT、RLHF、DPO）
- [[MOC - Inference]] — 推理优化
