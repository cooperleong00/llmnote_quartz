---
description: LLM 基础知识导航：信息论、优化器、Transformer 架构和 RL 基础
type: moc
aliases:
  - MOC - Foundations
tags:
  - foundations
  - mathematics
  - reinforcement-learning
created: 2026-01-27
updated: 2026-05-31T17:54
---

# MOC - Foundations

Foundations 组织理解 LLM 全栈所需的前置概念。这里的目标是建立可迁移的底层框架：信息论解释训练目标，优化器解释参数如何更新，Transformer 与 Attention 解释模型如何计算，RL 基础解释 post-training 中的策略优化。

---

## 概览

```
Foundations
├── 文本与表示
│   ├── [[Tokenization]] — 文本到 token 序列
│   ├── [[Embedding]] — 离散符号到连续向量
│   └── [[Positional Encoding]] — 为序列注入位置信息
│
├── 信息论与损失
│   ├── [[Entropy]] — 不确定性度量
│   ├── [[Cross-Entropy Loss]] — LLM 训练目标
│   └── [[KL Divergence]] — 分布差异与策略约束
│
├── 优化基础
│   ├── [[Adam]] — 自适应学习率优化器
│   └── [[AdamW]] — 解耦权重衰减
│
├── 架构基础
│   ├── [[Transformer]] — LLM 主干架构
│   ├── [[Attention]] / [[Multi-Head Attention]] — 动态上下文聚合
│   └── [[Mixture of Experts]] — 稀疏激活扩展模型容量
│
└── RL 基础
    ├── [[Markov Decision Process]] / [[Value Function]]
    ├── [[Policy Gradient]] / [[TD Learning]]
    └── [[Actor-Critic]] / [[GAE]] / [[PPO]]
```

---

## 文本与表示

[[Tokenization]] 决定原始文本如何进入模型，vocabulary 和切分规则会影响多语言、代码、数学符号和长尾实体的学习效率。token 序列进入模型后，[[Embedding]] 将离散 ID 映射为连续向量，让语义相似的对象在向量空间中接近。

[[Positional Encoding]] 解决 Attention 的顺序感知问题。具体位置编码方案在 [[02-MOC - Attention|MOC - Attention]] 中展开，基础入口包括 [[RoPE]] 和 [[ALiBi]]。

| 概念 | 核心问题 | 后续连接 |
|------|----------|----------|
| [[Tokenization]] | 文本如何变成可训练的离散序列 | 预训练数据、数学推理、多语言能力 |
| [[Embedding]] | 符号如何进入连续空间 | [[Dense Retrieval]]、语义搜索、RAG |
| [[Positional Encoding]] | Transformer 如何感知顺序 | [[RoPE]]、长度外推、长上下文 |

---

## 信息论与损失

信息论主线可以按这个顺序学习：[[Entropy]] 衡量分布自身的不确定性，[[Cross-Entropy Loss]] 衡量真实分布下模型预测的编码代价，[[KL Divergence]] 衡量两个分布之间的额外编码代价。

| 概念 | 一句话定位 | 常见用途 |
|------|------------|----------|
| [[Entropy]] | 随机变量不确定性的度量 | 解释探索、多样性、entropy collapse |
| [[Cross-Entropy Loss]] | 最大似然训练的损失形式 | 预训练、SFT、分类目标 |
| [[KL Divergence]] | 非对称的分布差异度量 | PPO KL penalty、DPO 推导、蒸馏 |
| [[Bradley-Terry Model]] | 成对偏好的概率模型 | [[Reward Model]]、[[DPO]]、偏好数据建模 |

> [!intuition] 学习锚点
> 当你能从 cross entropy 分解出 entropy 和 KL divergence 时，就能看懂很多 LLM 训练目标：SFT 是最大化目标回答的似然，RLHF 中的 KL penalty 限制策略偏移，DPO 用 Bradley-Terry 偏好模型把 reward learning 合并进策略优化。

---

## 优化基础

[[Adam]] 和 [[AdamW]] 是理解 LLM 训练配置的基础。Adam 用一阶矩和二阶矩估计自适应调整学习率，AdamW 将 weight decay 从梯度更新中解耦，成为现代 LLM 训练的常用选择。

| 概念 | 关注点 | 相关主题 |
|------|--------|----------|
| [[Adam]] | 动量、自适应学习率、偏差修正 | 稳定训练、学习率设置 |
| [[AdamW]] | 解耦 weight decay | LLM 预训练与微调 |
| [[Gradient Accumulation]] | 用多步累积模拟大 batch | 显存受限训练 |
| [[Mixed Precision Training]] | 用低精度提升吞吐并节省显存 | 分布式训练、数值稳定 |

更系统的训练效率问题见 [[03-MOC - Distributed Training|MOC - Distributed Training]]。

---

## 架构基础

### Transformer

[[Transformer]] 是现代 LLM 的主干架构。它通过 [[Attention]] 在序列内部动态聚合信息，通过 FFN 做逐位置非线性变换，再用 residual connection 和 normalization 保持深层训练稳定。

```
Transformer Block
├── Attention
│   ├── [[Attention]]
│   ├── [[Multi-Head Attention]]
│   └── [[Grouped-Query Attention]] / [[Multi-Query Attention]]
├── Feed-Forward Network
└── Residual + Normalization
```

### Mixture of Experts

[[Mixture of Experts]] 通过稀疏激活让模型拥有更大的参数容量，同时保持每个 token 的计算量可控。它连接三个方向：

- 架构设计：[[Loss-Free Load Balancing]] 处理专家负载与模型质量的冲突。
- 分布式训练：[[Expert Parallelism]] 将专家分布到多设备。
- RL 稳定性：[[IcePop]] 和 [[Rollout Routing Replay]] 处理 MoE 训练-推理概率不一致。

### Model Merging

[[Model Merging]] 在无需额外训练的条件下组合多个模型或 adapter 的能力，常与 [[LoRA]] 和 [[TIES-Merging]] 一起讨论。它适合作为 fine-tuning 后的模型组合工具。

---

## RL 基础

RL 基础服务于两类问题：经典序贯决策，以及 LLM post-training 中的策略优化。完整算法脉络见 [[06-MOC - Reinforcement Learning|MOC - Reinforcement Learning]]。

### 核心概念

| 概念 | 核心作用 | 为什么重要 |
|------|----------|------------|
| [[Markov Decision Process]] | 用状态、动作、奖励、转移定义序贯决策 | 所有 RL 算法的数学框架 |
| [[Value Function]] | 估计状态或动作的长期价值 | critic、Q-learning、GAE 的基础 |
| [[TD Learning]] | 用 bootstrap 更新价值估计 | Actor-Critic 的 critic 训练基础 |
| [[Policy Gradient]] | 直接对策略参数求梯度 | PPO、GRPO、VAPO 的理论起点 |
| [[Importance Sampling]] | 用一个策略的数据估计另一个策略的期望 | off-policy 学习和 PPO clip 的基础 |
| [[On-Policy vs Off-Policy]] | 区分采样策略和优化策略的关系 | 决定 sample efficiency 与训练稳定性 |

### 经典算法入口

| 路线 | 入口 | 学习重点 |
|------|------|----------|
| Value-based | [[Q-Learning]] → [[DQN]] → [[Experience Replay]] | Q 函数、bootstrap、样本复用 |
| Policy-based | [[REINFORCE]] → [[Policy Gradient]] | 直接优化策略、方差控制 |
| Actor-Critic | [[Actor-Critic]] → [[A2C]] / [[A3C]] → [[GAE]] | 用 value function 降低策略梯度方差 |
| Trust Region | [[TRPO]] → [[PPO]] | 限制策略更新幅度，连接 RLHF |

---

## 学习路径建议

**LLM 架构路线**：
1. [[Tokenization]] → [[Embedding]] — 理解输入表示。
2. [[Transformer]] → [[Attention]] → [[Multi-Head Attention]] — 理解模型主干。
3. [[RoPE]] → [[Grouped-Query Attention]] → [[Multi-head Latent Attention]] — 理解现代架构优化。

**训练目标路线**：
1. [[Entropy]] → [[Cross-Entropy Loss]] → [[KL Divergence]] — 建立信息论基础。
2. [[Adam]] → [[AdamW]] — 理解参数更新。
3. [[Mixed Precision Training]] → [[GPU Memory Calculation]] — 进入训练工程。

**RLHF 路线**：
1. [[Markov Decision Process]] → [[Value Function]] → [[Policy Gradient]] — 建立 RL 框架。
2. [[Actor-Critic]] → [[GAE]] → [[PPO]] — 理解 RLHF 中的策略优化。
3. [[Bradley-Terry Model]] → [[Reward Model]] → [[DPO]] — 理解偏好建模和 direct alignment。

---

## 相关 MOC

- [[02-MOC - Attention|MOC - Attention]] — Transformer 中 Attention 与位置编码的深入导航
- [[06-MOC - Reinforcement Learning|MOC - Reinforcement Learning]] — RL 算法从基础到 LLM alignment 的完整路径
- [[03-MOC - Distributed Training|MOC - Distributed Training]] — 训练效率、显存和并行策略
- [[07-MOC - Post-training|MOC - Post-training]] — SFT、RLHF、DPO、Online RL 与 Agentic RL
