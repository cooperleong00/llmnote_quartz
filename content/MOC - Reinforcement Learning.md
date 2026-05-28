---
type: moc
description: 强化学习知识导航，从经典 RL 基础到 LLM alignment 的完整学习路径
aliases:
  - MOC - RL
  - 强化学习导航
tags:
  - reinforcement-learning
  - moc
created: 2026-02-24
updated: 2026-04-26T22:34
---

# MOC - Reinforcement Learning

强化学习（Reinforcement Learning）是 agent 通过与环境交互、从奖励信号中学习最优行为策略的框架。在 LLM 时代，RL 的角色发生了根本转变：从训练 agent 玩游戏，变成了对齐语言模型与人类偏好的核心工具。这个 MOC 组织了从经典 RL 理论到 LLM alignment 前沿的完整知识网络。

---

## 概览

```
Reinforcement Learning
├── 基础理论
│   ├── [[Markov Decision Process]] — RL 的数学框架 ✅
│   ├── [[Value Function]] — 状态/动作的长期价值评估 ✅
│   ├── [[TD Learning]] — 用当前估计更新估计的学习方法 ✅
│   ├── [[Policy Gradient]] — 直接优化策略参数 ✅
│   ├── [[Importance Sampling]] — off-policy 学习的理论基础 ✅
│   └── [[On-Policy vs Off-Policy]] — 两大学习范式的核心区别 ✅
│
├── 经典算法
│   ├── Value-based
│   │   └── [[DQN]] — 深度 Q 网络 ✅
│   ├── Policy-based
│   │   └── [[REINFORCE]] — 最基础的 policy gradient 算法 ✅
│   ├── Actor-Critic 系列
│   │   ├── [[Actor-Critic]] — 策略网络 + 价值网络 ✅
│   │   ├── [[A2C]] — 同步并行 + Advantage function ✅
│   │   ├── [[GAE]] — 平衡 bias-variance 的优势估计 ✅
│   │   ├── [[TRPO]] — KL 约束的信任域方法 ✅
│   │   ├── [[PPO]] — clip 机制的工业标准算法 ✅
│   │   └── [[SAC]] — 最大熵 RL，平衡探索与利用 ✅
│   └── 核心问题
│       └── [[Credit Assignment]] — 延迟奖励的分配难题 ✅
│
├── LLM 中的 RL（Alignment）
│   ├── 经典框架
│   │   ├── [[RLHF]] — SFT → RM → RL 三阶段对齐 ✅
│   │   ├── [[Reward Model]] — 将人类偏好转化为标量信号 ✅
│   │   ├── [[Process Reward Model]] — 步骤级奖励信号 ✅
│   │   ├── [[RLAIF]] — AI 反馈替代人类反馈 ✅
│   │   └── [[Constitutional AI]] — 原则驱动的自我改进 ✅
│   ├── Direct Alignment（绕过 RM）
│   │   ├── [[DPO]] — 偏好数据直接优化 ✅
│   │   ├── [[SimPO]] — 去掉 reference model 的简化版 ✅
│   │   ├── [[IPO]] — 解决 DPO 过拟合的正则化变体 ✅
│   │   ├── [[KTO]] — 只需 pointwise 反馈 ✅
│   │   └── [[ORPO]] — SFT + 对齐单阶段合并 ✅
│   └── Online RL 算法（LLM 专用）
│       ├── Value-model-free
│       │   ├── [[GRPO]] — group relative advantage 替代 critic ✅
│       │   ├── [[DAPO]] — 解决 entropy collapse 的 GRPO 改进 ✅
│       │   ├── [[Dr. GRPO]] — 修正 GRPO 优化偏差的无偏变体 ✅
│       │   ├── [[GSPO]] — 序列级重要性采样替代 token 级 ✅
│       │   ├── [[CISPO]] — clip IS weights 保留所有 token 梯度 ✅
│       │   ├── [[SAPO]] — 平滑温度门替代硬裁剪 ✅
│       │   └── [[MIS-PO]] — 二元掩码解决 off-policy 方差 ✅
│       └── Value-model-based
│           └── [[VAPO (2025)]] — Length-adaptive GAE 解决 long-CoT 三大挑战 ✅
│
├── 训练稳定性
│   ├── [[Entropy Collapse]] — 策略熵急剧下降的核心挑战 ✅
│   ├── [[Reward Hacking]] — 欺骗奖励函数的失败模式 ✅
│   ├── [[Training-Inference Mismatch]] — 概率分布不一致 ✅
│   ├── [[IcePop]] — MoE 双边校准解决方案 ✅
│   └── [[Rollout Routing Replay]] — 路由重放解决 MoE 不一致 ✅
│
└── 前沿方向
    ├── [[Agentic RL]] — 训练 LLM Agent 的 RL 方法 ✅
    ├── [[RLTF (2025)]] — 文本反馈替代标量奖励 ✅
    ├── [[MOPD]] — 多教师在线蒸馏 ✅
    ├── [[JACKPOT]] — 小模型 rollout 训练大模型 ✅
    └── [[ASTRA]] — 自动合成 agentic 轨迹 ✅
```

> ✅ 已有笔记

> [!tip] 串讲入口
> 想从头到尾理解 RL 算法的演进逻辑？→ [[LLM RL Algorithm Evolution|LLM RL 算法大串讲]]

---

## 基础理论

RL 的理论基础可以用一条线串起来：[[Markov Decision Process]] 定义了问题框架（状态、动作、奖励、转移），[[Value Function]] 定义了"什么是好的"（长期累积奖励的期望），而 [[Policy Gradient]] 和 [[TD Learning]] 分别代表了两种学习路径——直接优化策略 vs 通过价值函数间接改进策略。

### 核心概念

| 概念 | 核心作用 | 为什么重要 |
|------|----------|------------|
| [[Markov Decision Process]] | 五元组 $(S, A, P, R, \gamma)$ 定义序贯决策 | 所有 RL 算法的数学基础 |
| [[Value Function]] | 评估状态/动作的长期价值 $V(s)$, $Q(s,a)$ | Critic 网络的优化目标 |
| [[TD Learning]] | 用 bootstrap 估计更新价值函数 | Actor-Critic 和 GAE 的理论基础 |
| [[Policy Gradient]] | 直接对策略参数求梯度 | PPO/GRPO 等现代算法的理论起点 |
| [[Importance Sampling]] | 用一个分布的样本估计另一个分布的期望 | Off-policy 学习和 PPO clip 的理论基础 |
| [[On-Policy vs Off-Policy]] | 数据收集策略与优化策略是否相同 | 决定 sample efficiency 与 stability 的根本权衡 |

> [!intuition] 两条学习路径
> - **Value-based**（DQN）：先学"什么状态/动作好"，再据此行动。适合离散动作空间。
> - **Policy-based**（REINFORCE → PPO）：直接学"怎么行动"。适合连续/高维动作空间，也是 LLM alignment 的主流路径。
> - **Actor-Critic**：两条路径的融合——Actor 学策略，Critic 学价值，互相辅助。

---

## 经典算法

### Value-based：从表格到深度网络

[[DQN]] 是深度 RL 的里程碑，用神经网络逼近 Q 函数，通过 Experience Replay 和 Target Network 解决了函数逼近下 Q-Learning 的不稳定性。但 DQN 只能处理离散动作空间，无法直接用于 LLM（token 空间虽然离散但维度极高）。

### Policy-based：从 REINFORCE 到信任域

[[REINFORCE]] 是最基础的 policy gradient 实现，用 Monte Carlo 采样估计梯度。简单但方差极高，需要完整 episode 才能更新。理解 REINFORCE 是理解所有后续算法的起点。

### Actor-Critic 演进链

这是 LLM RL 最核心的算法族，演进逻辑清晰：

```
REINFORCE（高方差）
    ↓ 引入 Critic 降低方差
Actor-Critic → A2C（同步并行 + Advantage）
    ↓ 引入 GAE 平衡 bias-variance
GAE（λ 参数控制 TD(0) 到 MC 的连续谱）
    ↓ 限制策略更新幅度
TRPO（KL 硬约束）→ PPO（clip 软约束）
    ↓ 引入最大熵目标
SAC（自动温度调节，平衡探索与利用）
```

| 算法 | 核心创新 | 关键权衡 |
|------|----------|----------|
| [[Actor-Critic]] | 策略网络 + 价值网络协同 | 比 REINFORCE 方差低，但引入 bias |
| [[A2C]] | 同步并行 + Advantage function | A3C 的确定性版本，更易调试 |
| [[GAE]] | $\lambda$ 参数平衡 bias-variance | $\lambda=0$ 低方差高偏差，$\lambda=1$ 高方差低偏差 |
| [[TRPO]] | KL 散度硬约束保证单调改进 | 理论优美但计算昂贵（需要二阶优化） |
| [[PPO]] | clip 机制替代 KL 约束 | 简单高效，RLHF 的工业标准 |
| [[SAC]] | 最大熵目标 + 自动温度 $\alpha$ | 探索能力强，但主要用于连续控制 |

### 核心难题

[[Credit Assignment]] 是 RL 的根本挑战：当奖励延迟到序列末尾时，如何判断哪个中间步骤贡献了最终结果？在 LLM 中，这表现为如何将 sequence-level reward 分配到每个 token。[[Process Reward Model]] 通过对每个推理步骤给出奖励信号来缓解这个问题。

---

## LLM 中的 RL

RL 在 LLM 中的应用可以分为三条路线，代表了不同的设计哲学。

### 路线一：经典 RLHF 框架

[[RLHF]] 建立了 SFT → Reward Model → RL 优化的三阶段范式，由 [[InstructGPT]] 首次大规模验证。这条路线的核心组件：

- [[Reward Model]] — 基于 [[Bradley-Terry Model]] 将人类偏好转化为标量奖励
- [[Process Reward Model]] — 步骤级奖励，解决 outcome RM 的 [[Credit Assignment]] 问题
- [[PPO]] — 策略优化算法，通过 clip 机制和 KL 惩罚稳定训练
- [[RLAIF]] — 用 AI 反馈替代人类反馈，解决标注成本和可扩展性
- [[Constitutional AI]] — Anthropic 的原则驱动对齐，AI 自我批评和修正

### 路线二：Direct Alignment（绕过 RM）

[[DPO]] 开创了直接从偏好数据优化策略的范式，将 RLHF 简化为单阶段监督学习。后续工作沿不同方向改进：

| 方法 | 相对 DPO 的改进 | 核心取舍 |
|------|-----------------|----------|
| [[DPO]] | 基线：偏好数据 → 闭式解 | 需要 reference model，假设 Bradley-Terry |
| [[SimPO]] | 去掉 reference model | 更简单，但需要 length normalization |
| [[IPO]] | identity mapping 正则化 | 更稳健，不依赖 Bradley-Terry 假设 |
| [[KTO]] | 只需 pointwise 反馈 | 数据需求更低，但信号更弱 |
| [[ORPO]] | SFT + 对齐合并为一阶段 | 训练更简单，但灵活性降低 |

### 路线三：Online RL 算法（LLM 专用）

随着 reasoning model 的兴起，online RL 重新成为主流。这里存在两条技术路线的竞争：

**Value-model-free（GRPO 系列）**：

[[GRPO]] 用 group relative advantage 替代 value function，省去 Critic 网络的显存开销。但在大规模训练中暴露出多个问题，催生了一系列改进：

- [[DAPO]] — 通过 Clip-Higher、Dynamic Sampling 等四项技术解决 [[Entropy Collapse]]
- [[Dr. GRPO]] — 移除长度和标准差归一化，修正 GRPO 的优化偏差
- [[GSPO]] — 用序列级重要性采样替代 token 级，从根本上解决 MoE 训练稳定性
- [[CISPO]] — clip importance sampling weights 而非 token updates，保留低概率重要 token 的梯度
- [[SAPO]] — 用平滑温度控制门替代硬裁剪，实现序列级连贯性
- [[MIS-PO]] — 二元掩码过滤 off-distribution 样本，解决 off-policy 梯度方差
- [[RLSD]] — 用自我蒸馏信号重分配 token-level credit，并让 verifier reward 锚定更新方向
- [[Near-Future Policy Optimization]] — 用同一训练轨迹的近未来 checkpoint 注入 verifier-correct 轨迹，平衡信号质量与 off-policy 方差

**Value-model-based**：

[[VAPO (2025)]] 论证了 value-model-based 方法在 long-CoT 推理任务上具有更高的性能上限——value model 能提供 token-level 的精细 [[Credit Assignment]]，而 GRPO 系列只能做 trajectory-level 的粗粒度优化。VAPO 通过 Length-Adaptive GAE、Value-Pretraining 等技术系统性解决了 value model 在长序列上的训练挑战，在 AIME 2024 上以 60.4 分大幅超越 DAPO（50 分）。

---

## 训练稳定性

大规模 RL 训练面临的核心挑战，尤其在 MoE 和 long CoT 场景下：

| 问题 | 表现 | 解决方案 |
|------|------|----------|
| [[Entropy Collapse]] | 策略熵急剧下降，丧失探索能力 | [[DAPO]] 的 Clip-Higher、[[SAC]] 的最大熵目标 |
| [[Reward Hacking]] | 模型欺骗奖励函数，高分低质 | 更好的 RM、KL 约束、多维奖励 |
| [[Training-Inference Mismatch]] | 训练和推理的概率分布不一致 | [[IcePop]]、[[Rollout Routing Replay]]、[[GSPO]] |

[[Stabilizing Reinforcement Learning with LLMs (2025)]] 提供了理论框架，证明 token-level 目标是 sequence-level 奖励的一阶近似，其有效性依赖于同时最小化训练-推理差异和策略陈旧度。

---

## 前沿方向

### Agentic RL

[[Agentic RL]] 将 RL 从单轮对齐扩展到多轮交互场景，训练 LLM 在工具调用、代码执行、网页浏览等环境中完成复杂任务。[[ASTRA]] 提供了自动合成 agentic 轨迹和 RL 环境的端到端框架。

### 反馈信号的演进

从标量奖励到更丰富的反馈形式：
- [[RLTF (2025)]] — 用文本反馈（text feedback）替代标量奖励，提供更细粒度的学习信号
- [[MOPD]] — 多教师在线蒸馏，通过领域专家提供 token-level KL 奖励

### 训练效率

- [[JACKPOT]] — 通过 optimal budget rejection sampling 让小模型 rollout 训练大模型，降低 rollout 成本
- [[On-Policy Distillation]] — 比 RL 高效 50-100 倍的替代范式

---

## 学习路径建议

**入门路线**（理解 RL 基础）：
1. [[Markov Decision Process]] → [[Value Function]] → [[TD Learning]] — 建立基本框架
2. [[Policy Gradient]] → [[REINFORCE]] — 理解策略优化的起点
3. [[Actor-Critic]] → [[GAE]] → [[PPO]] — 掌握核心算法演进

**RLHF 路线**（理解 LLM 对齐）：
1. [[RLHF]] → [[Reward Model]] → [[PPO]] — 经典三阶段框架
2. [[DPO]] → [[SimPO]] / [[KTO]] — Direct Alignment 简化路线
3. [[Entropy Collapse]] → [[Reward Hacking]] — 理解失败模式

**前沿路线**（理解最新进展）：
1. [[GRPO]] → [[DAPO]] → [[Dr. GRPO]] — Value-model-free 算法族
2. [[PPO]] → [[GAE]] → [[VAPO (2025)]] — Value-model-based 路线
3. [[Training-Inference Mismatch]] → [[IcePop]] — 大规模训练挑战
4. [[Agentic RL]] → [[ASTRA]] — Agent 训练方向

---

## 相关 MOC

- [[MOC - Post-training]] — Post-training 全景（SFT、RLHF、DPO 及对齐方法）
- [[MOC - Foundations]] — 数学基础（KL Divergence、Bradley-Terry Model 等 RL 的前置知识）
- [[MOC - Inference]] — 推理优化（与训练互补的另一面）
