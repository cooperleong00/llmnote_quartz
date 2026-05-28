---
type: moc
description: 强化学习知识导航，从经典 RL 基础到 LLM reasoning、RLHF 和 Agentic RL
aliases:
  - MOC - RL
  - 强化学习导航
tags:
  - reinforcement-learning
  - moc
created: 2026-02-24
updated: 2026-05-29T00:00
---

# MOC - Reinforcement Learning

强化学习（Reinforcement Learning）研究 agent 如何通过与环境交互、接收奖励并改进策略。LLM 时代的 RL 同时承担三类角色：经典序贯决策理论、RLHF 中的偏好优化、reasoning 和 agentic 任务中的在线探索。

---

## 概览

```
Reinforcement Learning
├── 基础理论
│   ├── [[Markov Decision Process]] — 序贯决策数学框架
│   ├── [[Value Function]] — 长期价值估计
│   ├── [[TD Learning]] — bootstrap 价值学习
│   ├── [[Policy Gradient]] — 直接优化策略
│   ├── [[Importance Sampling]] — off-policy 估计
│   └── [[On-Policy vs Off-Policy]] — 数据策略与优化策略的关系
│
├── 经典算法
│   ├── Value-based: [[Q-Learning]] → [[DQN]] → [[Experience Replay]]
│   ├── Policy-based: [[REINFORCE]] → [[RLOO]] → [[ReMax]]
│   └── Actor-Critic: [[Actor-Critic]] → [[A2C]] / [[A3C]] → [[GAE]] → [[TRPO]] → [[PPO]] → [[SAC]]
│
├── LLM Alignment
│   ├── [[RLHF]] — SFT → RM → RL
│   ├── [[Reward Model]] / [[Process Reward Model]]
│   ├── [[DPO]] / [[IPO]] / [[KTO]] / [[ORPO]] / [[SimPO]]
│   └── [[RLAIF]] / [[Constitutional AI]]
│
├── Reasoning RL
│   ├── [[GRPO]] / [[DAPO]] / [[Dr. GRPO]]
│   ├── [[GSPO]] / [[CISPO]] / [[SAPO]] / [[MIS-PO]]
│   ├── [[REINFORCE++]] / [[VC-PPO]] / [[VAPO (2025)]]
│   └── [[Near-Future Policy Optimization]] / [[RLSD]]
│
├── 稳定性与系统
│   ├── [[Entropy Collapse]] / [[Reward Hacking]]
│   ├── [[Training-Inference Mismatch]]
│   ├── [[IcePop]] / [[Rollout Routing Replay]]
│   └── [[ROLL (2025)]] / [[ROLL Flash (2025)]] / [[DORA]]
│
└── Agentic RL
    ├── [[Agentic RL]] / [[ASTRA]]
    ├── [[GiGPO]] / [[SAMPO]] / [[EMPG]]
    └── [[ARLArena (2026)]] / [[Agentic RL Survey (2025)]]
```

> [!tip] 串讲入口
> 想按算法演进理解 LLM RL，可以先读 [[LLM RL Algorithm Evolution]]，再回到本 MOC 查具体概念。

---

## 基础理论

[[Markov Decision Process]] 定义 RL 问题，[[Value Function]] 衡量状态或动作的长期价值，[[Policy Gradient]] 直接优化策略参数，[[TD Learning]] 用 bootstrap 更新价值估计。[[On-Policy vs Off-Policy]] 和 [[Importance Sampling]] 决定样本能否跨策略复用。

| 概念 | 核心作用 | 在 LLM RL 中的连接 |
|------|----------|--------------------|
| [[Markov Decision Process]] | 定义状态、动作、奖励、转移 | 将对话、工具调用、代码执行形式化为环境交互 |
| [[Value Function]] | 估计长期回报 | PPO critic、VAPO value model |
| [[TD Learning]] | 用当前估计更新当前估计 | critic 训练、GAE |
| [[Policy Gradient]] | 对策略 logprob 加权更新 | REINFORCE、PPO、GRPO 的共同基础 |
| [[Importance Sampling]] | 修正采样策略和目标策略差异 | PPO clip、GSPO、off-policy RL |
| [[Credit Assignment]] | 将延迟奖励分配到关键动作 | long-CoT、agentic multi-turn、PRM |

---

## 经典算法

### Value-based

[[Q-Learning]] 用 Bellman optimality 更新 $Q(s,a)$，[[DQN]] 用神经网络逼近 Q 函数，并通过 [[Experience Replay]] 和 target network 提升稳定性。这条路线适合理解 value learning、bootstrap 和样本复用。

### Policy-based

[[REINFORCE]] 用 Monte Carlo 采样估计 policy gradient，是理解 PPO/GRPO 的起点。[[RLOO]] 用 leave-one-out baseline 降低方差，[[ReMax]] 用 greedy decoding reward 作为 baseline，体现 LLM RL 中 critic-free 方法的早期思路。

### Actor-Critic

Actor-Critic 用 Actor 学策略，用 Critic 学价值。[[A2C]] 和 [[A3C]] 通过并行 worker 提升采样效率；[[GAE]] 在 bias 和 variance 之间平衡优势估计；[[TRPO]] 和 [[PPO]] 通过限制策略更新幅度稳定训练；[[SAC]] 通过最大熵目标鼓励探索。

| 算法 | 核心创新 | 适合理解 |
|------|----------|----------|
| [[REINFORCE]] | Monte Carlo policy gradient | 策略梯度基本形式 |
| [[Actor-Critic]] | 用 value baseline 降低方差 | PPO 的结构基础 |
| [[GAE]] | $\lambda$ 控制 bias-variance | token/trajectory advantage |
| [[TRPO]] | KL 约束的 trust region | 稳定策略更新 |
| [[PPO]] | clip objective | RLHF 工业标准 |
| [[SAC]] | 最大熵 RL | 探索与 entropy regularization |

---

## LLM Alignment

[[RLHF]] 将偏好数据转化为策略优化：先用 [[SFT]] 初始化行为，再训练 [[Reward Model]]，最后用 [[PPO]] 优化策略。[[Process Reward Model]] 将奖励从最终答案扩展到推理步骤，缓解 [[Credit Assignment]]。

Direct Alignment 直接在偏好数据上训练策略，减少显式 reward model 和在线 RL 环节：

| 方法 | 核心思想 | 关键依赖 |
|------|----------|----------|
| [[DPO]] | 用偏好数据直接优化策略 | [[Bradley-Terry Model]], [[KL Divergence]] |
| [[IPO]] | 用 identity mapping 正则化偏好优化 | 偏好噪声、过拟合 |
| [[KTO]] | 使用 pointwise 反馈 | thumbs up/down 数据 |
| [[ORPO]] | 将 SFT 和偏好对齐合并 | odds ratio |
| [[SimPO]] | 去掉 reference model | length normalization |

[[RLAIF]] 和 [[Constitutional AI]] 把反馈来源从人类扩展到 AI 或原则集合，提升可扩展性，也引入 feedback bias 和自我强化风险。

---

## Reasoning RL

reasoning model 的 RL 通常以可验证奖励为核心，模型从当前策略采样多条解题轨迹，再根据正确性或过程反馈更新。[[GRPO]] 是这条路线的关键节点：它用 group relative advantage 替代 value model，降低 critic 成本。

### Critic-free 路线

| 方法 | 解决的问题 | 连接 |
|------|------------|------|
| [[GRPO]] | 去掉 critic，降低 PPO 成本 | group advantage |
| [[DAPO]] | 缓解 entropy collapse 和训练不稳定 | Clip-Higher、Dynamic Sampling |
| [[Dr. GRPO]] | 修正长度和标准差归一化偏差 | 无偏 GRPO |
| [[GSPO]] | 用序列级 IS 稳定 MoE RL | [[Training-Inference Mismatch]] |
| [[CISPO]] | 保留低概率重要 token 梯度 | clip IS weights |
| [[SAPO]] | 平滑 clipping 决策 | soft temperature gate |
| [[MIS-PO]] | 控制 off-policy 方差 | 二元掩码 |
| [[REINFORCE++]] | global batch normalization | advantage 爆炸和任务过拟合 |

### Value-model-based 路线

[[VAPO (2025)]] 认为 long-CoT 推理需要更细粒度的 token-level credit assignment，通过 value model、Length-Adaptive GAE 和 Value Pretraining 改善 PPO 在长序列上的表现。[[VC-PPO]] 是理解这条路线的前置方法。

[[Near-Future Policy Optimization]] 用近未来 checkpoint 提供 verifier-filtered 正确轨迹，在信号质量和 off-policy 方差之间取得折中。[[RLSD]] 用自我蒸馏信号生成 token-level credit weights，让 verifier reward 决定更新方向。

---

## 稳定性与失败模式

大规模 LLM RL 的难点通常出现在训练动态和系统实现的交界处。

| 问题 | 表现 | 代表笔记 |
|------|------|----------|
| [[Entropy Collapse]] | 策略熵快速下降，探索不足 | [[DAPO]], [[SAC]] |
| [[Reward Hacking]] | 奖励高但输出质量低 | [[Reward Model]], [[RLHF]] |
| [[Training-Inference Mismatch]] | rollout 与训练 logprob 不一致 | [[IcePop]], [[Rollout Routing Replay]], [[GSPO]] |
| Off-policy drift | 旧策略数据带来梯度偏差或方差 | [[Importance Sampling]], [[MIS-PO]], [[Near-Future Policy Optimization]] |
| Long-tail rollout | 长序列生成拖慢同步训练 | [[ROLL Flash (2025)]], [[DORA]] |

[[Stabilizing Reinforcement Learning with LLMs (2025)]] 提供了理论入口：token-level 目标可以看作 sequence-level 奖励的一阶近似，其有效性依赖训练-推理差异和策略陈旧度的控制。

---

## Agentic RL

[[Agentic RL]] 把 RL 放入多轮环境：模型需要在工具、网页、代码、检索或游戏环境中行动，奖励经常延迟到任务结束后出现。

| 方法/报告 | 核心贡献 | 关注点 |
|-----------|----------|--------|
| [[ASTRA]] | 自动合成 agentic 轨迹和 RL 环境 | 环境与数据生成 |
| [[GiGPO]] | episode-level + step-level 两层分组 | 多轮信用分配 |
| [[SAMPO]] | sequence-level clipping、环境级优势、dynamic filtering | agentic 训练稳定性 |
| [[EMPG]] | entropy 调制 policy gradient | long-horizon exploration |
| [[ARLArena (2026)]] | Agentic RL 不稳定性的统一分析 | benchmark 与诊断 |
| [[Agentic RL Survey (2025)]] | POMDP 视角组织 agentic RL | 领域地图 |
| [[DAgger]] | 在学习者访问状态上查询教师 | imitation learning 和 covariate shift |

这条路线与 [[MOC - RAG]] 相连：检索、工具调用和网页浏览都可以成为环境动作。它也与 [[MOC - Distributed Training]] 相连：rollout 生成、reward 计算和 policy update 需要系统级调度。

---

## 训练系统

LLM RL 系统需要同时管理 rollout workers、policy model、reference model、reward/verifier、critic/value model 和训练器。

| 系统 | 关键点 | 连接 |
|------|--------|------|
| [[ROLL (2025)]] | 单控制器、模块化大规模 RL 框架 | 200B+ MoE RL 训练 |
| [[ROLL Flash (2025)]] | 异步训练与细粒度并行优化 | RLVR 与 agentic 加速 |
| [[DORA]] | 多版本流式 rollout 与 KV-cache 迁移 | long-tail generation bubble |

---

## 学习路径建议

**RL 基础路线**：
1. [[Markov Decision Process]] → [[Value Function]] → [[TD Learning]] — 建立序贯决策和价值学习框架。
2. [[Policy Gradient]] → [[REINFORCE]] → [[Actor-Critic]] — 理解策略更新和 baseline。
3. [[GAE]] → [[TRPO]] → [[PPO]] — 掌握 RLHF 的算法基础。

**LLM Alignment 路线**：
1. [[RLHF]] → [[Reward Model]] → [[PPO]] — 经典三阶段。
2. [[DPO]] → [[IPO]] / [[KTO]] / [[SimPO]] — Direct Alignment。
3. [[Reward Hacking]] → [[Entropy Collapse]] → [[Training-Inference Mismatch]] — 失败模式。

**Reasoning RL 路线**：
1. [[LLM RL Algorithm Evolution]] — 先建立整体演进线。
2. [[GRPO]] → [[DAPO]] → [[Dr. GRPO]] → [[GSPO]] — critic-free 路线。
3. [[PPO]] → [[GAE]] → [[VC-PPO]] → [[VAPO (2025)]] — value-model-based 路线。

**Agentic RL 路线**：
1. [[Agentic RL]] → [[Agentic RL Survey (2025)]] — 建立领域地图。
2. [[ASTRA]] → [[GiGPO]] → [[SAMPO]] → [[EMPG]] — 学习多轮训练方法。
3. [[ROLL (2025)]] → [[ROLL Flash (2025)]] → [[DORA]] — 理解系统实现。

---

## 相关 MOC

- [[MOC - Post-training]] — SFT、RLHF、DPO、OPD 与 Agentic RL 的后训练流程
- [[MOC - Foundations]] — MDP、value function、policy gradient 和 KL 的前置概念
- [[MOC - Distributed Training]] — LLM RL 训练系统和 rollout 编排
- [[MOC - RAG]] — 检索、工具调用与 agentic 环境
