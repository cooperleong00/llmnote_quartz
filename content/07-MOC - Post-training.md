---
description: Post-training 领域导航：SFT、RLHF、Direct Alignment、Online RL、On-Policy Distillation 与 Agentic RL
type: moc
aliases:
  - MOC - Post-training
tags:
  - post-training
  - alignment
  - reinforcement-learning
created: 2025-01-25
updated: 2026-05-31T17:54
---

# MOC - Post-training

Post-training 连接预训练模型和可用助手能力。它通过 SFT、偏好建模、RL、蒸馏、数据合成和环境交互，把基础模型转化为能遵循指令、解决复杂任务并适配实际产品形态的模型。

---

## 概览

```
Post-training
├── 指令与能力初始化
│   ├── [[SFT]] — 指令遵循基础
│   ├── [[SFT-then-RL]] — reasoning model 两阶段 recipe
│   ├── [[Data Synthesis]] — 合成训练数据
│   └── [[Mid-Training]] — pretraining 与 post-training 的桥梁
│
├── 经典 RLHF
│   ├── [[RLHF]] — SFT → RM → RL
│   ├── [[Preference Data]] — 偏好数据
│   ├── [[Reward Model]] / [[Process Reward Model]]
│   └── [[PPO]] / [[RLAIF]] / [[Constitutional AI]]
│
├── Direct Alignment
│   ├── [[DPO]] — 直接偏好优化
│   ├── [[IPO]] / [[KTO]] / [[ORPO]] / [[SimPO]]
│   └── [[Bradley-Terry Model]] / [[KL Divergence]]
│
├── Online RL for Reasoning
│   ├── [[GRPO]] / [[DAPO]] / [[Dr. GRPO]]
│   ├── [[GSPO]] / [[CISPO]] / [[SAPO]] / [[MIS-PO]]
│   └── [[VAPO (2025)]] / [[VC-PPO]]
│
├── On-Policy Distillation
│   ├── [[On-Policy Distillation]]
│   ├── [[MOPD]] / [[RLSD]] / [[TIP]] / [[TCOD]]
│   └── [[Post-Training Phase Transition]]
│
└── Agentic RL
    ├── [[Agentic RL]] / [[ASTRA]]
    ├── [[GiGPO]] / [[SAMPO]] / [[EMPG]]
    └── [[ARLArena (2026)]] / [[RAGEN-2 (2025)]]
```

---

## 指令与能力初始化

[[SFT]] 用高质量指令数据建立“会按指令回答”的基础行为。对于 reasoning model，[[SFT-then-RL]] 先用高质量推理轨迹建立可解题策略，再用可验证奖励强化已有推理能力。

[[Data Synthesis]] 解决高质量人类数据稀缺问题，常用于生成指令、推理轨迹、偏好对和 agentic 交互数据。[[Mid-Training]] 位于 pretraining 和 post-training 之间，常通过高质量 tokens、学习率退火和领域数据平衡，为后续 RL 创建更合适的权重状态。

| 笔记 | 角色 | 连接 |
|------|------|------|
| [[SFT]] | 指令遵循和格式学习 | [[Cross-Entropy Loss]], [[Preference Data]] |
| [[SFT-then-RL]] | reasoning model 的两阶段训练 recipe | [[GRPO]], [[VAPO (2025)]], verifiable reward |
| [[Data Synthesis]] | 扩展训练数据来源 | [[Self-Play]], [[ASTRA]], synthetic data |
| [[Mid-Training]] | 连接预训练和后训练 | [[PRISM (2026)]], [[LLM Mid-Training Survey (2024)]] |

---

## 经典 RLHF

[[RLHF]] 的经典流程是 SFT → [[Reward Model]] → RL 优化。[[Preference Data]] 通过成对比较或其他反馈形式表达偏好，[[Bradley-Terry Model]] 将比较结果建模为胜率，Reward Model 再把偏好转为标量奖励。

| 组件 | 作用 | 学习重点 |
|------|------|----------|
| [[Preference Data]] | 偏好信号来源 | 数据质量、标注一致性、pairwise/pointwise |
| [[Reward Model]] | 把偏好转成可优化奖励 | [[Bradley-Terry Model]], reward overoptimization |
| [[Process Reward Model]] | 对中间步骤给奖励 | [[Credit Assignment]], reasoning supervision |
| [[PPO]] | 稳定策略更新 | clip objective, [[KL Divergence]], [[GAE]] |
| [[RLAIF]] | 用 AI 反馈替代部分人类反馈 | 可扩展性、feedback bias |
| [[Constitutional AI]] | 用原则指导自我批评和修正 | safety、AI feedback、alignment |

[[Reward Hacking]] 是 RLHF 的核心失败模式：模型学会利用奖励模型漏洞获得高分。它与 [[Entropy Collapse]]、[[Training-Inference Mismatch]] 一起构成后训练稳定性主线。

---

## Direct Alignment

Direct Alignment 直接在偏好数据上优化策略，减少显式 Reward Model 与在线 RL 环节。[[DPO]] 是这条路线的代表方法，它用 [[Bradley-Terry Model]] 和 KL-regularized reward 的闭式关系，将偏好优化写成监督学习形式。

| 方法 | 核心改动 | 适合关注 |
|------|----------|----------|
| [[DPO]] | 偏好数据直接优化策略 | RLHF 简化、reference model、BT 假设 |
| [[IPO]] | identity mapping 正则化 | DPO 过拟合与偏好噪声 |
| [[KTO]] | pointwise 反馈 | thumbs up/down 数据 |
| [[ORPO]] | SFT 与偏好对齐合并 | 单阶段训练 |
| [[SimPO]] | 去掉 reference model 并引入 length-normalized reward | 简化训练、长度偏置 |

Direct Alignment 适合离线偏好数据充足的场景；当目标能力依赖模型自生成探索、长链推理或环境交互时，online RL 的价值会上升。

---

## Online RL for Reasoning

reasoning model 的兴起让 online RL 重新成为 post-training 的中心。核心变化是训练数据来自当前策略 rollout，奖励常来自 verifier 或环境反馈。

### Value-model-free 路线

[[GRPO]] 用 group relative advantage 替代 value function，降低 PPO 中 critic 的显存和训练成本。后续方法围绕稳定性、长度偏差、importance sampling 和 entropy collapse 展开：

- [[DAPO]] — 用 Clip-Higher、Dynamic Sampling 等技术缓解 [[Entropy Collapse]]。
- [[Dr. GRPO]] — 移除长度和标准差归一化，修正优化偏差。
- [[GSPO]] — 从 token-level IS 转向 sequence-level IS，改善 MoE RL 稳定性。
- [[CISPO]] — clip importance sampling weights，保留低概率重要 token 的梯度。
- [[SAPO]] — 用平滑温度门替代硬裁剪，兼顾序列级连贯性和 token 级适应。
- [[MIS-PO]] — 用二元掩码过滤 off-distribution 样本，控制 off-policy 方差。
- [[Soft Clipping]] / [[DGPO]] — 处理 hard clipping 的边界梯度和探索损失。

### Value-model-based 路线

[[VAPO (2025)]] 重新强调 value model 在 long-CoT 中的作用。它通过 Length-Adaptive GAE、Value Pretraining 等方法增强 token-level credit assignment。[[VC-PPO]] 是这一路线的重要前置工作。

---

## On-Policy Distillation

[[On-Policy Distillation]] 让学生从自身策略采样轨迹，并在这些轨迹上接收教师或自反馈信号。它缓解自回归生成中的 exposure bias，也连接 SFT、RL 和 distillation 的阶段切换。

| 方法 | 信号来源 | 关注点 |
|------|----------|--------|
| [[MOPD]] | 多个领域教师的 token-level KL 奖励 | 能力失衡、教师路由 |
| [[RLSD]] | 自我蒸馏得到 token-level credit weights | 密集监督与 verifier reward 结合 |
| [[TIP]] | student entropy 与 teacher-student divergence | 选择高价值 token 监督 |
| [[TCOD]] | 按轨迹深度推进的时间课程 | 多轮 agent 跨轮错误累积 |
| [[Post-Training Phase Transition]] | SFT、RL、OPD 的阶段分析 | 采样分布、性能上限、梯度几何 |

综述入口是 [[A Survey of On-Policy Distillation for Large Language Models (2026)]]。

---

## Agentic RL

[[Agentic RL]] 将 post-training 从单轮回答扩展到多轮环境交互。模型需要学会观察、计划、调用工具、处理失败并在延迟奖励下完成任务。

| 笔记 | 核心贡献 | 关联问题 |
|------|----------|----------|
| [[ASTRA]] | 自动合成 agentic 轨迹和 RL 环境 | 数据与环境生成 |
| [[GiGPO]] | episode-level + step-level 两层分组 | multi-turn credit assignment |
| [[SAMPO]] | sequence-level clipping、环境级优势和 dynamic filtering | Agentic RL 稳定性 |
| [[EMPG]] | entropy 调制 policy gradient | long-horizon credit assignment |
| [[ARLArena (2026)]] | 统一分析 agentic RL 训练不稳定性 | benchmark 与算法诊断 |
| [[RAGEN-2 (2025)]] | template collapse 诊断与 SNR-Aware Filtering | multi-turn collapse |
| [[DAgger]] | 在学习者访问状态上查询教师 | agent imitation 与 covariate shift |

Agentic RL 与 [[05-MOC - RAG|MOC - RAG]] 的连接点是 retrieval/tool use 成为环境动作；与 [[03-MOC - Distributed Training|MOC - Distributed Training]] 的连接点是 rollout 系统和异步训练框架。

---

## 稳定性与失败模式

| 问题 | 表现 | 相关笔记 |
|------|------|----------|
| Reward hacking | 高 reward 低质量输出 | [[Reward Hacking]], [[Reward Model]] |
| Entropy collapse | 策略过早变窄，探索能力下降 | [[Entropy Collapse]], [[DAPO]], [[SAC]] |
| Training-inference mismatch | rollout logprob 与训练 logprob 不一致 | [[Training-Inference Mismatch]], [[IcePop]], [[Rollout Routing Replay]] |
| Credit assignment | 序列级奖励难以分配到关键步骤 | [[Credit Assignment]], [[Process Reward Model]], [[VAPO (2025)]] |
| Off-policy drift | 数据来自旧策略或异步 rollout | [[Importance Sampling]], [[MIS-PO]], [[Near-Future Policy Optimization]] |

---

## 关键论文与报告入口

| 主题 | 入口 |
|------|------|
| RLHF 工业化实践 | [[InstructGPT]] |
| GRPO 来源与数学推理 RL | [[DeepSeekMath (2024)]] |
| RL 算法演进串讲 | [[LLM RL Algorithm Evolution]] |
| RL 稳定性理论 | [[Stabilizing Reinforcement Learning with LLMs (2025)]] |
| RL compute scaling | [[The Art of Scaling RL Compute for LLMs (2025)]] |
| On-Policy Distillation 综述 | [[A Survey of On-Policy Distillation for Large Language Models (2026)]] |
| 开源模型与 post-training 实践 | [[DeepSeek-V3 (2024)]], [[Kimi K2 (2025)]], [[Qwen3 (2025)]], [[DeepSeek-V4 (2026)]] |

---

## 学习路径建议

**入门路线**：
1. [[SFT]] → [[Preference Data]] → [[Reward Model]] — 理解后训练数据和奖励。
2. [[RLHF]] → [[PPO]] — 理解经典三阶段对齐。
3. [[DPO]] → [[IPO]] / [[KTO]] / [[SimPO]] — 理解 direct alignment。

**Reasoning RL 路线**：
1. [[LLM RL Algorithm Evolution]] — 先建立算法地图。
2. [[GRPO]] → [[DAPO]] → [[Dr. GRPO]] → [[GSPO]] — 理解 critic-free 路线。
3. [[PPO]] → [[GAE]] → [[VAPO (2025)]] — 理解 value-model-based 路线。

**Agentic 路线**：
1. [[Agentic RL]] → [[ASTRA]] — 理解环境和轨迹生成。
2. [[GiGPO]] → [[SAMPO]] → [[EMPG]] — 理解多轮 credit assignment 与稳定性。
3. [[ROLL (2025)]] → [[ROLL Flash (2025)]] → [[DORA]] — 理解系统层的 rollout 加速。

---

## 相关 MOC

- [[06-MOC - Reinforcement Learning|MOC - Reinforcement Learning]] — 从经典 RL 到 LLM RL 的算法主线
- [[01-MOC - Foundations|MOC - Foundations]] — KL、Bradley-Terry、Policy Gradient、PPO 的前置概念
- [[03-MOC - Distributed Training|MOC - Distributed Training]] — RL 训练系统、rollout 编排与显存效率
- [[05-MOC - RAG|MOC - RAG]] — Agentic RAG 与工具/检索环境
