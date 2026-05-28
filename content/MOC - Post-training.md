---
description: Post-training 领域导航：SFT、RLHF、DPO 及各类对齐方法
type: moc
created: 2025-01-25
updated: 2026-01-28T23:26
---

# MOC - Post-training

Post-training 是指在预训练（pretraining）之后，通过各种技术使模型更好地遵循指令、对齐人类偏好、提升安全性的阶段。这是当前 LLM 研究的核心领域之一。

---

## 概览

```
Pretrained Model
       ↓
   [[SFT]] (监督微调)
       ↓
   对齐方法
   ├── [[RLHF]] (强化学习路线)
   │   ├── [[Reward Model]]
   │   └── [[PPO]]
   │
   └── Direct Alignment (直接对齐路线)
       ├── [[DPO]] ← 开创性工作
       ├── [[IPO]]
       ├── [[KTO]]
       ├── [[ORPO]]
       └── [[SimPO]]
```

---

## 监督微调 (SFT)

Post-training 的第一步，通过高质量指令数据微调模型。

- [[SFT]] — 监督微调基础
- [[Instruction Tuning]] — 指令微调
- [[Chat Template]] — 对话格式与特殊 token
- [[SFT 数据质量]] — 数据筛选与构建

---

## RLHF 路线

基于强化学习的对齐方法，OpenAI 的 InstructGPT/ChatGPT 采用此路线。

### 核心组件
- [[RLHF]] — 整体框架
- [[Reward Model]] — 奖励模型的训练与设计
- [[PPO]] — 近端策略优化算法
- [[KL 散度]] — 约束策略偏离

### 关键概念
- [[Bradley-Terry Model]] — 偏好建模
- [[Preference Data]] — 偏好数据的构建
- [[Reward Hacking]] — 奖励欺骗问题
- [[KL Penalty]] — KL 惩罚的作用与调节

### 进阶话题
- [[Process Reward Model]] — 过程奖励 vs 结果奖励
- [[RLHF Scaling]] — RLHF 的规模化挑战
- [[MOPD]] — 多教师在线蒸馏，解决能力失衡问题
- [[Agentic RL]] — 智能体强化学习，多轮交互环境中的 RL 训练
- [[On-Policy Distillation]] — 在线蒸馏，学生在自生成序列上训练
- [[Training-Inference Mismatch]] — 训练与推理分布不匹配问题

---

## Direct Alignment 路线

绕过显式奖励模型，直接在偏好数据上优化策略。

### 核心方法
- [[DPO]] — 直接偏好优化（开创性工作）✅
- [[IPO]] — Identity Preference Optimization
- [[KTO]] — Kahneman-Tversky Optimization
- [[ORPO]] — Odds Ratio Preference Optimization
- [[SimPO]] — Simple Preference Optimization

### 方法对比
- [[DPO vs RLHF]] — 两条路线的对比分析
- [[Direct Alignment 方法对比]] — DPO 变体之间的异同

---

## 数据相关

- [[Preference Data]] — 偏好数据格式与构建
- [[Human Annotation]] — 人工标注流程
- [[AI Feedback]] — 用 AI 生成偏好数据（RLAIF）
- [[Data Quality vs Quantity]] — 数据质量与数量的权衡

---

## 评估与分析

- [[Alignment Evaluation]] — 对齐效果评估
- [[Win Rate]] — 胜率评估方法
- [[Reward Model Evaluation]] — RM 评估

---

## 关键论文

| 年份 | 论文 | 贡献 |
|------|------|------|
| 2022 | [[InstructGPT]] | RLHF 的工业化实践 |
| 2023 | [[DPO (2023)]] | 直接偏好优化 |
| 2023 | [[Constitutional AI (2023)]] | 基于原则的自我改进 |
| 2024 | [[IPO (2024)]] | 解决 DPO 的过拟合问题 |
| 2024 | [[KTO (2024)]] | 不需要成对偏好数据 |

---

## 学习路径建议

**入门路线：**
1. [[SFT]] — 理解微调基础
2. [[RLHF]] — 理解经典框架
3. [[DPO]] — 理解现代简化方法

**深入路线：**
1. 数学基础：[[KL 散度]]、[[Bradley-Terry Model]]
2. 算法细节：[[PPO]]、[[Reward Model]]
3. 前沿方法：[[IPO]]、[[KTO]]、[[SimPO]]
4. 实践问题：[[Reward Hacking]]、[[数据质量]]

---

## 待探索问题

- Online vs Offline preference learning
- Iterative DPO / Online DPO
- Multi-turn RLHF
- Constitutional AI 与 RLHF 的结合
