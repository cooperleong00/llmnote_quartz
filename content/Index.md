---
type: moc
created: 2025-01-25
updated: 2026-01-27T23:17
---

# LLM 全栈知识库

这是一个用于 LLM 研究/工程岗位面试准备的知识库。通过持续交互学习的方式迭代增长。

---

## 知识地图

### 核心领域

- [[MOC - Foundations]] — Transformer、Attention、优化、Scaling Laws ✅
- [[MOC - Attention]] — 注意力机制、MHA/MQA/GQA、位置编码 ✅
- [[MOC - Pretraining]] — 数据、训练策略、架构
- [[MOC - Post-training]] — SFT、RLHF、DPO 及对齐方法 ✅
- [[MOC - Inference]] — 量化、KV Cache、Serving ✅
- [[MOC - Evaluation]] — Benchmark、评估方法
- [[MOC - Applications]] — RAG、Agents、Tool Use

---

## 当前计划

### 阶段一：Post-training 核心

优先构建 post-training 相关知识，这是当前重点：

**对齐方法演进线：**
1. [[SFT]] — 监督微调基础 ✅
2. [[RLHF]] — 强化学习人类反馈 ✅
3. [[Reward Model]] — 奖励模型 ✅
4. [[PPO]] — 近端策略优化 ✅
5. [[DPO]] — 直接偏好优化 ✅
6. [[IPO]]、[[KTO]]、[[ORPO]]、[[SimPO]] — DPO 变体

**关键概念：**
- [[KL Divergence]] — 贯穿 RLHF 和 DPO 的核心概念
- [[Bradley-Terry Model]] — 偏好建模基础
- [[Preference Data]] — 偏好数据的构建与质量

### 阶段二：按需补充 Foundations

当 post-training 笔记引用到基础概念时，再回填：
- [[Transformer]] ✅
- [[Attention]] ✅
- [[Cross-Entropy Loss]]
- ...

---

## 快速入口

- **最近更新**：[[SFT]]、[[Transformer]]、[[MOC - Inference]]、[[MOC - Foundations]]
- **核心概念**：[[DPO]]、[[RLHF]]、[[PPO]]、[[Attention]]、[[Multi-Head Attention]]、[[KV Cache]]
- **前沿方法**：[[GRPO]]、[[GSPO]]、[[DAPO]]、[[SAPO]]、[[Dr. GRPO]]
- **知识空缺**：查看 Graph View 中的未创建链接

---

## 使用指南

参见 [[CLAUDE]] 了解本知识库的规范和交互方式。
