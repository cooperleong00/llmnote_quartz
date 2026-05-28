---
description: 将人类偏好转化为可优化的标量信号，是 RLHF 的核心组件
type: concept
prerequisites:
  - "[[Bradley-Terry Model]]"
  - "[[Cross-Entropy Loss]]"
tags:
  - post-training
  - alignment
created: 2025-01-25
updated: 2026-02-01T01:09
---

# Reward Model

奖励模型（Reward Model, RM）是 [[RLHF]] 的核心组件，负责将人类偏好转化为可优化的标量信号。它学习预测"人类会更喜欢哪个回答"。

---

## 核心思想

> [!intuition] 直觉理解
> 人类偏好难以直接写成数学公式。Reward Model 的作用是：**学习一个函数 $r(x, y)$，使得人类偏好的回答得分更高**。
>
> 有了 RM，我们就可以用 RL 算法（如 [[PPO]]）来优化模型，让它生成高分回答。

---

## 训练方式

### 数据格式

偏好数据是三元组：$(x, y_w, y_l)$
- $x$：prompt
- $y_w$：preferred response（人类更喜欢的）
- $y_l$：rejected response（人类不喜欢的）

### 训练目标

基于 [[Bradley-Terry Model]]，假设人类偏好 $y_w$ 的概率为：

$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$

最大化似然，等价于最小化：

> [!definition] Reward Model Loss
> $$
> \mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]
> $$

> [!intuition] 直觉
> 这个 loss 鼓励 RM 给 $y_w$ 打更高的分，给 $y_l$ 打更低的分。分差越大，loss 越小。

---

## 架构设计

### 基于 LLM 的 RM

最常见的做法：在 LLM 最后加一个 **scalar head**（线性层），输出单个标量作为 reward。

```
Input: [prompt] [response]
   ↓
LLM Backbone (frozen or fine-tuned)
   ↓
Last hidden state of [EOS] token
   ↓
Linear layer → scalar reward
```

### 设计选择

| 选择 | 选项 | 权衡 |
|------|------|------|
| Backbone | 与 policy 相同 / 独立模型 | 相同更一致，独立更灵活 |
| 初始化 | 从 SFT 模型 / 从预训练模型 | SFT 模型更懂任务 |
| 训练方式 | 全参数 / LoRA | 全参数效果好，LoRA 省资源 |

---

## 偏好数据的构建

### 数据来源

1. **人工标注**：质量高，成本高
2. **AI 标注**（[[RLAIF]]）：成本低，可能有偏差
3. **隐式反馈**：用户点赞/点踩、选择哪个回答

### 标注方式

- **Pairwise comparison**：A vs B，哪个更好？（最常用）
- **Ranking**：对 K 个回答排序
- **Rating**：给每个回答打分（1-5 分）

> [!warning] 标注一致性
> 不同标注员对同一对回答可能有不同判断。需要：
> - 清晰的标注指南
> - 多人标注取共识
> - 过滤低一致性样本

---

## 核心挑战

### 1. [[Reward Hacking]]

> [!warning] 定义
> 模型学会"欺骗" RM，生成高分但低质量的回答。

**常见表现**：
- 回答变得冗长（RM 可能偏好长回答）
- 重复某些高分 pattern
- 过度使用某些词汇或格式

**缓解方法**：
- KL 惩罚（限制偏离 reference）
- RM ensemble（多个 RM 投票）
- 定期用新数据更新 RM

### 2. Reward Overoptimization

> [!warning] 定义
> 过度优化 RM 分数，导致真实质量下降。

Goodhart's Law："当一个指标成为目标，它就不再是好指标。"

**现象**：RM 分数持续上升，但人类评估分数先升后降。

### 3. 分布偏移

RM 在 SFT 模型的输出分布上训练，但 RL 优化后模型的输出分布会变化。RM 在新分布上的泛化能力可能下降。

---

## Process RM vs Outcome RM

| 类型 | 评估对象 | 优点 | 缺点 |
|------|----------|------|------|
| **Outcome RM** | 最终回答 | 简单，数据易获取 | 无法指导中间步骤 |
| **Process RM** | 每个推理步骤 | 更细粒度的反馈 | 标注成本高 |

Process RM 在数学推理等任务中特别有用，可以识别"哪一步出错了"。

---

## 与 DPO 的关系

> [!intuition] DPO 的洞察
> [[DPO]] 证明了：在 KL 约束的 RL 目标下，最优策略和 reward 之间存在解析关系：
>
> $$
> r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
> $$
>
> 这意味着：**策略本身就隐式地定义了一个 reward function**。DPO 利用这一点，绕过了显式训练 RM 的步骤。

换句话说：
- RLHF：先学 RM，再用 RM 指导策略
- DPO：直接学策略，隐式学 RM

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: Reward Model 的训练目标是什么？**
> A: 基于 Bradley-Terry 模型，最大化偏好数据的似然。Loss 是 $-\log \sigma(r(y_w) - r(y_l))$，鼓励给 preferred response 更高的分。
>
> **Q2: 什么是 Reward Hacking？如何缓解？**
> A: 模型学会欺骗 RM，生成高分但低质量的回答。缓解方法：KL 惩罚、RM ensemble、定期更新 RM。
>
> **Q3: Process RM 和 Outcome RM 的区别？**
> A: Outcome RM 评估最终回答，Process RM 评估每个推理步骤。Process RM 提供更细粒度的反馈，但标注成本更高。
>
> **Q4: DPO 和 RM 的关系？**
> A: DPO 证明策略隐式定义了 RM。DPO 直接优化策略，绕过了显式训练 RM 的步骤。从某种意义上说，DPO 对标的是 RM 训练，而非 PPO。

---

## 相关概念

- [[RLHF]] — RM 的主要应用场景
- [[Bradley-Terry Model]] — RM loss 的理论基础
- [[DPO]] — 隐式学习 RM 的方法
- [[RLAIF]] — 用 AI 生成偏好数据

---

## 延伸阅读

- [[InstructGPT]] — RM 的工业实践
- [[Process Reward Model]] — 细粒度奖励建模
- [[Constitutional AI]] — 减少人类标注的方法