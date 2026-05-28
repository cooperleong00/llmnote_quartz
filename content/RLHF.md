---
description: 通过人类偏好信号对齐语言模型的三阶段方法：SFT → Reward Model → RL 优化
type: method
prerequisites:
  - "[[SFT]]"
  - "[[Reward Model]]"
  - "[[PPO]]"
tags:
  - post-training
  - alignment
created: 2025-01-25
updated: 2026-02-01T00:49
---

# RLHF (Reinforcement Learning from Human Feedback)

强化学习人类反馈（RLHF, Reinforcement Learning from Human Feedback）是一种通过人类偏好信号来对齐语言模型的方法。它是 ChatGPT、Claude 等现代对话模型的核心训练技术。

> [!paper] 论文出处
> Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), NeurIPS 2022

---

## 核心思想

> [!intuition] 直觉理解
> 预训练模型学会了"说话"，但不知道"该说什么"。RLHF 的目标是：**让模型学会人类喜欢什么样的回答**。
>
> 核心挑战：人类偏好难以直接写成数学公式。RLHF 的解决方案是：
> 1. 先让人类标注"哪个回答更好"
> 2. 训练一个 Reward Model 来模拟人类判断
> 3. 用 RL 优化模型，让它生成高分回答

---

## 三阶段流程

```
阶段 1: SFT (Supervised Fine-Tuning)
        预训练模型 → 高质量指令数据 → SFT 模型

阶段 2: RM (Reward Modeling)
        SFT 模型生成多个回答 → 人类标注偏好 → 训练 Reward Model

阶段 3: RL (Reinforcement Learning)
        SFT 模型 + Reward Model → PPO 优化 → 对齐后的模型
```

### 阶段 1: 监督微调 ([[SFT]])

用高质量的 (prompt, response) 数据对预训练模型进行微调，让模型学会基本的指令遵循能力。

- 数据来源：人工编写、众包标注
- 目标：让模型"会回答问题"，但不保证回答质量

### 阶段 2: 奖励模型训练 (RM)

训练一个模型来预测人类对回答的偏好。

**数据构建**：
1. 给定 prompt，让 SFT 模型生成多个回答
2. 人类标注员对回答进行排序（哪个更好）
3. 转化为 pairwise 偏好数据：$(x, y_w, y_l)$

**训练目标**（基于 [[Bradley-Terry Model]]）：

$$
\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]
$$

详见 [[Reward Model]]。

### 阶段 3: 强化学习优化 (RL)

用 [[PPO]] 算法优化策略，最大化 Reward Model 的打分，同时保持不偏离 SFT 模型太远。

**优化目标**：

> [!definition] RLHF 目标函数
> $$
> \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{KL} \left[ \pi_\theta(y|x) \| \pi_{ref}(y|x) \right]
> $$

其中：
- $\pi_\theta$：正在训练的策略（Actor）
- $\pi_{ref}$：参考策略（通常是 SFT 模型，frozen）
- $r_\phi$：Reward Model
- $\beta$：[[KL Divergence|KL]] 惩罚系数

> [!intuition] 为什么需要 KL 惩罚？
> 没有 KL 约束，模型会"钻空子"——找到 Reward Model 的漏洞，生成高分但无意义的回答（[[Reward Hacking]]）。KL 惩罚强制模型保持在"正常回答"的分布附近。

---

## 训练架构

RLHF 训练需要同时维护多个模型：

| 模型 | 作用 | 是否更新 |
|------|------|----------|
| **Actor** ($\pi_\theta$) | 生成回答的策略 | ✅ 更新 |
| **Critic** ($V_\psi$) | 估计状态价值，计算 advantage | ✅ 更新 |
| **Reward Model** ($r_\phi$) | 给回答打分 | ❌ 冻结 |
| **Reference** ($\pi_{ref}$) | 计算 KL 惩罚 | ❌ 冻结 |

> [!warning] 计算开销
> 同时运行 4 个大模型是 RLHF 的主要瓶颈。这也是 [[DPO]] 等方法试图解决的问题。

---

## 训练细节

### KL 惩罚的实现

实际训练中，KL 惩罚通常加到 reward 上：

$$
r_{total}(x, y) = r_\phi(x, y) - \beta \cdot \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

这样 PPO 只需要优化一个标量 reward。

### Advantage 估计

使用 GAE（Generalized Advantage Estimation）来估计 advantage，平衡 bias 和 variance。

---

## 核心挑战

### 1. Reward Hacking

模型学会"欺骗" Reward Model，生成高分但低质量的回答。

**表现**：
- 回答变长（RM 可能偏好长回答）
- 重复某些高分 pattern
- 生成 RM 训练数据中的"正确答案"

**缓解方法**：
- KL 惩罚
- 定期更新 RM
- 多样化的 RM ensemble

### 2. 训练不稳定

PPO 对超参数敏感，容易出现：
- reward 突然崩溃
- KL 散度爆炸
- 生成质量下降

### 3. 标注成本

高质量的人类偏好标注成本高昂，且标注员之间存在分歧。

### 4. Credit Assignment

序列级 reward 难以归因到具体 token，导致梯度信号稀疏。详见 [[Credit Assignment]]。

---

## 与 DPO 的对比

> [!comparison] RLHF vs DPO

| 方面 | RLHF | DPO |
|------|------|-----|
| **流程** | 三阶段：SFT → RM → RL | 两阶段：SFT → DPO |
| **Reward Model** | 显式训练 | 隐式（策略即 RM） |
| **优化方式** | 在线 RL（PPO） | 离线监督学习 |
| **计算开销** | 高（4 个模型） | 低（2 个模型） |
| **训练稳定性** | 较难调参 | 简单稳定 |
| **探索能力** | 有（在线采样） | 无（固定数据） |

> [!intuition] 本质区别
> RLHF 通过 **generate → evaluate → update** 的循环，把 RM 的 evaluate 能力转化为 generate 能力。DPO 跳过了这个循环，直接在偏好数据上学习，因此缺乏 online 和 explore 的特性。
>
> 详见 [[DPO#DPO 的局限性]]。

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: RLHF 的三个阶段分别是什么？**
> A: SFT（监督微调）→ RM（奖励模型训练）→ RL（强化学习优化）。SFT 让模型会回答，RM 学习人类偏好，RL 优化模型生成高偏好回答。
>
> **Q2: 为什么需要 KL 惩罚？**
> A: 防止 reward hacking。没有 KL 约束，模型会找到 RM 的漏洞，生成高分但无意义的回答。KL 惩罚强制模型保持在合理的输出分布内。
>
> **Q3: RLHF 训练需要哪些模型？**
> A: 四个：Actor（策略）、Critic（价值估计）、Reward Model（打分）、Reference（计算 KL）。其中 Actor 和 Critic 更新，RM 和 Reference 冻结。
>
> **Q4: RLHF 相比 DPO 的优缺点？**
> A: 优点是有 online 探索能力，能持续改进；缺点是计算开销大、训练不稳定、需要调参。

---

## 延伸阅读

- [[InstructGPT]] — RLHF 的工业化实践
- [[Constitutional AI]] — 减少人类标注的对齐方法
- [[RLAIF]] — 用 AI 反馈替代人类反馈

[[RLHF 常见的思维误区]]