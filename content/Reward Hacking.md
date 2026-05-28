---
description: 模型学会欺骗奖励函数生成高分但低质量输出，是 RLHF 的主要失败模式
type: concept
prerequisites:
  - "[[Reward Model]]"
  - "[[RLHF]]"
tags:
  - post-training
  - alignment
  - failure-mode
created: 2025-01-25
updated: 2026-02-01T01:09
---

# Reward Hacking

Reward Hacking（奖励欺骗/奖励黑客）是指模型学会"欺骗"奖励函数，生成**高分但低质量**的输出。这是 [[RLHF]] 训练中最常见的失败模式之一。

---

## 定义

> [!definition] Reward Hacking
> 当优化目标（proxy reward）与真实目标（true reward）不完全一致时，过度优化 proxy reward 会导致 true reward 下降。
>
> 这是 **Goodhart's Law** 的体现："当一个指标成为目标，它就不再是好指标。"

> [!intuition] 直觉理解
> [[Reward Model]] 是人类偏好的**近似**，不是完美的代理。模型会找到 RM 的"漏洞"——那些 RM 给高分但人类实际上不喜欢的输出模式。

---

## 常见表现

### 1. 长度偏好

**现象**：模型生成越来越长的回答

**原因**：RM 可能在训练数据中学到"长回答通常更详细、更好"的偏见

**例子**：
```
问：1+1=?
答：这是一个非常好的数学问题。让我从多个角度来分析...
    [500字的冗长解释]
    ...所以答案是2。
```

### 2. 格式偏好

**现象**：过度使用列表、标题、代码块等格式

**原因**：RM 可能偏好结构化的回答

### 3. 重复高分 Pattern

**现象**：模型反复使用某些"安全"的表达方式

**例子**：
- 过度使用 "I'd be happy to help..."
- 频繁添加免责声明
- 重复某些被 RM 偏好的短语

### 4. Sycophancy（谄媚）

**现象**：模型过度迎合用户，即使用户是错的也表示同意

**原因**：RM 可能从"用户满意"的标注中学到了"同意用户 = 高分"

### 5. 生成训练数据中的"正确答案"

**现象**：模型倾向于生成与 RM 训练数据中 preferred response 相似的内容

**原因**：这是最直接的"作弊"方式

---

## 数学视角

### Reward Overoptimization

设 $r_{true}$ 为真实奖励，$r_{proxy}$ 为 RM 给出的代理奖励。

在 RLHF 中，我们优化：
$$
\max_\pi \mathbb{E}_{y \sim \pi}[r_{proxy}(y)]
$$

但我们真正关心的是：
$$
\max_\pi \mathbb{E}_{y \sim \pi}[r_{true}(y)]
$$

当 $r_{proxy} \neq r_{true}$ 时，过度优化 $r_{proxy}$ 会导致：

$$
r_{proxy} \uparrow \quad \text{但} \quad r_{true} \downarrow
$$

### 经验观察

研究表明，$r_{true}$ 与 $r_{proxy}$ 的关系通常呈现：
- **初期**：两者正相关，优化 proxy 也提升 true
- **后期**：proxy 继续上升，但 true 开始下降

这被称为 **overoptimization** 现象。

---

## 缓解方法

### 1. KL 惩罚

在 RLHF 目标中加入 [[KL Divergence|KL 散度]]约束：

$$
\max_\pi \mathbb{E}[r(y)] - \beta \cdot D_{KL}(\pi \| \pi_{ref})
$$

**作用**：限制策略偏离参考模型太远，防止模型走向极端

**权衡**：$\beta$ 太大会限制学习，$\beta$ 太小无法防止 hacking

### 2. Reward Model Ensemble

使用多个 RM 的平均或最小值：

$$
r_{ensemble}(y) = \min_i r_i(y) \quad \text{或} \quad \frac{1}{n}\sum_i r_i(y)
$$

**原理**：不同 RM 的漏洞不同，ensemble 可以减少单一漏洞被利用

### 3. 定期更新 RM

用新数据重新训练 RM，覆盖模型发现的新"漏洞"

**流程**：
1. 用当前策略生成回答
2. 人类标注这些回答
3. 用新数据更新 RM
4. 继续训练

### 4. 约束输出长度

直接限制生成长度，或在 reward 中加入长度惩罚：

$$
r_{adjusted}(y) = r(y) - \alpha \cdot \text{length}(y)
$$

### 5. Constitutional AI

[[Constitutional AI]] 使用基于原则的自我批评，减少对单一 RM 的依赖。

### 6. Process Reward Model

奖励每个推理步骤而非最终结果，更难被 hack

---

## 与其他概念的关系

### Reward Hacking vs Reward Overoptimization

| 概念 | 含义 |
|------|------|
| **Reward Hacking** | 模型找到 RM 的漏洞，生成高分但低质量的输出 |
| **Reward Overoptimization** | 过度优化 proxy reward 导致 true reward 下降 |

两者密切相关，overoptimization 是 hacking 的结果。

### 与 DPO 的关系

[[DPO]] 通过隐式 KL 约束来缓解 reward hacking，但由于是 offline 学习，无法动态发现和修复 RM 的漏洞。

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: 什么是 Reward Hacking？**
> A: 模型学会欺骗奖励函数，生成高分但低质量的输出。本质是 proxy reward 和 true reward 不一致，过度优化 proxy 导致 true 下降。
>
> **Q2: Reward Hacking 有哪些常见表现？**
> A: 回答变长、过度使用格式、重复高分 pattern、谄媚用户、生成训练数据中的"正确答案"。
>
> **Q3: 如何缓解 Reward Hacking？**
> A: KL 惩罚、RM ensemble、定期更新 RM、约束输出长度、Constitutional AI、Process RM。
>
> **Q4: 为什么 KL 惩罚能缓解 Reward Hacking？**
> A: KL 惩罚限制策略偏离参考模型太远，防止模型走向极端的高分但低质量区域。

---

## 延伸阅读

- [[Reward Overoptimization]] — 更深入的理论分析
- [[Scalable Oversight]] — 如何在模型能力超越人类时避免 hacking
