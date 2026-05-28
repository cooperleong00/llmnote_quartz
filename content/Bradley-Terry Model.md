---
description: 成对比较的概率模型，是 Reward Model 训练和 DPO 推导的数学基础
type: concept
prerequisites:
  - "[[概率论]]"
  - "[[最大似然估计]]"
tags:
  - math
  - preference-modeling
created: 2025-01-25
updated: 2026-02-01T00:59
---

# Bradley-Terry Model

Bradley-Terry 模型是一种用于**成对比较**（pairwise comparison）的概率模型，最初用于体育比赛排名。在 LLM 领域，它是 [[Reward Model]] 训练和 [[DPO]] 推导的数学基础。

---

## 定义

> [!definition] Bradley-Terry 模型
> 给定两个选项 $i$ 和 $j$，选项 $i$ 被偏好的概率为：
>
> $$
> P(i \succ j) = \frac{p_i}{p_i + p_j}
> $$
>
> 其中 $p_i > 0$ 是选项 $i$ 的"强度"参数。

等价地，如果定义 $s_i = \log p_i$（对数强度/分数），则：

$$
P(i \succ j) = \frac{e^{s_i}}{e^{s_i} + e^{s_j}} = \frac{1}{1 + e^{-(s_i - s_j)}} = \sigma(s_i - s_j)
$$

其中 $\sigma$ 是 sigmoid 函数。

> [!intuition] 直觉理解
> 偏好概率只取决于**分数之差** $s_i - s_j$。分数差越大，偏好概率越接近 1 或 0。

---

## 历史背景

Bradley-Terry 模型由 Ralph Bradley 和 Milton Terry 于 1952 年提出，最初用于：
- 体育比赛排名（哪支队伍更强）
- 消费者偏好研究（哪个产品更受欢迎）
- 心理学实验（哪个刺激更强）

类似的思想也出现在 [[Elo Rating|Elo 评分系统]] 中，后者广泛用于国际象棋等竞技排名。

**核心假设**：每个选项有一个内在的"强度"，偏好概率由强度比值决定。

---

## 在 LLM 中的应用

### 1. Reward Model 训练

[[Reward Model]] 学习一个函数 $r_\phi(x, y)$，给 prompt $x$ 和 response $y$ 打分。

假设人类偏好遵循 Bradley-Terry 模型：

$$
P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

**训练目标**：最大化 [[Preference Data|偏好数据]] 的似然

$$
\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

> [!intuition] 为什么用 Bradley-Terry？
> 1. **简单**：只需要学习一个标量分数
> 2. **可解释**：分数差直接对应偏好强度
> 3. **与 sigmoid 自然结合**：便于神经网络优化

### 2. DPO Loss 推导

[[DPO]] 的核心推导依赖于 Bradley-Terry 假设。

从 RLHF 目标的最优策略：
$$
\pi^*(y|x) \propto \pi_{ref}(y|x) \exp(r(x,y)/\beta)
$$

反解 reward：
$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

代入 Bradley-Terry 模型，$\log Z(x)$ 项抵消，得到 DPO loss：

$$
\mathcal{L}_{DPO} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

---

## 模型假设与局限性

### 假设

1. **传递性**：如果 $A \succ B$ 且 $B \succ C$，则 $A \succ C$
2. **独立性**：对 $(A, B)$ 的偏好不受其他选项影响
3. **一致性**：每个选项有固定的内在强度

### 局限性

> [!warning] 实际中的问题

1. **非传递偏好**：人类偏好可能不满足传递性（A > B > C > A）
2. **上下文依赖**：偏好可能随 prompt 变化，不是固定的
3. **标注噪声**：不同标注员可能有不同偏好
4. **Tie 处理**：原始模型不处理"一样好"的情况

**缓解方法**：
- 使用更复杂的模型（如 Thurstone 模型）
- 多人标注取共识
- 引入 tie 选项

---

## 扩展：Plackett-Luce 模型

Bradley-Terry 只处理成对比较。对于 **K 个选项的排序**，可以使用 [[Plackett-Luce Model]]：

$$
P(\text{排序 } \pi) = \prod_{i=1}^{K} \frac{p_{\pi(i)}}{\sum_{j=i}^{K} p_{\pi(j)}}
$$

**应用**：当标注员对多个回答进行排序时，可以用 Plackett-Luce 提取更多信息。

---

## 参数估计

### 最大似然估计

给定 $N$ 个成对比较结果 $\{(i_n, j_n, y_n)\}$，其中 $y_n = 1$ 表示 $i_n$ 被偏好：

$$
\mathcal{L} = \sum_{n=1}^{N} \left[ y_n \log P(i_n \succ j_n) + (1-y_n) \log P(j_n \succ i_n) \right]
$$

这是一个凸优化问题，可以用梯度下降求解。

### 在神经网络中

Reward Model 将 $s_i = r_\phi(x, y_i)$ 参数化为神经网络输出，通过反向传播优化。

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: Bradley-Terry 模型是什么？**
> A: 一种成对比较的概率模型。给定两个选项的分数 $s_i, s_j$，偏好概率为 $P(i \succ j) = \sigma(s_i - s_j)$。
>
> **Q2: 为什么 Reward Model 使用 Bradley-Terry？**
> A: 简单、可解释、与 sigmoid 自然结合。只需要学习标量分数，分数差直接对应偏好强度。
>
> **Q3: Bradley-Terry 的假设是什么？有什么局限？**
> A: 假设偏好满足传递性、独立性、一致性。局限是人类偏好可能非传递、上下文依赖、有噪声。
>
> **Q4: DPO 推导中 Bradley-Terry 起什么作用？**
> A: 将 reward 差转化为偏好概率，使得 $\log Z(x)$ 项在相减时抵消，从而得到不依赖 partition function 的 loss。

---

## 相关概念

**延伸阅读**：
- [[Plackett-Luce Model]] — 多选项排序的扩展（见上文"扩展"章节）
- [[Elo Rating]] — 另一种基于成对比较的排名系统（见上文"历史背景"）
