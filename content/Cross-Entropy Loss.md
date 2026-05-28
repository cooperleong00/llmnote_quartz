---
description: 衡量预测分布与真实分布差异的损失函数，是 LLM 训练的核心目标，最小化交叉熵等价于最大化似然
type: concept
aliases:
  - 交叉熵损失
  - Cross-Entropy
  - CE Loss
  - 交叉熵
prerequisites:
  - "[[Probability Distribution]]"
  - "[[KL Divergence]]"
  - "[[Entropy]]"
tags:
  - math
  - foundations
  - training
created: 2026-01-29
updated: 2026-02-01T01:02
---

# Cross-Entropy Loss

交叉熵损失（Cross-Entropy Loss）是衡量预测概率分布与真实分布差异的标准方法，也是几乎所有语言模型训练的核心目标。它的本质是：**用模型的编码方案来表示真实数据需要多少信息量**。

---

## 动机

> [!intuition] 为什么需要交叉熵？
> 训练模型的目标是让预测分布 $Q$ 尽可能接近真实分布 $P$。我们需要一个可微分的度量来衡量"接近程度"。
>
> - **信息论视角**：交叉熵衡量用 $Q$ 的编码方案表示 $P$ 的数据需要多少比特
> - **概率视角**：最小化交叉熵等价于最大化数据的对数似然
> - **实用视角**：梯度计算简单，与 [[Softmax]] 配合优雅

---

## 定义

> [!definition] 交叉熵
> 对于真实分布 $P$ 和预测分布 $Q$，交叉熵定义为：
>
> **离散情况**：
> $$
> H(P, Q) = -\sum_{x} P(x) \log Q(x)
> $$
>
> **连续情况**：
> $$
> H(P, Q) = -\int P(x) \log Q(x) \, dx
> $$

在分类任务中，真实分布 $P$ 通常是 one-hot 编码（只有正确类别概率为 1），此时交叉熵简化为：

$$
H(P, Q) = -\log Q(y_{true})
$$

即**负对数似然**（Negative Log-Likelihood, NLL）。

---

## 与 KL 散度的关系

> [!math] 核心等式
> 交叉熵可以分解为熵和 KL 散度：
>
> $$
> H(P, Q) = H(P) + D_{KL}(P \| Q)
> $$
>
> 其中：
> - $H(P) = -\sum_x P(x) \log P(x)$ 是真实分布的[[Entropy|熵]]
> - $D_{KL}(P \| Q)$ 是 [[KL Divergence|KL 散度]]

> [!intuition] 直觉理解
> - $H(P)$：真实分布的"内在复杂度"，是不可压缩的下界
> - $D_{KL}(P \| Q)$：用 $Q$ 近似 $P$ 的"额外代价"
> - $H(P, Q)$：总代价 = 内在复杂度 + 近似代价

**关键推论**：当 $P$ 固定时（训练数据不变），最小化交叉熵 $H(P, Q)$ 等价于最小化 KL 散度 $D_{KL}(P \| Q)$。

---

## 为什么最小化交叉熵等价于最大化似然

> [!math] 等价性证明
> 给定数据集 $\{x_1, x_2, ..., x_N\}$，经验分布为 $\hat{P}(x) = \frac{1}{N} \sum_{i=1}^N \delta(x - x_i)$。
>
> **交叉熵**：
> $$
> H(\hat{P}, Q) = -\frac{1}{N} \sum_{i=1}^N \log Q(x_i)
> $$
>
> **对数似然**：
> $$
> \log \mathcal{L}(Q) = \sum_{i=1}^N \log Q(x_i)
> $$
>
> 因此：
> $$
> H(\hat{P}, Q) = -\frac{1}{N} \log \mathcal{L}(Q)
> $$
>
> **最小化交叉熵 = 最大化对数似然 = 最大化似然**

> [!intuition] 直觉
> 交叉熵越小，意味着模型给真实数据分配的概率越高。这正是"模型认为数据更可能"的含义。

---

## 在 LLM 中的应用

### 1. 语言建模目标

语言模型的训练目标是预测下一个 token。给定序列 $y = (y_1, y_2, ..., y_T)$：

$$
\mathcal{L}_{LM} = -\sum_{t=1}^{T} \log p_\theta(y_t | y_{<t})
$$

这就是**自回归交叉熵损失**，也称为 **Causal Language Modeling Loss**。评估时常用 [[Perplexity]]（困惑度），即交叉熵的指数形式 $\text{PPL} = e^{\mathcal{L}_{LM}/T}$。

### 2. SFT 训练

[[SFT]] 使用交叉熵损失，但只在 response 部分计算：

$$
\mathcal{L}_{SFT} = -\sum_{t \in \text{response}} \log p_\theta(y_t | y_{<t}, x)
$$

Prompt 部分被 mask 掉，不参与 loss 计算。

### 3. 知识蒸馏

在 [[On-Policy Distillation]] 中，学生模型学习匹配教师的输出分布：

$$
\mathcal{L}_{KD} = H(P_{teacher}, Q_{student}) = -\sum_v P_{teacher}(v) \log Q_{student}(v)
$$

这里 $P_{teacher}$ 不是 one-hot，而是教师模型的 soft distribution。

---

## 与其他损失函数的对比

> [!comparison] Cross-Entropy vs MSE

| 特性 | Cross-Entropy | MSE |
|------|---------------|-----|
| **适用场景** | 分类、概率分布 | 回归、连续值 |
| **输出层** | Softmax | 线性 |
| **梯度特性** | 错误越大梯度越大 | 梯度与误差成正比 |
| **概率解释** | 最大化似然 | 假设高斯噪声 |
| **对异常值** | 相对鲁棒 | 敏感（平方放大） |

> [!warning] 为什么分类不用 MSE？
> 1. **梯度消失**：Softmax + MSE 在预测错误时梯度很小，学习慢
> 2. **概率语义**：MSE 不保证输出是有效概率分布
> 3. **理论基础**：交叉熵有信息论和最大似然的理论支撑

---

## 常见变体

### 1. Label Smoothing

> [!definition] Label Smoothing
> 将 one-hot 标签"软化"，防止模型过度自信：
>
> $$
> P_{smooth}(y) = (1 - \epsilon) \cdot P_{one-hot}(y) + \frac{\epsilon}{K}
> $$
>
> 其中 $\epsilon$ 是平滑系数（通常 0.1），$K$ 是类别数。

**作用**：
- 防止模型输出过于尖锐的分布
- 提高泛化能力
- 在 [[Transformer]] 原论文中使用

### 2. Focal Loss

针对类别不平衡问题，降低易分类样本的权重：

$$
\mathcal{L}_{focal} = -(1 - p_t)^\gamma \log(p_t)
$$

其中 $p_t$ 是正确类别的预测概率，$\gamma$ 控制聚焦程度。

### 3. Weighted Cross-Entropy

对不同类别赋予不同权重：

$$
\mathcal{L}_{weighted} = -\sum_c w_c \cdot P(c) \log Q(c)
$$

常用于处理类别不平衡。

---

## 数值稳定性

> [!warning] 实现注意事项
> 直接计算 $-\log(\text{softmax}(z))$ 可能导致数值问题：
> - Softmax 可能下溢（输出接近 0）
> - Log 对接近 0 的值不稳定

**解决方案**：使用 `log_softmax` 或 `cross_entropy_with_logits`：

$$
\log \text{softmax}(z_i) = z_i - \log \sum_j e^{z_j}
$$

通过 log-sum-exp trick 保证数值稳定：

$$
\log \sum_j e^{z_j} = z_{max} + \log \sum_j e^{z_j - z_{max}}
$$

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: 交叉熵的定义是什么？**
> A: $H(P, Q) = -\sum_x P(x) \log Q(x)$，衡量用分布 $Q$ 编码分布 $P$ 的数据所需的平均信息量。
>
> **Q2: 交叉熵和 KL 散度的关系？**
> A: $H(P, Q) = H(P) + D_{KL}(P \| Q)$。当 $P$ 固定时，最小化交叉熵等价于最小化 KL 散度。
>
> **Q3: 为什么最小化交叉熵等价于最大化似然？**
> A: 交叉熵 $= -\frac{1}{N} \sum_i \log Q(x_i) = -\frac{1}{N} \log \mathcal{L}$，二者只差一个常数和符号。
>
> **Q4: 为什么分类任务用交叉熵而不是 MSE？**
> A: (1) 梯度特性更好，错误越大梯度越大；(2) 有最大似然的理论基础；(3) 与 softmax 配合数值稳定。
>
> **Q5: Label Smoothing 的作用？**
> A: 防止模型过度自信，提高泛化能力。将 one-hot 标签软化为 $(1-\epsilon, \epsilon/K, ...)$。
