---
description: 在 SDPA 输出后添加 sigmoid gating，引入非线性和稀疏性，消除 attention sink 并提升训练稳定性
type: method
aliases:
  - GA
  - SDPA Gating
prerequisites:
  - "[[Attention]]"
  - "[[Attention Sink]]"
tags:
  - attention
  - architecture
  - training-stability
created: 2026-02-04
updated: 2026-02-04
---

# Gated Attention

Gated Attention (GA) 是一种在标准 [[Attention|softmax attention]] 中引入 gating 机制的方法。核心发现是：在 Scaled Dot-Product Attention (SDPA) 输出后添加 **head-specific sigmoid gating** 能显著提升性能、训练稳定性，并消除 [[Attention Sink]] 现象。

---

## 动机

标准 attention 中，value projection ($W_V$) 和 output projection ($W_O$) 是两个连续的线性层，可以合并为一个 **low-rank 线性变换**。这限制了 attention 的表达能力。

> [!intuition] 为什么需要 Gating？
> 1. **打破线性瓶颈**：在 $W_V$ 和 $W_O$ 之间引入非线性，增强表达能力
> 2. **提供显式 rescaling**：让模型不再依赖 [[Attention Sink|attention sink]] 来实现 rescaling
> 3. **引入稀疏性**：query-dependent 的 gating 可以过滤与当前 query 无关的信息

---

## 方法

### Gating 机制的形式化

给定输入 $Y$（待调制）和 $X$（用于计算 gating score）：

$$
Y' = g(Y, X, W_\theta, \sigma) = Y \odot \sigma(XW_\theta)
$$

其中 $\sigma$ 是激活函数（通常是 sigmoid），$W_\theta$ 是可学习参数。

### Gating 位置的选择

论文系统比较了 5 个位置：

| 位置 | 描述 | 效果 |
|------|------|------|
| $G_1$ (SDPA 后) | 在 SDPA 输出后 gating | **最佳**：PPL 降低 0.2，MMLU +2 |
| $G_2$ (Value 后) | 在 V projection 后 gating | 次优：PPL 改善明显 |
| $G_3$ (Key 后) | 在 K projection 后 gating | 效果有限 |
| $G_4$ (Query 后) | 在 Q projection 后 gating | 效果有限 |
| $G_5$ (Dense 后) | 在最终输出后 gating | 效果有限 |

> [!intuition] 为什么 $G_1$ 最有效？
> SDPA 输出位于 $W_V$ 和 $W_O$ 之间，在此处引入非线性能最大程度打破 low-rank 瓶颈。同时，此位置的 gating 是 query-dependent 的，能根据当前 query 动态过滤信息。

### 最佳配置

```
Head-specific + Elementwise + Sigmoid + Multiplicative
```

- **Head-specific**：每个 attention head 有独立的 gating 参数
- **Elementwise**：gating score 与 SDPA 输出同维度，细粒度调制
- **Sigmoid**：输出范围 [0,1]，bounded 且在 0 附近有细粒度控制
- **Multiplicative**：$Y' = Y \cdot \sigma(XW_\theta)$

---

## 效果

### 性能提升

在 15B MoE 模型（3.5T tokens）上：
- PPL 降低约 0.2
- MMLU 提升约 2 points
- 参数开销仅约 1.3%（201M / 15B）

### 训练稳定性

> [!example] 训练稳定性改善
> - 几乎消除 loss spikes
> - 支持更大的 learning rate
> - 更好的 scaling properties

### 消除 Attention Sink

> [!intuition] 为什么 Gating 能消除 Attention Sink？
> Sigmoid gating 引入了 **query-dependent 稀疏性**：
> - 当 gating score 接近 0 时，对应的 SDPA 输出被抑制
> - 模型不再需要通过 attention sink 来"倾倒"无关注意力
> - 第一个 token 的 attention score 显著降低

实验验证：
- 第一个 token 的平均 attention score 从 baseline 的高值降至正常水平
- Massive Activation 也随之减少（因为不再需要生成 near one-hot 向量）

### 长度外推

消除 attention sink 后，模型在 [[Length Extrapolation|长度外推]] 上表现更好：

| 方法 | 4k | 32k | 64k | 128k |
|------|-----|------|------|------|
| Baseline + YaRN | 82.9 | 37.9 | 37.5 | 31.7 |
| **GA + YaRN** | 88.1 | 72.9 | 66.6 | 58.8 |

> [!intuition] 为什么 GA 有助于长度外推？
> Baseline 模型依赖 attention sink 来调节 attention 分布。当使用 YaRN 等方法修改 RoPE base 时，attention sink 模式难以适应，导致性能下降。GA 模型主要依赖 input-dependent gating 来控制信息流，对 RoPE 修改更鲁棒。

---

## 与其他方法的关系

### 与 GatedNorm 的区别

| 方面 | Gated Attention | [[Attention Sink#GatedNorm\|GatedNorm]] |
|------|-----------------|-----------|
| **位置** | Attention 内部（SDPA 后） | RMSNorm 后 |
| **目标** | 消除 attention sink | 消除 residual sink |
| **机制** | Head-specific sigmoid gating | Low-rank self-gating |

两者可以组合使用，分别处理 attention 和 residual stream 中的 outliers。

### 与 Learnable Sink Bias 的区别

| 方面 | Gated Attention | Learnable Sink Bias |
|------|-----------------|---------------------|
| **方法** | 显式 gating 提供 rescaling | 在 softmax 分母添加 bias |
| **效果** | 消除 attention sink | 提供"虚拟 sink" |
| **参数** | 每层约 201M（15B 模型） | 每个 head 一个标量 |

---

## 面试要点

> [!interview] 面试视角
> **Q: Gated Attention 的核心思想是什么？**
> A: 在 SDPA 输出后添加 head-specific sigmoid gating，引入非线性打破 $W_V$-$W_O$ 的 low-rank 瓶颈，同时通过 query-dependent 稀疏性消除 attention sink。
>
> **Q: 为什么 Gating 能消除 Attention Sink？**
> A: Sigmoid gating 引入稀疏性，当 gating score 接近 0 时抑制 SDPA 输出。模型不再需要通过 attention sink 来"倾倒"无关注意力，因为 gating 本身就提供了这个功能。
>
> **Q: Gated Attention 对长度外推有什么帮助？**
> A: 消除 attention sink 后，模型不再依赖特定 token 的 attention 模式。当使用 YaRN 等方法扩展上下文长度时，GA 模型更鲁棒，在 128k 长度上比 baseline 高约 27 points。

---

## 参考资料

> [!paper] 原始论文
> [[Clippings/Paper/250506708v1/250506708v1|Gated Attention for Large Language Models]] (Qiu et al., 2025)
> - 系统比较 30+ 种 gating 变体
> - 在 15B MoE 和 1.7B dense 模型上验证
> - 开源 attention-sink-free 模型
