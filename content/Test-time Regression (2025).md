---
type: paper
description: 将 associative recall 形式化为回归问题，统一 softmax attention、linear attention、SSM、fast-weight programmers 等序列模型为 test-time regression 的特例
aliases:
  - TTR
  - Test-time Regression
  - 测试时回归
prerequisites:
  - "[[Attention]]"
  - "[[Linear Attention]]"
tags:
  - architecture
  - attention
  - efficient-attention
  - sequence-modeling
created: 2026-02-10
updated: 2026-02-11T00:23
---

# Test-time Regression (2025)

> [!paper] 论文信息
> **Test-time Regression: a Unifying Framework for Designing Sequence Models**
> Ke Alexander Wang, Jiaxin Shi, Emily B. Fox (Stanford)
> arXiv: 2501.12352

序列模型（Transformer、SSM、Linear Attention 等）看似各自独立发展，但本文揭示了它们共享一个统一的底层原理：**associative recall 可以形式化为一个回归问题**。在前向传播中，模型先将 key-value 对"记忆"为一个回归器（memorization），再用 query 从中"检索"输出（retrieval）——这个过程就是 **test-time regression (TTR)**。不同的架构只是在三个设计选择上做了不同的决定。

## 动机：为什么需要统一框架？

近年来序列模型百花齐放——[[Attention|Softmax Attention]]、[[Linear Attention]]、[[State Space Model|SSM]]（如 [[Mamba]]）、[[DeltaNet]]、[[GLA|Gated Linear Attention]] 等——但它们各自从不同动机出发，使用不同的数学语言，彼此关系模糊。

> [!intuition] 核心洞察
> 实证研究发现，模型的 **associative recall 能力**（给定 cue 检索对应 value 的能力）与语言建模性能高度相关。TTR 框架的出发点是：如果 associative recall 如此重要，能否从这个能力出发，**系统性地推导**出各种序列模型？

经典例子："Hakuna Matata, it means no worries... Hakuna ___"。要预测下一个词，模型需要：
1. **记忆**之前出现过的 "Hakuna" → "Matata" 关联
2. **检索**：再次遇到 "Hakuna" 时，从记忆中取出 "Matata"

这就是 associative recall 的两步过程，而 TTR 将第一步形式化为**回归问题**。

## TTR 框架：两步过程

### Step 1: Memorization = 回归问题

给定 key-value 对 $(\mathbf{k}_1, \mathbf{v}_1), \dots, (\mathbf{k}_t, \mathbf{v}_t)$，找一个函数 $m: \mathbb{R}^{D_k} \to \mathbb{R}^{D_v}$ 使得 $m(\mathbf{k}_i) \approx \mathbf{v}_i$。这等价于求解加权回归问题：

$$m_t \approx \underset{m \in \mathcal{M}}{\operatorname{argmin}} \frac{1}{2} \sum_{i=1}^t \gamma_i^{(t)} \|\mathbf{v}_i - m(\mathbf{k}_i)\|_2^2$$

其中 $\gamma_i^{(t)}$ 控制每个关联的重要性权重。

### Step 2: Retrieval = 函数求值

用 query $\mathbf{q}_t$ 从记忆中检索：$\mathbf{y}_t = m_t(\mathbf{q}_t)$。

> [!intuition] 为什么叫 "test-time" regression？
> 因为回归器 $m_t$ 不是在训练阶段固定的，而是在**每次前向传播时**根据输入 token 重新生成。每个 sequence layer 的前向传播都隐式地在做一次回归。

### 三个设计选择

TTR 框架的核心贡献是：**任何 associative recall 序列层都由三个选择决定**：

| 设计选择 | 含义 | 影响 |
|----------|------|------|
| **回归权重** $\gamma_i^{(t)}$ | 每个 key-value 对的重要性 | 等权 vs 时间衰减（遗忘） |
| **函数类** $\mathcal{M}$ | 回归器的参数化形式 | 线性 vs 非参数 |
| **优化算法** | 如何求解回归问题 | 解析解 vs 梯度下降 vs 在线学习 |

## 统一视角：现有架构的推导

这是论文最精彩的部分——通过改变三个设计选择，**推导出**（而非事后解释）各种已有架构。

### Vignette 1: Linear Attention = 次优的线性最小二乘

**设计选择**：等权 + 线性函数类 + 解析解

线性最小二乘的最优解是：

$$\mathbf{M}_t = \mathbf{V}_t^\top \mathbf{K}_t (\mathbf{K}_t^\top \mathbf{K}_t)^{-1}$$

但 $(\mathbf{K}_t^\top \mathbf{K}_t)^{-1}$ 的计算代价高。[[Linear Attention]] 做了一个粗暴的近似 $\mathbf{K}_t^\top \mathbf{K}_t \approx \mathbf{I}$（忽略 key 之间的协方差），得到：

$$\mathbf{y}_t = \mathbf{V}_t^\top \mathbf{K}_t \mathbf{q}_t = \sum_{i=1}^t \mathbf{v}_i \mathbf{k}_i^\top \mathbf{q}_t$$

> [!warning] Linear Attention 为什么不如 Softmax Attention
> 从 TTR 视角看，Linear Attention 是一个**忽略了 key 协方差结构**的次优回归器。当 key 之间存在相关性时（现实中几乎总是如此），这个近似会导致：
> 1. **检索精度下降**：无法区分相似但不同的 key
> 2. **训练不稳定**：输出范数可能随序列长度无界增长（缺少 $(\mathbf{K}_t^\top \mathbf{K}_t)^{-1}$ 的自归一化效应）
>
> 这解释了为什么 Qin et al. (2022) 提出的 output normalization 能改善 Linear Attention——它近似恢复了被丢弃的自归一化性质。

### Vignette 2: Gated Linear Attention / SSM = 加权线性最小二乘

**设计选择**：时间衰减权重 + 线性函数类 + 解析解（近似）

加入几何衰减权重 $\gamma_i^{(t)} = \prod_{j=i+1}^t \gamma_j$，再做同样的协方差近似，得到递推：

$$\mathbf{M}_t = \gamma_t \mathbf{M}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$$

这正是 [[GLA|Gated Linear Attention]]、RetNet、RWKV-6、mLSTM、以及 [[Mamba|Mamba-2]] 等模型的核心递推。遗忘门（forget gate）在 TTR 框架下有了数学解释：**它是加权最小二乘中对旧关联降权的自然结果**。

### Vignette 3: Fast Weight Programmers / Online Learners = 一阶优化

**设计选择**：等权 + 线性函数类 + 梯度下降

不用解析解，改用梯度下降来最小化回归目标。不同的优化策略对应不同的架构：

| 优化方法 | 对应架构 | 递推形式 |
|----------|----------|----------|
| 1 步 batch GD（从零初始化） | [[Linear Attention]] | $\mathbf{M}_t = \mathbf{V}_t^\top \mathbf{K}_t$ |
| 1 步 batch GD + 完美预条件 | 最优线性记忆 | $\mathbf{M}_t = \mathbf{V}_t^\top \mathbf{K}_t (\mathbf{K}_t^\top \mathbf{K}_t)^{-1}$ |
| 单样本 SGD（warm start） | [[DeltaNet]] | $\mathbf{M}_t = \mathbf{M}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$ |
| SGD + L2 正则化 | [[Gated DeltaNet]] | SGD + 遗忘 = Leaky LMS |
| 自适应步长 SGD | [[Longhorn]] | 保守步长 $\beta_t = \delta_t/(1+\delta_t\|\mathbf{k}_t\|^2)$ |
| SGD + 非线性回归器 | [[TTT-Linear\|TTT]] | MLP 作为记忆函数 |

> [!intuition] 从优化视角看 Linear Attention vs 最优解
> Linear Attention 和最优线性记忆是**同一优化问题的两个极端**：
> - Linear Attention = 1 步 batch GD，**无预条件**
> - 最优解 = 1 步 batch GD，**完美预条件**（Newton 法）
>
> 性能差异完全来自优化器是否考虑了目标函数的曲率（即 key 的协方差）。

### Vignette 4: Softmax Attention = 非参数回归

**设计选择**：基于相似度的权重 + 局部常数函数类 + 解析解

这是最优雅的推导。Softmax attention 对应 **local constant regression**（Nadaraya-Watson 估计器）：

$$m_{0,t}(\mathbf{q}) = \sum_{i=1}^t \frac{s(\mathbf{k}_i, \mathbf{q})}{\sum_{j=1}^t s(\mathbf{k}_j, \mathbf{q})} \mathbf{v}_i$$

其中 $s(\mathbf{k}, \mathbf{q})$ 是相似度函数。当使用指数平滑核 $s(\mathbf{k}, \mathbf{q}) = \exp(-\|\mathbf{k} - \mathbf{q}\|^2 / B)$ 时，**如果 key 和 query 被归一化到单位球上**（$\|\mathbf{k}\| = \|\mathbf{q}\| = 1$），则：

$$\exp\left(\frac{-\|\mathbf{k} - \mathbf{q}\|^2}{2\sqrt{D_k}}\right) = \exp\left(\frac{-\|\mathbf{k}\|^2 - \|\mathbf{q}\|^2 + 2\mathbf{k}^\top\mathbf{q}}{2\sqrt{D_k}}\right) \propto \exp\left(\frac{\mathbf{k}^\top\mathbf{q}}{\sqrt{D_k}}\right)$$

这就是标准的 [[Attention|softmax attention]]。

> [!math] QKNorm 的数学证明
> Query-Key Normalization（QKNorm）在实践中被发现能稳定大模型训练（Dehghani et al., 2023），但一直缺乏理论解释。TTR 框架给出了简洁的证明：
>
> **QKNorm 是确保 softmax attention 是 proper local constant regressor 的数学必要条件。**
>
> 没有归一化时，指数点积 $\exp(\mathbf{k}^\top \mathbf{q} / \sqrt{D_k})$ 不是距离的单调函数，因此不满足局部近似的要求。归一化后，点积与欧氏距离等价，局部性得到保证。

> [!comparison] 参数化 vs 非参数化记忆
> | 维度 | 参数化（Linear Attention, SSM 等） | 非参数化（Softmax Attention） |
> |------|------|------|
> | 记忆形式 | 固定大小矩阵 $\mathbf{M} \in \mathbb{R}^{D_v \times D_k}$ | 完整 KV 缓存 |
> | 记忆容量 | 固定（受限于矩阵维度） | 随序列长度增长 |
> | 推理复杂度 | $O(1)$ per token | $O(t)$ per token |
> | 回归类型 | 参数化回归（线性/非线性） | 非参数回归（核回归/局部多项式） |
> | 类比 | 压缩记忆 | 完整记忆 |

## Higher-Order Attention：超越 Softmax

TTR 框架自然地指向一个新方向：如果 softmax attention 是 $p=0$（局部常数）的非参数回归，那 $p=1$（局部线性）会怎样？

$p=1$ 的 higher-order attention 输出为：

$$m_{1,t}(\mathbf{q}) = \bar{\mathbf{v}} - \mathbf{M}^{[1]}\bar{\mathbf{d}}$$

其中 $\bar{\mathbf{v}}$ 是标准 softmax attention 的输出，$\mathbf{M}^{[1]}\bar{\mathbf{d}}$ 是基于 key-query 距离的**修正项**。与标准 softmax attention 不同，$p=1$ 的版本通过逆协方差项考虑了 key 之间的相关性——类似于线性回归相对于 linear attention 的改进。

> [!warning] 计算瓶颈
> Higher-order attention 目前计算代价过高（需要 $O(T^2 D_k)$ 空间和 $O(TD_k^3)$ 时间），尚不实用。论文指出开发类似 [[Flash Attention]] 的硬件优化是重要的未来方向。

## Short Convolution 的理论解释

论文还解释了为什么 short convolution 对循环模型至关重要。在 multi-query associative recall (MQAR) 任务中：

- 构造 key 为 $\mathbf{k}_t = \mathbf{x}_{t-1}$（通过 short conv "回看"一个 token）
- 构造 value 为 $\mathbf{v}_t = \mathbf{x}_t$
- 这样记忆的是 **bigram 关联** $\{(\mathbf{x}_{t-1}, \mathbf{x}_t)\}$

> [!intuition] 一层 short conv + 一层序列层就够了
> 单个 TTR 层（即使是最简单的 linear attention）配合一个 short convolution，就足以解决 MQAR 任务。Short convolution 的作用是让模型能"回看"前一个 token，从而构造出有效的 bigram key-value 对。这解释了为什么去掉 short convolution 会导致循环模型性能严重下降。

## 统一全景图

```
                    TTR Framework
                    ┌─────────────────────────────────────┐
                    │  Memorize: solve regression problem  │
                    │  Retrieve: apply regressor to query  │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
        参数化回归              一阶优化             非参数回归
              │                    │                     │
    ┌─────────┴─────────┐   ┌─────┴──────┐        ┌────┴────┐
    │                   │   │            │        │         │
  等权              时间衰减  SGD      Batch GD   p=0      p=1
    │                   │   │            │        │         │
 Linear          GLA/SSM  DeltaNet  Lin.Attn  Softmax  Higher-
 Attention       Mamba-2  TTT      (from 0)  Attention  Order
 (crude)         RetNet   Gated               (QKNorm)  Attention
                 RWKV-6   DeltaNet
```

## 局限性

> [!warning] 框架的边界
> 1. **仅覆盖 QKV 模式**：不包含纯卷积架构（如 S4 的原始形式）、非 QKV 的计算模式
> 2. **近似的代价**：论文推导中多次使用 $\mathbf{K}_t^\top \mathbf{K}_t \approx \mathbf{I}$ 的近似，这个近似的质量在不同场景下差异很大
> 3. **Higher-order attention 不实用**：$p \geq 1$ 的版本计算代价过高，缺乏高效实现
> 4. **解释性 vs 预测性**：框架主要是事后统一（unification），对"哪种设计选择最优"的预测能力有限

## 延伸阅读

**原始论文与相关 clipping**：
- [[Clippings/Paper/250112352v3/250112352v3|论文原文 clipping]] -- 完整的数学推导和实验

**框架中涉及的具体架构**：
- [[DeltaNet]] -- SGD 视角下的 delta rule 更新
- [[Fast Weight Programming]] -- 快速权重编程的经典视角
- [[TTT-Linear|Test-Time Training]] -- 非线性回归器的 SGD 扩展
- [[Longhorn]] -- 自适应步长的在线学习层

**相关理论背景**：
- [[Kernel Regression]] -- 非参数回归的经典方法
- [[Nadaraya-Watson Estimator]] -- softmax attention 对应的统计估计器
- [[Recursive Least Squares]] -- 在线线性回归的经典算法
