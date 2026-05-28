---
type: concept
description: 通过去掉 softmax 并利用矩阵乘法结合律，将 Attention 复杂度从 O(n^2) 降到 O(n) 的方法族
aliases:
  - 线性注意力
  - Linear Transformer
prerequisites:
  - "[[Attention]]"
  - "[[Transformer]]"
tags:
  - efficient-attention
  - linear-attention
  - architecture
created: 2026-02-11
updated: 2026-02-11T00:06
---

# Linear Attention

Linear Attention 是一类将 [[Attention]] 复杂度从 $O(n^2)$ 降到 $O(n)$ 的方法。核心思想是**去掉 softmax，利用矩阵乘法的结合律改变计算顺序**。这使得 Attention 可以写成 RNN 形式，在推理时实现常数时间的增量更新。

> [!intuition] 一句话理解
> Softmax Attention 必须先算 $n \times n$ 的注意力矩阵；Linear Attention 通过去掉 softmax，可以先算 $d \times d$ 的中间结果，从而绕过平方复杂度。

---

## 动机：为什么需要 Linear Attention？

### Softmax Attention 的平方复杂度瓶颈

标准 Softmax Attention 的计算：

$$
\mathbf{O} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d}) \mathbf{V}
$$

必须显式计算 $n \times n$ 的注意力矩阵 $\mathbf{Q}\mathbf{K}^\top$，导致：
- **时间复杂度**：$O(n^2 d)$
- **空间复杂度**：$O(n^2)$（存储注意力矩阵）

当序列长度 $n$ 增长时，计算量和内存需求呈平方增长。[[Flash Attention]] 通过 IO-aware 算法降低了空间需求，但**平方的时间复杂度无法避免**。

### 与 Sparse Attention 的不同思路

处理长序列有两条主要路线：

| 方法 | 核心思想 | 复杂度 | 精度 |
|------|----------|--------|------|
| [[Sparse Attention]] | 跳过不重要的计算（稀疏化） | $O(n \cdot k)$ | 精确（在稀疏模式内） |
| **Linear Attention** | 改变计算顺序（去掉 softmax） | $O(n \cdot d^2)$ | 近似 |

Sparse Attention 保留 softmax，通过只计算部分注意力权重降低复杂度；Linear Attention 则从根本上改变计算方式。

---

## 核心机制：结合律的魔法

### 最初形式：去掉 softmax

最简单的 Linear Attention 直接去掉 $\exp$：

$$
\mathbf{O} = (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V}
$$

其中 $\mathbf{M}$ 是 causal mask（下三角矩阵），$\odot$ 是逐元素乘法。

**为什么这能实现线性复杂度？** 先考虑非 causal 情况（去掉 mask）：

$$
\mathbf{O} = (\mathbf{Q}\mathbf{K}^\top)\mathbf{V} = \mathbf{Q}(\mathbf{K}^\top\mathbf{V})
$$

关键在于**结合律**：
- 先算 $\mathbf{Q}\mathbf{K}^\top$：得到 $n \times n$ 矩阵，复杂度 $O(n^2 d)$
- 先算 $\mathbf{K}^\top\mathbf{V}$：得到 $d \times d$ 矩阵，复杂度 $O(nd^2)$

当 $n \gg d$ 时，后者显著更快。

### RNN 形式：Causal 场景的关键

对于 causal attention，写成分量形式：

$$
\mathbf{o}_t = \sum_{j=1}^t \mathbf{v}_j (\mathbf{k}_j^\top \mathbf{q}_t) = \sum_{j=1}^t (\mathbf{v}_j \mathbf{k}_j^\top) \mathbf{q}_t = \left(\sum_{j=1}^t \mathbf{v}_j \mathbf{k}_j^\top\right) \mathbf{q}_t
$$

定义 **state 矩阵** $\mathbf{S}_t = \sum_{j=1}^t \mathbf{v}_j \mathbf{k}_j^\top$，得到递归形式：

> [!math] Linear Attention 的 RNN 形式
> $$
> \mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top, \quad \mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t
> $$

这是一个以 $\mathbf{S}_t \in \mathbb{R}^{d \times d}$ 为状态的**线性 RNN**：
- **训练时**：可以并行计算（利用 cumsum 或 associative scan）
- **推理时**：每步只需 $O(d^2)$ 更新，无需存储历史 KV

> [!intuition] 为什么叫"线性" RNN？
> "线性"指的是递归公式对 state 变量 $\mathbf{S}$ 的依赖是线性的（只有一次方）。对输入 $\mathbf{q}, \mathbf{k}, \mathbf{v}$ 的依赖可以是非线性的。

---

## 发展脉络：从模仿到创新

### 第一阶段：模仿 Softmax（早期探索）

早期工作试图**近似** softmax attention：
- **Performer**、**RFA**：用随机特征近似 $\exp(\mathbf{Q}\mathbf{K}^\top)$
- 给 $\mathbf{Q}, \mathbf{K}$ 加非负激活函数，保证注意力权重非负
- 加分母归一化，模仿 softmax 的概率分布特性

后来研究发现，序列长度维度的归一化并不能完全避免数值不稳定性，不如直接用 RMSNorm 事后归一化。而且不加激活函数效果也足够好。

### 第二阶段：引入遗忘门（RetNet 等）

原始 Linear Attention 的问题：纯加性更新 $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ 本质上是 cumsum，当叠加的 token 足够多时，每个 token 的信息占比变得极小，**记忆变得模糊**。

**RetNet** 引入衰减因子：

$$
\mathbf{S}_t = \gamma \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top, \quad \gamma \in (0, 1)
$$

这体现了语言模型的**就近原则（Recency Bias）**：倾向于遗忘久远的历史，保证最近 token 的分辨率。

后续发展：
- **Mamba**、**GLA**：将 $\gamma$ 做成 data-dependent（类似 LSTM 的遗忘门）
- 为保持线性性，去掉了遗忘门对 state 的依赖

### 第三阶段：Delta Rule（DeltaNet）

[[DeltaNet]] 从 **test-time training** 视角重新理解 Linear Attention：将 RNN 的状态更新视为在线学习问题。

原始 Linear Attention 对应的"损失函数"是 $-\mathbf{v}^\top(\mathbf{S}\mathbf{k})$，这是无下界的，可能导致 $\mathbf{S}$ 趋于无穷。更好的目标是平方损失 $\frac{1}{2}\|\mathbf{S}\mathbf{k} - \mathbf{v}\|^2$，对应的更新规则是：

> [!math] Delta Rule
> $$
> \mathbf{S}_t = \mathbf{S}_{t-1} - (\mathbf{S}_{t-1}\mathbf{k}_t - \mathbf{v}_t)\mathbf{k}_t^\top = \mathbf{S}_{t-1}(\mathbf{I} - \mathbf{k}_t\mathbf{k}_t^\top) + \mathbf{v}_t\mathbf{k}_t^\top
> $$

直观理解："先减后加"——先移除模型对 $\mathbf{k}_t$ 的旧认知，再补充新认知，实现**除旧迎新**。

### 第四阶段：反哺 Softmax Attention

Linear Attention 的发展甚至开始"反哺" Softmax Attention。通过核技巧，存在映射 $\phi$ 使得 $\exp(\mathbf{Q}\mathbf{K}^\top) = \phi(\mathbf{Q})\phi(\mathbf{K})^\top$，可以将 Linear Attention 的改进迁移回 Softmax Attention：

- **ALIBI**：将常数衰减因子加到 Softmax Attention
- **FoX**：data-dependent 的遗忘门版本
- **DeltaFormer**：Softmax Attention 的 DeltaNet 版本

---

## 局限性与边界

### 1. Key Collision 问题

原始 Linear Attention 的致命缺陷：**无法删除过时的 key-value 关联**。当序列长度 $L > d$ 时，新 key 与旧 key 冲突，导致记忆容量饱和。

> [!warning] 类比：只能写入不能删除的内存
> 想象一个只能写入、不能删除的内存系统——后期新 key 不可避免地与旧 key 冲突，覆盖重要信息。

DeltaNet 通过 delta rule 部分解决了这个问题。

### 2. 状态大小的可扩展性

Linear Attention 的 state 大小是 $d \times d$，而 Softmax Attention 的 KV cache 大小是 $n \times d$。当 $n < d$ 时，Linear Attention 反而更占内存。

### 3. 与 Transformer 的性能差距

尽管 Linear Attention 在效率上有优势，但在语言建模质量上仍与 Softmax Attention 有差距。这也是为什么主流 LLM 仍使用 Softmax Attention + [[Sliding Window Attention]] 等组合。

### 4. 并行训练的挑战

虽然 Linear Attention 理论上可以并行训练，但高效实现需要精心设计的算法（如 DeltaNet 的 WY 表示）。通用的 associative scan 并不是 GPU 高效的。

---

## 统一视角：Test-time Regression

[[Test-time Regression (2025)]] 提出了统一框架，将 Linear Attention、SSM、Fast-weight Programmers 等都视为 test-time regression 问题的不同实例：

$$
\mathbf{o}_t = f(\mathbf{S}_t; \mathbf{q}_t), \quad \mathbf{S}_t = \mathbf{S}_{t-1} - \eta_t \nabla_{\mathbf{S}_{t-1}} \mathcal{L}(f(\mathbf{S}_{t-1}; \mathbf{k}_t), \mathbf{v}_t)
$$

不同的模型对应不同的 $f$（模型结构）和 $\mathcal{L}$（损失函数）选择。

---

## 面试要点

> [!interview] 常见问题
> **Q: Linear Attention 如何实现线性复杂度？**
> A: 去掉 softmax 后，利用矩阵乘法结合律改变计算顺序。先算 $\mathbf{K}^\top\mathbf{V}$（$d \times d$）再乘 $\mathbf{Q}$，避免计算 $n \times n$ 的注意力矩阵。
>
> **Q: Linear Attention 的主要问题是什么？**
> A: Key collision——纯加性更新无法删除过时信息，导致记忆模糊。DeltaNet 通过 delta rule 部分解决。
>
> **Q: Linear Attention 和 Sparse Attention 的区别？**
> A: Sparse Attention 保留 softmax，跳过部分计算；Linear Attention 去掉 softmax，改变计算顺序。前者精确但需要稀疏先验，后者是近似但复杂度更低。

---

## 延伸阅读

**原始论文与深入材料**：
- [[Clippings/Article/线性注意力简史：从模仿、创新到反哺|苏剑林的线性注意力简史]] — 非常全面的中文综述

**基于 Linear Attention 的后续发展**：
- [[DeltaNet]] — 使用 delta rule 解决 key collision
- [[Gated DeltaNet]] — 结合遗忘门和 delta rule
- [[Mamba]] — SSM 视角的线性架构
- [[RetNet]] — 引入衰减因子的线性 Transformer
