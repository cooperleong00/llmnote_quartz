---
type: method
description: 使用 delta rule 更新的线性 Transformer，通过 WY 表示实现硬件高效的并行训练，在保持线性复杂度的同时解决 key collision 问题
aliases:
  - Delta Rule Transformer
prerequisites:
  - "[[Attention]]"
  - "[[Transformer]]"
tags:
  - efficient-attention
  - linear-attention
  - architecture
created: 2026-02-10
updated: 2026-02-10T23:55
---

# DeltaNet

DeltaNet 是一种线性 Transformer 变体，使用 **delta rule**（Widrow-Hoff 学习规则）替代传统线性注意力的加性更新。核心创新是提出了硬件高效的并行训练算法，利用 Householder 矩阵的 WY 表示，使得 delta rule 的优势能够在实际训练中发挥。

> [!paper] 原始论文
> **Parallelizing Linear Transformers with the Delta Rule over Sequence Length**
> - 作者：Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim (MIT)
> - 年份：2024
> - arXiv: [2406.06484](https://arxiv.org/abs/2406.06484)
> - GitHub: [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)

---

## 动机：为什么需要 Delta Rule？

### 线性注意力的 Key Collision 问题

传统 [[Linear Attention]] 使用简单的加性更新：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top
$$

这种纯加性更新存在致命缺陷：**无法删除过时的 key-value 关联**。当序列长度 $L > d$（特征维度）时，会发生 **key collision**——新的 key 与旧的 key 冲突，导致记忆容量饱和。

> [!intuition] 类比：只能写入不能删除的内存
> 想象一个只能写入、不能删除的内存系统：
> - 初期：每个 key 都能找到独立的存储位置
> - 后期：新 key 不可避免地与旧 key 冲突，覆盖重要信息
> - 结果：模型无法学会"遗忘"不重要的信息，记忆容量受限

### Delta Rule 的解决方案

Delta rule（也称 Widrow-Hoff 学习规则）引入了 **基于预测误差的更新**：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t (\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t) \mathbf{k}_t^\top
$$

其中：
- $\mathbf{S}_{t-1} \mathbf{k}_t$：当前预测值（用旧记忆检索 $\mathbf{k}_t$）
- $\mathbf{v}_t$：目标值
- $\beta_t \in (0, 1)$：学习率（写入强度），由 $\beta_t = \sigma(\mathbf{W}_\beta \mathbf{x}_t)$ 学习得到
- $(\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t)$：预测误差（delta）

> [!intuition] 三种理解视角
>
> **1. 在线学习视角**：优化平方损失的单步 SGD
> $$
> \mathcal{L}_t(\mathbf{S}) = \frac{1}{2} \|\mathbf{S} \mathbf{k}_t - \mathbf{v}_t\|^2
> $$
> $$
> \mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \nabla_{\mathbf{S}_{t-1}} \mathcal{L}_t(\mathbf{S}_{t-1})
> $$
>
> **2. 记忆更新视角**：先删除旧值，再写入新值
> $$
> \mathbf{S}_t = \mathbf{S}_{t-1} \underbrace{- \mathbf{v}_t^{\text{old}} \mathbf{k}_t^\top}_{\text{删除旧关联}} \underbrace{+ \mathbf{v}_t^{\text{new}} \mathbf{k}_t^\top}_{\text{写入新关联}}
> $$
> 其中 $\mathbf{v}_t^{\text{new}} = \beta_t \mathbf{v}_t + (1 - \beta_t) \mathbf{v}_t^{\text{old}}$（插值）
>
> **3. 快速权重视角**：用 delta rule 更新快速权重
> - 线性注意力：Hessian-like 更新（记忆容量有限）
> - DeltaNet：Delta rule 更新（记忆容量更大）

**为什么 delta rule 更好**：
- **自适应删除**：根据预测误差自动调整更新幅度
- **更强梯度**：平方损失的梯度与误差成正比，误差大时修正更强
- **更大容量**：理论和实验都证明 delta rule 的记忆容量优于加性更新

---

## 核心机制：并行训练算法

### 挑战：Delta Rule 的顺序依赖

Delta rule 的递归形式本质上是顺序的：

$$
\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top
$$

每一步都需要显式计算 $\mathbf{S}_{t-1}$（$d \times d$ 矩阵），导致：
- **内存开销**：$O(Ld^2)$（需要存储所有时间步的状态）
- **无法并行**：必须按时间步顺序计算
- **硬件低效**：无法利用 GPU 的并行计算能力

这使得早期的 DeltaNet 实现（Schlag et al., 2021）只能在小规模任务上验证，无法扩展到实际的语言模型训练。

### 解决方案：WY 表示 + 分块并行

本文的核心贡献是证明：**可以在 $O(d)$ 内存下计算 delta rule，并实现分块并行**。

#### 步骤 1：伪值向量（Pseudo-value Vectors）

关键观察：$\mathbf{S}_t$ 可以表示为加性形式

$$
\mathbf{S}_t = \sum_{i=1}^t \mathbf{u}_i \mathbf{k}_i^\top
$$

其中 $\mathbf{u}_i$ 是"伪值向量"，定义为：

$$
\mathbf{u}_t = \beta_t \left( \mathbf{v}_t - \sum_{i=1}^{t-1} \mathbf{u}_i (\mathbf{k}_i^\top \mathbf{k}_t) \right)
$$

**关键性质**：
- $\mathbf{u}_t$ 只需要 $O(d)$ 内存（不需要显式存储 $\mathbf{S}_{t-1}$）
- 计算 $\mathbf{u}_t$ 的复杂度是 $O(td)$（需要遍历历史）
- 一旦得到所有 $\mathbf{u}_i$，后续计算与线性注意力相同

> [!math] 推导：为什么 $\mathbf{u}_t$ 不需要显式的 $\mathbf{S}_{t-1}$？
>
> 从 delta rule 出发：
> $$
> \mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top
> $$
>
> 代入 $\mathbf{S}_{t-1} = \sum_{i=1}^{t-1} \mathbf{u}_i \mathbf{k}_i^\top$：
> $$
> \mathbf{S}_t = \sum_{i=1}^{t-1} \mathbf{u}_i \mathbf{k}_i^\top - \beta_t \left( \sum_{i=1}^{t-1} \mathbf{u}_i \mathbf{k}_i^\top \right) \mathbf{k}_t \mathbf{k}_t^\top + \beta_t \mathbf{v}_t \mathbf{k}_t^\top
> $$
>
> 整理得：
> $$
> \mathbf{S}_t = \sum_{i=1}^{t-1} \mathbf{u}_i \mathbf{k}_i^\top + \underbrace{\beta_t \left( \mathbf{v}_t - \sum_{i=1}^{t-1} \mathbf{u}_i (\mathbf{k}_i^\top \mathbf{k}_t) \right)}_{\mathbf{u}_t} \mathbf{k}_t^\top
> $$
>
> 这就是 Householder 矩阵的 **WY 表示**：将矩阵乘积表示为低秩更新的累积。

#### 步骤 2：分块并行（Chunkwise Parallel Form）

将序列分成 $L/C$ 个长度为 $C$ 的块，定义块级变量：

$$
\mathbf{P}_{[t]}^r = \prod_{i=1}^r (\mathbf{I} - \beta_{[t]}^i \mathbf{k}_{[t]}^i \mathbf{k}_{[t]}^{i\top}), \quad \mathbf{H}_{[t]}^r = \sum_{i=1}^r \beta_{[t]}^i \mathbf{v}_{[t]}^i \mathbf{k}_{[t]}^{i\top} \mathbf{P}_{[t+1]}^r
$$

块级递归：

$$
\mathbf{S}_{[t+1]} = \mathbf{S}_{[t]} \mathbf{P}_{[t]} + \mathbf{H}_{[t]}
$$

**关键技巧**：用 WY 表示压缩 $\mathbf{P}_{[t]}$ 和 $\mathbf{H}_{[t]}$

$$
\mathbf{P}_{[t]}^r = \mathbf{I} - \sum_{i=1}^r \mathbf{w}_{[t]}^i \mathbf{k}_{[t]}^{i\top}, \quad \mathbf{H}_{[t]}^r = \sum_{i=1}^r \mathbf{u}_{[t]}^i \mathbf{k}_{[t]}^{i\top}
$$

其中 $\mathbf{w}_{[t]}^i$ 和 $\mathbf{u}_{[t]}^i$ 可以通过递归计算（类似步骤 1）。

**最终算法**：
1. **块内并行**：每个块内独立计算 $\mathbf{u}_{[t]}^i$ 和 $\mathbf{w}_{[t]}^i$（可并行）
2. **块间递归**：按块顺序更新 $\mathbf{S}_{[t]}$（必须顺序）
3. **输出计算**：与线性注意力相同

**复杂度分析**：
- 时间：$O(L^2 d / C + Ld^2)$（块内 $O(C^2d)$，块间 $O(d^2)$）
- 空间：$O(Ld + d^2)$（不需要存储所有 $\mathbf{S}_t$）
- 并行度：块内完全并行，块间顺序（$L/C$ 步）

---

## 实验结果

### 语言建模性能

在 SlimPajama 数据集上训练，与 Mamba、GLA（Gated Linear Attention）等基线对比：

| 模型 | 规模 | Tokens | Wikitext PPL | 平均准确率 | SWDE | SQuAD | FDA |
|------|------|--------|--------------|------------|------|-------|-----|
| Transformer++ | 340M | 15B | 28.39 | 41.2% | 42.2% | 22.1% | 21.4% |
| Mamba | 340M | 15B | 28.39 | 41.8% | 12.4% | 23.0% | 2.1% |
| GLA | 340M | 15B | 28.65 | 41.5% | 18.6% | 27.2% | 8.1% |
| **DeltaNet** | 340M | 15B | **28.24** | **42.1%** | **26.4%** | **28.9%** | **12.8%** |
| **DeltaNet + Sliding Attn** | 340M | 15B | **27.06** | **42.1%** | **39.3%** | **32.5%** | **18.8%** |

**关键发现**：
1. **纯 DeltaNet**：在相同状态大小下，显著优于 Mamba 和 GLA
2. **召回任务**：在 SWDE、SQuAD、FDA 等需要精确检索的任务上，DeltaNet 的优势更明显（验证了 delta rule 的记忆容量优势）
3. **混合架构**：DeltaNet + [[Sliding Window Attention]] 进一步提升性能，接近 Transformer

### 1.3B 模型（100B tokens）

| 模型 | Wikitext PPL | 平均准确率 | SWDE | SQuAD | FDA |
|------|--------------|------------|------|-------|-----|
| Transformer++ | 16.85 | 50.9% | 66.6% | 31.5% | 27.4% |
| Mamba | 17.06 | 50.0% | 41.4% | 35.2% | 6.2% |
| GLA | 17.22 | 51.0% | 50.6% | 42.6% | 19.9% |
| **DeltaNet** | **16.87** | **51.6%** | 49.5% | 37.4% | 17.2% |
| **DeltaNet + Global Attn** | **16.55** | **51.8%** | **71.0%** | **43.0%** | **29.8%** |

**观察**：
- 在 1.3B 规模，DeltaNet 在召回任务上不如 GLA（因为 GLA 使用了更大的状态大小 256x vs 128x）
- 混合 2 层全局注意力后，DeltaNet 在召回任务上超越所有基线

---

## 局限性与边界

### 1. 状态大小的可扩展性

DeltaNet 的状态大小扩展性不如 GLA：
- **原因**：Delta rule 的更新涉及 $\mathbf{S}_{t-1} \mathbf{k}_t$，状态大小增加时计算开销更大
- **影响**：在需要大状态的召回密集型任务上，大规模时可能不如 GLA
- **缓解**：混合架构（+ 少量全局注意力层）

### 2. 训练速度

虽然本文提出了并行算法，但 DeltaNet 的训练速度仍略慢于 GLA：
- **原因**：块内需要计算 $\mathbf{u}_t$ 和 $\mathbf{w}_t$（$O(C^2d)$）
- **对比**：GLA 的块内计算是 $O(Cd^2)$，在 $C < d$ 时更快

### 3. 与 Transformer 的差距

在标准语言建模任务上，纯 DeltaNet 仍不如 Transformer：
- **Wikitext PPL**：28.24 vs 28.39（340M）、16.87 vs 16.85（1.3B）
- **需要混合架构**：加入滑动窗口或全局注意力后才能超越

---

## 与相关方法的对比

### DeltaNet vs 线性注意力

| 维度 | 线性注意力 | DeltaNet |
|------|-----------|----------|
| 更新规则 | 加性：$\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | Delta rule：$\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t (\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t) \mathbf{k}_t^\top$ |
| 记忆容量 | 有限（key collision） | 更大（自适应删除） |
| 训练复杂度 | $O(Ld^2)$ | $O(L^2d/C + Ld^2)$ |
| 并行化 | 完全并行 | 分块并行 |
| 召回任务 | 较弱 | 较强 |

### DeltaNet vs Mamba

| 维度 | Mamba | DeltaNet |
|------|-------|----------|
| 架构类型 | 状态空间模型（SSM） | 线性 Transformer |
| 核心机制 | 选择性扫描（Selective Scan） | Delta rule 更新 |
| 状态大小 | 小（64x） | 中等（128x） |
| 召回任务 | 弱 | 强 |
| 训练速度 | 快 | 中等 |

**关键区别**：
- Mamba 通过选择性机制动态调整状态，但状态大小受限
- DeltaNet 通过 delta rule 提升记忆容量，但需要更大的状态

### DeltaNet vs GLA

| 维度 | GLA | DeltaNet |
|------|-----|----------|
| 更新规则 | 门控线性注意力 | Delta rule |
| 状态扩展性 | 好（可扩展到 256x） | 中等（128x） |
| 召回任务（小状态） | 中等 | 强 |
| 召回任务（大状态） | 强 | 中等 |
| 训练速度 | 快 | 中等 |

**选择建议**：
- 需要强召回 + 小状态：DeltaNet
- 需要强召回 + 大状态：GLA
- 需要最快训练：GLA

---

## 实现细节

### 特征映射与归一化

**特征映射**：$\phi(\mathbf{x})$，将输入映射到非负空间（线性注意力的要求）

实验发现最佳组合：
- **特征映射**：SiLU（$\text{silu}(x) = x \cdot \sigma(x)$）
- **归一化**：L2-norm（而非 L1-norm）

### 混合架构

**两种混合方式**：
1. **Sliding Window Attention**：每隔一层插入滑动窗口注意力
2. **Global Attention**：在特定层（如第 2 层和倒数第 2 层）使用全局注意力

**效果**：
- 混合架构在召回任务上显著提升（SWDE: 26.4% → 39.3%）
- 在标准任务上也有提升（Wikitext PPL: 28.24 → 27.06）

---

## 延伸阅读

**线性注意力的演进**：
- [[Linear Attention]] — 基础概念（待创建）
- [[GLA]] — 门控线性注意力（待创建）
- [[Mamba]] — 状态空间模型（待创建）

**混合架构**：
- [[Sliding Window Attention]] — 滑动窗口注意力
- [[Sparse Attention]] — 稀疏注意力

**理论基础**：
- [[Fast Weight Programming]] — 快速权重视角（待创建）
- [[Widrow-Hoff Rule]] — Delta rule 的起源（待创建）

---

> [!interview] 面试要点
>
> **Q1: DeltaNet 解决了什么问题？**
> A: 线性注意力的 key collision 问题。传统线性注意力只能加性更新，无法删除过时信息，导致记忆容量受限。DeltaNet 用 delta rule 实现自适应删除，根据预测误差调整更新幅度。
>
> **Q2: Delta rule 的核心思想是什么？**
> A: 基于预测误差更新：$\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t (\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t) \mathbf{k}_t^\top$。可以理解为：先用旧记忆预测，计算误差，然后按误差大小修正。类似在线学习的单步 SGD。
>
> **Q3: 为什么 delta rule 需要特殊的并行算法？**
> A: 因为 delta rule 的递归形式需要显式计算 $\mathbf{S}_{t-1}$（$d \times d$ 矩阵），导致 $O(Ld^2)$ 内存和无法并行。本文用 Householder 矩阵的 WY 表示，将状态表示为低秩更新的累积，实现 $O(d)$ 内存和分块并行。
>
> **Q4: DeltaNet vs Mamba 的核心区别？**
> A:
> - **架构**：Mamba 是 SSM，DeltaNet 是线性 Transformer
> - **记忆机制**：Mamba 用选择性扫描动态调整状态，DeltaNet 用 delta rule 自适应删除
> - **召回能力**：DeltaNet 在召回任务上显著优于 Mamba（因为更大的记忆容量）
>
> **Q5: DeltaNet 的局限性？**
> A:
> 1. 状态大小扩展性不如 GLA（在大状态时计算开销更大）
> 2. 训练速度略慢于 GLA（块内计算复杂度更高）
> 3. 纯 DeltaNet 不如 Transformer，需要混合架构
