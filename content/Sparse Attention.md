---
description: 通过只计算部分注意力权重降低复杂度的方法族，是处理长序列的重要范式
type: concept
aliases:
  - 稀疏注意力
prerequisites:
  - "[[Attention]]"
  - "[[Multi-Head Attention]]"
tags:
  - attention
  - efficient-attention
  - long-context
created: 2026-01-29
updated: 2026-02-01T01:12
---

# Sparse Attention

Sparse Attention（稀疏注意力）是一类通过**只计算部分注意力权重**来降低复杂度的方法。核心思想是：不是所有 token 对都需要交互，通过预定义或学习的稀疏模式，将 $O(N^2)$ 复杂度降至 $O(N \cdot k)$，其中 $k \ll N$。

---

## 动机

> [!intuition] 为什么需要 Sparse Attention？
> 标准 [[Attention]] 的 $O(N^2)$ 复杂度是处理长序列的主要瓶颈。但观察发现：**大多数注意力权重接近于零**——模型实际上只关注少数关键位置。Sparse Attention 利用这一先验，跳过不重要的计算。

### 标准 Attention 的问题

| 序列长度 | Attention 计算量 | 内存占用 |
|----------|------------------|----------|
| 1K | $1 \times$ | $1 \times$ |
| 8K | $64 \times$ | $8 \times$ |
| 32K | $1024 \times$ | $32 \times$ |
| 128K | $16384 \times$ | $128 \times$ |

当序列长度增加时，计算量以平方增长，内存以线性增长，两者都成为瓶颈。

---

## 核心思想

> [!definition] Sparse Attention
> 定义稀疏连接模式 $\mathcal{S}(i) \subseteq \{1, ..., N\}$，位置 $i$ 只 attend 到 $\mathcal{S}(i)$ 中的位置：
> $$\text{SparseAttn}(Q, K, V)_i = \text{softmax}\left(\frac{q_i K_{\mathcal{S}(i)}^T}{\sqrt{d_k}}\right) V_{\mathcal{S}(i)}$$

不同的稀疏模式定义了不同的 $\mathcal{S}(i)$，形成了 Sparse Attention 的各种变体。

### 稀疏模式可视化

```
Full Attention:          Sparse Attention (示例):
■ ■ ■ ■ ■ ■ ■ ■          ■ □ □ □ ■ □ □ □
■ ■ ■ ■ ■ ■ ■ ■          ■ ■ □ □ □ ■ □ □
■ ■ ■ ■ ■ ■ ■ ■          ■ ■ ■ □ □ □ ■ □
■ ■ ■ ■ ■ ■ ■ ■    →     ■ ■ ■ ■ □ □ □ ■
■ ■ ■ ■ ■ ■ ■ ■          ■ □ □ □ ■ □ □ □
■ ■ ■ ■ ■ ■ ■ ■          □ ■ □ □ ■ ■ □ □
■ ■ ■ ■ ■ ■ ■ ■          □ □ ■ □ ■ ■ ■ □
■ ■ ■ ■ ■ ■ ■ ■          □ □ □ ■ ■ ■ ■ ■

■ = 计算 attention    □ = 跳过（设为 0）
```

---

## 主要稀疏模式

### 1. Local/Sliding Window

> [!intuition] 局部窗口
> 每个 token 只关注固定大小的局部窗口，假设局部上下文最重要。

$$\mathcal{S}_{\text{local}}(i) = \{j : |i - j| \leq w\}$$

这正是 [[Sliding Window Attention]] 的核心思想。

```
窗口大小 w=2:
□ □ □ □ □ □ □ □
■ □ □ □ □ □ □ □
■ ■ □ □ □ □ □ □
■ ■ ■ □ □ □ □ □
□ ■ ■ ■ □ □ □ □
□ □ ■ ■ ■ □ □ □
□ □ □ ■ ■ ■ □ □
□ □ □ □ ■ ■ ■ □
```

**优点**：简单高效，适合语言建模
**缺点**：无法直接捕捉长程依赖

### 2. Strided (跳跃式)

> [!intuition] 跳跃采样
> 以固定步长采样，用少量位置覆盖更大范围。

$$\mathcal{S}_{\text{strided}}(i) = \{j : (i - j) \mod s = 0\}$$

```
步长 s=2:
■ □ ■ □ ■ □ ■ □
□ ■ □ ■ □ ■ □ ■
■ □ ■ □ ■ □ ■ □
□ ■ □ ■ □ ■ □ ■
■ □ ■ □ ■ □ ■ □
□ ■ □ ■ □ ■ □ ■
■ □ ■ □ ■ □ ■ □
□ ■ □ ■ □ ■ □ ■
```

**优点**：覆盖范围大
**缺点**：可能错过重要的相邻信息

### 3. Global Tokens

> [!intuition] 全局 token
> 指定某些位置（如 [CLS]、句首）可以 attend 到所有位置，也被所有位置 attend。

$$\mathcal{S}_{\text{global}}(i) = \begin{cases} \{1, ..., N\} & \text{if } i \in \mathcal{G} \\ \mathcal{S}_{\text{local}}(i) \cup \mathcal{G} & \text{otherwise} \end{cases}$$

其中 $\mathcal{G}$ 是全局 token 的位置集合。

**优点**：保留全局信息聚合能力
**缺点**：需要预先指定哪些位置是全局的

---

## 经典方法

### Sparse Transformer (OpenAI, 2019)

> [!paper] 原始论文
> "Generating Long Sequences with Sparse Transformers" (Child et al., 2019)

组合 Local + Strided 两种模式：

- **Local head**：关注局部窗口
- **Strided head**：以固定步长采样

```
Local + Strided 组合:
■ □ □ □ □ □ □ □     ■ □ □ □ □ □ □ □     ■ □ □ □ □ □ □ □
■ ■ □ □ □ □ □ □  +  □ ■ □ □ □ □ □ □  =  ■ ■ □ □ □ □ □ □
■ ■ ■ □ □ □ □ □     ■ □ ■ □ □ □ □ □     ■ ■ ■ □ □ □ □ □
■ ■ ■ ■ □ □ □ □     □ ■ □ ■ □ □ □ □     ■ ■ ■ ■ □ □ □ □
□ ■ ■ ■ ■ □ □ □     ■ □ ■ □ ■ □ □ □     ■ ■ ■ ■ ■ □ □ □
□ □ ■ ■ ■ ■ □ □     □ ■ □ ■ □ ■ □ □     □ ■ ■ ■ ■ ■ □ □
□ □ □ ■ ■ ■ ■ □     ■ □ ■ □ ■ □ ■ □     ■ □ ■ ■ ■ ■ ■ □
□ □ □ □ ■ ■ ■ ■     □ ■ □ ■ □ ■ □ ■     □ ■ □ ■ ■ ■ ■ ■
   Local              Strided             Combined
```

**复杂度**：$O(N \sqrt{N})$

### Longformer (2020)

> [!paper] 原始论文
> "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)

组合 Local + Global：

- **Local attention**：所有位置使用滑动窗口
- **Global attention**：特定位置（如 [CLS]）使用全局注意力

```
Longformer 模式 (G = global token):
G ■ ■ ■ ■ ■ ■ ■ ■    ← Global token attend 所有
■ ■ ■ □ □ □ □ □ G    ← 普通 token: local + global
■ ■ ■ ■ □ □ □ □ G
□ ■ ■ ■ ■ □ □ □ G
□ □ ■ ■ ■ ■ □ □ G
□ □ □ ■ ■ ■ ■ □ G
□ □ □ □ ■ ■ ■ ■ G
□ □ □ □ □ ■ ■ ■ G
G G G G G G G G ■    ← Global token 被所有 attend
```

**复杂度**：$O(N \cdot (w + g))$，其中 $w$ 是窗口大小，$g$ 是全局 token 数量

### BigBird (2020)

> [!paper] 原始论文
> "Big Bird: Transformers for Longer Sequences" (Zaheer et al., 2020)

组合三种模式：Local + Global + Random

- **Local**：滑动窗口
- **Global**：特定位置全局连接
- **Random**：随机采样一些位置

```
BigBird 模式:
Local + Global + Random = 更丰富的连接模式
```

**理论贡献**：证明了这种组合是图灵完备的，且能近似任意序列到序列函数。

**复杂度**：$O(N)$

---

## 与 Sliding Window Attention 的关系

> [!comparison] SWA 是 Sparse Attention 的特例
> [[Sliding Window Attention]] 是最简单的 Sparse Attention 形式：
> - **稀疏模式**：纯 Local（只有滑动窗口）
> - **优点**：实现简单，[[KV Cache]] 固定
> - **缺点**：无法直接捕捉长程依赖
>
> 更复杂的 Sparse Attention（如 Longformer、BigBird）通过添加 Global/Random 模式来弥补这一缺陷。

| 方法 | 稀疏模式 | 长程依赖 | 实现复杂度 |
|------|----------|----------|------------|
| SWA | Local only | 需要多层传播 | 简单 |
| Longformer | Local + Global | 通过 Global token | 中等 |
| BigBird | Local + Global + Random | 多种路径 | 复杂 |

---

## 与 Linear Attention 的对比

> [!comparison] 两种不同的效率化思路
> | 维度 | Sparse Attention | [[Linear Attention]] |
> |------|------------------|---------------------|
> | **核心思想** | 跳过不重要的计算 | 改变计算顺序 |
> | **Softmax** | 保留 | 去掉/近似 |
> | **复杂度** | $O(N \cdot k)$ | $O(N \cdot d^2)$ |
> | **精度** | 精确（在稀疏模式内） | 近似 |
> | **稳定性** | 稳定 | 可能不稳定 |
> | **实现** | 需要稀疏矩阵支持 | 需要 kernel 近似 |

**选择建议**：
- 需要精确 attention 且有明确稀疏先验 -> Sparse Attention
- 需要线性复杂度且能接受近似 -> Linear Attention

---

## 与 Flash Attention 的关系

> [!intuition] 正交的优化方向
> - **Sparse Attention**：改变**计算什么**（跳过部分 attention）
> - **[[Flash Attention]]**：改变**如何计算**（IO-aware 实现）
>
> 两者可以**同时使用**：用 Flash Attention 高效实现 Sparse Attention 的稀疏模式。

实际上，Flash Attention 2/3 已经支持多种稀疏模式的高效实现。

---

## 在现代 LLM 中的应用

### 主流模型的选择

| 模型 | 方法 | 说明 |
|------|------|------|
| GPT-4 | 未公开 | 推测使用某种稀疏模式 |
| Mistral 7B | SWA | 纯滑动窗口 |
| Llama 3 | Full Attention | 依赖 Flash Attention 优化 |
| MiMo-V2-Flash | Hybrid SWA | 5:1 SWA + Global |
| LongCat-Flash | Zigzag Attention | MLA + Streaming Sparse |

### 趋势观察

> [!intuition] 现代趋势
> 1. **Hybrid 架构流行**：纯稀疏效果不佳，混合 Local + Global 更实用
> 2. **与 MLA 结合**：[[Multi-head Latent Attention]] + Sparse 进一步压缩
> 3. **Flash Attention 降低需求**：高效实现让 Full Attention 在中等长度可行
> 4. **长上下文需求增加**：128K+ 上下文让 Sparse Attention 重新重要

---

## 局限性

> [!warning] Sparse Attention 的边界条件
> 1. **稀疏模式选择困难**：不同任务可能需要不同模式，预定义模式可能不最优
> 2. **实现复杂**：稀疏矩阵运算在 GPU 上效率不如 dense，需要特殊优化
> 3. **信息丢失风险**：如果重要的 token 对被跳过，会损失关键信息
> 4. **训练-推理一致性**：训练用 Full Attention，推理用 Sparse 可能有性能差距
> 5. **与 KV Cache 的交互**：某些稀疏模式（如 Random）难以利用 KV Cache

---

## 面试要点

> [!interview] 面试视角
> **Q: Sparse Attention 的核心思想是什么？**
> A: 只计算部分 attention 权重，跳过不重要的 token 对。通过预定义的稀疏模式（如 Local、Strided、Global）将 $O(N^2)$ 降至 $O(N \cdot k)$。
>
> **Q: 常见的稀疏模式有哪些？**
> A: Local（滑动窗口）、Strided（跳跃采样）、Global（特定位置全局连接）、Random（随机采样）。实际方法通常组合多种模式。
>
> **Q: Sparse Attention 和 Flash Attention 的区别？**
> A: Sparse Attention 改变计算什么（跳过部分 attention），Flash Attention 改变如何计算（IO-aware 实现）。两者正交，可以同时使用。
>
> **Q: 为什么现代 LLM 不都用 Sparse Attention？**
> A: 1) Flash Attention 让 Full Attention 在中等长度可行；2) 稀疏模式选择困难；3) 实现复杂度高；4) 可能损失重要信息。但在超长上下文（128K+）场景，Sparse Attention 仍然重要。

---

## 相关概念

- [[Attention]] — 标准注意力机制
- [[Sliding Window Attention]] — 最简单的 Sparse Attention 形式
- [[Flash Attention]] — IO-aware 的高效实现，可与 Sparse 结合
- [[Linear Attention]] — 另一种效率化思路，去掉 softmax
- [[Multi-head Latent Attention]] — 通过压缩 KV 降低复杂度
- [[Length Extrapolation]] — 长度外推问题

---

## 参考资料

- Generating Long Sequences with Sparse Transformers (Child et al., 2019) — OpenAI 的 Sparse Transformer
- Longformer: The Long-Document Transformer (Beltagy et al., 2020) — Local + Global 组合
- Big Bird: Transformers for Longer Sequences (Zaheer et al., 2020) — Local + Global + Random，理论分析
- Mistral 7B (Jiang et al., 2023) — 在主流模型中使用 SWA
