---
description: Attention 机制导航：核心机制、KV Cache 变体、位置编码、长上下文与高效注意力
type: moc
tags:
  - attention
  - architecture
  - inference
created: 2025-01-26
updated: 2026-05-29T00:00
---

# MOC - Attention

Attention 是 Transformer 的核心计算机制，也是推理成本、长上下文能力和现代架构改进的交汇点。本 MOC 组织从基础公式到 KV Cache、位置编码、高效注意力和长序列建模的学习路径。

---

## 概览

```
Attention
├── 核心机制
│   ├── [[Attention]] — Scaled Dot-Product Attention
│   └── [[Multi-Head Attention]] — 多头并行关注不同子空间
│
├── KV Cache 与推理效率
│   ├── [[KV Cache]] — 自回归生成的缓存机制
│   ├── [[Multi-Query Attention]] — 所有 head 共享 K/V
│   ├── [[Grouped-Query Attention]] — 分组共享 K/V
│   └── [[Multi-head Latent Attention]] — DeepSeek 的低秩 KV 压缩
│
├── 位置与长度
│   ├── [[Positional Encoding]] — 位置信息注入
│   ├── [[RoPE]] / [[ALiBi]] — 主流位置编码
│   └── [[Length Extrapolation]] — 超出训练长度的推理
│
├── 高效注意力
│   ├── [[Flash Attention]] — IO-aware 精确算法
│   ├── [[Sparse Attention]] / [[Sliding Window Attention]]
│   └── [[Linear Attention]] / [[DeltaNet]]
│
└── 训练与稳定性
    ├── [[Attention Sink]] — attention outlier 与 sink 现象
    ├── [[Gated Attention]] — 用 gating 缓解 sink 并提升稳定性
    └── [[Attention Residuals]] — 用 attention 聚合跨层残差
```

---

## 核心机制

### Scaled Dot-Product Attention

[[Attention]] 用 Query 和 Key 的相似度决定每个位置应从 Value 中读取多少信息：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

这里有三个学习重点：

- $QK^T$ 建立 token 间的动态相关性。
- $\sqrt{d_k}$ 缩放避免 dot product 随维度增大导致 softmax 饱和。
- softmax 让每个位置形成对上下文的加权读取。

### Multi-Head Attention

[[Multi-Head Attention]] 让多个 head 并行学习不同子空间中的依赖关系：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

它带来更强的表达能力，也带来推理时 KV Cache 随 head 数增长的问题。后续的 MQA、GQA、MLA 都围绕这个瓶颈展开。

---

## KV Cache 与推理效率

自回归生成每次只新增一个 token，历史 token 的 Key/Value 可以复用。[[KV Cache]] 将这些 K/V 缓存在显存中，避免重复计算历史前缀。

| 方案 | 核心思想 | KV Cache 规模 | 典型用途 |
|------|----------|---------------|----------|
| [[Multi-Head Attention|MHA]] | 每个 head 有独立 K/V | 最大 | 基础结构，质量稳定 |
| [[Grouped-Query Attention|GQA]] | 多个 query head 共享一组 K/V | 中等 | Llama、Mistral、Qwen 常用 |
| [[Multi-Query Attention|MQA]] | 所有 query head 共享 K/V | 最小 | 强调推理效率 |
| [[Multi-head Latent Attention|MLA]] | 将 K/V 压缩到 latent 表示 | 极小 | DeepSeek 系列的核心优化 |

[[Paged Attention]] 和 [[Radix Attention]] 属于服务层的 KV Cache 管理：前者解决显存碎片，后者复用跨请求公共前缀。它们在 [[MOC - Inference]] 中展开。

---

## 位置编码与长度外推

[[Attention]] 本身对位置顺序不敏感，需要 [[Positional Encoding]] 注入顺序信息。

| 方法 | 位置类型 | 核心特点 | 适合关注 |
|------|----------|----------|----------|
| Sinusoidal PE | 绝对位置 | 原始 Transformer 方案 | 基础理解 |
| Learnable PE | 绝对位置 | 位置向量可学习 | BERT/GPT 早期模型 |
| [[RoPE]] | 相对位置信息 | 通过旋转编码相对距离 | Llama/Qwen/Mistral 主流方案 |
| [[ALiBi]] | 相对偏置 | 对远距离 attention 加线性惩罚 | 长度外推 |

[[Length Extrapolation]] 关注模型在超出训练长度时为何退化，以及 PI、NTK scaling、YaRN 等方法如何调整位置编码或上下文分布。

---

## 高效 Attention

Attention 的效率优化可以分成三层：

| 层级 | 代表方法 | 解决的问题 |
|------|----------|------------|
| Kernel 层 | [[Flash Attention]] | 用 tiling 和 online softmax 降低 HBM 读写 |
| 稀疏模式层 | [[Sparse Attention]], [[Sliding Window Attention]] | 降低参与 attention 的 token 数 |
| 线性化层 | [[Linear Attention]], [[DeltaNet]] | 将二次复杂度改写为线性复杂度 |

### Flash Attention

[[Flash Attention]] 保持精确 attention 结果，通过重排计算和减少 HBM 访问提升速度，并把显存复杂度从 $O(n^2)$ 降到 $O(n)$。它是训练和推理框架中的基础优化。

### Sparse 与 Sliding Window

[[Sparse Attention]] 只计算部分注意力边，[[Sliding Window Attention]] 让每个 token 只关注局部窗口。它们适合长上下文，但需要处理远距离信息传递和全局 token 设计。

[[Attention Sink]] 解释了为什么一些 token 会吸收大量 attention 权重。[[Gated Attention]] 通过在 SDPA 输出后加入 gating，引入非线性和稀疏性，缓解 sink 现象并提升稳定性。

### Linear Attention 与 Test-Time Regression

[[Linear Attention]] 通过 kernel trick 或状态递推降低复杂度。[[DeltaNet]] 用 delta rule 更新线性 Transformer 状态，[[Test-time Regression (2025)]] 进一步把 softmax attention、linear attention、SSM 和 fast-weight programmers 统一到 test-time regression 视角。

---

## 跨层与长推理

[[Attention Residuals]] 用 softmax attention 替代固定残差累加，让每层通过可学习 pseudo-query 选择性聚合前面层输出，解决 PreNorm 下隐藏状态随层数增长带来的层贡献稀释。

[[MEMENTO]] 将推理链分段并生成压缩摘要，通过原位 KV Cache 遮蔽降低长推理峰值内存。它把 attention 的上下文管理问题连接到 reasoning model 的推理效率。

---

## 学习路径建议

**基础路线**：
1. [[Attention]] — 理解 $QK^T$、softmax 和 Value 加权。
2. [[Multi-Head Attention]] — 理解多头如何扩展表达能力。
3. [[Positional Encoding]] → [[RoPE]] — 理解顺序信息如何进入模型。

**推理效率路线**：
1. [[KV Cache]] — 建立自回归推理的显存模型。
2. [[Grouped-Query Attention]] / [[Multi-Query Attention]] / [[Multi-head Latent Attention]] — 理解架构层的 KV 压缩。
3. [[Paged Attention]] / [[Radix Attention]] — 理解 serving 层的缓存管理。

**长上下文路线**：
1. [[Length Extrapolation]] — 理解训练长度外推的挑战。
2. [[Sliding Window Attention]] → [[Attention Sink]] → [[Gated Attention]] — 理解局部注意力与稳定性。
3. [[Linear Attention]] → [[DeltaNet]] → [[Test-time Regression (2025)]] — 理解线性序列模型的统一视角。

---

## 相关 MOC

- [[MOC - Foundations]] — Transformer、信息论和基础表示
- [[MOC - Inference]] — KV Cache、serving、推理加速与量化
- [[MOC - Distributed Training]] — Flash Attention 与训练系统优化
