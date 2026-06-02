---
description: 基于 Attention 的序列建模架构，抛弃 RNN 的递归结构，实现完全并行化训练
type: concept
aliases:
  - Transformer 架构
prerequisites: []
tags:
  - architecture
  - foundations
  - transformer
created: 2026-01-27
updated: 2026-05-31T17:53
---

# Transformer

Transformer 是一种基于 [[Attention]] 机制的序列到序列架构，完全抛弃了 RNN 的递归结构，实现了训练的完全并行化。它是现代大语言模型（LLM）的基础架构。

> [!paper] 论文出处
> Vaswani et al., "Attention Is All You Need", NeurIPS 2017
> 标题本身就是核心贡献：只用 Attention，不需要 RNN/CNN。

---

## 核心思想

> [!intuition] 为什么要抛弃 RNN？
> RNN 的核心问题是**顺序依赖**：必须先算 $h_1$，才能算 $h_2$，再算 $h_3$...
>
> 这导致：
> 1. **无法并行**：训练时间随序列长度线性增长
> 2. **长距离依赖困难**：信息需要逐步传递，容易丢失
> 3. **梯度问题**：梯度消失/爆炸
>
> **Transformer 的解决方案**：用 Attention 直接建模任意两个位置的关系，完全并行。

---

## 架构变体

```
Transformer 架构
├── Encoder-Decoder（原始 Transformer）
│   └── 用于：机器翻译、T5、BART
│
├── Encoder-only
│   └── 用于：BERT、文本分类、NLU
│
└── Decoder-only ⭐ LLM 主流
    └── 用于：GPT 系列、Llama、Claude
```

### Encoder-Decoder

原始 Transformer 架构，用于序列到序列任务（如机器翻译）：

- **Encoder**：双向 Self-Attention，理解输入序列
- **Decoder**：Causal Self-Attention + Cross-Attention，生成输出序列
- **代表模型**：T5、BART、原始 Transformer

### Encoder-only

只保留 Encoder，用于理解任务：

- **特点**：双向 Attention，每个位置可以看到所有其他位置
- **代表模型**：BERT、RoBERTa
- **典型任务**：文本分类、NER、句子相似度

### Decoder-only

只保留 Decoder，用于生成任务：

- **特点**：Causal Attention，每个位置只能看到之前的位置
- **代表模型**：GPT 系列、Llama、Claude、Qwen
- **为什么成为 LLM 主流**：自回归生成天然适合语言建模，Scaling 效果好

> [!intuition] 为什么 LLM 都用 Decoder-only？
> 1. **统一的预训练目标**：Next Token Prediction 简单有效
> 2. **Scaling 友好**：参数量增加时性能持续提升
> 3. **生成能力强**：天然支持自回归生成
> 4. **In-context Learning**：涌现出强大的上下文学习能力

---

## 核心组件

一个标准的 Transformer Block 包含以下组件：

```
输入 X
    │
    ▼
┌─────────────────────────────┐
│  [[Multi-Head Attention]]   │ ← Self-Attention / Cross-Attention
└─────────────────────────────┘
    │
    ▼ Add & Norm (残差连接 + LayerNorm)
    │
    ▼
┌─────────────────────────────┐
│  Feed-Forward Network (FFN) │ ← 两层 MLP
└─────────────────────────────┘
    │
    ▼ Add & Norm (残差连接 + LayerNorm)
    │
    ▼
输出
```

### Multi-Head Attention

[[Multi-Head Attention]] 是 Transformer 的核心，让模型能够同时关注不同子空间的信息：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

详见 [[Attention]] 和 [[Multi-Head Attention]]。

### Feed-Forward Network (FFN)

每个位置独立应用的两层 MLP：

> [!definition] FFN 定义
> $$
> \text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
> $$
>
> 或使用 GELU/SwiGLU 等激活函数（现代 LLM 常用 SwiGLU）。

**维度设计**：
- 输入/输出维度：$d_{model}$
- 隐藏层维度：$d_{ff} = 4 \times d_{model}$（原始设计）

> [!intuition] FFN 的作用
> - **Attention** 负责 token 之间的信息交换
> - **FFN** 负责每个位置的非线性变换，可以理解为"思考"
>
> 有研究认为 FFN 存储了大量的世界知识（Key-Value Memory 假说）。

### Layer Normalization

> [!definition] LayerNorm
> $$
> \text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
> $$
>
> 其中 $\mu$、$\sigma$ 是沿特征维度计算的均值和标准差。

**Pre-Norm vs Post-Norm**：

| 方式 | 公式 | 特点 |
|------|------|------|
| **Post-Norm** | $x + \text{Sublayer}(\text{LN}(x))$ | 原始 Transformer |
| **Pre-Norm** | $\text{LN}(x + \text{Sublayer}(x))$ | 训练更稳定，现代 LLM 主流 |

> [!intuition] 为什么现代 LLM 用 Pre-Norm？
> Pre-Norm 让残差路径更"干净"，梯度可以直接流过，训练更稳定。
> 代价是最终表示可能略弱，但对于大模型来说稳定性更重要。

### 残差连接

每个子层都有残差连接：

$$
\text{output} = x + \text{Sublayer}(x)
$$

**作用**：
- 缓解梯度消失
- 允许堆叠很深的网络
- 让模型可以学习"恒等映射"

---

## 位置编码

[[Attention]] 是置换不变的，无法区分 token 顺序。[[Positional Encoding]] 注入位置信息。

**原始 Transformer 使用 Sinusoidal PE**：

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{aligned}
$$

**现代 LLM 主流方案**：
- [[RoPE]]（Llama、Qwen、Mistral）— 旋转位置编码
- [[ALiBi]]（BLOOM）— 注意力线性偏置

详见 [[Positional Encoding]]。

---

## 与 RNN/LSTM 的对比

| 方面 | RNN/LSTM | Transformer |
|------|----------|-------------|
| **并行性** | 顺序计算，无法并行 | 完全并行 |
| **长距离依赖** | 信息逐步传递，容易丢失 | 直接建模任意距离 |
| **计算复杂度** | $O(n)$ | $O(n^2)$（Attention） |
| **内存** | $O(1)$ 隐状态 | $O(n^2)$ 注意力矩阵 |
| **训练效率** | 慢（顺序） | 快（并行） |
| **推理效率** | 快（增量） | 需要 [[KV Cache]] 优化 |

> [!comparison] 核心权衡
> Transformer 用**更高的计算复杂度**换取了**并行训练能力**和**更好的长距离建模**。
>
> 在 GPU 时代，并行性带来的训练加速远超 $O(n^2)$ 的代价。

> [!warning] Transformer 的局限
> - **二次复杂度**：长序列时计算量和内存爆炸（[[Flash Attention]] 通过分块计算缓解内存问题）
> - **位置编码外推**：超出训练长度后性能下降
> - **推理效率**：自回归生成时需要 KV Cache，内存占用大（[[Multi-Query Attention]] 和 [[Grouped-Query Attention]] 通过共享 KV 减少内存）

---

## 参数量分析

对于一个 $L$ 层的 Decoder-only Transformer：

| 组件 | 参数量 |
|------|--------|
| Embedding | $V \times d_{model}$ |
| 每层 Attention | $4 d_{model}^2$（Q/K/V/O） |
| 每层 FFN | $2 \times d_{model} \times d_{ff} \approx 8 d_{model}^2$ |
| 每层 LayerNorm | $4 d_{model}$（可忽略） |
| **总计** | $\approx V \cdot d + 12 L d_{model}^2$ |

> [!example] GPT-3 175B 参数分布
> - $d_{model} = 12288$, $L = 96$, $V = 50257$
> - Embedding: ~0.6B
> - Attention: ~58B
> - FFN: ~116B
> - 总计: ~175B

---

## 在 LLM 中的应用

### GPT 系列架构

GPT（Generative Pre-trained Transformer）使用 Decoder-only 架构：

```
Token Embedding + Positional Encoding
        │
        ▼
┌─────────────────────────────┐
│   Transformer Block × L     │
│   (Causal Self-Attention)   │
└─────────────────────────────┘
        │
        ▼
    LayerNorm
        │
        ▼
    LM Head (Linear → Vocab)
        │
        ▼
    Next Token Probabilities
```

**关键设计**：
- **Causal Mask**：确保每个位置只能看到之前的 token
- **Tied Embedding**：输入 Embedding 和输出 LM Head 共享权重（可选）
- **Pre-Norm**：LayerNorm 在子层之前

### 主流 LLM 架构对比

| 模型 | 架构 | 位置编码 | Attention | 激活函数 |
|------|------|----------|-----------|----------|
| GPT-3 | Decoder-only | 可学习 | MHA | GELU |
| Llama 2 | Decoder-only | [[RoPE]] | MHA/GQA | SwiGLU |
| Llama 3 | Decoder-only | [[RoPE]] | GQA | SwiGLU |
| Mistral | Decoder-only | [[RoPE]] | GQA + SWA | SwiGLU |
| Qwen | Decoder-only | [[RoPE]] | GQA | SwiGLU |

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: Transformer 的核心创新是什么？**
> A: 完全基于 Attention，抛弃 RNN 的递归结构，实现训练的完全并行化。
>
> **Q2: Transformer 有哪些架构变体？LLM 用哪种？**
> A: Encoder-Decoder（T5）、Encoder-only（BERT）、Decoder-only（GPT）。LLM 主要用 Decoder-only，因为自回归生成适合语言建模，Scaling 效果好。
>
> **Q3: Transformer Block 包含哪些组件？**
> A: Multi-Head Attention、FFN、LayerNorm、残差连接。
>
> **Q4: 为什么需要位置编码？**
> A: Attention 是置换不变的，无法区分 token 顺序。位置编码注入位置信息。
>
> **Q5: Transformer 相比 RNN 的优缺点？**
> A: 优点：并行训练、长距离建模好。缺点：$O(n^2)$ 复杂度、推理需要 KV Cache。
>
> **Q6: Pre-Norm 和 Post-Norm 的区别？**
> A: Pre-Norm 把 LayerNorm 放在子层之前，训练更稳定，是现代 LLM 的主流选择。
>
> **Q7: FFN 的作用是什么？**
> A: 每个位置独立的非线性变换，可以理解为"思考"。有研究认为 FFN 存储了世界知识。

---

## 相关概念

**核心组件**：
- [[Attention]] — Transformer 的核心机制
- [[Multi-Head Attention]] — 多头注意力
- [[Positional Encoding]] — 位置编码
- [[RoPE]] — 旋转位置编码
- [[ALiBi]] — 注意力线性偏置

**效率优化**：
- [[KV Cache]] — 推理时的 Attention 优化

**相关 MOC**：
- [[02-MOC - Attention|MOC - Attention]] — Attention 机制导航
