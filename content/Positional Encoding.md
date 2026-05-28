---
description: 为置换不变的 Attention 注入序列位置信息的机制
type: concept
aliases:
  - 位置编码
  - PE
prerequisites:
  - "[[Attention]]"
tags:
  - transformer
  - architecture
  - foundations
created: 2025-01-26
updated: 2026-01-31T22:42
---

# Positional Encoding

位置编码（Positional Encoding）是 Transformer 中用于注入序列位置信息的机制。由于 [[Attention]] 本身是置换不变的（permutation invariant），无法区分不同位置的 token，因此需要额外的位置信息。

> [!paper] 论文出处
> Vaswani et al., "Attention Is All You Need", NeurIPS 2017（Sinusoidal PE）

---

## 为什么需要位置编码？

> [!intuition] Attention 的置换不变性
> Self-Attention 的计算只涉及 token 之间的相似度，与它们的顺序无关：
>
> $$
> \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
> $$
>
> 如果打乱输入序列的顺序，只要相应地打乱 Q、K、V，输出也会相应打乱，但每个位置的输出值不变。
>
> **问题**：语言是有顺序的！"狗咬人" 和 "人咬狗" 意思完全不同。

> [!example] 置换不变性示例
> 输入 "I love you" 和 "you love I"，如果没有位置编码，Self-Attention 无法区分这两个句子。

---

## 位置编码的分类

```
Positional Encoding
├── Absolute PE
│   ├── Sinusoidal PE（原始 Transformer）
│   └── Learnable PE（BERT、GPT）
│
└── Relative PE
    ├── [[RoPE]]（Llama、Qwen）
    ├── [[ALiBi]]（BLOOM）
    └── T5 Relative Bias
```

---

## 绝对位置编码

### Sinusoidal 位置编码

原始 Transformer 使用固定的正弦/余弦函数：

> [!definition] Sinusoidal PE
> $$
> \begin{aligned}
> PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
> PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
> \end{aligned}
> $$
>
> 其中 $pos$ 是位置，$i$ 是维度索引，$d$ 是总维度。

**使用方式**：直接加到输入 embedding 上：

$$
X' = X + PE
$$

> [!intuition] 为什么用正弦/余弦？
> 1. **有界性**：值域在 $[-1, 1]$，不会影响 embedding 的尺度
> 2. **相对位置可表示**：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性变换
> 3. **外推能力**：理论上可以处理任意长度（但实际效果有限）

### 可学习位置编码

BERT、GPT 等模型使用可学习的位置 embedding：

$$
PE = \text{Embedding}(pos) \in \mathbb{R}^{L_{max} \times d}
$$

**优点**：模型可以学习到任务相关的位置表示
**缺点**：无法外推到训练时未见过的长度

---

## 相对位置编码

### 核心思想

> [!intuition] 绝对 vs 相对
> - **绝对位置**：token 在序列中的位置（第 1 个、第 2 个...）
> - **相对位置**：两个 token 之间的距离（相邻、间隔 3 个...）
>
> 相对位置编码认为：**相对距离比绝对位置更重要**。"the cat sat on the mat" 中，"cat" 和 "sat" 的关系不应该因为它们出现在句首还是句中而改变。

### 相对位置编码的优势

1. **更好的泛化**：相对距离的模式更容易迁移
2. **长度外推**：不依赖绝对位置，更容易处理长序列
3. **符合语言直觉**：语法关系通常是局部的

### 主要方法

| 方法 | 核心思想 | 代表模型 |
|------|----------|----------|
| [[RoPE]] | 旋转矩阵编码相对位置 | Llama, Qwen, Mistral |
| [[ALiBi]] | 注意力分数加线性偏置 | BLOOM, MPT |
| T5 Relative Bias | 可学习的相对位置偏置 | T5, Flan-T5 |

---

## 位置编码的注入方式

### 1. 加到输入 (Additive)

$$
X' = X + PE
$$

**代表**：Sinusoidal PE、可学习 PE

### 2. 修改注意力分数 (Attention Bias)

$$
\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right) V
$$

**代表**：[[ALiBi]]、T5 Relative Bias

### 3. 旋转 Q/K (Rotary)

$$
\text{Attention} = \text{softmax}\left(\frac{(R_m Q)(R_n K)^T}{\sqrt{d_k}}\right) V
$$

**代表**：[[RoPE]]

---

## 长度外推问题

> [!warning] 长度外推 (Length Extrapolation)
> 模型在短序列上训练，能否在长序列上保持性能？
>
> - **绝对位置编码**：外推能力差，超出训练长度后性能急剧下降
> - **相对位置编码**：外推能力更好，但也有上限

**解决方案**：
- [[RoPE]] + Position Interpolation / NTK-aware Scaling
- [[ALiBi]] 天然支持外推
- 长度扩展微调（如 LongLoRA）

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: 为什么 Transformer 需要位置编码？**
> A: 因为 Self-Attention 是置换不变的，无法区分 token 的顺序。位置编码注入位置信息，让模型能理解序列结构。
>
> **Q2: Sinusoidal PE 的公式是什么？为什么这样设计？**
> A: $PE_{(pos, 2i)} = \sin(pos/10000^{2i/d})$。正弦/余弦有界、可表示相对位置、理论上支持外推。
>
> **Q3: 绝对位置编码和相对位置编码的区别？**
> A: 绝对 PE 编码 token 在序列中的位置；相对 PE 编码 token 之间的距离。相对 PE 泛化更好，外推能力更强。
>
> **Q4: 现代 LLM 主要用什么位置编码？**
> A: RoPE（Llama、Qwen、Mistral）和 ALiBi（BLOOM）是主流。RoPE 更常见。
>
> **Q5: 什么是长度外推问题？**
> A: 模型在短序列训练，在长序列推理时性能下降。相对位置编码和专门的外推技术可以缓解。

---

## 相关概念

- [[Attention]] — 位置编码的应用场景
- [[RoPE]] — 旋转位置编码
- [[ALiBi]] — 注意力线性偏置
- [[Transformer]] — 位置编码的宿主架构
- [[Length Extrapolation]] — 位置编码的核心挑战
