---
description: 在注意力分数上添加线性偏置编码位置，无需训练且支持长度外推
type: method
prerequisites:
  - "[[Positional Encoding]]"
  - "[[Attention]]"
tags:
  - transformer
  - positional-encoding
  - architecture
created: 2025-01-26
updated: 2026-02-01T00:58
---

# ALiBi (Attention with Linear Biases)

ALiBi（Attention with Linear Biases）是一种简单高效的[[Positional Encoding|相对位置编码]]方法，通过在注意力分数上添加线性偏置来编码位置信息。它的核心优势是**无需训练位置编码**且**天然支持长度外推**。

> [!paper] 论文出处
> Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", ICLR 2022

---

## 核心思想

> [!intuition] 直觉理解
> ALiBi 的想法极其简单：**距离越远，注意力越低**。
>
> 不需要学习复杂的位置编码，只需要在注意力分数上减去一个与距离成正比的惩罚：
> - 相邻 token：惩罚小
> - 远距离 token：惩罚大
>
> 这符合语言的局部性假设：相邻的词通常更相关。

---

## 数学定义

> [!definition] ALiBi
> $$
> \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + m \cdot B\right) V
> $$
>
> 其中偏置矩阵 $B$ 是：
> $$
> B_{ij} = -|i - j|
> $$
>
> $m$ 是每个头的斜率（slope），不同头使用不同的 $m$。

### 偏置矩阵示例

对于序列长度 4：

$$
B = \begin{pmatrix}
0 & -1 & -2 & -3 \\
-1 & 0 & -1 & -2 \\
-2 & -1 & 0 & -1 \\
-3 & -2 & -1 & 0
\end{pmatrix}
$$

### 斜率设计

对于 $h$ 个头，斜率按几何级数设置（这是 [[Multi-Head Attention]] 的关键设计）：

$$
m_i = 2^{-8/h \cdot i}, \quad i = 1, 2, ..., h
$$

> [!example] 8 头的斜率
> $m = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]$
>
> - 大斜率的头：强烈偏好局部信息
> - 小斜率的头：可以关注较远的信息

---

## ALiBi 的优势

### 1. 无需训练位置编码

> [!intuition] 零参数位置编码
> ALiBi 不需要学习任何位置相关的参数：
> - 偏置矩阵 $B$ 是预定义的
> - 斜率 $m$ 是预定义的
>
> 这意味着模型可以直接应用到任意长度的序列。

### 2. 天然支持[[长度外推]]

> [!warning] 长度外推能力
> ALiBi 在长度外推上表现优异：
> - 在 1024 token 上训练
> - 可以直接推理 2048、4096 甚至更长的序列
> - 性能下降很小

**原因**：ALiBi 只依赖于相对距离，不依赖于绝对位置。无论序列多长，相邻 token 的偏置始终是 $-m$。

### 3. 计算高效

ALiBi 只需要在注意力分数上加一个预计算的矩阵，几乎不增加计算开销。

### 4. 实现简单

```python
def alibi_bias(seq_len, num_heads):
    # 计算斜率
    slopes = 2 ** (-8 / num_heads * torch.arange(1, num_heads + 1))

    # 计算距离矩阵
    positions = torch.arange(seq_len)
    distances = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)

    # 偏置 = 斜率 * 距离
    bias = slopes.view(-1, 1, 1) * distances.abs().unsqueeze(0)  # (heads, seq, seq)
    return -bias
```

---

## 与 [[RoPE]] 的对比

| 方面 | RoPE | ALiBi |
|------|------|-------|
| **编码方式** | 旋转 Q/K | 注意力偏置 |
| **位置信息** | 编码到向量中 | 加到注意力分数上 |
| **长度外推** | 需要额外技术（PI、NTK） | 天然支持 |
| **额外参数** | 无 | 无 |
| **计算开销** | 低（逐元素旋转） | 低（加偏置） |
| **主流程度** | 更主流（Llama 系列） | 较少使用（BLOOM） |

> [!intuition] 为什么 RoPE 更主流？
> 尽管 ALiBi 外推能力更强，但：
> 1. RoPE 在短序列上效果更好
> 2. Llama 的成功带动了 RoPE 的流行
> 3. RoPE 的外推问题可以通过 PI/NTK 缓解

---

## 使用 ALiBi 的模型

| 模型 | 位置编码 | 训练长度 | 外推能力 |
|------|----------|----------|----------|
| BLOOM | ALiBi | 2048 | 好 |
| MPT | ALiBi | 2048/8192 | 好 |
| Falcon | RoPE | 2048 | 中等 |

---

## 局限性

> [!warning] ALiBi 的局限
> 1. **固定的衰减模式**：线性衰减可能不适合所有任务
> 2. **短序列性能**：在短序列上可能不如 RoPE
> 3. **缺乏位置感知**：只编码相对距离，不编码绝对位置
> 4. **采用率低**：主流模型（Llama、Qwen、Mistral）都选择了 RoPE

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: ALiBi 的核心思想是什么？**
> A: 在注意力分数上加线性偏置，距离越远惩罚越大。$\text{score}_{ij} = QK^T/\sqrt{d} - m|i-j|$。
>
> **Q2: ALiBi 为什么能外推？**
> A: 因为它只依赖相对距离，不依赖绝对位置。无论序列多长，相邻 token 的偏置始终相同。
>
> **Q3: ALiBi 和 RoPE 的主要区别？**
> A: RoPE 通过旋转 Q/K 编码位置，ALiBi 通过注意力偏置编码位置。ALiBi 外推更好，RoPE 短序列效果更好。
>
> **Q4: ALiBi 的斜率是怎么设计的？**
> A: 几何级数 $m_i = 2^{-8/h \cdot i}$。不同头使用不同斜率，有的关注局部，有的关注远距离。
>
> **Q5: 为什么 ALiBi 没有成为主流？**
> A: 虽然外推好，但短序列性能可能不如 RoPE，且 Llama 的成功带动了 RoPE 的流行。

---

## 相关概念

- [[Positional Encoding]] — 位置编码的基础概念
- [[Attention]] — ALiBi 的应用场景
