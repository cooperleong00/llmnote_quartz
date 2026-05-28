---
description: 所有注意力头共享同一组 KV，大幅减少 KV Cache 内存占用
type: method
prerequisites:
  - "[[Multi-Head Attention]]"
tags:
  - transformer
  - architecture
  - inference
  - efficiency
created: 2025-01-26
updated: 2026-01-31T22:42
---

# Multi-Query Attention (MQA)

多查询注意力（Multi-Query Attention, MQA）是一种高效的注意力变体，通过让所有注意力头**共享同一组 Key 和 Value**，大幅减少推理时的 [[KV Cache]] 内存占用。

> [!paper] 论文出处
> Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019

---

## 动机：KV Cache 的内存瓶颈

> [!intuition] 为什么需要 MQA？
> 在自回归生成时，为了避免重复计算，需要缓存历史 token 的 Key 和 Value（[[KV Cache]]）。
>
> 对于标准 [[Multi-Head Attention]]：
> - 每个头都有独立的 K 和 V
> - KV Cache 大小 = $2 \times \text{layers} \times \text{heads} \times \text{seq\_len} \times d_k$
>
> 当序列变长、batch 变大时，KV Cache 成为显存瓶颈。

> [!example] KV Cache 内存计算
> Llama-7B（32 层，32 头，$d_k = 128$），序列长度 4096，batch size 32：
> $$
> \text{KV Cache} = 2 \times 32 \times 32 \times 4096 \times 128 \times 2 \text{ bytes} = 64 \text{ GB}
> $$
> 这已经超过了单张 A100 的显存！

---

## 核心思想

> [!definition] Multi-Query Attention
> 所有注意力头共享同一组 Key 和 Value，但保留独立的 Query：
>
> $$
> \text{head}_i = \text{Attention}(Q_i, K, V)
> $$
>
> 其中：
> - $Q_i = XW_i^Q$：每个头有独立的 Query 投影
> - $K = XW^K$：所有头共享同一个 Key
> - $V = XW^V$：所有头共享同一个 Value

```
标准 MHA:
Q1, K1, V1 → head_1
Q2, K2, V2 → head_2
...
Qh, Kh, Vh → head_h

MQA:
Q1, K, V → head_1
Q2, K, V → head_2    (K, V 共享)
...
Qh, K, V → head_h
```

---

## 内存节省分析

### KV Cache 对比

| 方法 | KV Cache 大小 | 相对 MHA |
|------|---------------|----------|
| MHA | $2 \times L \times h \times n \times d_k$ | 1x |
| **MQA** | $2 \times L \times n \times d_k$ | $1/h$ |
| GQA ($g$ 组) | $2 \times L \times g \times n \times d_k$ | $g/h$ |

> [!math] MQA 的内存节省
> MQA 将 KV Cache 减少到原来的 $1/h$。
>
> 对于 32 头的模型，内存减少 **32 倍**！

### 参数量对比

| 参数 | MHA | MQA |
|------|-----|-----|
| $W^Q$ | $d \times d$ | $d \times d$ |
| $W^K$ | $d \times d$ | $d \times d_k$ |
| $W^V$ | $d \times d$ | $d \times d_v$ |
| $W^O$ | $d \times d$ | $d \times d$ |
| **总计** | $4d^2$ | $2d^2 + 2d \cdot d_k$ |

MQA 的参数量也有所减少（约 $2d^2$ vs $4d^2$）。

---

## 质量 vs 效率权衡

> [!warning] MQA 的代价
> 共享 K/V 意味着不同头无法学习不同的"关注模式"（至少在 K/V 层面）。
>
> 实验表明：
> - MQA 在某些任务上性能略有下降
> - 但在大模型上，下降通常可以接受
> - 推理速度提升显著（尤其是长序列）

### 性能对比

| 模型 | 方法 | 质量 | 推理速度 |
|------|------|------|----------|
| 基线 | MHA | 100% | 1x |
| MQA | MQA | ~98-99% | 1.5-2x |
| GQA | GQA | ~99-100% | 1.3-1.5x |

---

## 实现

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 每个头独立的 Q
        self.W_q = nn.Linear(d_model, d_model)
        # 共享的 K, V
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq, _ = x.shape

        # Q: (batch, heads, seq, d_k)
        Q = self.W_q(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        # K, V: (batch, 1, seq, d_k) - 广播到所有头
        K = self.W_k(x).unsqueeze(1)
        V = self.W_v(x).unsqueeze(1)

        # 注意力计算（K, V 自动广播）
        attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
        out = attn @ V

        # 合并头
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.W_o(out)
```

---

## 使用 MQA 的模型

| 模型 | 注意力类型 | 备注 |
|------|------------|------|
| PaLM | MQA | Google 的大模型 |
| Falcon-40B | MQA | 开源模型 |
| StarCoder | MQA | 代码模型 |
| Llama 2 | GQA | MQA 的折中方案 |

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: MQA 的核心思想是什么？**
> A: 所有注意力头共享同一组 Key 和 Value，只保留独立的 Query。这样 KV Cache 减少到 $1/h$。
>
> **Q2: MQA 解决什么问题？**
> A: 解决推理时 KV Cache 的内存瓶颈。长序列、大 batch 时，KV Cache 是显存的主要消耗。
>
> **Q3: MQA 的代价是什么？**
> A: 不同头无法学习不同的 K/V 表示，可能导致模型质量略有下降。但在大模型上通常可以接受。
>
> **Q4: MQA 能节省多少内存？**
> A: KV Cache 减少到 $1/h$。对于 32 头的模型，减少 32 倍。
>
> **Q5: MQA 和 GQA 的关系？**
> A: GQA 是 MQA 和 MHA 的折中。MQA 是 GQA 的极端情况（只有 1 组 K/V）。

---

## 相关概念

- [[Multi-Head Attention]] — MQA 的基础
- [[Grouped-Query Attention]] — MQA 和 MHA 的折中
- [[KV Cache]] — MQA 优化的目标
- [[Transformer]] — MQA 的宿主架构
- [[MOC - Inference]] — MQA 的应用场景
