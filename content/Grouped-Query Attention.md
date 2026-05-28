---
description: MHA 和 MQA 的折中，分组共享 KV 平衡质量与效率，是 Llama 2 的标准配置
type: method
prerequisites:
  - "[[Multi-Head Attention]]"
  - "[[Multi-Query Attention]]"
tags:
  - transformer
  - architecture
  - inference
  - efficiency
created: 2025-01-26
updated: 2026-02-01T00:57
---

# Grouped-Query Attention (GQA)

分组查询注意力（Grouped-Query Attention, GQA）是 [[Multi-Head Attention]] 和 [[Multi-Query Attention]] 的折中方案，通过将注意力头分组共享 Key/Value，在模型质量和推理效率之间取得平衡。它是 Llama 2 等主流模型的标准配置。

> [!paper] 论文出处
> Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP 2023

---

## 核心思想

> [!intuition] MHA 和 MQA 的折中
> - **MHA**：每个头有独立的 K/V → 质量好，但 [[KV Cache]] 大
> - **MQA**：所有头共享 K/V → KV Cache 小，但质量可能下降
> - **GQA**：将头分成 $g$ 组，组内共享 K/V → 平衡质量和效率

```
MHA (h=8, g=8):  每个头独立 K/V
K1 K2 K3 K4 K5 K6 K7 K8
V1 V2 V3 V4 V5 V6 V7 V8

GQA (h=8, g=2):  每 4 个头共享 K/V
K1 K1 K1 K1 | K2 K2 K2 K2
V1 V1 V1 V1 | V2 V2 V2 V2

MQA (h=8, g=1):  所有头共享 K/V
K1 K1 K1 K1 K1 K1 K1 K1
V1 V1 V1 V1 V1 V1 V1 V1
```

---

## 数学定义

> [!definition] Grouped-Query Attention
> 将 $h$ 个注意力头分成 $g$ 组，每组 $h/g$ 个头共享同一组 K/V：
>
> $$
> \text{head}_i = \text{Attention}(Q_i, K_{\lfloor i \cdot g / h \rfloor}, V_{\lfloor i \cdot g / h \rfloor})
> $$
>
> 其中：
> - $Q_i$：第 $i$ 个头的独立 Query
> - $K_j, V_j$：第 $j$ 组的共享 Key/Value
> - $g$：组数（$1 \leq g \leq h$）

**特殊情况**：
- $g = h$：退化为 MHA
- $g = 1$：退化为 MQA

---

## KV Cache 分析

| 方法 | 组数 | KV Cache 大小 | 相对 MHA |
|------|------|---------------|----------|
| MHA | $g = h$ | $2Lhn d_k$ | 1x |
| GQA | $g$ | $2Lgn d_k$ | $g/h$ |
| MQA | $g = 1$ | $2Ln d_k$ | $1/h$ |

> [!example] Llama 2 的配置
> Llama 2 70B：$h = 64$ 头，$g = 8$ 组
> - KV Cache 减少到 MHA 的 $8/64 = 1/8$
> - 比 MQA 多 8 倍，但质量更好

---

## 从 MHA 转换到 GQA

> [!intuition] Uptraining 策略
> 可以从已训练的 MHA 模型转换到 GQA，而不需要从头训练：
>
> 1. **Mean Pooling**：将同组的 K/V 投影矩阵取平均
> 2. **继续训练**：用少量数据微调，恢复性能

```python
# 从 MHA 转换到 GQA
def convert_mha_to_gqa(mha_weights, num_groups):
    h = mha_weights['W_k'].shape[0] // d_k
    heads_per_group = h // num_groups

    # 对每组的 K/V 权重取平均
    gqa_W_k = mha_weights['W_k'].reshape(h, d_k, d_model)
    gqa_W_k = gqa_W_k.reshape(num_groups, heads_per_group, d_k, d_model)
    gqa_W_k = gqa_W_k.mean(dim=1)  # (num_groups, d_k, d_model)

    return gqa_W_k
```

---

## 质量 vs 效率权衡

### 实验结果

| 配置 | 组数 | 质量（相对 MHA） | KV Cache |
|------|------|------------------|----------|
| MHA | 64 | 100% | 1x |
| GQA-8 | 8 | ~99.5% | 1/8 |
| GQA-4 | 4 | ~99% | 1/16 |
| GQA-2 | 2 | ~98.5% | 1/32 |
| MQA | 1 | ~98% | 1/64 |

> [!intuition] 为什么 GQA 质量损失小？
> 1. 不同头的 K/V 本身就有冗余
> 2. Query 仍然是独立的，保留了多样性
> 3. 组内共享可能起到正则化作用

---

## 使用 GQA 的模型

| 模型 | 头数 $h$ | 组数 $g$ | 比例 |
|------|----------|----------|------|
| Llama 2 7B | 32 | 32 | MHA |
| Llama 2 13B | 40 | 40 | MHA |
| **Llama 2 70B** | 64 | 8 | GQA |
| **Llama 3 8B** | 32 | 8 | GQA |
| **Llama 3 70B** | 64 | 8 | GQA |
| Mistral 7B | 32 | 8 | GQA |
| Qwen 2 | 变化 | 变化 | GQA |

> [!intuition] 为什么大模型更倾向于 GQA？
> 大模型的 KV Cache 更大，内存压力更大。GQA 在大模型上的质量损失相对更小，收益更大。

---

## 实现

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.d_k = d_model // num_heads

        # 每个头独立的 Q
        self.W_q = nn.Linear(d_model, d_model)
        # 每组共享的 K, V
        self.W_k = nn.Linear(d_model, num_groups * self.d_k)
        self.W_v = nn.Linear(d_model, num_groups * self.d_k)
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq, _ = x.shape

        # Q: (batch, num_heads, seq, d_k)
        Q = self.W_q(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        # K, V: (batch, num_groups, seq, d_k)
        K = self.W_k(x).view(batch, seq, self.num_groups, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq, self.num_groups, self.d_k).transpose(1, 2)

        # 扩展 K, V 到每个头: (batch, num_heads, seq, d_k)
        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)

        # 注意力计算
        attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
        out = attn @ V

        # 合并头
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.W_o(out)
```

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: GQA 的核心思想是什么？**
> A: 将注意力头分成 $g$ 组，组内共享 K/V。是 MHA 和 MQA 的折中，平衡质量和效率。
>
> **Q2: GQA、MHA、MQA 的关系？**
> A: GQA 是通用形式。$g=h$ 时是 MHA，$g=1$ 时是 MQA。
>
> **Q3: GQA 能节省多少内存？**
> A: KV Cache 减少到 $g/h$。Llama 2 70B 用 $g=8, h=64$，减少到 1/8。
>
> **Q4: 为什么大模型更倾向于用 GQA？**
> A: 大模型 KV Cache 更大，内存压力更大。GQA 在大模型上质量损失相对更小，收益更大。
>
> **Q5: 如何从 MHA 模型转换到 GQA？**
> A: 将同组的 K/V 权重取平均（mean pooling），然后用少量数据微调恢复性能。

---

## 相关概念

- [[Multi-Head Attention]] — GQA 的基础
- [[Multi-Query Attention]] — GQA 的极端情况
- [[Transformer]] — GQA 的宿主架构
- [[MOC - Inference]] — GQA 的应用场景
