---
description: 通过低秩压缩 KV 到 latent 向量大幅减少 KV Cache，是 DeepSeek-V2/V3 的核心架构创新
type: method
aliases:
  - MLA
  - 多头潜在注意力
prerequisites:
  - "[[Multi-Head Attention]]"
  - "[[KV Cache]]"
  - "[[RoPE]]"
tags:
  - transformer
  - architecture
  - inference
  - efficiency
created: 2026-01-28
updated: 2026-01-31T22:42
---

# Multi-head Latent Attention (MLA)

多头潜在注意力（Multi-head Latent Attention, MLA）是 DeepSeek-V2/V3 提出的高效注意力机制，通过**低秩联合压缩** Key 和 Value 到一个共享的 latent 向量，大幅减少推理时的 [[KV Cache]] 内存占用，同时保持与标准 [[Multi-Head Attention]] 相当的模型质量。

> [!paper] 论文出处
> DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", 2024
> DeepSeek-AI, "DeepSeek-V3 Technical Report", 2024

---

## 动机：KV Cache 的内存瓶颈

> [!intuition] 为什么需要 MLA？
> 在自回归生成时，[[KV Cache]] 是主要的内存瓶颈。现有的解决方案各有局限：
>
> | 方法 | 策略 | 问题 |
> |------|------|------|
> | [[Multi-Query Attention\|MQA]] | 所有头共享 K/V | 质量下降明显 |
> | [[Grouped-Query Attention\|GQA]] | 分组共享 K/V | 仍需存储多组 K/V |
>
> **MLA 的洞察**：K 和 V 本身存在冗余，可以用**低秩压缩**将它们联合编码到一个更小的 latent 向量中。推理时只需缓存这个 latent 向量，需要时再解压。

---

## 核心机制

MLA 的核心是**低秩联合压缩**：将 K 和 V 投影到一个共享的低维 latent 空间，推理时只缓存 latent 向量。

### 符号定义

| 符号 | 含义 |
|------|------|
| $d$ | embedding 维度 |
| $n_h$ | 注意力头数 |
| $d_h$ | 每头维度 |
| $d_c$ | KV 压缩维度（$d_c \ll d_h n_h$） |
| $d_h^R$ | RoPE key 的维度 |
| $\mathbf{h}_t$ | 第 $t$ 个 token 的 attention 输入 |

### KV 压缩（Down-projection）

> [!math] KV Latent 压缩
> 将输入 $\mathbf{h}_t$ 投影到低维 latent 向量：
> $$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t$$
>
> 其中 $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ 是压缩后的 latent 向量，$W^{DKV} \in \mathbb{R}^{d_c \times d}$ 是 down-projection 矩阵。

### KV 解压（Up-projection）

> [!math] K/V 解压
> 从 latent 向量恢复 K 和 V：
> $$\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}$$
> $$\mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}$$
>
> 其中 $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ 是 up-projection 矩阵。

### RoPE 兼容性：Decoupled Key

> [!warning] RoPE 与低秩压缩的冲突
> [[RoPE]] 需要对 K 应用位置相关的旋转变换。但如果直接对压缩后的 $\mathbf{c}_t^{KV}$ 应用 RoPE，会破坏低秩结构，导致无法正确解压。

> [!intuition] 解决方案：Decoupled Key
> MLA 将 Key 分成两部分：
> - **Content Key** $\mathbf{k}_t^C$：从 latent 解压，不带位置信息
> - **RoPE Key** $\mathbf{k}_t^R$：单独计算，携带位置信息
>
> 最终 Key 是两者的拼接：$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R]$

> [!math] Decoupled RoPE Key
> $$\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t)$$
>
> 其中 $W^{KR} \in \mathbb{R}^{d_h^R \times d}$，$\mathbf{k}_t^R$ 是所有头共享的 RoPE key。

### Query 压缩（可选）

MLA 也对 Query 进行低秩压缩，减少训练时的 activation 内存：

> [!math] Query 压缩
> $$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t$$
> $$\mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q$$
> $$\mathbf{q}_t^R = \text{RoPE}(W^{QR} \mathbf{c}_t^Q)$$
> $$\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R]$$

### 完整 Attention 计算

> [!math] MLA Attention
> $$\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\left(\frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}}\right) \mathbf{v}_{j,i}^C$$
> $$\mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; ...; \mathbf{o}_{t,n_h}]$$

---

## KV Cache 节省分析

> [!intuition] 需要缓存什么？
> 推理时只需缓存：
> 1. **Latent 向量** $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$
> 2. **RoPE Key** $\mathbf{k}_t^R \in \mathbb{R}^{d_h^R}$
>
> 总共 $d_c + d_h^R$ 维，远小于 MHA 的 $2 \times n_h \times d_h$ 维。

### 内存对比

| 方法 | 每 token 缓存维度 | 相对 MHA |
|------|------------------|----------|
| MHA | $2 n_h d_h$ | 1x |
| GQA ($g$ 组) | $2 g d_h$ | $g / n_h$ |
| MQA | $2 d_h$ | $1 / n_h$ |
| **MLA** | $d_c + d_h^R$ | $\approx (d_c + d_h^R) / (2 n_h d_h)$ |

> [!example] DeepSeek-V3 配置
> DeepSeek-V3 的 MLA 配置：
> - $n_h = 128$，$d_h = 128$
> - $d_c = 512$（KV 压缩维度）
> - $d_h^R = 64$（RoPE key 维度）
>
> KV Cache 节省：
> - MHA：$2 \times 128 \times 128 = 32768$ 维
> - MLA：$512 + 64 = 576$ 维
> - **压缩比：约 57 倍**

---

## 与 MHA/MQA/GQA 的对比

> [!comparison] 注意力变体对比
>
> | 方法 | 核心思想 | KV Cache | 质量 | 代表模型 |
> |------|----------|----------|------|----------|
> | [[Multi-Head Attention\|MHA]] | 每头独立 K/V | 最大 | 最好 | GPT-3, Llama 1 |
> | [[Multi-Query Attention\|MQA]] | 所有头共享 K/V | 最小 | 较差 | PaLM, Falcon |
> | [[Grouped-Query Attention\|GQA]] | 分组共享 K/V | 中等 | 很好 | Llama 2/3, Mistral |
> | **MLA** | 低秩压缩 K/V | 极小 | 很好 | DeepSeek-V2/V3 |

> [!intuition] MLA vs GQA 的本质区别
> - **GQA**：减少 K/V 的**数量**（共享）
> - **MLA**：减少 K/V 的**维度**（压缩）
>
> MLA 保留了每个头的独立性（通过 up-projection 恢复），同时实现更激进的压缩。

---

## 实现细节

### 训练时的矩阵吸收

> [!intuition] 计算优化
> 训练时，up-projection 矩阵可以与 attention 的 Q/K/V 投影矩阵合并，避免显式计算 latent 向量。这使得 MLA 的训练计算量与 MHA 相当。

### 推理时的流程

```
输入 h_t
    │
    ├─→ W^DKV ─→ c_t^KV ─→ [缓存]
    │                │
    │                ├─→ W^UK ─→ k_t^C ─┐
    │                └─→ W^UV ─→ v_t^C ─┼─→ Attention
    │                                   │
    └─→ W^KR ─→ RoPE ─→ k_t^R ─→ [缓存] ─┘
```

### Scale-Correction（LongCat-Flash 改进）

> [!math] 方差对齐
> LongCat-Flash 发现 MLA 的低秩分解会导致方差不对齐，影响模型缩放。解决方案是添加 scale-correction 因子：
> $$\mathbf{c}_t^Q = \alpha_q W^{DQ} \mathbf{h}_t$$
> $$\mathbf{c}_t^{KV} = \alpha_{kv} W^{DKV} \mathbf{h}_t$$
>
> 其中 $\alpha_q = \sqrt{d / d_c'}$，$\alpha_{kv} = \sqrt{d / d_c}$。

---

## 局限性

> [!warning] MLA 的局限
> 1. **推理时需要解压**：虽然缓存小，但每步需要 up-projection 计算
> 2. **与 Flash Attention 的兼容性**：需要特殊实现来支持 latent 解压
> 3. **RoPE 处理复杂**：decoupled key 增加了实现复杂度
> 4. **训练时 activation 内存**：需要 recomputation 策略来减少 up-projection 的 activation

---

## 使用 MLA 的模型

| 模型 | 参数量 | MLA 配置 | 备注 |
|------|--------|----------|------|
| DeepSeek-V2 | 236B | $d_c = 512$ | 首次提出 MLA |
| DeepSeek-V3 | 671B | $d_c = 512$, $d_h^R = 64$ | 当前最强开源模型之一 |
| LongCat-Flash | 193B | 带 scale-correction | 添加方差对齐 |
| Kimi K2 | 1T | MLA + QK-Clip | 添加训练稳定性优化 |

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: MLA 的核心思想是什么？**
> A: 通过低秩联合压缩将 K 和 V 编码到一个共享的 latent 向量，推理时只缓存 latent 向量，需要时再解压。这比 GQA 的"共享"策略实现更激进的压缩。
>
> **Q2: MLA 如何处理 RoPE？**
> A: 使用 decoupled key：将 Key 分成 content key（从 latent 解压）和 RoPE key（单独计算）两部分拼接。RoPE key 是所有头共享的，也需要缓存。
>
> **Q3: MLA 的 KV Cache 能节省多少？**
> A: DeepSeek-V3 中约 57 倍压缩（从 32768 维降到 576 维）。比 GQA 更激进。
>
> **Q4: MLA 和 GQA 的本质区别？**
> A: GQA 减少 K/V 的数量（共享），MLA 减少 K/V 的维度（压缩）。MLA 保留每个头的独立性，通过 up-projection 恢复。
>
> **Q5: MLA 的代价是什么？**
> A: 推理时每步需要 up-projection 解压计算；与 Flash Attention 的兼容需要特殊实现；decoupled key 增加实现复杂度。

---

## 相关概念

- [[Multi-Head Attention]] — MLA 的基础
- [[Multi-Query Attention]] — 另一种 KV Cache 优化
- [[Grouped-Query Attention]] — MQA 和 MHA 的折中
- [[KV Cache]] — MLA 优化的目标
- [[RoPE]] — MLA 需要特殊处理的位置编码
- [[Flash Attention]] — 与 MLA 配合的高效 Attention 实现
- [[MOC - Attention]] — Attention 机制导航
