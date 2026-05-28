---
description: MoE 通过稀疏激活实现参数量与计算量的解耦，用路由机制选择性激活部分专家，实现"大模型容量 + 小模型推理成本"
type: concept
aliases:
  - MoE
  - 混合专家
  - 稀疏激活
prerequisites:
  - "[[Transformer]]"
  - "[[Multi-Head Attention]]"
tags:
  - architecture
  - efficiency
  - scaling
created: 2026-01-27
updated: 2026-02-01T01:06
---

# Mixture of Experts

Mixture of Experts（MoE，混合专家）是一种**稀疏激活架构**，通过路由机制（Router/Gate）选择性地激活部分专家网络，实现参数量与计算量的解耦。核心价值：**大模型的容量 + 小模型的推理成本**。

> [!paper] 论文出处
> - 原始概念：Jacobs et al., "Adaptive Mixtures of Local Experts", 1991
> - 现代 MoE：Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR 2017
> - 代表工作：Switch Transformer (2021), GShard (2020), Mixtral (2024), DeepSeek-V2/V3 (2024-2025)

---

## 动机

> [!intuition] 为什么需要 MoE？
> **Dense 模型的困境**：参数量 ≈ 计算量
> - 想要更强的模型 → 增加参数
> - 增加参数 → 计算量线性增长
> - 训练和推理成本都随参数量增长
>
> **MoE 的解决方案**：打破这个等式
> - 总参数量可以很大（存储知识的容量）
> - 每个 token 只激活少量专家（实际计算量小）
> - 实现 **Conditional Computation**：根据输入动态决定计算路径

**核心洞察**：不是所有参数对所有输入都同等重要。让不同的专家专注于不同类型的输入，按需激活。

---

## 核心机制

### 架构概览

```
输入 token x
      │
      ▼
┌─────────────────┐
│   Router/Gate   │ ← 决定激活哪些专家
│  softmax(W·x)   │
└─────────────────┘
      │
      ▼ Top-K 选择
      │
┌─────┴─────┬─────────┬─────────┐
│           │         │         │
▼           ▼         ▼         ▼
Expert 1  Expert 2  Expert 3  ... Expert N
(FFN)     (FFN)     (FFN)         (FFN)
│           │         │         │
└─────┬─────┴─────────┴─────────┘
      │
      ▼ 加权求和
      │
输出
```

### 三个核心组件

#### 1. Experts（专家网络）

多个并行的子网络，通常是 FFN（Feed-Forward Network）：

> [!definition] Expert 定义
> 每个 Expert 通常是一个标准的 FFN：
> $$
> \text{Expert}_i(x) = W_2^{(i)} \cdot \sigma(W_1^{(i)} \cdot x)
> $$
> 其中 $\sigma$ 是激活函数（如 SwiGLU）。

**设计选择**：
- 专家数量：8, 16, 64, 256, 甚至更多
- 专家结构：通常与 Dense 模型的 FFN 相同
- 专家容量：每个专家可以处理的 token 数有上限

#### 2. Router/Gate（路由器）

决定每个 token 应该被哪些专家处理：

> [!definition] Router 定义
> $$
> G(x) = \text{softmax}(W_g \cdot x)
> $$
> 输出一个 $N$ 维向量，表示每个专家的权重。

**Top-K 选择**：只激活权重最高的 K 个专家
- $K=1$：Switch Transformer
- $K=2$：Mixtral、GShard
- $K=8$：DeepSeek-V3

#### 3. 加权组合

> [!math] MoE 层输出
> $$
> y = \sum_{i \in \text{TopK}(G(x))} G(x)_i \cdot \text{Expert}_i(x)
> $$
> 只计算被选中的 K 个专家，其他专家不参与计算。

---

## MoE 在 Transformer 中的位置

MoE 通常**替换 FFN 层**，[[Multi-Head Attention]] 层保持不变（这是 [[Transformer]] 架构的核心组件之一）：

```
标准 Transformer Block          MoE Transformer Block
┌─────────────────────┐        ┌─────────────────────┐
│  Multi-Head Attn    │        │  Multi-Head Attn    │
└─────────────────────┘        └─────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────────┐        ┌─────────────────────┐
│       FFN           │   →    │    MoE Layer        │
│   (Dense, 全激活)    │        │  (Sparse, 部分激活)  │
└─────────────────────┘        └─────────────────────┘
```

**常见配置**：
- 每层都用 MoE（如 Switch Transformer）
- 交替使用：Dense FFN 和 MoE 层交替（如 Mixtral）
- 只在部分层使用 MoE

---

## 训练挑战与解决方案

### 1. Load Balancing（负载均衡）

> [!warning] 核心问题：专家坍塌
> 如果不加约束，Router 可能学会把所有 token 都路由到少数几个专家：
> - 这些专家被过度训练，其他专家闲置
> - 失去了 MoE 的多样性优势
> - 计算负载不均衡，影响并行效率

**解决方案 1：Auxiliary Loss（辅助损失）**

> [!math] 负载均衡损失
> $$
> L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
> $$
> 其中：
> - $f_i$：实际路由到专家 $i$ 的 token 比例
> - $P_i$：Router 分配给专家 $i$ 的平均概率
> - $\alpha$：平衡系数（通常 0.01）
> - $N$：专家数量
>
> 这个损失鼓励 $f_i$ 和 $P_i$ 都趋向均匀分布 $1/N$。

**解决方案 2：[[Loss-Free Load Balancing]]（无损负载均衡）**

传统 Auxiliary Loss 存在两难困境：小的 $\alpha$ 导致负载不均衡，大的 $\alpha$ 引入干扰梯度损害性能。Loss-Free Balancing 通过动态调整 expert-wise bias 影响 top-K 选择，而非引入额外损失，打破了这个困境。已被 DeepSeek-V3、Ling 2.0 等大规模模型采用。

### 2. Expert Capacity（专家容量）

限制每个专家在一个 batch 中能处理的 token 数：

$$
\text{Capacity} = \frac{\text{tokens\_per\_batch}}{N} \times \text{capacity\_factor}
$$

- `capacity_factor > 1`：允许一定程度的不均衡
- 超出容量的 token 会被丢弃或路由到其他专家

### 3. 训练不稳定性

MoE 训练比 Dense 模型更容易不稳定：
- Router 的离散选择导致梯度估计困难
- 专家利用率的波动

**缓解方法**：
- 使用更小的学习率
- Router z-loss：惩罚 Router logits 过大
- 专家并行 + 数据并行的混合策略

---

## 典型配置示例

| 模型 | 总参数 | 激活参数 | 专家数 | Top-K | 特点 |
|------|--------|----------|--------|-------|------|
| Switch Transformer | 1.6T | ~100B | 2048 | 1 | 极端稀疏 |
| Mixtral 8x7B | 47B | 13B | 8 | 2 | 开源标杆 |
| DeepSeek-V2 | 236B | 21B | 160 | 6 | 高效推理 |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | SOTA 性能 |
| Qwen2-MoE | 57B | 14B | 64 | 8 | 中等规模 |

> [!example] DeepSeek-V3 配置详解
> - 总参数：671B（存储容量大）
> - 激活参数：37B（实际计算量）
> - 专家数：256 个 routed experts + 1 个 shared expert
> - Top-K：8（每个 token 激活 8 个专家）
> - 效率：推理成本接近 37B Dense 模型，但性能接近 671B

---

## 与 Dense 模型的对比

> [!comparison] MoE vs Dense

| 方面 | Dense 模型 | MoE 模型 |
|------|------------|----------|
| **参数效率** | 参数量 ≈ 计算量 | 参数量 >> 计算量 |
| **训练成本** | 与参数量成正比 | 与激活参数成正比 |
| **推理成本** | 与参数量成正比 | 与激活参数成正比 |
| **内存占用** | 存储全部参数 | 需存储所有专家（更大） |
| **训练稳定性** | 相对稳定 | 需要额外技巧 |
| **专业化** | 所有参数处理所有输入 | 不同专家可专业化 |
| **Scaling** | 线性扩展 | 可更高效扩展 |

**关键权衡**：
- MoE 用**更大的内存占用**换取**更低的计算成本**
- 相同计算量下，MoE 通常性能更好
- 但 MoE 需要更多工程优化（通信、负载均衡）

---

## 推理优化

MoE 推理面临独特挑战：

### 1. 内存带宽瓶颈

- 所有专家参数都需要加载到内存
- 但每个 token 只用少量专家
- 内存带宽可能成为瓶颈

### 2. 专家并行

将不同专家放在不同 GPU 上：
- 需要 All-to-All 通信
- 通信开销可能抵消计算节省

### 3. 优化策略

- **Expert Offloading**：不活跃的专家放在 CPU/SSD
- **Expert Caching**：缓存常用专家
- **Speculative Expert Loading**：预测并预加载专家
- **[[Quantization|量化]]**：减少专家参数的内存占用，对 MoE 尤其有效（因为专家数量多）
- **[[KV Cache]]**：虽然 MoE 不改变 Attention 层，但 KV Cache 仍是推理优化的关键
- **[[Flash Attention]]**：优化 Attention 计算，与 MoE 的 FFN 优化互补

---

## 局限性

> [!warning] MoE 的边界条件
>
> 1. **内存占用大**：需要存储所有专家参数，即使大部分不被激活
>
> 2. **通信开销**：分布式训练/推理时，专家并行需要大量通信
>
> 3. **训练不稳定**：负载不均衡、专家坍塌等问题需要额外处理
>
> 4. **Batch Size 敏感**：小 batch 时负载均衡更难，效率下降
>
> 5. **不适合所有场景**：
>    - 内存受限环境（边缘设备）
>    - 需要确定性计算的场景
>    - 小规模模型（MoE 开销不值得）

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: MoE 的核心思想是什么？**
> A: 稀疏激活——总参数量大但每个 token 只激活少量专家，实现参数量与计算量的解耦。
>
> **Q2: MoE 如何解决负载均衡问题？**
> A: 传统方法用 Auxiliary Loss 惩罚不均匀的专家利用率；更先进的方法如 [[Loss-Free Load Balancing]] 通过动态 bias 调整路由，避免干扰梯度。
>
> **Q3: MoE 相比 Dense 模型的优缺点？**
> A: 优点：相同计算量下性能更好，可高效扩展。缺点：内存占用大，训练不稳定，需要额外工程优化。
>
> **Q4: MoE 在 Transformer 中替换哪个组件？**
> A: 通常替换 FFN 层，Attention 层保持不变。
>
> **Q5: 什么是 Expert Capacity？为什么需要它？**
> A: 限制每个专家处理的 token 数上限，防止单个专家过载，保证计算负载均衡。
>
> **Q6: DeepSeek-V3 的 MoE 配置是什么？**
> A: 671B 总参数，37B 激活参数，256 个专家，Top-8 选择。

---

## 延伸阅读

**负载均衡**：
- [[Loss-Free Load Balancing]] — 无损负载均衡策略

**相关方法**：
- [[Sparse Attention]] — 另一种稀疏化方法（待创建）
- [[Expert Parallelism]] — MoE 的分布式训练策略（待创建）

**代表论文**：
- [[Switch Transformer (2021)]] — Google 的大规模 MoE（待创建）
- [[Mixtral (2024)]] — Mistral 的开源 MoE（待创建）
- [[DeepSeek-V3 (2025)]] — 当前 SOTA MoE 模型（待创建）

---

## 参考资料

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR 2017
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR 2022
- Jiang et al., "Mixtral of Experts", 2024
- DeepSeek-AI, "DeepSeek-V3 Technical Report", 2025

