---
description: Loss-Free Balancing 通过动态调整 expert-wise bias 实现 MoE 负载均衡，避免辅助损失引入的干扰梯度，打破负载均衡与模型性能的两难困境
type: method
aliases:
  - Loss-Free Balancing
  - LFB
  - Auxiliary-Loss-Free Load Balancing
  - 无损负载均衡
prerequisites:
  - "[[Mixture of Experts]]"
tags:
  - architecture
  - moe
  - efficiency
  - optimization
created: 2026-01-28
updated: 2026-01-31T22:42
---

# Loss-Free Load Balancing

Loss-Free Load Balancing（无损负载均衡）是一种 **auxiliary-loss-free** 的 MoE 负载均衡策略。核心思想：在 top-K 路由决策前，为每个专家的路由分数添加一个动态调整的 bias，通过 bias 影响专家选择而非引入额外损失函数，从而实现负载均衡且不产生干扰梯度。

> [!paper] 论文出处
> - Wang et al., "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts", 2024
> - 机构：DeepSeek-AI, Peking University
> - 已被 DeepSeek-V3、Ling 2.0、LongCat-Flash、GLM-4.5 等大规模 MoE 模型采用

---

## 动机

> [!intuition] 为什么需要 Loss-Free Balancing？
> **传统方法的困境**：MoE 模型使用 [[Mixture of Experts#训练挑战与解决方案|Auxiliary Loss]] 来鼓励负载均衡，但存在根本性的两难：
> - **小的 $\alpha$**：负载不均衡，导致 routing collapse 或计算瓶颈
> - **大的 $\alpha$**：引入强干扰梯度，损害模型性能
>
> **核心洞察**：能否在不引入额外损失的情况下，直接控制路由决策？
> - 答案是：通过动态调整 bias 来"诱导"均衡的路由选择
> - bias 只影响 top-K 选择，不参与梯度计算

---

## 核心机制

### Biased Gating Score

> [!math] 数学形式
> 在标准 MoE 中，top-K 选择基于原始 gating score $s_{i,t}$。Loss-Free Balancing 引入 expert-wise bias $b_i$：
>
> $$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \text{Topk}(\{s_{j,t} + b_j | 1 \le j \le N\}, K), \\ 0, & \text{otherwise.} \end{cases}$$
>
> **关键点**：
> - $b_i$ 只用于 top-K 选择决策
> - 最终的 gating weight 仍然是原始的 $s_{i,t}$，不包含 $b_i$
> - 因此 $b_i$ 不参与梯度计算，不引入干扰梯度

> [!intuition] 直觉理解
> 想象一个"招聘市场"：
> - 每个专家是一个"雇主"，每个 token 是一个"求职者"
> - 原始 gating score 是"匹配度"
> - bias 是"招聘补贴"——负载轻的专家获得更高补贴，吸引更多 token
> - 但实际"工资"（gating weight）仍由匹配度决定

### 动态 Bias 更新

> [!definition] 更新算法
> **初始化**：所有 $b_i = 0$
>
> **每个 batch 后更新**：
> 1. 统计每个专家的 token 数 $c_i$ 和平均数 $\bar{c}$
> 2. 计算 violation error：$e_i = c_i - \bar{c}$
> 3. 更新 bias：$b_i = b_i + u \cdot \text{sign}(e_i)$
>
> 其中 $u$ 是更新率（推荐 $u = 0.001$）。

```
Algorithm: Loss-Free Balancing

Input: MoE model θ, training batch iterator B, update rate u
Initialize: b_i = 0 for each expert

for each batch {(x_k, y_k)} in B:
    1. Forward pass with biased top-K selection
    2. Count tokens per expert: c_i
    3. Compute average: c̄ = (1/N) Σ c_i
    4. Compute error: e_i = c_i - c̄
    5. Update bias: b_i = b_i + u * sign(e_i)
    6. Backward pass (gradients only from LM loss)

Output: trained model θ, bias {b_i}
```

> [!intuition] 为什么用 sign(e_i) 而非 e_i？
> - 使用 $\text{sign}(e_i)$ 使更新步长固定，避免大 violation 导致的剧烈波动
> - 实验表明 $\text{sign}(e_i)$ 版本性能更好（PPL 9.50 vs 9.51-9.53）

---

## 与其他方法对比

> [!comparison] 负载均衡方法对比

| 方法 | 负载均衡 | 干扰梯度 | 未来 token 泄露 |
|------|---------|---------|----------------|
| Loss-Controlled (强 $\alpha$) | 均衡 | 强 | 无 |
| Loss-Controlled (弱 $\alpha$) | 不均衡 | 弱 | 无 |
| Expert Choice | 均衡 | 无 | **有泄露** |
| **Loss-Free (本方法)** | **均衡** | **无** | **无** |

> [!warning] Expert Choice 的问题
> Expert Choice 让每个专家选择固定数量的 token，看似完美均衡，但：
> - 违反因果约束：未来 token 会影响当前 token 的专家分配
> - 导致信息泄露：理论上每个 token 可泄露 $K \log_2 \frac{1-R}{R}$ bits
> - 破坏模型泛化能力

---

## 实验结果

> [!example] 性能对比

| 模型规模 | 方法 | Validation PPL | MaxVio_global |
|---------|------|----------------|---------------|
| 1B | Loss-Controlled | 9.56 | 0.72 |
| 1B | **Loss-Free** | **9.50** | **0.04** |
| 3B | Loss-Controlled | 7.97 | 0.52 |
| 3B | **Loss-Free** | **7.92** | **0.04** |

**关键发现**：
- Loss-Free 同时实现更好的性能和更好的负载均衡
- MaxVio_global 从 0.52-0.72 降至 0.04（提升 10x+）
- 打破了"负载均衡 vs 模型性能"的两难困境

### 超参数研究

| 配置 | PPL | MaxVio_global |
|------|-----|---------------|
| $u = 0.001$（推荐） | 9.50 | 0.044 |
| $u = 0.01$（过大） | 波动 | 后期恶化 |
| $u = 0.0001$（过小） | 收敛慢 | 早期不均衡 |
| 乘性 bias | 9.52 | 0.036-0.048 |
| $b_i + u \cdot e_i$ | 9.51-9.53 | 0.028-0.040 |

---

## 与 Expert Parallelism 的兼容性

> [!intuition] 为什么 Loss-Free 更适合大规模训练？
> Expert Parallelism 将专家分布在多个设备上，每个计算步骤涉及：
> $$\text{computation\_batch} = \text{micro\_batch\_size} \times \text{ep\_data\_parallel\_size}$$
>
> - Loss-Free 的全局负载均衡随 batch size 增大而改善
> - Auxiliary Loss 方法的均衡度在大 batch 时趋于常数
> - 因此 Loss-Free 在大规模 Expert Parallelism 场景下优势更明显

---

## 后续应用

Loss-Free Balancing 已被多个大规模 MoE 模型采用：

| 模型 | 参数量 | 专家数 | 应用方式 |
|------|--------|--------|----------|
| [[DeepSeek-V3]] | 671B | 256 | 首次大规模应用，结合 sequence-wise aux loss |
| Ling 2.0 | 1T | 256 | 配合 [[Multi-Token Prediction\|MTP]] |
| LongCat-Flash | 560B | - | 扩展为零计算专家，动态计算预算 |
| GLM-4.5 | 355B | - | 直接采用 |

> [!example] DeepSeek-V3 的实践
> DeepSeek-V3 在 Loss-Free Balancing 基础上增加了 sequence-wise auxiliary loss 作为补充：
> - 主要依赖 Loss-Free Balancing 控制全局均衡
> - 辅助损失仅用于序列级别的微调
> - 消融实验显示在 15.7B 和 228.7B 模型上都有性能提升

---

## 局限性

> [!warning] 边界条件
>
> 1. **需要历史信息**：bias 更新依赖前一个 batch 的负载统计，训练初期可能不稳定
>
> 2. **超参数敏感**：更新率 $u$ 需要调优，过大导致波动，过小导致收敛慢
>
> 3. **不保证完美均衡**：只能趋近均衡，不像 Expert Choice 那样强制均衡
>
> 4. **推理时的处理**：推理时 bias 是固定的，需要使用训练结束时的 bias 值

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: Loss-Free Balancing 的核心思想是什么？**
> A: 通过动态调整 expert-wise bias 影响 top-K 选择，而非引入辅助损失。bias 只参与路由决策，不参与梯度计算，因此不产生干扰梯度。
>
> **Q2: 为什么传统 Auxiliary Loss 方法有问题？**
> A: 存在两难困境——小的 $\alpha$ 导致负载不均衡，大的 $\alpha$ 引入干扰梯度损害性能。Loss-Free 打破了这个困境。
>
> **Q3: bias 是如何更新的？**
> A: 每个 batch 后，统计每个专家的 token 数与平均值的差异 $e_i$，然后 $b_i = b_i + u \cdot \text{sign}(e_i)$。重负载专家降低 bias，轻负载专家提高 bias。
>
> **Q4: 为什么 Expert Choice 不是好的替代方案？**
> A: Expert Choice 违反因果约束，未来 token 会影响当前 token 的专家分配，导致信息泄露，破坏模型泛化能力。
>
> **Q5: Loss-Free Balancing 在哪些模型中被采用？**
> A: DeepSeek-V3、Ling 2.0、LongCat-Flash、GLM-4.5 等大规模 MoE 模型。

---

## 相关概念

**前置知识**：
- [[Mixture of Experts]] — MoE 架构基础，负载均衡问题的来源

**相关方法**：
- [[Expert Parallelism]] — MoE 分布式训练策略（待创建）
- [[DeepSeek-V3]] — 首个大规模应用 Loss-Free Balancing 的模型（待创建）

**对比方法**：
- [[Expert Choice]] — 另一种无损方法，但有信息泄露问题（待创建）

---

## 参考资料

- Wang et al., "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts", arXiv:2408.15664, 2024
- DeepSeek-AI, "DeepSeek-V3 Technical Report", 2025
