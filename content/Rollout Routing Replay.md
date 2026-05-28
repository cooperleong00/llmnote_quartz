---
type: method
description: 通过记录推理时的路由分布并在训练时重放，解决 MoE 模型 RL 训练中的训练-推理不一致问题
aliases:
  - R3
  - 路由重放
prerequisites:
  - "[[RLHF]]"
  - "[[PPO]]"
  - "[[Mixture of Experts]]"
tags:
  - post-training
  - rlhf
  - moe
  - optimization
created: 2026-01-27
updated: 2026-02-01T01:11
---

# Rollout Routing Replay

Rollout Routing Replay (R3) 是一种稳定 [[Mixture of Experts|MoE]] 模型强化学习训练的方法。核心思想是将推理阶段的路由分布（routing distribution）记录下来，并在训练阶段重放（replay），从而消除训练-推理之间的路由不一致性。这种方法将 MoE 模型的训练-推理 KL 散度降低了约 50%，使其接近 Dense 模型的水平。

## 动机

> [!intuition] 为什么需要 R3？
> 在 LLM 强化学习（[[RLHF]]）中，通常使用**分离的引擎**：推理引擎（如 SGLang）生成样本，训练引擎（如 Megatron）更新策略。这种分离在 MoE 模型中会导致严重问题：
>
> **问题根源**：MoE 的路由器（router）对输入敏感，即使输入微小扰动也可能选择完全不同的专家。当训练和推理引擎的数值精度、计算顺序略有差异时，约 10% 的路由器会选择不同的专家，导致 94% 的 token 至少在一层中选择了不同专家。
>
> **后果**：
> - 训练-推理 KL 散度显著增大（MoE 是 Dense 的 2.4 倍）
> - 极端 token（概率比值 > 2）的比例高一个数量级
> - RL 训练不稳定，甚至崩溃（collapse）
>
> **关键洞察**：既然路由不一致是根本原因，为什么不直接让训练时使用推理时的路由决策？

## 核心机制

> [!intuition] 直觉理解
> 想象你在两个不同的厨房（推理引擎和训练引擎）做同一道菜。传统方法是：在厨房 A 做菜并记录结果，然后在厨房 B 重新决定用哪些食材（路由）。R3 的做法是：在厨房 A 做菜时记录下"用了哪些食材"，然后在厨房 B 直接使用相同的食材清单，只是重新计算配比（权重）。
>
> 这样做的好处：
> - **专家选择一致**：训练和推理使用相同的专家
> - **梯度流保留**：权重仍然基于训练引擎的 logits 计算，梯度可以正常回传到路由器

### 传统 MoE 前向传播

在 MoE 层中，对于第 $t$ 个 token 和第 $l$ 层：

1. **计算路由 logits**：
$$\mathbf{s}_{\text{train}} = \mathbf{x}_{\text{train}} \mathbf{W}_r$$

2. **选择 Top-K 专家**（产生二值掩码）：
$$\mathbf{I}_{\text{train}} = \text{TopKMask}(\mathbf{s}_{\text{train}}, K)$$

3. **计算门控权重**（只对选中的专家做 softmax）：
$$g_{\text{train},i} = \frac{I_{\text{train},i} \exp(s_{\text{train},i})}{\sum_{j=1}^{M} I_{\text{train},j} \exp(s_{\text{train},j})}$$

4. **加权组合专家输出**：
$$\mathbf{y}_{\text{train}} = \sum_{i=1}^{M} g_{\text{train},i} \, \mathcal{E}_i(\mathbf{x}_{\text{train}})$$

**问题**：训练和推理的 $\mathbf{I}_{\text{train}}$ 和 $\mathbf{I}_{\text{infer}}$ 可能不同！

### R3 的改进

R3 在推理阶段记录路由掩码 $\mathbf{I}_{\text{infer}}$，在训练阶段：

1. **使用推理的路由掩码**：
$$\mathbf{I}_{\text{replay}} = \mathbf{I}_{\text{infer}} \quad \text{(来自推理引擎)}$$

2. **但用训练的 logits 计算权重**：
$$g_{\text{replay},i} = \frac{I_{\text{infer},i} \exp(s_{\text{train},i})}{\sum_{j=1}^{M} I_{\text{infer},j} \exp(s_{\text{train},i})}$$

3. **输出**：
$$\mathbf{y}_{\text{replay}} = \sum_{i=1}^{M} g_{\text{replay},i} \, \mathcal{E}_i(\mathbf{x}_{\text{train}})$$

> [!math] 关键设计
> - **掩码来自推理**：$\mathbf{I}_{\text{infer}}$ 确保专家选择一致
> - **权重来自训练**：$\exp(s_{\text{train},i})$ 确保梯度可以回传到路由器
> - **两全其美**：既对齐了训练-推理，又保留了优化能力

### 路由掩码缓存

R3 支持与 KV Cache 类似的**路由掩码缓存**：

- 对于相同的 prefix token，路由决策应该相同
- 将 $\mathbf{I}_{\text{infer}}$ 与 KV Cache 一起缓存
- 多轮对话场景下，直接复用缓存的路由掩码
- 对 Agent 任务（如代码生成、网页浏览）尤其重要

## 实验效果

### 训练-推理一致性

在 Qwen3-30B-A3B（MoE 模型）上：

| 指标 | MoE (无 R3) | MoE (R3) | Dense 基线 |
|------|-------------|----------|-----------|
| KL 散度 | $1.535 \times 10^{-3}$ | $7.5 \times 10^{-4}$ | $6.4 \times 10^{-4}$ |
| 极端 token 比例 ($\tau > 2$) | ~10× Dense | ~1× Dense | 基线 |

**结论**：R3 将 MoE 的训练-推理一致性提升到接近 Dense 模型的水平。

### RL 训练稳定性

论文使用 [[PPO]]/GRPO 作为 RL 算法，在数学推理任务上（AIME24, AIME25, AMC23, MATH500）：

**Single Mini-Step + SFT 模型**：
- [[GRPO]]：在 60 步崩溃
- GRPO + TIS：在 105 步崩溃
- GRPO + R3：**不崩溃**，最终性能 71.83（比 TIS 高 5.58）

**Single Mini-Step + Base 模型**：
- GRPO：在 105 步崩溃
- GRPO + TIS：不崩溃，性能 69.22
- GRPO + R3：**不崩溃**，性能 70.73（比 TIS 高 1.51）

**Multi Mini-Step + SFT 模型**：
- GRPO：在 120 步崩溃
- GSPO：不崩溃，性能 66.76
- GRPO + R3：性能 68.05（比 GSPO 高 1.29）
- GSPO + R3：性能 **69.00**（最佳）

### 训练动态

R3 改善了优化行为：
- **梯度范数更小**：优化更稳定
- **序列长度增长更平滑**：更快捕捉正确优化方向
- **熵增长更早更稳定**：更早开始探索，波动更小

## 局限性

> [!warning] 适用边界
> 1. **仅适用于 MoE 模型**：Dense 模型没有路由，不需要 R3
> 2. **需要缓存基础设施**：需要在推理引擎中记录路由掩码，并传递给训练引擎
> 3. **与 TIS 组合效果不明显**：R3 已经大幅降低了训练-推理差异，TIS 的额外修正收益有限，甚至可能负面（SFT 模型上 TIS+R3 比 R3 低 1.69）
> 4. **不解决其他不稳定性来源**：如 reward hacking、梯度爆炸等问题仍需其他方法

> [!warning] 实现要求
> - 推理引擎需要支持路由掩码导出
> - 训练引擎需要支持路由掩码注入
> - 需要修改 MoE 层的前向传播逻辑

## 相关概念

**相关方法**（解决训练-推理不一致）：
- [[Truncated Importance Sampling]] (TIS) — 通过截断重要性权重缓解不一致
- [[GSPO]] — 序列级重要性采样
- Batch-Invariant Operations — 通过确定性计算减少不一致（性能开销大）

**延伸阅读**：
- [[Agent Training]] — 多轮对话场景下路由缓存尤其重要

## 速查

> [!example] 关键要点
> **核心公式**：
> $$g_{\text{replay},i} = \frac{I_{\text{infer},i} \exp(s_{\text{train},i})}{\sum_{j=1}^{M} I_{\text{infer},j} \exp(s_{\text{train},i})}$$
>
> **实现要点**：
> - 推理阶段：记录每层每个 token 的 TopK 掩码 $\mathbf{I}_{\text{infer}}$
> - 训练阶段：用 $\mathbf{I}_{\text{infer}}$ 替换 $\mathbf{I}_{\text{train}}$，但保留 $s_{\text{train}}$
> - 缓存：与 KV Cache 一起缓存路由掩码
>
> **效果**：
> - KL 散度降低 ~50%（从 $1.5 \times 10^{-3}$ 到 $7.5 \times 10^{-4}$）
> - 极端 token 减少约 10 倍
> - 防止 RL 训练崩溃

> [!interview] 面试视角
> **Q: R3 如何解决 MoE RL 训练不稳定问题？**
> A: R3 识别出路由不一致是 MoE 训练-推理差异的根本原因。通过记录推理时的路由掩码并在训练时重放，确保训练和推理使用相同的专家，同时保留梯度流到路由器。这将 KL 散度降低约 50%，极端 token 减少 10 倍，有效防止训练崩溃。
>
> **Q: R3 与 TIS 有什么区别？**
> A: TIS 通过截断重要性权重来缓解训练-推理不一致的**后果**，而 R3 直接解决不一致的**根源**（路由差异）。R3 更彻底，效果更好。两者可以组合，但 R3 已经大幅降低差异后，TIS 的额外收益有限。
>
> **Q: R3 的计算开销如何？**
> A: 几乎没有额外开销。只需要：(1) 推理时记录路由掩码（已经计算过）；(2) 训练时读取掩码（替换 TopK 操作）。支持缓存后，多轮对话场景下甚至可以复用掩码。
>
> **Q: R3 为什么不影响路由器的优化？**
> A: 虽然使用推理的掩码 $\mathbf{I}_{\text{infer}}$，但门控权重仍基于训练的 logits $s_{\text{train}}$ 计算。梯度可以正常回传到 $\mathbf{W}_r$，路由器仍然可以学习。这是 R3 的关键设计。

## 参考资料

- 论文：[[251011370v1|Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers (2025)]]
- 实现框架：VeRL (训练), SGLang (推理), Megatron (训练引擎)

