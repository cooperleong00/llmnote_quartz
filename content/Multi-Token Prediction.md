---
type: method
description: MTP 训练目标让模型同时预测多个未来 token，提升训练效率和模型性能，并可用于 Speculative Decoding 加速推理
aliases:
  - MTP
  - 多 token 预测
prerequisites:
  - "[[Transformer]]"
  - "[[Cross-Entropy Loss]]"
  - "[[KV Cache]]"
tags:
  - pretraining
  - inference
  - efficiency
created: 2026-01-28
updated: 2026-01-31T22:42
---

# Multi-Token Prediction

Multi-Token Prediction（MTP，多 token 预测）是一种训练目标，让模型在每个位置同时预测多个未来 token，而非仅预测下一个 token。这种方法可以**提升训练信号密度**、**改善模型表征**，并且训练得到的 MTP 模块可以直接用于 [[Speculative Decoding]] 加速推理。

## 动机

> [!intuition] 为什么需要 MTP？
> 标准的 Next-Token Prediction（NTP）每个位置只提供一个训练信号。MTP 通过预测多个未来 token，**增加训练信号密度**，可能提升数据效率。更重要的是，MTP 可能促使模型**提前规划表征**（pre-plan representations），为预测更远的 token 做准备，从而学到更好的内部表示。

**核心优势**：
1. **训练效率**：更密集的训练信号，可能提升数据效率
2. **表征质量**：模型需要考虑更长的上下文依赖，学到更好的表征
3. **推理加速**：MTP 模块可直接用于 Speculative Decoding，无需额外训练 draft model

## 为什么有效：理论解释

Gloeckle et al. (2024) 提出了两个理论视角解释 MTP 的有效性。

### Choice Points 强化

> [!intuition] 不是所有 token 决策都同等重要
> 有些 token 只是风格变化（如同义词选择），不影响后续文本；而有些是**关键决策点（choice points）**，决定了生成质量是否会"跑偏"（derailing）。

MTP 通过 lookahead 隐式地给不同 token 赋予不同权重：

- **Choice point**：由于其后果（consequences）同样难以预测，n-token prediction 会通过多个相关 loss term 累积权重，总权重为 $\frac{n(n+1)}{2}$
- **Inconsequential point**：只获得 $n$ 倍的权重

因此，MTP 平均给 choice points 赋予 $\frac{n+1}{2}$ 倍于 inconsequential points 的重要性。

### 信息论视角

> [!math] Mutual Information 的权重增加
> 设 X 为下一个 token，Y 为第二个 token。标准 NTP 优化 H(X)，而 2-token prediction 优化 H(X) + H(Y)。分解后：
> $$H(X) + H(Y) = H(X \mid Y) + 2I(X;Y) + H(Y \mid X)$$
>
> 关键观察：**mutual information I(X;Y) 的权重翻倍**。这意味着 MTP 更准确地预测那些对后续文本有影响的 token X。

直觉上，MTP 缓解了 teacher forcing 训练和 autoregressive 推理之间的分布不匹配：模型被迫考虑更长期的依赖，而非只关注短期预测。

## 两种实现方式

MTP 有两种主要的实现范式，核心区别在于**是否保持因果链**。

### 并行预测（Independent Heads）

> [!paper] Gloeckle et al. (2024)
> "Better & Faster Large Language Models via Multi-token Prediction" 提出使用**独立的 output heads** 并行预测多个 token。

**架构**：在主模型的最后一层表征上，添加 D 个独立的 output heads，每个 head 预测不同位置的未来 token。

```
Position i 的表征 h_i
    |-- Head 1 -> 预测 t_{i+1}
    |-- Head 2 -> 预测 t_{i+2}
    +-- Head D -> 预测 t_{i+D}
```

**特点**：
- 各 head 独立预测，**不保持因果链**
- 实现简单，训练高效
- 但推理时用于 Speculative Decoding 效果可能受限

**Memory-efficient 实现**：

> [!math] 内存优化技巧
> 朴素实现需要同时 materialize 所有 n 个 head 的 logits 和梯度，内存占用 O(nV + d)，其中 V 是词表大小，d 是隐藏维度。
>
> Gloeckle et al. 提出**顺序 forward/backward**：
> 1. 完成 shared trunk 的 forward pass
> 2. 对每个 head i **依次**执行 forward 和 backward
> 3. 在处理 head i+1 前释放 head i 的 logits 和梯度
> 4. 只保留 d 维的 trunk gradient $\partial L_n / \partial f_s$
>
> 这将内存从 O(nV + d) 降到 O(V + d)，**无运行时开销**。

### 顺序预测（Sequential with Causal Chain）

> [!paper] DeepSeek-V3 (2024)
> DeepSeek-V3 采用**顺序预测**方式，保持完整的因果链。这种设计类似于 EAGLE 的思路，但目标是改善训练而非仅用于推理。

**架构**：使用 D 个顺序的 MTP 模块，每个模块包含：
- 共享的 Embedding 层
- 独立的 Transformer block
- 共享的 Output Head

**关键设计**：第 k 层的输入结合了第 k-1 层的表征和第 i+k 个 token 的 embedding：

$$\mathbf{h}_{i}^{\prime k} = M_{k}[\text{RMSNorm}(\mathbf{h}_{i}^{k-1}); \text{RMSNorm}(\text{Emb}(t_{i+k}))]$$

其中 $[\cdot;\cdot]$ 表示拼接，$M_k \in \mathbb{R}^{d \times 2d}$ 是投影矩阵。

> [!comparison] 两种方式对比
> | 特性 | 并行预测 | 顺序预测 |
> |------|----------|----------|
> | 因果链 | 不保持 | 保持完整 |
> | 训练复杂度 | 较低 | 较高 |
> | Speculative Decoding 效果 | 一般 | 更好 |
> | 代表模型 | Gloeckle et al. | DeepSeek-V3, Ling 2.0 |

## 训练目标

> [!math] MTP Loss
> 对于每个预测深度 k，计算交叉熵损失：
> $$\mathcal{L}_{\text{MTP}}^{k} = -\frac{1}{T} \sum_{i=2+k}^{T+1} \log P_{i}^{k}[t_{i}]$$
>
> 总 MTP 损失为各深度损失的加权平均：
> $$\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{\text{MTP}}^{k}$$
>
> 其中 $\lambda$ 是权重因子，$D$ 是预测深度。

**训练时的灵活性**：MTP 主要用于改善主模型的训练。推理时可以**直接丢弃 MTP 模块**，主模型独立工作；也可以保留 MTP 模块用于 Speculative Decoding。

## 推理应用：Speculative Decoding

MTP 模块可以直接作为 [[Speculative Decoding]] 的 draft model，无需额外训练。

### 工作流程

1. **Draft 阶段**：MTP 模块快速生成 D 个候选 token
2. **Verify 阶段**：主模型并行验证这些候选 token
3. **Accept/Reject**：根据验证结果接受或拒绝候选 token

### 实际效果

> [!example] DeepSeek-V3 的 MTP 效果
> - **预测深度**：D = 1（预测 2 个 token）
> - **Acceptance Rate**：85-90%（跨不同生成任务）
> - **推理加速**：1.8x TPS（Tokens Per Second）

> [!example] MiMo-V2-Flash 的 MTP 效果
> - **预测深度**：D = 3（3 层 MTP）
> - **MTP 模块参数**：仅 0.33B（轻量级设计，使用 Dense FFN + [[Sliding Window Attention]]）
> - **推理加速**：2.0-2.7x（取决于 batch size 和 acceptance length）

### Acceptance Length 与任务 Entropy 的关系

> [!intuition] 低 entropy 任务 → 更长的 acceptance length
> MiMo-V2-Flash 的实验表明，acceptance length 与 next-token cross-entropy 呈强负相关：
> - **WebDev**（低 entropy）：~3.6 tokens
> - **MMLU Pro**（高 entropy）：~2.0 tokens
>
> 拟合公式：$y = 4(1 - 0.58x^{0.58})$，$R^2 = 0.995$

| Batch Size | Acceptance 2.8 | Acceptance 3.2 | Acceptance 3.6 |
|------------|----------------|----------------|----------------|
| 32 | 1.86× | 2.12× | 2.39× |
| 64 | 1.97× | 2.25× | 2.53× |
| 96 | 1.99× | 2.28× | 2.56× |

*表：MiMo-V2-Flash 3-layer MTP 在不同 batch size 和 acceptance length 下的加速比*

### MTP 对 RL 训练的加速

MTP 特别适合加速 RL 训练中的 rollout 阶段：

1. **Rollout 是瓶颈**：RL 训练中，生成 rollout 的推理成本往往是主要瓶颈
2. **缓解 long-tail stragglers**：当 rollout 进入后期，部分序列变长导致 batch size 接近 1，GPU 利用率下降。MTP 在这种 memory-bound 场景下能显著提升效率

## 训练效果

> [!example] DeepSeek-V3 Ablation Study
> | Benchmark | Small MoE Baseline | Small MoE w/ MTP | Large MoE Baseline | Large MoE w/ MTP |
> |-----------|-------------------|------------------|-------------------|------------------|
> | HumanEval | 20.7 | **26.8** | 44.5 | **53.7** |
> | GSM8K | 25.4 | **31.4** | 72.3 | **74.0** |
> | MATH | 10.7 | **12.6** | 38.6 | **39.8** |
>
> MTP 在代码和数学推理任务上提升尤为明显。

## Scaling 行为

> [!warning] MTP 只在大模型上有效
> Gloeckle et al. 发现 MTP 的效果**随模型规模增大而增强**，这可能是 MTP 之前被忽视的原因。

| 模型规模 | n=1 (baseline) | n=4 (MTP) | 变化 |
|---------|----------------|-----------|------|
| 0.3B | 1.8% | 1.0% | -44% ❌ |
| 1.3B | 6.8% | 7.4% | +9% |
| 3B | 11.1% | 12.7% | +14% |
| 7B | 23.9% | 26.0% | +9% |
| 13B | 26.0% | 30.5% | +17% ✓ |

*表：MBPP pass@1 结果，训练于 91B tokens 代码数据*

**最优 n 的选择**：
- **Token-level**：n=4 在 HumanEval/MBPP 上最优
- **Byte-level**：n=8 最优
- 最优 n 依赖数据分布，APPS/Intro 上 n=6 更好

## Byte-level 模型

MTP 对 byte-level 模型的提升尤为显著，因为 byte-level tokenization 更容易陷入局部模式。

> [!example] 7B Byte-level 模型结果（314B bytes 训练）
> | Benchmark | n=1 | n=8 | 提升 |
> |-----------|-----|-----|------|
> | MBPP pass@1 | 19.3% | 32.3% | **+67%** |
> | HumanEval pass@1 | 18.1% | 21.8% | **+20%** |
> | APPS/Intro pass@1 | 0.1% | 1.2% | **+1100%** |
>
> 8-byte prediction 模型配合 self-speculative decoding 可实现 **6.4× 推理加速**，完全补偿 byte-level 序列更长的开销。

## 轻量级 MTP 设计

为了在推理时保持高效，MTP 模块通常采用轻量级设计：

- **Dense FFN**：而非 [[Mixture of Experts|MoE]]，减少路由开销
- **[[Sliding Window Attention]]**：而非 Global Attention，减少 KV Cache
- **参数共享**：Embedding 和 Output Head 与主模型共享

> [!example] MiMo-V2-Flash 的轻量级 MTP
> 每个 MTP block 仅 0.33B 参数，使用 SWA（窗口大小 128）+ Dense FFN，相比主模型的 MoE 层大幅降低计算量。

## 局限性

> [!warning] MTP 的局限性
> 1. **训练复杂度增加**：需要额外的 MTP 模块和损失计算
> 2. **Pipeline 调度复杂**：MTP 层与主模型层的异构性增加了分布式训练的调度难度
> 3. **Acceptance Rate 依赖任务**：在高不确定性任务（如开放式生成）上，acceptance rate 可能下降
> 4. **小模型无效**：<1B 参数的模型使用 MTP 反而会降低性能
> 5. **自然语言 benchmark 效果有限**：在选择题类 benchmark 上无提升或略有下降，但在生成任务（summarization）上有提升

## 相关概念

- [[Speculative Decoding]] - MTP 的主要推理应用
- [[KV Cache]] - MTP 需要考虑的内存优化
- [[Sliding Window Attention]] - 轻量级 MTP 常用的注意力机制
- [[Mixture of Experts]] - 与 MTP 结合的稀疏架构

## 参考资料

- Gloeckle et al. (2024). "Better & Faster Large Language Models via Multi-token Prediction"
- DeepSeek-V3 Technical Report (2024), Section 2.2
- MiMo-V2-Flash Technical Report (2025), Section 2.3
- Ling 2.0 Technical Report (2025)

> [!interview] 面试视角
> **Q: MTP 和标准 NTP 的核心区别是什么？**
> A: NTP 每个位置只预测下一个 token，MTP 同时预测多个未来 token。这增加了训练信号密度，可能促使模型学习更好的表征。
>
> **Q: MTP 如何用于推理加速？**
> A: MTP 模块可以作为 Speculative Decoding 的 draft model。先用 MTP 快速生成多个候选 token，再用主模型并行验证。DeepSeek-V3 实现了 85-90% 的 acceptance rate 和 1.8x 的加速。
>
> **Q: 并行预测和顺序预测的区别？**
> A: 并行预测使用独立的 output heads，不保持因果链；顺序预测保持完整因果链，每层的输入依赖上一层的输出。顺序预测在 Speculative Decoding 中效果更好。
>
> **Q: 为什么 MTP 能提升模型性能？**
> A: 两个理论解释：(1) MTP 给"关键决策点"（choice points）赋予更高权重，这些点决定生成质量；(2) 从信息论角度，MTP 增加了 token 间 mutual information 的权重，促使模型学习更长期的依赖。
>
> **Q: MTP 有什么局限性？**
> A: 主要是：(1) 只在大模型（>1B）上有效，小模型反而变差；(2) 在选择题类 benchmark 上效果有限，主要提升生成任务；(3) 增加训练复杂度和 pipeline 调度难度。
