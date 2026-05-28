---
description: Transformer 中的 outlier 现象，通过与 normalization 协同实现 rescaling 功能；可通过 learnable bias 或 gating 机制缓解
type: concept
aliases:
  - 注意力汇聚
  - Sink Token
prerequisites:
  - "[[Attention]]"
  - "[[Sliding Window Attention]]"
  - "[[RMSNorm]]"
tags:
  - attention
  - efficient-attention
  - inference
  - architecture
created: 2026-01-27
updated: 2026-02-04T00:19
---

# Attention Sink

Attention Sink 是 Transformer 中的一种 outlier 现象：少数 token（通常是序列开头的 BOS）持续获得异常高的 attention score。这一现象与 [[RMSNorm]] 后的 **Residual Sink**（固定维度的大激活值）共同构成了 **Outlier-Driven Rescaling** 机制——outliers 与 normalization 协同工作，实现对非 outlier 成分的 rescaling。

> [!intuition] Outlier-Driven Rescaling 的统一视角
> Attention sink 和 residual sink 本质上都是 **rescaling 因子**，而非信息贡献者：
> - Attention sink 的 value vector norm 很小，实际贡献远小于其 attention score 暗示的程度
> - Residual sink 维度对应的 RMSNorm weight 很小（如 0.006 vs 平均 1），说明这些维度主要用于调节 norm 而非传递特征
>
> 移除 normalization 会消除 outliers，但同时损害训练稳定性和性能——说明 outlier-driven rescaling 是训练所必需的。

---

## 动机

> [!intuition] 为什么需要 Attention Sink？
> Softmax 强制注意力权重和为 1。当窗口内没有相关信息时，模型被迫关注无关 token，导致信息污染。Attention Sink 提供一个"垃圾桶"位置，让模型可以选择"不关注任何 token"。

### Softmax 的强制分配问题

标准 [[Attention]] 的 softmax 归一化：

$$
\alpha_{ij} = \frac{\exp(a_{ij})}{\sum_k \exp(a_{ik})}
$$

**问题**：无论 $a_{ij}$ 多小，$\sum_j \alpha_{ij} = 1$ 恒成立。模型**必须**把注意力分配给某些 token。

### 在 Sliding Window Attention 中的表现

[[Sliding Window Attention]] 限制每个 token 只能 attend 局部窗口。当窗口内没有语义相关的 token 时：

```
序列:  [The] [cat] [sat] [on] [the] [mat] [.] [It] [was] [soft]
                                              ↑
                                          当前位置 "It"

W=4:   [mat] [.] [It] [was]  ← 窗口内没有 "cat" 的信息
       └─────────────────┘
         被迫关注这些 token
```

模型被迫关注 `[mat]`、`[.]` 等无关 token，导致：
- **信息污染**：无关信息混入表示
- **性能退化**：纯 SWA 效果明显差于 Global Attention

---

## Attention Sink 现象

> [!intuition] 自然涌现的 Sink
> 研究发现，LLM 会自发地将大量注意力分配给序列开头的 token（如 BOS），即使这些 token 本身没有语义意义。它们充当了"注意力垃圾桶"。

### StreamingLLM 的发现

StreamingLLM (Xiao et al., 2023) 观察到：

1. **初始 token 吸引大量注意力**：BOS 或第一个 token 获得异常高的注意力权重
2. **移除初始 token 导致崩溃**：在 streaming 推理中，如果滑动窗口丢弃初始 token，模型输出会严重退化
3. **保留初始 token 稳定推理**：只需保留前几个 token（即使窗口滑动），就能维持稳定输出

> [!example] StreamingLLM 的解决方案
> ```
> 传统 SWA:  [t1] [t2] [t3] [t4] [t5] [t6] [t7] [t8]
>                      └─────────────────────────┘
>                           窗口滑动，丢弃 t1, t2
>                           → 性能崩溃
>
> StreamingLLM: [t1] [t2] ... [t5] [t6] [t7] [t8]
>               └──┘          └─────────────────┘
>             保留 sink        滑动窗口
>                           → 稳定输出
> ```

### 为什么初始 token 成为 Sink？

> [!intuition] Sink 的形成机制
> 1. **位置优势**：初始 token 在所有位置的 causal mask 中都可见
> 2. **训练适应**：模型学会利用这些"总是可见"的位置作为注意力缓冲
> 3. **语义无关**：BOS 等 token 本身没有强语义，不会干扰其他计算

### Attention Sink 的功能角色

> [!intuition] Sink 是 Rescaling 因子，不是信息贡献者
> 研究发现 sink token 的 value vector norm 显著小于非 sink token。这意味着：
> - 虽然 sink 获得很高的 attention score，但其对 attention output 的实际贡献很小
> - Sink 的真正作用是**通过 softmax 归一化来 rescale 其他 token 的贡献**
>
> 这解释了为什么直接 clip attention logits 会损害性能——破坏了 rescaling 机制。

---

## Residual Sink

除了 attention sink，Transformer 的 residual stream 中还存在另一类 outlier：**Residual Sink**。

> [!definition] Residual Sink
> Residual stream 中**固定的少数维度**在几乎所有 token 上都表现出异常大的激活值（比其他维度大几个数量级）。这些维度与输入无关，在不同样本间保持一致。

### 与 Attention Sink 的关系

| 特性 | Attention Sink | Residual Sink |
|------|----------------|---------------|
| **位置** | Attention logits | Residual stream |
| **模式** | 少数 token 获得高 score | 少数维度有大激活值 |
| **输入依赖** | 与特定 token 绑定 | 与输入无关，固定维度 |
| **协同的 normalization** | Softmax | RMSNorm |
| **缓解方法** | Gated Attention / Learnable Bias | GatedNorm / PreAffine |

### Massive Activation (MA)

> [!definition] Massive Activation
> 与 attention sink 关联的 token（如 BOS）在特定维度上出现极端大的激活值（可达 1000+）。MA 是 attention sink 和 residual sink 之间的桥梁：
> - MA 经过 normalization 后产生近似 one-hot 的向量
> - 投影到 key space 后只激活固定的矩阵列
> - 这促进了 attention sink 的形成

当引入 learnable sink bias 后，模型不再需要从真实 token 生成 MA，因此 MA 消失——但 residual sink 仍然存在。

---

## Learnable Attention Sink Bias

从"被动利用"到"主动学习"——引入可学习的 sink bias，显式建模这一机制。

### 核心思想

> [!definition] Learnable Sink Bias
> 在 softmax 分母中添加一个可学习的标量 $s$（每个 attention head 一个）：
>
> $$
> \alpha_{ij} = \frac{\exp(a_{ij})}{\exp(s) + \sum_k \exp(a_{ik})}
> $$
>
> 当 $s$ 较大时，所有 $\alpha_{ij}$ 都会变小，允许模型输出接近零向量。

### 数值稳定的实现

> [!math] MiMo-V2-Flash 的实现
> 为了数值稳定，使用 log-sum-exp 技巧：
>
> $$
> \begin{aligned}
> a_{ij} &= \frac{q_i \cdot k_j^T}{\sqrt{d}} \\
> m_i &= \max\left(\max_j(a_{ij}), s\right) \\
> \alpha_{ij} &= \frac{\exp(a_{ij} - m_i)}{\exp(s - m_i) + \sum_{j'} \exp(a_{ij'} - m_i)} \\
> o_i &= \sum_j \alpha_{ij} \cdot v_j
> \end{aligned}
> $$
>
> 其中 $s$ 是可学习的 sink bias，$m_i$ 是用于数值稳定的最大值。

### 直觉理解

> [!intuition] Sink Bias 的作用
> 想象 sink 是一个"虚拟 token"，它的 attention score 是 $s$，但它的 value 是零向量。
>
> - 当 $s \gg a_{ij}$：大部分注意力流向 sink，输出接近零
> - 当 $s \ll a_{ij}$：sink 被忽略，正常计算 attention
> - 模型学习何时"不关注任何 token"

---

## 效果验证

### MiMo-V2-Flash 实验

> [!example] Hybrid SWA + Sink Bias 的效果
> | 配置 | 相对性能 |
> |------|----------|
> | All-GA (baseline) | 100% |
> | W=512, no sink | 95% |
> | W=128, no sink | 90% |
> | **W=128, with sink** | **101%** |
>
> 关键发现：
> - 没有 sink bias 的 SWA 性能明显退化
> - W=128 + sink bias **超越** All-GA baseline
> - Sink bias 不只是修复问题，而是让模型学到更好的注意力模式

### 为什么 Sink Bias 能超越 Global Attention？

> [!intuition] 可能的解释
> 1. **正则化效果**：强制模型区分"需要关注"和"不需要关注"
> 2. **更清晰的分工**：SWA 层专注局部，GA 层处理长程依赖
> 3. **减少噪声**：避免无关 token 的信息污染

---

## 与 StreamingLLM 的对比

> [!comparison] 两种方法的对比
> | 方面 | StreamingLLM | Learnable Sink Bias |
> |------|--------------|---------------------|
> | **方法** | 保留初始 token | 引入可学习 bias |
> | **训练** | 无需训练 | 需要训练 |
> | **灵活性** | 固定位置 | 每个 head 独立学习 |
> | **额外开销** | 存储初始 token 的 [[KV Cache\|KV]] | 每个 head 一个标量 |
> | **适用场景** | 推理时的 streaming | 训练时的架构设计 |

**关键区别**：
- StreamingLLM 是**推理时的 hack**，利用已有模型的 sink 现象，实现 [[Length Extrapolation|长度外推]]
- Learnable Sink Bias 是**训练时的设计**，显式学习何时不关注

---

## 局限性

> [!warning] 边界条件
> 1. **需要训练**：不能直接应用于已训练的模型
> 2. **超参数**：sink bias 的初始化可能影响训练稳定性
> 3. **与其他技术的交互**：与 [[Flash Attention]] 等实现的兼容性需要验证

---

## Gating-Based 缓解方法

除了 learnable sink bias，还可以通过显式的 gating 机制来缓解 outliers，减少模型对 outlier-driven rescaling 的依赖。

### [[Gated Attention]] (GA)

在 SDPA 输出后添加 **head-specific sigmoid gating**，显式提供 rescaling 能力：

$$
Y' = Y \odot \sigma(XW_\theta)
$$

其中 $Y$ 是 SDPA 输出，$\sigma$ 是 sigmoid，$W_\theta$ 是可学习参数。

> [!intuition] GA 消除 Attention Sink 的机制
> Sigmoid gating 引入 **query-dependent 稀疏性**：
> - 当 gating score 接近 0 时，对应的 SDPA 输出被抑制
> - 模型不再需要通过 attention sink 来"倾倒"无关注意力
> - 同时打破了 $W_V$-$W_O$ 的 low-rank 瓶颈，增强表达能力

**GA 的效果**（15B MoE，3.5T tokens）：
- PPL 降低约 0.2，MMLU +2 points
- 几乎消除 loss spikes，支持更大 learning rate
- 第一个 token 的 attention score 降至正常水平
- 长度外推性能显著提升（128k 上比 baseline 高约 27 points）

### GatedNorm

> [!definition] GatedNorm
> 在每个 [[RMSNorm]] 层后添加 low-rank self-gating 机制，显式提供 rescaling 能力：
>
> $$
> \mathbf{y}_g = \sigma(\mathbf{W}_{up}(\text{swish}(\mathbf{W}_{down}(\mathbf{y})))), \quad \mathbf{y}' = \mathbf{y}_g \odot \mathbf{y}
> $$
>
> 其中 $\mathbf{W}_{down} \in \mathbb{R}^{d \times r}$，$\mathbf{W}_{up} \in \mathbb{R}^{r \times d}$，$r \ll d$（如 $r=16$），$\sigma$ 是 sigmoid。

**GatedNorm 的效果**：
- 有效减少 residual sink 的幅度
- 训练 loss 平均降低约 2 points
- W4A4 量化性能提升约 1.2 points（因为激活更平滑）
- 参数开销仅约 2%（2B 模型增加 3.7M 参数）

> [!intuition] 为什么 Gating 有效？
> 当模型有显式的 rescaling 机制时，不再需要通过 outliers 来实现这一功能。GatedNorm 后，RMSNorm 的 learned weights 在各维度间更均匀（最大偏离 1 仅 0.73，而 baseline 是 0.004），说明模型不再需要抑制特定维度。

---

## 面试要点

> [!interview] 面试视角
> **Q: 什么是 Attention Sink？**
> A: LLM 会将大量注意力分配给序列开头的 token（如 BOS），即使它们没有语义意义。这些 token 充当"注意力垃圾桶"，让模型在不需要关注任何 token 时有地方"倾倒"注意力。
>
> **Q: Attention Sink 的真正功能是什么？**
> A: Sink 本质上是 **rescaling 因子**，而非信息贡献者。Sink token 的 value vector norm 很小，实际贡献远小于其 attention score 暗示的程度。它通过 softmax 归一化来 rescale 其他 token 的贡献。
>
> **Q: 什么是 Residual Sink？它和 Attention Sink 有什么关系？**
> A: Residual sink 是 residual stream 中固定维度持续出现的大激活值。两者都是 outlier-driven rescaling 的表现：attention sink 与 softmax 协同，residual sink 与 RMSNorm 协同，共同实现对非 outlier 成分的 rescaling。
>
> **Q: 如何缓解 Sink 现象？**
> A: 三种方法：(1) Learnable sink bias：在 softmax 分母添加可学习标量；(2) Gated Attention：显式 gating 提供 rescaling；(3) GatedNorm：在 RMSNorm 后添加 low-rank self-gating。这些方法通过显式提供 rescaling 能力，减少模型对 outliers 的依赖。

---

## 参考资料

- Efficient Streaming Language Models with Attention Sinks (Xiao et al., 2023) — 发现 attention sink 现象
- gpt-oss (Agarwal et al., 2025) — 提出 learnable sink bias 设计
- MiMo-V2-Flash (2025) — 在 Hybrid SWA 中验证 sink bias 的效果
- [[Clippings/Paper/260122966v1/260122966v1|A Unified View of Attention and Residual Sinks]] (Qiu et al., 2025) — 提出 outlier-driven rescaling 统一视角和 GatedNorm
- [[Clippings/Paper/250506708v1/250506708v1|Gated Attention for Large Language Models]] (Qiu et al., 2025) — 系统研究 gating 机制，提出 SDPA output gating 消除 attention sink

