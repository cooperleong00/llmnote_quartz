---
type: method
description: 用小模型快速生成候选 token、大模型并行验证的推理加速技术，通过 rejection sampling 保证输出分布与原模型完全一致
aliases:
  - 投机解码
  - 推测解码
prerequisites:
  - "[[KV Cache]]"
  - "[[Transformer]]"
  - "[[Attention]]"
tags:
  - inference
  - optimization
  - efficiency
created: 2026-01-28
updated: 2026-02-01T01:13
---

# Speculative Decoding

Speculative Decoding（投机解码）是一种 LLM 推理加速技术，核心思想是用一个**快速的 draft model 生成多个候选 token**，然后用**目标大模型并行验证**。通过精心设计的 rejection sampling 机制，这种方法能**数学上保证输出分布与原模型完全一致**，实现无损加速。

## 动机

> [!intuition] 为什么需要 Speculative Decoding？
> LLM 自回归生成的瓶颈不是计算量，而是**内存带宽**。每生成一个 token 都需要加载整个模型权重，但实际计算量很小（尤其是 batch size = 1 时）。这意味着 GPU 大部分时间在等待数据传输，而非计算。
>
> Speculative Decoding 的洞察是：既然每步都要加载模型权重，不如**一次验证多个 token**，摊薄内存访问开销。

**核心问题**：
- 自回归生成是 **memory-bound**，GPU 利用率低
- 每个 token 的生成延迟受限于模型加载时间，而非计算时间
- 增加 batch size 可以提升吞吐，但会增加延迟

**Speculative Decoding 的解决方案**：
- 用小模型快速"猜测"多个 token
- 大模型一次性验证这些猜测
- 正确的猜测直接接受，错误的重新采样

---

## 核心算法

### 基本流程

```
1. Draft 阶段：小模型自回归生成 K 个候选 token
   x_1, x_2, ..., x_K ~ q(x|context)

2. Verify 阶段：大模型并行计算所有位置的概率
   p(x_1|context), p(x_2|context, x_1), ..., p(x_K|context, x_1:K-1)

3. Accept/Reject：对每个位置，决定接受或拒绝
   - 接受：继续验证下一个
   - 拒绝：从修正分布重新采样，停止

4. Bonus Token：无论接受多少，最后都从大模型采样一个新 token
```

### Rejection Sampling 机制

> [!math] 数学保证
> 设 $q(x)$ 是 draft model 的分布，$p(x)$ 是 target model 的分布。对于 draft model 生成的 token $x$：
>
> **接受概率**：
> $$P(\text{accept}) = \min\left(1, \frac{p(x)}{q(x)}\right)$$
>
> **拒绝时的修正分布**：
> $$p'(x) = \text{norm}\left(\max(0, p(x) - q(x))\right)$$
>
> 这个机制保证最终采样的 token 服从 $p(x)$ 分布。

> [!intuition] 直觉理解
> - 如果 draft model 对某个 token 的概率 $q(x)$ 低于 target model $p(x)$，说明 draft model "低估"了这个 token，应该**总是接受**
> - 如果 $q(x) > p(x)$，说明 draft model "高估"了，需要**按比例拒绝**一部分
> - 拒绝后从修正分布采样，补偿被 draft model 低估的 token

### 为什么能保证分布一致？

> [!math] 证明思路
> 对于任意 token $x$，其最终被采样的概率为：
>
> $$P(x) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) + P(\text{reject}) \cdot p'(x)$$
>
> 展开计算可以证明 $P(x) = p(x)$，即最终分布与 target model 完全一致。

---

## 加速原理

### 期望加速比

设 draft model 的 acceptance rate 为 $\alpha$（每个 token 被接受的概率），每次 draft $K$ 个 token：

> [!math] 期望接受 token 数
> $$E[\text{accepted tokens}] = \sum_{i=0}^{K} i \cdot \alpha^i (1-\alpha) + K \cdot \alpha^K = \frac{1 - \alpha^{K+1}}{1 - \alpha} - 1$$
>
> 加上 bonus token，每轮期望生成：
> $$E[\text{tokens per round}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

**加速比估算**：
- 假设 draft model 速度是 target model 的 $c$ 倍（$c > 1$）
- 每轮需要 $K$ 次 draft + 1 次 verify
- 加速比 $\approx \frac{E[\text{tokens}]}{K/c + 1}$

> [!example] 数值例子
> - $\alpha = 0.8$, $K = 4$, $c = 10$
> - 期望 tokens = $(1 - 0.8^5) / 0.2 = 3.36$
> - 每轮成本 = $4/10 + 1 = 1.4$（相对于 target model 单步）
> - 加速比 $\approx 3.36 / 1.4 = 2.4\times$

### 什么时候有效？

> [!warning] 适用条件
> 1. **Memory-bound 场景**：batch size 小，GPU 利用率低
> 2. **Draft model 足够快**：通常需要比 target model 快 5-10 倍以上
> 3. **Acceptance rate 足够高**：draft model 与 target model 分布相近
>
> **不适用场景**：
> - 大 batch size（已经 compute-bound）
> - Draft model 与 target model 差异大（acceptance rate 低）
> - 高度不确定的生成任务（如开放式创作）

---

## 变体方法

### 1. Draft Model Based（独立小模型）

最经典的方法，使用独立训练的小模型作为 draft model。

**优点**：
- 实现简单
- Draft model 可以针对特定任务优化

**缺点**：
- 需要额外训练/维护 draft model
- Draft model 与 target model 可能分布不匹配

**代表工作**：
- Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
- Chen et al. (2023) "Accelerating Large Language Model Decoding with Speculative Sampling"

### 2. Self-Speculative（模型自身早期层）

利用 target model 自身的早期层作为 draft model，无需额外模型。

**核心思想**：
- Transformer 的早期层已经包含足够信息预测下一个 token
- 在早期层添加一个轻量级 head，用于快速 draft

**优点**：
- 无需额外模型，部署简单
- Draft 与 target 天然对齐

**缺点**：
- 需要修改模型架构
- 早期层的预测能力有限

**代表工作**：
- Draft & Verify (2023)
- LayerSkip (2024)

### 3. Medusa（多头并行预测）

在 target model 上添加多个预测头，并行预测多个未来 token。

**架构**：
```
Target Model 最后一层表征
    |-- Head 1 -> 预测 t+1
    |-- Head 2 -> 预测 t+2
    +-- Head K -> 预测 t+K
```

**特点**：
- 不需要独立的 draft model
- 多个头可以并行预测，效率高
- 使用 tree attention 验证多个候选序列

**优点**：
- 单模型，部署简单
- 可以生成多个候选路径

**缺点**：
- 需要额外训练 Medusa heads
- 各头独立预测，不保持因果链

**代表工作**：
- Medusa (2024)

### 4. Lookahead Decoding

利用 Jacobi 迭代的思想，并行生成和验证。

**核心思想**：
- 将自回归生成视为求解方程组
- 用 Jacobi 迭代并行更新所有位置
- 收敛的位置即为正确的 token

**特点**：
- 不需要 draft model
- 理论上可以无限并行

**缺点**：
- 收敛速度不稳定
- 实际加速效果依赖任务

### 5. Multi-Token Prediction (MTP)

训练时让模型预测多个未来 token，推理时 MTP 模块直接作为 draft model。

> [!comparison] 与 Medusa 的区别
> | 特性 | Medusa | MTP |
> |------|--------|-----|
> | 训练方式 | 冻结主模型，只训练 heads | 与主模型联合训练 |
> | 因果链 | 不保持 | 可保持（顺序预测） |
> | 对主模型的影响 | 无 | 可能改善主模型表征 |

详见 [[Multi-Token Prediction]]。

---

## 实际效果

> [!example] 典型加速效果
> | 方法 | 模型 | 加速比 | Acceptance Rate |
> |------|------|--------|-----------------|
> | Draft Model | Llama 2 70B + 7B | 2-3x | 70-80% |
> | Medusa | Vicuna 7B/13B | 2-3x | - |
> | MTP | DeepSeek-V3 | 1.8x | 85-90% |
> | MTP | MiMo-V2-Flash | 2.0-2.7x | - |

**影响因素**：
- **任务类型**：确定性高的任务（如代码补全）加速更明显
- **Batch size**：小 batch 加速更明显
- **Draft model 质量**：与 target model 越接近，acceptance rate 越高

---

## 局限性

> [!warning] Speculative Decoding 的局限性
> 1. **大 batch 场景无效**：已经 compute-bound，无法从并行验证中获益
> 2. **Draft model 开销**：需要额外的模型加载和计算
> 3. **[[KV Cache]] 管理复杂**：需要处理 draft 和 verify 两套 KV Cache
> 4. **Acceptance rate 不稳定**：高不确定性任务效果差
> 5. **实现复杂度**：需要修改推理框架，支持 rejection sampling

---

## 面试要点

> [!interview] 面试视角
> **Q: Speculative Decoding 的核心思想是什么？**
> A: 用快速的小模型生成多个候选 token，大模型并行验证。通过 rejection sampling 保证输出分布与原模型完全一致，实现无损加速。
>
> **Q: 为什么能保证分布一致？**
> A: 使用 rejection sampling：接受概率为 $\min(1, p(x)/q(x))$，拒绝后从修正分布 $\text{norm}(\max(0, p(x)-q(x)))$ 采样。数学上可以证明最终分布等于 target model 分布。
>
> **Q: 什么场景下 Speculative Decoding 最有效？**
> A: Memory-bound 场景（小 batch size）、draft model 与 target model 分布接近、确定性高的任务（如代码补全）。大 batch 或高不确定性任务效果有限。
>
> **Q: 有哪些主要变体？**
> A: Draft model based（独立小模型）、Self-speculative（早期层）、Medusa（多头预测）、Lookahead decoding（Jacobi 迭代）、MTP（多 token 预测）。
>
> **Q: Speculative Decoding 与 MTP 的关系？**
> A: MTP 训练的模块可以直接作为 Speculative Decoding 的 draft model，无需额外训练。DeepSeek-V3 和 MiMo-V2-Flash 都采用这种方式。

---

## 相关概念

- [[Multi-Token Prediction]] - MTP 模块可作为 draft model
- [[Flash Attention]] - 可与 Speculative Decoding 结合使用
- [[Continuous Batching]] - Serving 层面的优化，与 Speculative Decoding 正交

---

## 参考资料

- Leviathan et al. (2023). "Fast Inference from Transformers via Speculative Decoding"
- Chen et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling"
- Cai et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
- Fu et al. (2024). "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"
- DeepSeek-V3 Technical Report (2024)
