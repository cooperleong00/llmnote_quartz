---
description: 修正 GRPO 优化偏差的无偏变体，移除长度和标准差归一化，防止错误响应变长并提升 token 效率
type: method
aliases:
  - Dr. GRPO
  - GRPO Done Right
  - 无偏 GRPO
prerequisites:
  - "[[GRPO]]"
  - "[[PPO]]"
tags:
  - post-training
  - reinforcement-learning
  - rlhf
  - optimization
created: 2026-01-27
updated: 2026-02-01T01:02
---

# Dr. GRPO (GRPO Done Right)

Dr. GRPO 是 [[GRPO]] 的**无偏优化变体**，通过移除长度归一化和标准差归一化来修正 GRPO 中的优化偏差，防止模型在训练过程中生成越来越长的错误响应，从而提升 token 效率。

> [!paper] 论文出处
> Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective", 2025
> - 批判性分析 DeepSeek-R1-Zero 训练
> - 识别 GRPO 的优化偏差
> - 提出 Dr. GRPO 并在 AIME 2024 上达到 43.3% (7B 模型)

---

## 动机

> [!intuition] 为什么需要 Dr. GRPO？

[[GRPO]] 在实践中观察到一个现象：**响应长度随训练持续增长**，尤其是错误响应。这通常被解释为"长思维链（long-CoT）的涌现"，但论文发现这可能是**优化偏差**导致的：

1. **响应级长度偏差**：GRPO 对每个 token 的 loss 除以响应长度 $|o_i|$，导致：
   - 正确响应：更短的响应获得更大的梯度更新（鼓励简洁）
   - **错误响应**：更长的响应惩罚更小（鼓励冗长）

2. **问题级难度偏差**：GRPO 用组内标准差归一化 advantage，导致：
   - 太简单或太难的问题（标准差小）获得更大权重
   - 引入难度偏差

3. **开源实现的普遍问题**：论文发现几乎所有开源 PPO 实现（trl, OpenRLHF, verl 等）都存在长度归一化偏差，与 PPO 的理论公式不一致。

> [!warning] 长度偏差的后果
> GRPO 倾向于让**错误响应变长**，因为长响应的每个 token 惩罚更小。这导致：
> - Token 效率下降（生成更多无用 token）
> - 可能误导对"长 CoT 涌现"的理解
> - 训练不稳定

---

## 核心机制

### GRPO 的偏差来源

> [!math] GRPO 的目标函数

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim p_{\mathcal{Q}}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t} \hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right) \right]
$$

其中：
- $r_{i,t} = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}$ 是重要性权重
- $\hat{A}_{i,t} = \frac{R(q, o_i) - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$ 是归一化的 advantage

**两个偏差项**：
1. $\frac{1}{|o_i|}$ — 响应级长度归一化
2. $\frac{1}{\text{std}(\mathbf{R})}$ — 问题级标准差归一化

> [!intuition] 为什么长度归一化有偏？
>
> 考虑两个错误响应（$\hat{A} < 0$）：
> - 短响应（10 tokens）：每个 token 的梯度权重 = $\frac{1}{10} \times |\hat{A}|$
> - 长响应（100 tokens）：每个 token 的梯度权重 = $\frac{1}{100} \times |\hat{A}|$
>
> 长响应的每个 token 惩罚更小，模型学会"用长度稀释惩罚"。

### Dr. GRPO 的修正

> [!definition] Dr. GRPO 的核心改动
>
> 1. **移除长度归一化**：用常数（如最大生成长度 `MAX_TOKENS`）替代 $|o_i|$
> 2. **移除标准差归一化**：只做中心化，不除以 $\text{std}(\mathbf{R})$

**修正后的 advantage**：

$$
\tilde{A}_{i,t} = R(q, o_i) - \text{mean}(\mathbf{R})
$$

**修正后的目标函数**：

$$
\mathcal{J}_{Dr.GRPO}(\theta) = \mathbb{E}_{q \sim p_{\mathcal{Q}}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{\text{MAX\_TOKENS}} \sum_{t=1}^{|o_i|} \min \left( r_{i,t} \tilde{A}_{i,t}, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \tilde{A}_{i,t} \right) \right]
$$

> [!math] 理论基础
>
> Dr. GRPO 恢复了 [[PPO]] 的理论目标函数（Eq. 2），使用 Monte Carlo return 和无偏 baseline 估计 advantage。

### 实现细节

> [!example] 代码修改

```python
def masked_mean(tensor, mask, dim):
    # ❌ GRPO (biased): 除以实际响应长度
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)

    # ✅ Dr. GRPO (unbiased): 除以常数
    return (tensor * mask).sum(axis=-1) / MAX_TOKENS
```

**关键点**：
- `MAX_TOKENS` 是全局常数（如 2048），在整个训练过程中固定
- 也可以用其他常数，只影响梯度范数，不影响优化方向

---

## 实验结果

### Token 效率提升

> [!example] Dr. GRPO vs GRPO

论文在 Qwen2.5-1.5B 上对比了两种方法：

| 指标 | GRPO | Dr. GRPO | 改进 |
|------|------|----------|------|
| **训练 reward** | 持续增长 | 持续增长 | 相似 |
| **响应长度** | 持续增长（即使 reward 饱和） | 增长后稳定 | ✅ 防止野蛮增长 |
| **错误响应长度** | 显著增长 | 显著减少 | ✅ 缓解 overthinking |
| **Benchmark 准确率** | 基线 | 相当或更好 | ✅ 保持性能 |

**关键发现**：
- GRPO 在 reward 饱和后仍继续增加响应长度（尤其是错误响应）
- Dr. GRPO 有效防止这种"长度爆炸"，提升 token 效率

### 最小化 R1-Zero 配方

论文使用 Dr. GRPO 在 Qwen2.5-Math-7B 上实现了 SOTA：

| Benchmark | 准确率 | 训练成本 |
|-----------|--------|----------|
| AIME 2024 | 43.3% | 27 小时 (8×A100) |

**配方**：
- 基础模型：Qwen2.5-Math-7B
- 算法：Dr. GRPO
- 数据：MATH level 3-5 问题
- 模板：Qwen-Math template

---

## 与其他方法的关系

> [!comparison] Dr. GRPO vs GRPO vs PPO

| 方面 | PPO | GRPO | Dr. GRPO |
|------|-----|------|----------|
| **Advantage 估计** | Value function + [[GAE]] | Group relative | Group relative (无偏) |
| **长度归一化** | 无 | 有（引入偏差） | 无（修正） |
| **标准差归一化** | 无 | 有（引入偏差） | 无（修正） |
| **Token 效率** | 基线 | 差（错误响应变长） | 好（防止变长） |
| **内存开销** | 高（需要 Critic） | 低 | 低 |
| **理论一致性** | ✅ | ❌ | ✅ |

> [!intuition] 核心洞察
>
> - **GRPO** 的初衷是简化 PPO（省去 value function），但实现中引入了偏差
> - **Dr. GRPO** 修正偏差，真正恢复 PPO 的理论目标
> - 开源实现的普遍问题：从 pretraining 的长度归一化习惯延续到 RL，但 RL 中响应长度不是常数

Dr. GRPO 本质上是 [[RLHF]] 的一种实现方式，与 [[GSPO]] 类似都是对 GRPO 的改进——GSPO 解决序列级重要性采样问题，而 Dr. GRPO 解决长度和标准差归一化偏差。

---

## 其他发现

论文还分析了 R1-Zero 训练的其他方面：

### 1. Base Model 的影响

> [!warning] Qwen2.5 的特殊性
>
> - Qwen2.5-Math 模型在**不使用任何模板**时表现最好（~60% 提升）
> - 论文假设：Qwen2.5 可能在预训练时使用了拼接的问答对
> - 这意味着 Qwen2.5 base 模型已经类似 SFT，需谨慎解读"纯 RL"的效果

### 2. "Aha Moment" 的真相

> [!warning] 自我反思的涌现？
>
> - 论文发现几乎所有 base 模型（包括 DeepSeek-V3-Base）都已经展现自我反思行为
> - "Aha moment" 可能不是 RL 训练的涌现，而是 base 模型已有的能力
> - 自我反思行为与准确率**不正相关**

### 3. 模板与数据集的交互

> [!intuition] 模板-数据集二重奏
>
> - 当模板与 base 模型**匹配**时（如 Qwen-Math template + Qwen2.5），即使在简单的 out-of-domain 数据（GSM8K）上训练也能达到最佳性能
> - 当模板与 base 模型**不匹配**时（如 R1 template + Qwen2.5），需要更大的数据集覆盖
> - 这说明：RL 主要是**强化已有的推理行为**，而非注入新知识

---

## 局限性

> [!warning] Dr. GRPO 的局限

1. **仍需 Reward Model**：与 GRPO 一样，仍需要训练 [[Reward Model]]
2. **采样开销**：每个问题需要采样多个输出（如 G=64）
3. **未解决所有问题**：
   - 仍可能出现训练不稳定（虽然比 GRPO 好）
   - 在大规模 MoE 模型上的表现未充分验证
4. **开源实现需更新**：几乎所有现有 PPO/GRPO 实现都需要修正长度偏差

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: Dr. GRPO 相比 GRPO 改了什么？**
> A: 移除了两个归一化项：(1) 响应级长度归一化 $\frac{1}{|o_i|}$，(2) 问题级标准差归一化 $\frac{1}{\text{std}(\mathbf{R})}$。用常数替代长度，只做中心化不做标准化。
>
> **Q2: 为什么 GRPO 的长度归一化有问题？**
> A: 对于错误响应（负 advantage），长度归一化导致更长的响应每个 token 惩罚更小，模型学会"用长度稀释惩罚"，导致错误响应越来越长。
>
> **Q3: Dr. GRPO 如何提升 token 效率？**
> A: 通过移除长度偏差，防止模型生成越来越长的错误响应，从而减少无用 token 的生成。
>
> **Q4: Dr. GRPO 的理论基础是什么？**
> A: 恢复 PPO 的理论目标函数，使用 Monte Carlo return 和无偏 baseline 估计 advantage，与 PPO 的数学形式一致。
>
> **Q5: 论文对"长 CoT 涌现"的解释是什么？**
> A: 响应长度增长可能部分是优化偏差导致的，而非纯粹的推理能力涌现。Dr. GRPO 能防止这种偏差驱动的长度增长。

---

## 参考资料

- [[250320783v2|Understanding R1-Zero (2025)]] — Dr. GRPO 的原始论文
- [[240203300v3|DeepSeekMath (2024)]] — GRPO 的原始论文
