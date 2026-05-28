---
description: 用序列级重要性采样替代 GRPO 的 token 级采样，从根本上解决大规模 MoE 模型 RL 训练的稳定性问题
type: method
aliases:
  - Group Sequence Policy Optimization
  - 组序列策略优化
prerequisites:
  - "[[GRPO]]"
  - "[[PPO]]"
tags:
  - post-training
  - reinforcement-learning
  - rlhf
created: 2026-01-27
updated: 2026-02-01T00:54
---

# GSPO (Group Sequence Policy Optimization)

组序列策略优化（GSPO, Group Sequence Policy Optimization）是 [[GRPO]] 的改进版本，通过**将重要性采样从 token 级提升到序列级**，从根本上解决了大规模 MoE 模型 [[RLHF|RL 训练]]的稳定性问题。GSPO 是 Qwen3 模型 RL 训练的核心算法。

> [!paper] 论文出处
> Zheng et al., "Group Sequence Policy Optimization", 2025 (Qwen Team, Alibaba)
> - 解决了 GRPO 在大规模 MoE 模型上的训练崩溃问题
> - 无需 Routing Replay 等复杂稳定化策略
> - 成功应用于 Qwen3 系列模型的 RL 训练

---

## 动机

> [!intuition] 为什么需要 GSPO？

[[GRPO]] 在大规模 RL 训练中存在严重的稳定性问题，尤其是训练大型 MoE 模型时，经常出现**灾难性且不可逆的模型崩溃**。

### GRPO 的根本问题

GRPO 的 token 级重要性采样存在**理论缺陷**：

1. **[[Importance Sampling|重要性采样]]的基本原理**：
$$\mathbb{E}_{z \sim \pi_{\text{tar}}} [f(z)] = \mathbb{E}_{z \sim \pi_{\text{beh}}} \left[ \frac{\pi_{\text{tar}}(z)}{\pi_{\text{beh}}(z)} f(z) \right]$$

   这要求从行为分布 $\pi_{\text{beh}}$ 采样**多个样本**（$N \gg 1$）来校正分布差异。

2. **GRPO 的误用**：GRPO 在每个 token 位置 $t$ 应用重要性权重 $\frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})}$，但这个权重基于**单个样本** $y_{i,t}$，无法执行预期的分布校正。

3. **后果**：引入高方差噪声，随序列长度累积，被 clipping 机制放大，最终导致模型崩溃。

> [!warning] 核心洞察
> **优化目标的单位应该与 reward 的单位匹配**。既然 reward 是给整个序列的，在 token 级做 off-policy 校正是有问题的。

### MoE 模型的特殊挑战

MoE 模型的**专家激活波动性**（expert-activation volatility）使问题更加严重：

- 每次梯度更新后，同一 response 激活的专家可能显著变化
- 在 48 层的 Qwen3-30B-A3B 模型中，每次更新约有 **10% 的专家**发生变化
- 这导致 token 级重要性比率剧烈波动，进一步失效

---

## 核心机制

### 序列级重要性比率

> [!definition] GSPO 的核心创新
> 将重要性比率从 token 级提升到**序列级**，使其与序列级的 reward 和优化目标对齐。

**序列级重要性比率**（带长度归一化）：

$$s_i(\theta) = \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \right)^{\frac{1}{|y_i|}} = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})} \right)$$

> [!intuition] 为什么要长度归一化？
> - 减少方差
> - 将 $s_i(\theta)$ 控制在统一的数值范围内
> - 否则，少数 token 的 likelihood 变化会导致序列级比率剧烈波动
> - 不同长度的 response 需要不同的 clipping 范围

### GSPO 目标函数

> [!math] 数学形式

$$\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( s_i(\theta) \hat{A}_i, \text{clip}(s_i(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_i \right) \right]$$

这个目标函数继承了 [[PPO]] 的 clipping 机制，但将其应用于序列级重要性比率而非 token 级。

其中 advantage 估计与 GRPO 相同：

$$\hat{A}_i = \frac{r(x, y_i) - \text{mean}(\{r(x, y_i)\}_{i=1}^G)}{\text{std}(\{r(x, y_i)\}_{i=1}^G)}$$

> [!comparison] GSPO vs GRPO
>
> | 方面 | GRPO | GSPO |
> |------|------|------|
> | **重要性比率** | Token 级 $w_{i,t}(\theta)$ | 序列级 $s_i(\theta)$ |
> | **Clipping 单位** | 每个 token | 整个 response |
> | **梯度权重** | 各 token 权重不等 | 所有 token 权重相等 |
> | **MoE 稳定性** | 需要 Routing Replay | 原生稳定 |
> | **Clipping 范围** | ~0.2 | ~3e-4 到 4e-4 |

### 梯度分析

> [!math] 梯度对比

**GSPO 梯度**（省略 clipping）：

$$\nabla_\theta \mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^G \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \right)^{\frac{1}{|y_i|}} \hat{A}_i \cdot \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \nabla_\theta \log \pi_\theta(y_{i,t}|x, y_{i,<t}) \right]$$

**GRPO 梯度**：

$$\nabla_\theta \mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^G \hat{A}_i \cdot \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})} \nabla_\theta \log \pi_\theta(y_{i,t}|x, y_{i,<t}) \right]$$

> [!intuition] 关键区别
> - **GRPO**：各 token 的梯度被其各自的重要性权重加权，权重可在 $(0, 1+\varepsilon]$ 或 $[1-\varepsilon, +\infty)$ 之间变化，影响累积导致不可预测的后果
> - **GSPO**：所有 token 的梯度权重相等，消除了这一不稳定因素

---

## GSPO-token 变体

对于需要更细粒度 advantage 调整的场景（如多轮 RL），GSPO 提供了 token 级变体：

$$\mathcal{J}_{\text{GSPO-token}}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left( s_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(s_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t} \right) \right]$$

其中：

$$s_{i,t}(\theta) = \text{sg}[s_i(\theta)] \cdot \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\text{sg}[\pi_\theta(y_{i,t}|x, y_{i,<t})]}$$

$\text{sg}[\cdot]$ 表示 stop gradient（PyTorch 中的 detach）。

> [!intuition] 设计巧思
> - $s_{i,t}(\theta)$ 的数值等于 $s_i(\theta)$（因为 $\frac{\pi_\theta}{\text{sg}[\pi_\theta]} = 1$）
> - 当所有 token 的 advantage 相同时，GSPO-token 与 GSPO 完全等价
> - 但 GSPO-token 允许逐 token 调整 advantage

---

## 实验结果

### 训练效率

在 Qwen3-30B-A3B-Base 的 cold-start 模型上：

- **GSPO 训练全程稳定**
- **训练效率显著优于 GRPO**：相同计算量和 query 数下，GSPO 达到更高的训练准确率和 benchmark 性能
- **持续提升**：通过增加训练计算、更新 query 集、延长生成长度，GSPO 能持续带来性能提升

### 反直觉的 Clipping 发现

> [!warning] 有趣的观察
> GSPO 的 clipped token 比例比 GRPO **高两个数量级**，但训练效率反而更高。

这说明 GRPO 的 token 级梯度估计本身就是**噪声大、样本利用效率低**的。GSPO 的序列级方法提供了更可靠、更有效的学习信号。

---

## 对 MoE 训练的优势

### 问题回顾

GRPO 训练 MoE 模型时，专家激活波动导致 token 级重要性比率失效，需要 **Routing Replay** 策略：
- 缓存 $\pi_{\theta_{old}}$ 中激活的专家
- 在计算 $\pi_\theta$ 时"重放"这些路由模式
- 引入额外的内存和通信开销
- 限制了 MoE 模型的实际容量

### GSPO 的解决方案

> [!intuition] 为什么 GSPO 原生稳定？
> GSPO 只关注**序列 likelihood**（$\pi_\theta(y_i|x)$），对单个 token likelihood 不敏感。由于 MoE 模型始终保持语言建模能力，序列 likelihood 不会剧烈波动。

**结果**：GSPO 从根本上解决了 MoE 的专家激活波动问题，无需 Routing Replay，训练过程更简单、更稳定，模型可以充分利用其全部容量。

---

## 对 RL 基础设施的简化

由于训练引擎（如 Megatron）和推理引擎（如 SGLang、vLLM）之间存在精度差异，通常需要用训练引擎重新计算 $\pi_{\theta_{old}}$ 下的 likelihood。

> [!intuition] GSPO 的优势
> GSPO 只使用**序列级** likelihood，对精度差异的容忍度更高。因此可以直接使用推理引擎返回的 likelihood，避免重新计算。

这在以下场景特别有益：
- Partial rollout
- 多轮 RL
- 训练-推理分离架构

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: GSPO 相比 GRPO 的核心改进是什么？**
> A: 将重要性采样从 token 级提升到序列级。GRPO 的 token 级重要性权重基于单样本，无法执行有效的分布校正，引入高方差噪声。GSPO 使用序列级重要性比率，与序列级 reward 对齐。
>
> **Q2: GSPO 的重要性比率怎么计算？**
> A: $s_i(\theta) = \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \right)^{\frac{1}{|y_i|}}$，即序列 likelihood 比率的长度归一化。
>
> **Q3: 为什么 GSPO 能解决 MoE 训练的稳定性问题？**
> A: MoE 模型的专家激活在梯度更新后会变化，导致 token 级重要性比率剧烈波动。GSPO 只关注序列级 likelihood，对单 token 变化不敏感，因此原生稳定。
>
> **Q4: GSPO 的 clipping 范围为什么比 GRPO 小很多？**
> A: 因为重要性比率的定义不同。GSPO 使用长度归一化的序列级比率，数值范围更紧凑，所以 clipping 范围（~3e-4）比 GRPO（~0.2）小两个数量级。
>
> **Q5: GSPO 和 GRPO 的梯度有什么本质区别？**
> A: GRPO 中各 token 的梯度被其各自的重要性权重加权，权重不等且可能累积；GSPO 中所有 token 的梯度权重相等，消除了这一不稳定因素。

---

## 相关概念

- [[Reward Model]] — 提供 reward 信号
- [[SAPO]] — 用平滑软门替代 GSPO 的硬裁剪，进一步提升稳定性

---

## 参考资料

- [[250718071v2|GSPO Paper (2025)]] — GSPO 的原始论文
- [[GRPO]] — GRPO 的详细介绍
- [[240203300v3|DeepSeekMath (2024)]] — GRPO 的原始论文
