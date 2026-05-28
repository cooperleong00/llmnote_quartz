---
description: 通过双边校准和 masking 机制解决 MoE 模型 RL 训练中的训练-推理概率不一致问题
type: method
aliases:
  - IcePop Calibration
prerequisites:
  - "[[GRPO]]"
  - "[[Mixture of Experts]]"
tags:
  - post-training
  - reinforcement-learning
  - moe
  - optimization
created: 2026-01-28
updated: 2026-02-01T01:33
---

# IcePop

IcePop 是 [[GRPO]] 的变体，专门解决 [[Mixture of Experts|MoE]] 模型 RL 训练中的 **training-inference mismatch** 问题。核心思想是通过**双边校准（double-sided calibration）**  和 **masking** 机制，过滤掉概率偏差过大的 token 的梯度更新，从而稳定训练过程。

> [!paper] 论文出处
> Ring-1T Team, "Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model", 2025
> - 在 Ring-mini-2.0 上，IcePop 在 AIME25 上比 TIS 高 6%
> - 在 Ring-1T 上显著降低了梯度范数和概率差异的波动

---

## 动机

> [!intuition] 为什么需要 IcePop？

在大规模 RL 训练中，通常使用**分离的引擎**：推理引擎（如 SGLang）生成样本，训练引擎（如 Megatron）更新策略。这种分离在 MoE 模型中会导致严重的 **training-inference mismatch**：

1. **路由不一致**：MoE 的路由器对输入敏感，训练和推理引擎的数值精度差异会导致不同的专家选择
2. **概率偏差**：同一 token 在训练和推理时的概率可能显著不同
3. **梯度噪声**：概率偏差会引入高方差梯度，导致训练不稳定

> [!comparison] 与 GSPO 的互补关系
> - [[GSPO]]：解决 **专家激活波动** 问题（同一模型内部的不稳定性）
> - **IcePop**：解决 **训练-推理引擎差异** 问题（不同引擎间的不一致性）
>
> 两者解决的是 MoE 训练稳定性的不同层面，可以结合使用。

---

## 核心机制

### 核心思想

IcePop 的关键洞察：**与其试图消除训练-推理差异，不如直接过滤掉差异过大的 token**。

定义概率比率：
$$k = \frac{\pi_{\text{train}}(y_{i,t} \mid x, y_{i, < t}; \theta_{\text{old}})}{\pi_{\text{infer}}(y_{i,t} \mid x, y_{i, < t}; \theta_{\text{old}})}$$

当 $k$ 偏离 1 过多时，说明该 token 的训练-推理差异过大，其梯度不可靠。

### Masking 函数

IcePop 使用双边校准的 masking 函数：

$$\mathcal{M}(k; \alpha, \beta) = \begin{cases} k & \text{if } k \in [\alpha, \beta] \\ 0 & \text{otherwise} \end{cases}$$

- $\alpha < 1$：下界，过滤训练概率过低的 token
- $\beta > 1$：上界，过滤训练概率过高的 token
- 典型值：$\alpha = 0.5$, $\beta = 5$（Ring-1T 使用）

### 目标函数

$$\mathcal{J}_{\text{IcePop}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^{G} \sim \pi_{\text{infer}}(\cdot | x; \theta_{\text{old}})} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \mathcal{M}\left(\frac{\pi_{\text{train}}(y_{i,t} \mid x, y_{i, < t}; \theta_{\text{old}})}{\pi_{\text{infer}}(y_{i,t} \mid x, y_{i, < t}; \theta_{\text{old}})}; \alpha, \beta\right) \cdot \hat{A}_i \cdot \log \pi_\theta(y_{i,t} \mid x, y_{i,<t}) \right]$$

其中 $\hat{A}_i$ 是 GRPO 风格的 group relative advantage。

> [!intuition] 直觉理解
> IcePop 就像一个**质量过滤器**：
> - 概率比率在 $[\alpha, \beta]$ 范围内的 token：正常参与梯度更新
> - 概率比率超出范围的 token：完全丢弃其梯度贡献
>
> 这样可以确保只有"可信"的 token 参与训练，避免噪声梯度破坏模型。

---

## 与其他方法的对比

> [!comparison] MoE 训练稳定化方法对比

| 方法 | 解决的问题 | 机制 | 开销 |
|------|-----------|------|------|
| [[Rollout Routing Replay]] | 路由不一致 | 记录并重放推理时的路由决策 | 存储开销 |
| [[GSPO]] | 专家激活波动 | 序列级重要性采样 | 无额外开销 |
| **IcePop** | 训练-推理概率差异 | 双边校准 + masking | 无额外开销 |
| TIS (Token-level IS) | 概率差异 | Token 级重要性采样校正 | 无额外开销 |

> [!warning] IcePop vs TIS
> TIS（Token-level [[Importance Sampling]]）试图用重要性采样**校正**概率差异，但这在单样本情况下理论上不成立（参见 [[GSPO]] 的分析）。IcePop 采取更保守的策略：**直接丢弃**不可靠的 token，而非试图校正。

---

## 实验结果

### Ring-mini-2.0 上的效果

| 方法 | AIME25 |
|------|--------|
| TIS | baseline |
| IcePop | **+6%** |

### Ring-1T 上的效果

- 显著降低了梯度范数的波动
- 显著降低了概率差异的波动
- 训练更加稳定

---

## 超参数

> [!example] 典型配置

| 参数 | Ring-1T 配置 | 说明 |
|------|-------------|------|
| $\alpha$ | 0.5 | 下界，过滤 $\pi_{\text{train}} < 0.5 \cdot \pi_{\text{infer}}$ 的 token |
| $\beta$ | 5 | 上界，过滤 $\pi_{\text{train}} > 5 \cdot \pi_{\text{infer}}$ 的 token |

---

## 局限性

> [!warning] 边界条件

1. **信息丢失**：被 mask 掉的 token 不参与训练，可能丢失有价值的梯度信号
2. **超参数敏感**：$\alpha$, $\beta$ 的选择需要根据具体模型和训练设置调整
3. **治标不治本**：IcePop 是一种缓解策略，并未从根本上消除训练-推理差异

---

## 参考资料

- [[Clippings/Paper/251018855v2/251018855v2|Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model (2025)]]
- IcePop 博客：https://ringtech.notion.site/icepop
