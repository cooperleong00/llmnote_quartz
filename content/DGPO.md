---
type: method
description: 用 probability 梯度替代 log-probability 梯度，通过双边解耦衰减机制解决 soft clipping 在边界处的梯度发散问题，兼顾探索与稳定性
aliases:
  - Decoupled Gradient Policy Optimization
  - 解耦梯度策略优化
prerequisites:
  - "[[GRPO]]"
  - "[[CISPO]]"
  - "[[Importance Sampling]]"
tags:
  - post-training
  - reinforcement-learning
  - rlhf
  - optimization
created: 2026-04-05
updated: 2026-04-05
---

# DGPO (Decoupled Gradient Policy Optimization)

解耦梯度策略优化（DGPO, Decoupled Gradient Policy Optimization）是 [[GRPO]] 的改进算法，核心创新是**将优化原语从 log-probability 梯度 ($\nabla_\theta\log\pi_\theta$) 切换到 probability 梯度 ($\nabla_\theta\pi_\theta$)**，并通过双边解耦衰减机制解决 soft clipping 在边界处的梯度发散问题。

> [!paper] 论文出处
> Fu et al., "From $\log \pi$ to $\pi$: Taming Divergence in Soft Clipping via Bilateral Decoupled Decay of Probability Gradient Weight", 2026
> - 在 DeepSeek-R1-Distill-Qwen 系列模型上验证（1.5B/7B/14B）
> - 1.5B 模型在 AIME24/25 上较 GRPO 提升 +4.3%，较最佳基线 CE-GPPO 提升 +3.5%
> - 7B 模型较 GRPO 提升 +3.1%
> - 开源实现：github.com/VenomRose-Juri/DGPO-RL

---

## 动机

> [!intuition] 为什么需要 DGPO？

### 现有 Soft Clipping 方法的问题

[[CISPO]]、GPPO 等 "soft clipping" 方法尝试保留被 [[PPO]]/[[GRPO]] 丢弃的边界 token 梯度，但它们有一个根本缺陷：

**依赖 log-probability 梯度 ($\nabla_\theta\log\pi_\theta$) 会导致梯度权重发散**

当概率 $\pi_\theta \to 0$ 时，log-probability 梯度的权重会无限增长：

$$\text{gradient weight} \propto \frac{1}{\pi_\theta} \to \infty \quad \text{as } \pi_\theta \to 0$$

这导致**左边界（低 IS ratio）的训练崩溃**——概率越低的 token 受到越大的梯度惩罚，造成灾难性不稳定。

### 核心洞察：Probability 作为优化原语

论文提出了一个范式转换：**probability 梯度 ($\nabla_\theta\pi_\theta$) 才是更优的优化原语**。

> [!intuition] 为什么 probability 梯度更好？

1. **RL 目标的自然对齐**：RL 实际上是在最大化 expert token 的概率（推导见论文 Appendix A.1），而非对数概率
   - SFT 目标：$\nabla_\theta \log \pi_\theta$（最大化对数概率的均值）
   - RL 目标：$\nabla_\theta \pi_\theta$（最大化概率的均值）

2. **几何对称性**：概率空间是 $(0,1)$ 的有界对称区间，便于设计稳定的梯度机制
   - Log-probability 空间是 $(-\infty, 0)$ 的无界非对称区间，梯度设计复杂

---

## 核心机制

### DGPO 目标函数

$$\mathcal{J}_{\text{DGPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \mathcal{W}_{i,t}^{\text{DGPO}}(\theta) \hat{A}_i \pi_\theta(o_{i,t}|q, o_{i,<t}) \right]$$

### 双边解耦衰减权重

DGPO 根据 IS ratio $w_{i,t} = \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}$ 定义权重函数：

$$\mathcal{W}_{i,t}^{\text{DGPO}}(\theta) = \begin{cases}
C_{\text{left}} \cdot \text{sg}[\pi_\theta^n] & \text{if } w < 1 - \varepsilon_{\text{low}} \text{ (LN: 低 ratio, 负 advantage)} \\
C_{\text{right}} \cdot \text{sg}[\pi_\theta^{-1/m}] & \text{if } w > 1 + \varepsilon_{\text{high}} \text{ (HP: 高 ratio, 正 advantage)} \\
\frac{1}{\pi_{\theta_{\text{old}}}} & \text{otherwise (边界内)}
\end{cases}$$

其中：
- $n, m \in \mathbb{Z}^+$ 是控制衰减率的超参数（默认 $n=1, m=2$）
- $\text{sg}[\cdot]$ 是 stop-gradient 操作符
- $C_{\text{left}} = (1-\varepsilon_{\text{low}})^{-n} \pi_{\theta_{\text{old}}}^{-(n+1)}$ 和 $C_{\text{right}} = (1+\varepsilon_{\text{high}})^{1/m} \pi_{\theta_{\text{old}}}^{1/m-1}$ 是确保梯度连续性的常数

### 衰减机制的直觉

> [!intuition] 双边衰减设计

| 边界 | 场景 | 衰减形式 | 效果 |
|------|------|----------|------|
| **左边界** (LN) | 低 IS ratio + 负 advantage | $\pi_\theta^n$（多项式衰减） | 概率越小，梯度衰减越强，防止发散 |
| **右边界** (HP) | 高 IS ratio + 正 advantage | $\pi_\theta^{-1/m}$（倒数根式衰减） | 概率越大，梯度衰减越强，控制探索 |
| **边界内** | 正常更新 | 常数权重 | 标准策略梯度 |

> [!intuition] "Slow Down" vs "Slow Down Gently"
- **左边界**：强衰减（Slow Down）确保稳定性
- **右边界**：温和衰减（Slow Down Gently）保留探索能力

---

## 理论优势

### 梯度连续性

DGPO 通过精心设计的常数 $C_{\text{left}}$ 和 $C_{\text{right}}$ 确保权重函数在边界处的连续性，避免梯度突变导致的训练不稳定。

### 最小偏差估计

| 偏差条件 | DGPO 相对优势 |
|----------|---------------|
| 边界内偏差 | 0（与 GRPO 相同） |
| 左边界偏差 | 最小化（$n=1$ 时优于 CISPO/GPPO） |
| 右边界偏差 | 最小化且收敛 |
| 反向左边界 | 0 |
| 反向右边界 | 0 |

> [!warning] CISPO/GPPO 的理论缺陷
> CISPO 和 GPPO 在 $n>1$ 时理论偏差更小，但它们的梯度权重在左边界发散，导致训练崩溃。DGPO 在 $n=1$ 时就实现了最小偏差，同时保证梯度收敛。

### 熵动态平衡

通过调整 $n$ 和 $m$ 控制熵动态：
- **增大 $n$**：加强左边界衰减，减缓熵下降（减少 exploration）
- **增大 $m$**：减弱右边界衰减，加快熵上升（增加 exploration）

默认配置 $n=1, m=2$ 是一个鲁棒的保守选择。

---

## 实验结果

### 主要结果

| 模型 | 方法 | Avg@32 | Pass@32 | vs GRPO |
|------|------|--------|---------|---------|
| **1.5B** | GRPO | 48.4 | 70.1 | — |
| | CISPO | 47.8 | 71.5 | -0.6 |
| | GPPO | 45.6 | 68.3 | -2.8 |
| | CE-GPPO | 49.2 | 70.7 | +0.8 |
| | ASPO | 48.9 | 70.6 | +0.5 |
| | **DGPO** | **52.7** | **73.0** | **+4.3** |
| **7B** | GRPO | 58.9 | 77.2 | — |
| | **DGPO** | **62.0** | **77.9** | **+3.1** |
| **14B** | GRPO | 53.6 | 67.4 | — |
| | **DGPO** | **56.7** | **70.4** | **+3.1** |

### 训练动态分析

> [!intuition] 为什么 DGPO 更好？

**稳定性**：
- CISPO、GPPO、CE-GPPO 在左边界梯度发散，导致训练崩溃
- DGPO 的梯度权重始终收敛

**探索能力**：
- GRPO 过早 drop entropy（ exploitation 过早）
- ASPO entropy 过高（过度 exploration）
- DGPO 保持适中且稳定的 entropy 下降

### 超参数敏感性

| 配置 | 1.5B AIME25 | 7B AIME25 | 熵稳定性 |
|------|-------------|-----------|----------|
| (1, 1) | 基准 | 基准 | 最稳定 |
| (1, 2) | + | ++ | 稳定 |
| (2, 1) | ++ | + | 较稳定 |
| (2, 2) | +++ | ~ (波动) | 7B 波动 |

**调参建议**：从 (1, 2) 开始，如果 entropy 稳定则可尝试增大 $n$ 或 $m$。

---

## 与其他方法的对比

> [!comparison] 边界梯度处理对比

| 方法 | 左边界处理 | 右边界处理 | 梯度发散？ | 主要问题 |
|------|-----------|-----------|-----------|----------|
| **GRPO** | Hard clip (0) | Hard clip (0) | 否 | 丢弃边界 token，限制探索 |
| **CISPO** | Constant log-prob weight | Constant log-prob weight | **是**（左边界）| 左边界发散导致崩溃 |
| **GPPO** | 同上 | 同上 | **是** | 同上 |
| **CE-GPPO** | Scaled constant | Scaled constant | **是** | 缓解但未解决发散 |
| **ASPO** | Reversed ratio | Reversed ratio | 否 | 大偏差 |
| **DGPO** | Polynomial decay | Reciprocal radical decay | **否** | 最优平衡 |

---

## 局限性

> [!warning] 边界条件

1. **领域特定性**：实验主要在数学推理（可验证 reward）上进行，在稀疏或主观 reward 领域（如创意写作）的效果待验证

2. **计算资源限制**：扩展性分析仅限于 14B 模型，更大模型（70B+）的效果待验证

3. **超参数选择**：虽然 (1, 2) 是鲁棒默认值，但最优配置随模型规模变化

4. **概率 vs Log-prob 的深层影响**：probability 空间的对称性假设在极端分布下可能不成立

---

## 面试视角

> [!interview] 常见问题

**Q1: DGPO 相比 GRPO/CISPO 的核心改进是什么？**
A: 将优化原语从 log-probability 梯度 ($\nabla\log\pi$) 切换到 probability 梯度 ($\nabla\pi$)。这解决了 CISPO/GPPO 在左边界（低概率 token）的梯度发散问题，同时通过双边解耦衰减机制兼顾探索与稳定性。

**Q2: 为什么 log-probability 梯度会导致发散？**
A: 因为 log-probability 梯度的权重与 $1/\pi_\theta$ 成正比。当概率趋近于 0 时，权重无限增长，导致低概率 token 受到不成比例的梯度惩罚，训练崩溃。

**Q3: DGPO 的双边衰减机制如何工作？**
A: 左边界（低 IS ratio + 负 advantage）：使用 $\pi_\theta^n$ 多项式衰减，概率越小衰减越强；右边界（高 IS ratio + 正 advantage）：使用 $\pi_\theta^{-1/m}$ 倒数根式衰减，概率越大衰减越强。这确保了梯度在所有边界条件下都收敛。

**Q4: DGPO 为什么用 probability 梯度而不是 log-probability？**
A: 两个原因：(1) RL 目标实际上是最大化概率而非对数概率，SFT 目标才是对数概率；(2) 概率空间 $(0,1)$ 有界对称，便于设计稳定的梯度机制；log-probability 空间 $(-\infty,0)$ 无界非对称。

**Q5: DGPO 的超参数 $n$ 和 $m$ 如何影响训练？**
A: $n$ 控制左边界衰减强度（越大越保守），$m$ 控制右边界衰减强度（越大越开放）。推荐默认值 $n=1, m=2$，这是一个兼顾性能和稳定性的保守配置。

---

## 相关概念

- [[GRPO]] — DGPO 的基础算法
- [[CISPO]] — Soft clipping 先驱，但有发散问题
- [[PPO]] — Hard clipping 机制
- [[Importance Sampling]] — IS ratio 的理论基础
- [[Entropy Collapse]] — DGPO 试图解决的核心问题
- [[Soft Clipping]] — 相关方法的统一框架

## 延伸阅读

- [[2603.14389|Fu et al., 2026]] — DGPO 原始论文
- [[MiniMax-M1 (2025)]] — CISPO 的来源
- [[DeepSeekMath (2024)]] — GRPO 的来源
- [[DAPO]] — 其他解决 entropy collapse 的方法
