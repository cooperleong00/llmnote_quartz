---
type: method
description: 用 global batch normalization 替代 local group normalization 的 critic-free 策略优化算法，解决 GRPO/RLOO 的 advantage 爆炸和任务过拟合问题
aliases:
  - REINFORCE Plus Plus
prerequisites:
  - "[[REINFORCE]]"
  - "[[PPO]]"
  - "[[GRPO]]"
tags:
  - post-training
  - reinforcement-learning
  - rlhf
  - variance-reduction
created: 2026-02-26
updated: 2026-02-26
---

# REINFORCE++

REINFORCE++ 是一种 critic-free 的策略优化算法，核心创新是用 **global batch normalization** 替代 [[GRPO]]/[[RLOO]] 中的 prompt-level (local) normalization 来计算 advantage。这个看似简单的改动解决了 local normalization 的理论偏差和实际不稳定性，使得 $k=1$（每个 prompt 只采样一个输出）就能稳定训练，同时保持更好的泛化能力。

> [!paper] 论文出处
> Jian Hu et al., "REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Normalization", arXiv: 2501.03262, 2025.
> [[Clippings/Paper/250103262v9/250103262v9|原文]]

---

## 动机：Local Normalization 的三个问题

[[GRPO]] 和 [[RLOO]] 等 critic-free 方法的共同做法是：对每个 prompt 采样 $k$ 个输出，用组内 reward 的均值和标准差来归一化 advantage。这种 **prompt-level (local) normalization** 存在三个根本问题：

### 1. 理论偏差：分子分母不独立

Local normalization 的公式是：

$$
\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}
$$

这里 $\bar{r}$ 和 $\sigma_r$ 都是从同一组样本计算的。问题在于分子 $r_i - \bar{r}$（centered reward）和分母 $\sigma_r$（local std）**不是独立的**——它们来自同一组样本。这导致归一化后的 advantage 是一个**有偏估计**。

### 2. 实际不稳定：小组 + 相似 reward = 爆炸

当 group size $k$ 很小（如 4 或 8）时，如果同一 prompt 的所有输出恰好获得相似的 reward，local std $\sigma_r \to 0$，归一化后的 advantage 会**爆炸到极大值**。这在实践中并不罕见——简单问题的所有回答可能都是对的，难问题的所有回答可能都是错的。

> [!intuition] 直觉理解
> 想象一个班级内部排名的场景：如果全班都考了 95-98 分，那么考 98 分的人在班内排名"遥遥领先"，但放到全校来看只是正常水平。Local normalization 就是这种"班内排名"，它会放大微小差异，制造虚假的信号强度。

### 3. 任务过拟合：优化"局部胜出"而非"全局好"

Local normalization 让策略被优化为"在同一 prompt 的 $k$ 个输出中胜出"，而非"在全局意义上产生好的输出"。这导致：
- 策略学会利用 local group 的统计特性，而非学习通用能力
- 在小数据集上尤其严重——[[GRPO]] 在 30 题 AIME-24 上训练后，OOD 泛化完全失败（Pass@1 降到 0.0）

---

## 核心机制：Global Batch Normalization

REINFORCE++ 的解决方案直截了当：**把 normalization 的统计量从 prompt-level 提升到 batch-level**。

> [!comparison] Local vs Global Normalization
>
> | | Local (GRPO) | Global (REINFORCE++) |
> |---|---|---|
> | **统计量来源** | 同一 prompt 的 $k$ 个输出 | 整个 batch 的所有输出 |
> | **最小采样** | $k \geq 2$（需要组内对比） | $k \geq 1$（单样本即可） |
> | **偏差** | 有偏（分子分母不独立） | 渐近无偏（batch 足够大） |
> | **稳定性** | $k$ 小时 std 可能趋零 | batch 大，std 稳定 |
> | **优化目标** | 在 local group 中胜出 | 在全局分布中表现好 |

REINFORCE++ 提供两个变体，适用于不同场景。

### 变体一：REINFORCE++（$k \geq 1$）

适用于 general-domain RLHF，$k=1$ 时效率最高（每个 prompt 只需一个输出）。

**Advantage 计算**：

$$
A_{q,o_t} = r(o_{1:T}, q) - \beta \cdot \sum_{i=t}^{T} \text{KL}(i)
$$

其中 KL penalty 直接嵌入 advantage（类似 [[PPO]] 在 reward 中加 KL 的做法），而非像 GRPO 那样放在 loss 中。

**Global normalization**：

$$
A_{q,o_t}^{\text{norm}} = \frac{A_{q,o_t} - \text{mean}(A \mid A \in \mathcal{D}_{\text{batch}})}{\text{std}(A \mid A \in \mathcal{D}_{\text{batch}}) + \epsilon}
$$

统计量在整个 batch 上计算，batch 通常包含数百到数千个样本，std 估计非常稳定。最终使用 [[PPO]] 的 clip objective 优化。

### 变体二：REINFORCE++ w/ Baseline（$k > 1$）

适用于 complex reasoning 和 agentic tasks（如 multi-turn tool-calling），需要更强的 credit assignment。

**两步 normalization**：

**Step 1 — Group mean subtraction（reward reshaping）**：

$$
A'_{q,o_t} = R_{q,o_t} - \text{mean}_{\text{group}}(R_{q,o_t})
$$

先减去组内均值，这一步的作用是 **reward reshaping**——消除不同 prompt 难度差异带来的 reward 偏移。注意这里只减均值，不除以 std，避免了 local normalization 的偏差问题。

**Step 2 — Global batch normalization**：

$$
A_{q,o_t}^{\text{norm}} = \frac{A'_{q,o_t} - \text{mean}_{\text{batch}}(A')}{\text{std}_{\text{batch}}(A') + \epsilon}
$$

再在整个 batch 上做标准化，确保 advantage 的尺度一致。

> [!intuition] 两步 normalization 的直觉
> Step 1 解决的是"不同题目难度不同"的问题——难题全错、简单题全对，减去组内均值后都变成"相对于这道题的表现"。Step 2 解决的是"不同题目的 reward 方差不同"的问题——统一到全局尺度。
>
> 关键区别：GRPO 把这两步合成一步（除以 local std），引入了偏差；REINFORCE++ 把它们分开，Step 1 只做 centering（无偏），Step 2 用 global std（稳定）。

**KL 正则化**：

w/ Baseline 变体使用 k2 KL estimator，而非 GRPO 的 k3 estimator：

$$
\mathcal{J}_{k_2}(\theta) = \mathbb{E}\left[\frac{1}{2}\left(\log \frac{\pi_\theta}{\pi_{ref}}\right)^2\right]
$$

k2 estimator 的梯度更平滑，避免了 k3 在 $\pi_\theta / \pi_{ref}$ 偏离 1 时的梯度爆炸。最终目标函数：

$$
\mathcal{L} = \mathcal{L}_{PPO}(A^{\text{norm}}) - \lambda \cdot \mathcal{J}_{k_2}(\theta)
$$

---

## 与 PPO 的关系

REINFORCE++ w/ Baseline 可以理解为一个**简化的 PPO**：

1. **去掉 critic**：不需要 value function，用 group mean 作为 baseline
2. **[[GAE]] 参数极端化**：$\lambda = 1, \gamma = 1$，等价于 Monte Carlo return
3. **用 two-step global normalization 替代 learned value function**

这个视角揭示了一个重要洞察：在 LLM 对齐场景中，精确的 value function 可能不是必需的——一个好的 normalization 策略就足够了。

---

## 适用场景选择

> [!example] 实践指南
>
> | 场景 | 推荐变体 | 原因 |
> |------|----------|------|
> | General RLHF / PRM / online sampling | REINFORCE++（$k=1$） | 最高效，无需多次采样 |
> | Complex reasoning（数学、代码） | REINFORCE++ w/ Baseline | 需要 group baseline 做 credit assignment |
> | Multi-turn tool-calling / agentic | REINFORCE++ w/ Baseline | void samples 多，需要 baseline 稳定训练 |
> | RLVR（rule-based reward） | 两者均可，推荐 symmetric reward (-1/1) | -1/1 比 0/1 提供更强的负信号 |

---

## 实验结果

### General RLHF

$k=1$ 的 REINFORCE++ 与 $k=4$ 的 GRPO 性能持平，但 KL divergence 更低更稳定。更低的 KL 意味着策略没有过度偏离 reference model，降低了 reward hacking 的风险。

### Reasoning（小数据集）

这是 REINFORCE++ 优势最明显的场景。GRPO 在仅 30 题的 AIME-24 上训练后，OOD 泛化完全崩溃（Pass@1 = 0.0），而 REINFORCE++ 保持了泛化能力。这直接验证了 local normalization 导致任务过拟合的论断。

### Multi-step RL（tool-use）

REINFORCE++ w/ Baseline 在 AIME 24/25、HMMT 等 benchmark 上超越 GRPO 和 PPO，尤其在需要多步推理和工具调用的场景中优势显著。

### 第三方验证

LitePPO、ScaleRL、DLER 等独立工作验证了 global normalization 的优势，说明这不是特定实验设置下的偶然结果。

---

## 局限性

> [!warning] 需要注意的边界

1. **依赖 batch 大小**：global normalization 的质量取决于 batch 中样本的多样性。如果 batch 太小或 prompt 分布偏斜，global 统计量也可能不稳定
2. **w/ Baseline 仍需多次采样**：虽然基础版 $k=1$ 即可，但 w/ Baseline 版本仍需 $k > 1$，采样开销与 GRPO 相当
3. **KL 嵌入 advantage 的权衡**：基础版将 KL penalty 嵌入 advantage 而非 loss，这意味着 KL 的影响会被 normalization 缩放，可能需要仔细调节 $\beta$

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: REINFORCE++ 和 GRPO 的核心区别是什么？**
> A: Normalization 的范围不同。GRPO 用 prompt-level（local）normalization，在同一 prompt 的 $k$ 个输出内归一化；REINFORCE++ 用 batch-level（global）normalization，在整个 batch 上归一化。Global normalization 解决了 local 的三个问题：有偏估计、小组 std 趋零导致 advantage 爆炸、以及优化"局部胜出"导致的泛化差。
>
> **Q2: 为什么 global normalization 能用 $k=1$？**
> A: Local normalization 需要 $k \geq 2$ 才能计算组内统计量。Global normalization 的统计量来自整个 batch，即使每个 prompt 只有一个输出，batch 中仍有足够样本计算稳定的均值和标准差。
>
> **Q3: REINFORCE++ w/ Baseline 的两步 normalization 为什么比 GRPO 的一步更好？**
> A: GRPO 同时用 local mean 和 local std 归一化，分子分母不独立导致有偏。REINFORCE++ 把 centering（减组内均值）和 scaling（除以 global std）分开：Step 1 只做 centering 消除题目难度差异，Step 2 用 global std 统一尺度。分开处理避免了偏差，且 global std 比 local std 稳定得多。

---

## 延伸阅读

**后续发展与第三方验证**：
- [[VAPO (2025)|VAPO]] — 结合 value model 和 advantage filtering 的进一步改进
- [[VC-PPO]] — 另一种 critic-free PPO 变体

**原始论文**：
- [[Clippings/Paper/250103262v9/250103262v9|REINFORCE++ 原文]] — 完整实验和消融分析
