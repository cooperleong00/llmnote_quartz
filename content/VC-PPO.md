---
type: method
description: 通过 Value Pretraining 和 Decoupled GAE 解决 PPO 在 long-CoT 任务中崩溃问题的方法，是 VAPO 的基础工作
aliases:
  - VC-PPO
  - Value-Calibrated PPO
  - Value-Calibrated Proximal Policy Optimization
prerequisites:
  - "[[PPO]]"
  - "[[GAE]]"
  - "[[Value Function]]"
  - "[[Reward Model]]"
tags:
  - reinforcement-learning
  - post-training
  - reasoning
created: 2026-02-25
updated: 2026-02-25T14:08
---

# VC-PPO

Value-Calibrated PPO（VC-PPO）是 ByteDance Seed 提出的方法，解决了 [[PPO]] 在 long-CoT 推理任务中崩溃的根本问题。核心发现是：PPO 崩溃不是算法本身的缺陷，而是 value model 在长序列场景下的两个系统性偏差——初始化偏差和 reward 信号衰减。VC-PPO 通过两个针对性技术（Value Pretraining + Decoupled GAE）修复了这些偏差，在 AIME 上将 baseline PPO 从 5.6 提升到 49.0。

这是 [[VAPO (2025)]] 的前身工作，VAPO 在 VC-PPO 的基础上进一步引入了 Length-Adaptive GAE 等技术。

## 动机：PPO 为什么在 Long-CoT 中崩溃

直接将 [[PPO]] 应用于 long-CoT 任务（如数学竞赛题）会出现一个典型的失败模式：训练刚开始，模型的输出长度就急剧下降，验证性能随之崩溃。由于 long-CoT 任务的性能与输出长度强相关（模型需要足够长的思维链来推理），输出长度的崩溃直接导致了性能的崩溃。

> [!intuition] 为什么输出会变短？
> 根本原因是 advantage estimation 出现了系统性偏差：越靠前的 token 被赋予了越大的正 advantage。这让模型"学到"了一个错误的信号——前面的 token 比后面的更有价值，于是倾向于尽早结束生成。

这个偏差有两个独立的来源，VC-PPO 分别诊断并解决了它们。

## 问题一：Value Initialization Bias

### 诊断

标准 RLHF 实践中，value model 通常从 [[Reward Model]] 初始化。这看起来合理——两者都预测 response 的标量评分。但它们的目标有本质区别：

- **Reward model**：只在 `<EOS>` token 上训练，对整个 response 打分
- **Value model**：需要估计每个 token 位置的期望累积回报

由于 reward model 从未在中间 token 上训练过，它对越靠前的 token（信息越不完整）倾向于给出越低的分数。当用这样的模型初始化 value model 时，前面 token 的 value 被系统性低估。

根据 [[GAE]] 的计算公式：

$$\hat{A}_t = \sum_{l=0}^{T-t-1} \lambda^l (r_{t+l} + V(s_{t+l+1}) - V(s_{t+l}))$$

当 $V(s_t)$ 被低估时，$\hat{A}_t$ 就会被正向偏置。而且这种偏差沿 trajectory 累积——越靠前的 token，累积的正偏差越大。这就解释了为什么模型会倾向于生成更短的 response。

### 解决方案：Value Pretraining

在正式 RL 训练之前，先离线预训练 value model 直到收敛：

1. **固定策略采样**：用 SFT policy $\pi_{\text{sft}}$ 持续生成 response
2. **Monte Carlo return 训练**：使用 $\lambda = 1.0$ 的 GAE（即 Monte Carlo return）更新 value model。$\lambda = 1.0$ 将优化问题转化为稳定的梯度下降，避免 bootstrapping 带来的不稳定性
3. **监控收敛指标**：训练直到 value loss 和 explained variance 达到合理水平
4. **保存 checkpoint**：将预训练好的 value model 作为后续 RL 训练的起点

> [!intuition] 为什么不直接在 RL 训练中 warm-up？
> 论文发现 Value Pretraining 的收敛过程分为两个阶段：第一阶段是快速的 range alignment（类似传统的 value warm-up），第二阶段是更慢的 knowledge injection——模型开始学习哪些 token 更有价值。第二阶段对最终性能有显著影响，但传统的 warm-up 通常只覆盖第一阶段。

> [!warning] 过拟合风险
> 消融实验显示 100 步预训练是最优的，超过 150 步后性能开始下降，可能是因为 value model 过拟合到固定策略的分布上。

## 问题二：Reward Signal Decay

### 诊断

在 RLHF 中，reward 通常只在 trajectory 末尾（`<EOS>` token）给出。根据 [[GAE]] 的计算，reward 信号传播到第 $t$ 个 token 时的强度为 $\lambda^{T-t} r_{\langle EOS \rangle}$。

当 $\lambda = 0.95$ 且序列长度 $T$ 很大时，这个衰减是灾难性的：

| 距离 $T-t$ | 衰减系数 $0.95^{T-t}$ |
|------------|----------------------|
| 10         | 0.60                 |
| 50         | 0.08                 |
| 100        | 0.006                |
| 200        | 0.00004              |

对于 long-CoT 任务（response 长度可达数千 token），前面的 token 几乎完全接收不到 reward 信号。这意味着 value model 无法学习到前面 token 的真实价值。

> [!intuition] 传统 RL 为什么用 $\lambda < 1.0$？
> 在 Mujoco、Atari 等传统 RL 环境中，reward 沿 trajectory 累积（每步都有 reward），return 的方差很高，需要通过 $\lambda < 1.0$ 来降低方差。但 RLHF 的 reward 是 trajectory-level 的、非累积的、well-defined 的值，方差本身就不高。这是一个关键的 domain difference。

### 解决方案：Decoupled GAE

核心洞察：value model 和 policy model 对 bias-variance trade-off 的偏好不同。

- **Value model**（MSE loss）：对方差更 tolerant，更需要无偏的学习目标 → 偏好 $\lambda = 1.0$
- **Policy model**（policy gradient）：对方差更敏感，需要降低方差来加速收敛 → 偏好 $\lambda < 1.0$

Decoupled GAE 的做法很简单：为 actor 和 critic 分别计算 GAE，使用不同的 $\lambda$：

$$\lambda_{\text{critic}} = 1.0, \quad \lambda_{\text{actor}} \in [0.95, 1.0)$$

> [!math] 数学正当性
> 论文证明了使用不同 $\lambda$ 训练出的 value function 不会给 policy gradient 引入额外 bias。关键在于 policy gradient 的期望不依赖于 baseline 的选择——任意 value function $\bar{V}$ 作为 baseline 都不改变梯度的期望方向：
>
> $$\mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) A_t \right] = \mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \left( (1-\lambda) \sum_{l=1}^{T-t-1} \lambda^{l-1} G_{t:t+l} + \lambda^{T-t-1} G_{t:T} \right) \right]$$
>
> 其中 $G_{t:t+h}$ 是用 $\bar{V}$ 计算的 n-step return。$\bar{V}$ 项在期望下消失，因此不引入额外 bias。

## 实验结果

基于 Qwen2.5-32B 模型，使用 rule-based reward（正确 +1，错误 -1）：

**主实验（16K context length）**：

| 模型 | AIME pass@1 | AIME pass@32 | GPQA pass@1 |
|------|-------------|--------------|-------------|
| [[GRPO]] | 38.9 | 70.0 | 49.4 |
| VC-PPO | **48.8** | **73.3** | 48.8 |

**消融实验（8K context length）**：

| 配置 | AIME pass@1 | AIME pass@32 |
|------|-------------|--------------|
| Baseline PPO | 5.6 | 36.7 |
| VC-PPO w/o Decoupled-GAE | 29.4 | 66.7 |
| VC-PPO（完整） | **41.9** | **76.6** |
| GRPO | 35.8 | 63.0 |

> [!warning] 两个技术缺一不可
> Value Pretraining 单独使用就能将 PPO 从崩溃中拯救出来（5.6 → 29.4），但加上 Decoupled GAE 后进一步提升到 41.9。两者解决的是不同层面的问题：前者修复初始化偏差，后者修复训练过程中的信号传播。

**$\lambda_{\text{actor}}$ 消融**：

| $\lambda_{\text{actor}}$ | AIME pass@1 |
|--------------------------|-------------|
| 0.90 | 34.3 |
| 0.95 | 41.3 |
| 0.99 | **41.9** |
| 1.00 | 29.4 |

$\lambda_{\text{actor}} = 1.0$ 性能显著下降，验证了 policy 确实需要方差控制。推荐设置 $\lambda_{\text{actor}} \in [0.95, 1.0)$。

## 与 GRPO 的对比

[[GRPO]] 通过完全去掉 value model 来回避了 value estimation 的所有问题，用 group relative advantage 替代。VC-PPO 则选择修复 value model，保留 token-level 的细粒度反馈。

> [!comparison] VC-PPO vs GRPO
> - **GRPO**：value-model-free，实现简单，内存开销低，但只能利用 response-level 反馈
> - **VC-PPO**：保留 value model，能利用 token-level 反馈，在复杂推理任务上潜力更大，但需要额外的 value pretraining 步骤和 value model 的显存开销
>
> 实验显示 VC-PPO 在 AIME 上显著优于 GRPO（48.8 vs 38.9），支持了 token-level 反馈在复杂推理任务中更有价值的假设。

## 局限性

- **额外计算开销**：Value Pretraining 需要额外的离线训练阶段，增加了整体训练时间
- **Value model 显存**：相比 [[GRPO]]，需要维护一个与 policy 同等规模的 value model
- **超参数敏感性**：Value Pretraining 的步数需要仔细调节（过少不够，过多过拟合）
- **固定 $\lambda$ 的局限**：Decoupled GAE 仍然使用固定的 $\lambda_{\text{actor}}$，无法适应不同长度的序列——这正是后续 [[VAPO (2025)]] 通过 Length-Adaptive GAE 解决的问题

> [!paper] 论文出处
> Yufeng Yuan, Yu Yue, Ruofei Zhu, Tiantian Fan, Lin Yan. *What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret.* ByteDance Seed, March 2025. ArXiv: 2503.01491

## 延伸阅读

**后续发展**：
- [[VAPO (2025)]] — 在 VC-PPO 基础上引入 Length-Adaptive GAE、Token-level Loss 等技术，进一步提升到 AIME 60.4
- [[DAPO]] — 另一条路线：完全去掉 value model 的 on-policy 方法
