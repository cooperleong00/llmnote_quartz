---
type: method
description: 通过二元掩码过滤 off-distribution 样本替代 importance sampling 的连续权重，解决大规模 off-policy RL 训练的梯度方差问题
aliases:
  - Metropolis Independence Sampling-Filtered Policy Optimization
  - MIS-Filtered Policy Optimization
prerequisites:
  - "[[Importance Sampling]]"
  - "[[PPO]]"
  - "[[Training-Inference Mismatch]]"
tags:
  - post-training
  - rlhf
  - off-policy
created: 2026-02-12
updated: 2026-02-12T20:35
---

# MIS-PO

MIS-PO（Metropolis Independence Sampling-Filtered Policy Optimization）是 [[Step 3.5 Flash (2025)]] 提出的 RL 训练方法。核心创新是用**二元掩码**替代 [[Importance Sampling]] 的连续权重，将通过过滤的样本视为 effectively on-policy，从而大幅降低梯度方差。

## 动机

> [!intuition] 为什么需要 MIS-PO？
> 大规模 LLM 的 RL 训练面临严重的梯度方差问题，根源在于两个层面的 off-policy 不匹配：
> 1. **Infrastructure divergence**：高吞吐推理引擎（如 vLLM）与训练框架计算的概率分布存在差异
> 2. **Off-policy misalignment**：迭代更新中策略不断变化，采样时的策略与训练时的策略不同
>
> 传统的 [[Importance Sampling]] 通过 bounded ratio 缩放梯度来修正这种不匹配，但 token 级别的概率偏移会**累积**成噪声梯度，导致训练不稳定。这个问题在长链推理（long-horizon reasoning）和大规模 MoE 模型中尤为严重。

这个问题的本质是 [[Training-Inference Mismatch]]。现有方法如 [[GRPO]] 和 [[PPO]] 都依赖 importance sampling，在大规模场景下表现出高方差和训练不稳定。

## 核心机制

### 从 Metropolis Independence Sampling 获得灵感

MIS-PO 的名字来源于 Metropolis Independence Sampling（MIS），一种 MCMC 采样方法。核心思想是：

- 将 **inference policy**（推理引擎的策略）作为 proposal distribution
- 将 **training policy**（训练框架的策略）作为 target distribution
- 只保留**足够接近** target distribution 的样本

> [!intuition] 关键洞察
> Importance sampling 的问题在于它试图用**连续权重**修正分布差异，但权重的方差很高。MIS-PO 的解决方案更激进：直接**丢弃** off-distribution 样本，将保留的样本视为 on-policy。
>
> 这是一种 trade-off：牺牲一部分样本利用率，换取显著降低的梯度方差。

### 双层过滤机制

MIS-PO 在两个粒度上应用二元掩码：

**1. Token-level 过滤**

对每个 token 计算概率比：

$$x_t = \frac{\pi_{\theta_{\text{old}}}(a_t|s_t)}{\pi_{\theta_{\text{vllm}}}(a_t|s_t)}$$

其中 $\pi_{\theta_{\text{old}}}$ 是训练策略，$\pi_{\theta_{\text{vllm}}}$ 是推理引擎的策略。

**2. Trajectory-level 过滤**

对整条轨迹计算几何平均比：

$$\bar{\rho}(\tau) = \left(\prod_t x_t\right)^{1/T}$$

几何平均比序列长度归一化，避免长序列被系统性地过滤掉。

**指示函数**：

$$\mathbb{I}(x) = \mathbb{I}[\rho_{\min} \le x \le \rho_{\max}]$$

论文中使用的边界：
- Token-level: $[\rho_{\min}, \rho_{\max}] = [0.5, 2]$
- Trajectory-level: $[\rho_{\min}, \rho_{\max}] = [0.996, 1.001]$

> [!warning] 为什么 trajectory-level 边界这么紧？
> Trajectory-level 的边界 [0.996, 1.001] 非常接近 1，这意味着只有整体分布非常接近的轨迹才会被保留。这是因为 trajectory-level 的几何平均已经对 token-level 的偏差做了平滑，如果几何平均仍然偏离 1，说明整条轨迹的分布偏移是系统性的。

### Actor Loss

> [!math] MIS-PO 目标函数
> $$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{\tau \sim \pi_{\theta_{\text{vllm}}}} \left[ \mathbb{I}(x_t) \cdot \mathbb{I}(\bar{\rho}(\tau)) \cdot \log \pi_{\theta}(a_t | s_t) \cdot \hat{A}_t \right]$$
>
> 其中：
> - $\mathbb{I}(x_t)$：token-level 掩码
> - $\mathbb{I}(\bar{\rho}(\tau))$：trajectory-level 掩码
> - $\hat{A}_t$：advantage 估计（通常用 [[GAE]]）
> - 两个掩码相乘意味着：只有同时通过两层过滤的 token 才参与梯度更新

## 与 PPO 的对比

论文通过约 5000 步的 ablation study（Figure 5）展示了 MIS-PO 相对于 [[PPO]] 的优势：

| 维度 | PPO | MIS-PO |
|------|-----|--------|
| **Sample Efficiency** | 较低 | 更高的 reward plateau，更快收敛 |
| **Stability** | 梯度 norm 有大量噪声和 spike | 显著降低噪声，消除 spike |
| **Exploration** | 熵快速衰减 | 更慢的熵衰减，更好的 exploration-exploitation 平衡 |

> [!intuition] 为什么 MIS-PO 能保持更好的 exploration？
> PPO 的 clip 机制虽然限制了更新幅度，但高方差的梯度仍然会导致策略快速收敛到局部最优。MIS-PO 通过过滤 off-distribution 样本，保证每次更新都是高质量的，避免了噪声梯度导致的过早收敛。

## 配套技术

### 1. Truncation-Aware Value Bootstrapping

长链推理任务中，轨迹经常因为 context length 限制被截断。如果简单地给截断轨迹分配零奖励，会错误地惩罚长链推理。

解决方案：用 value estimate 替代截断轨迹的零奖励：

$$\hat{R}_i = \begin{cases} V_{\phi}(s_T) & \text{if truncated} \\ R_i & \text{otherwise} \end{cases}$$

这将截断视为 horizon interruption 而非 task failure。

### 2. Routing Confidence 监控

针对 [[Mixture of Experts|MoE]] 架构，论文提出用 Routing Confidence（$\Sigma_k$，激活专家的平均概率质量）作为训练稳定性的代理指标：

- 低 $\Sigma_k$ → 高路由不确定性 → 放大 training-inference mismatch → 需要严格的稳定化措施
- 高 $\Sigma_k$ → 路由稳定 → 可以容忍 off-policy 训练

## 与其他方法的对比

> [!comparison] MIS-PO vs 其他 off-policy 稳定化方法
>
> | 方法 | 核心思想 | 特点 |
> |------|----------|------|
> | [[PPO]] | Clip importance ratio | 连续权重，方差仍然较高 |
> | [[GRPO]] | Group relative advantage | 去掉 value function，但仍用 IS |
> | [[GSPO]] | 序列级 IS 替代 token 级 | 降低方差，但仍是连续权重 |
> | [[IcePop]] | 双边校准 + masking | 类似思路，针对 MoE |
> | **MIS-PO** | 二元掩码替代连续权重 | 最激进的过滤，最低方差 |

## 局限性

> [!warning] 边界条件
> 1. **样本利用率**：被过滤的样本不参与训练，在样本稀缺场景下可能是问题
> 2. **超参数敏感**：$\rho_{\min}$, $\rho_{\max}$ 的选择需要调优
> 3. **依赖 value function**：仍需要 critic 网络估计 advantage，不如 [[GRPO]] 简洁

## 面试要点

> [!interview] 常见问题
> **Q: MIS-PO 和 PPO 的核心区别是什么？**
> A: PPO 用 clip 限制 importance ratio 的范围但仍使用连续权重；MIS-PO 更激进，直接用二元掩码过滤 off-distribution 样本，将保留的样本视为 on-policy。
>
> **Q: 为什么 MIS-PO 能降低梯度方差？**
> A: Importance sampling 的方差来源于权重的不确定性。MIS-PO 通过过滤，确保参与训练的样本都是"足够接近"target distribution 的，消除了权重带来的方差。
>
> **Q: MIS-PO 的双层过滤分别解决什么问题？**
> A: Token-level 过滤处理局部的概率偏移；trajectory-level 过滤处理整体的分布漂移。两者结合确保样本在局部和全局都是 on-policy 的。

## 延伸阅读

**原始论文**：
- [[Step 3.5 Flash (2025)]] — MIS-PO 的提出论文

**相关方法**：
- [[GSPO]] — 另一种解决 off-policy 稳定性的方法，使用序列级 IS
- [[IcePop]] — 针对 MoE 的 masking 方法
- [[SAPO]] — 用平滑温度门替代硬裁剪

**背景知识**：
- [[Training-Inference Mismatch]] — MIS-PO 解决的核心问题
- [[Importance Sampling]] — MIS-PO 替代的传统方法
