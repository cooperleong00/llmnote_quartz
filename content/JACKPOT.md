---
type: method
description: 通过 Optimal Budget Rejection Sampling 缩小 decoupled RL 中 rollout 模型与 policy 模型的分布差距，使小模型 rollout 训练大模型成为可能
aliases:
  - Jackpot
  - Optimal Budgeted Rejection Sampling for RL
prerequisites:
  - "[[PPO]]"
  - "[[Importance Sampling]]"
  - "[[KL Divergence]]"
  - "[[Training-Inference Mismatch]]"
tags:
  - rl-for-llm
  - training-efficiency
  - distribution-alignment
created: 2026-02-11
updated: 2026-02-11T22:50
---

# JACKPOT

JACKPOT（Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch Reinforcement Learning）解决的是 RL for LLM 中一个很实际的问题：rollout 占训练成本的 80%，如果能用更小的模型做 rollout（比如用 1.7B 给 8B 生成轨迹），成本会大幅下降——但两个模型的输出分布差异太大，直接训练会崩溃。JACKPOT 通过 Optimal Budget Rejection Sampling（OBRS）在 token 级别筛选和重加权 rollout 样本，将 actor 分布拉近 policy 分布，同时联合更新两个模型防止分布差距随训练扩大。

> [!paper] 论文信息
> Zhuoming Chen, Hongyi Liu, Yang Zhou, Haizhong Zheng, Beidi Chen (CMU)
> ArXiv: 2602.06107 (2025)
> GitHub: https://github.com/Infini-AI-Lab/jackpot

## 动机

> [!intuition] 为什么需要 JACKPOT？
> RL 训练 LLM 时，rollout（自回归生成轨迹）占了绝大部分计算成本。一个自然的想法是：用一个更小、更快的模型来做 rollout，然后用生成的轨迹训练大模型。
>
> 问题在于，小模型和大模型的输出分布差异巨大——KL 散度比异步训练或量化带来的 mismatch 大一个数量级。现有的 [[Importance Sampling|重要性采样]] 校正方法（如 TIS）在这种极端 mismatch 下力不从心，训练直接崩溃。

JACKPOT 的核心洞察是：与其在事后用 importance weight 校正分布差异（post-hoc correction），不如在源头直接缩小差距——通过 rejection sampling 筛掉那些"分布差异最大"的 token，让剩余样本的分布更接近目标分布。但经典 rejection sampling 在 LLM 的 100K+ 词表上几乎不可用（acceptance rate 趋近于零），所以 JACKPOT 采用了放松版本的 OBRS，在给定 budget 下做到最优的分布对齐。

### 现有方法为什么不够？

[[Training-Inference Mismatch]] 在 decoupled RL 中有三类解决思路：

1. **Post-hoc IS 校正**（如 TIS）：用 $\min(\frac{p_{\text{ref}}}{p_{\text{inf}}}, C)$ 截断重要性权重。当 mismatch 小（异步训练、量化）时有效，但在极端 mismatch（不同模型）下，截断太多信息丢失，截断太少方差爆炸
2. **Masking**（如 [[IcePop]]）：丢弃概率比率异常的 token。简单有效但没有理论最优性保证
3. **JACKPOT 的方案**：在 token 级别做 optimal rejection sampling，理论上保证在任意 budget 下调整后的分布严格更接近目标分布。与 TIS 正交互补——先用 OBRS 缩小分布差距，再用 TIS 校正残余 mismatch

## 核心机制

JACKPOT 包含三个组件，分别解决 decoupled RL 的三个挑战。

### 组件一：Optimal Budget Rejection Sampling (OBRS)

**问题**：经典 rejection sampling 要求 $\lambda \geq \max_i \frac{p_i}{q_i}$，在 LLM 词表上这个 $\lambda$ 极大，导致几乎所有 token 都被拒绝。

**OBRS 的放松**：不要求 $\tilde{q} = p$（完美对齐），而是在给定 acceptance budget 下，找到最优的 rejection 规则使 $\tilde{q}$ 尽可能接近 $p$。

> [!math] OBRS 接受规则
> 对于从 rollout 模型 $q$ 采样的 token $i$，以概率接受：
>
> $$a_i = \min\left(1, \frac{p_i}{\lambda q_i}\right)$$
>
> 其中 $\lambda > 0$ 是用户指定的参数，控制 rejection budget（$\lambda$ 越小，拒绝越多，对齐越好）。
>
> 接受后的分布为：
> $$\tilde{q}_i \propto q_i \cdot a_i = \min\left(q_i, \frac{p_i}{\lambda}\right)$$

> [!intuition] 直觉理解
> OBRS 的接受规则本质上在做一件事：对于那些 rollout 模型过度生成的 token（$q_i \gg p_i$），按比例降低它们的接受概率；对于 rollout 模型生成概率已经低于目标的 token（$q_i \leq \frac{p_i}{\lambda}$），全部接受。这相当于"削峰"——把 $q$ 中超出 $p$ 的部分削掉，让分布形状更接近目标。

**理论保证**：

> [!math] Theorem 3.3 — OBRS 严格改善分布对齐
> 对任意 $\lambda > 0$（只要 $\lambda < \max_i \frac{p_i}{q_i}$），OBRS 调整后的分布 $\tilde{q}$ 严格比原始 $q$ 更接近目标 $p$：
>
> $$D_{\text{KL}}(p \| \tilde{q}) \leq D_{\text{KL}}(p \| q)$$

> [!math] Theorem 3.4 — OBRS 在固定 budget 下最优
> 在所有满足给定 acceptance rate 约束的 rejection 规则中，OBRS 是 $D_{\text{KL}}(p \| \hat{q})$ 的唯一最小化器：
>
> $$\tilde{q} = \underset{\hat{q}}{\operatorname{argmin}} D_{\text{KL}}(p \| \hat{q}), \quad \text{s.t.} \quad \hat{q}_i \propto q_i a_i, \sum_i q_i a_i = \bar{a}$$

这意味着 OBRS 不只是"某种 rejection 方法"，而是在给定 budget 下理论上最好的选择。

**在 RL 中的应用**：被拒绝的 token 从 loss 计算和梯度传播中 mask 掉，被接受的 token 按 OBRS 调整后的分布 $P_{\text{OBRS}}$ 重加权，确保梯度忠实反映调整后的分布。

### 组件二：统一训练目标

JACKPOT 联合优化三个损失函数：

> [!math] JACKPOT 联合目标
> $$\mathcal{L}^{\text{Jackpot}}(\theta, \omega) = \underbrace{\mathcal{L}^{\text{PPO-OBRS}}(\theta)}_{\text{policy RL}} + \underbrace{\mathcal{L}^{\text{PPO}}(\omega)}_{\text{rollout RL}} + \lambda_{\text{distill}} \underbrace{\mathcal{L}^{\text{distill}}(\omega)}_{\text{on-policy distillation}}$$

**(1) OBRS-adjusted PPO loss（训练 policy 模型 $\theta$）**：

在标准 [[PPO]] 目标上加入 OBRS mask 和重加权：

$$\mathcal{L}^{\text{PPO-OBRS}}(\theta) = \mathbb{E}_{x \sim p_{\text{inf}}} \left[ \text{Mask}(x) \cdot \text{SG}\left(\rho_{\text{jackpot}}\right) \cdot \min\left(r_\theta(x)\hat{A}(x), \text{clip}(r_\theta(x), 1-\epsilon, 1+\epsilon)\hat{A}(x)\right) \right]$$

其中 $\rho_{\text{jackpot}}$ 包含 OBRS 权重和 IS 校正。

**(2) 标准 PPO loss（训练 rollout 模型 $\omega$）**：

rollout 模型也用 PPO 更新，但不需要 OBRS（因为它自己生成的轨迹对自己是 on-policy 的）。

**(3) On-policy distillation loss（对齐 rollout 模型到 policy 模型）**：

$$\mathcal{L}^{\text{distill}}(\omega) = \mathbb{E}_{x \sim p_{\text{inf}}} \left[ D_{\text{KL}}\left(\text{SG}(p_{\theta_{\text{new}}}(x)) \| p_\omega(x)\right) \right]$$

> [!intuition] 为什么需要 distillation loss？
> 如果只用 OBRS + PPO 训练 policy 模型，随着 policy 不断改进，它和固定的 rollout 模型之间的分布差距会越来越大，最终 OBRS 也无法弥补。Distillation loss 让 rollout 模型"追着" policy 模型走，保持两者的分布差距在可控范围内。这本质上是一种 [[On-Policy Distillation]] 的思想。

**$p_{\text{target}}$ 的选择**：OBRS 中的目标分布可以选 $p_{\text{ref}}$（reference policy）或 $p_{\theta_{\text{new}}}$（当前 policy）。当 policy staleness 严重时（如大 batch 训练），用 $p_{\theta_{\text{new}}}$ 效果更好，因为 $p_{\text{ref}}$ 可能已经过时。

### 组件三：高效系统实现

OBRS 的归一化常数 $Z = \sum_{x'} \min(p_{\text{inf}}(x'), \frac{p_{\text{target}}(x')}{\lambda})$ 需要遍历整个词表（100K+ tokens），存储 batch_size $\times$ seq_len $\times$ vocab_size 的 logit 张量会导致内存瓶颈。

**Top-k 近似**：利用 LLM 输出分布的集中性，只在 top-k token 的并集上计算：

$$\mathcal{V}_k = \text{top-k}(p_{\text{inf}}) \cup \text{top-k}(p_{\theta_{\text{new}}})$$

$$Z_{\text{approx}} = \sum_{x \in \mathcal{V}_k} \min\left(p_{\text{inf}}(x), \frac{p_{\text{target}}(x)}{\lambda}\right)$$

**Batch-level bias correction**：top-k 近似系统性地低估 $Z$（因为省略了尾部 token）。巧妙的是，$Z$ 恰好等于期望 acceptance rate $\bar{\alpha}$，可以从实际采样中无偏估计。用 batch 级别的校准因子 $\kappa = \frac{\hat{\alpha}}{\frac{1}{B}\sum_i Z_{\text{approx}}^{(i)}}$ 修正偏差。

> [!intuition] 效率关键点
> - 不需要额外的 rollout——所有 loss 都在同一批轨迹上计算
> - 不需要额外的 forward pass——$p_{\text{ref}}$ 和 $p_{\theta_{\text{new}}}$ 在标准 PPO 计算中已经产生
> - 不需要修改 vLLM——rollout 阶段完全标准，只在训练阶段做 OBRS 重加权
> - 不需要重采样轨迹——与 speculative decoding 不同，被拒绝的 token 只是 mask 掉，不需要重新生成后续 token

## 实验结果

在三组 actor-policy 配对上验证（数学推理任务）：

| 配置 | 数据集 | 结果 |
|------|--------|------|
| Qwen2.5-1.5B → Qwen2.5-3B | MATH | 接近 3B on-policy 性能 |
| Qwen3-1.7B → Qwen3-4B | DeepScaleR | 匹配 4B on-policy 性能 |
| Qwen3-1.7B → Qwen3-8B | DeepScaleR | 300 步内接近 8B on-policy 性能 |

对比 baseline：
- **无对齐**：训练直接崩溃（KL 散度爆炸）
- **TIS + Reverse KL**：有改善但仍不稳定，与 on-policy 有明显差距
- **JACKPOT**：稳定训练 300 步，性能接近甚至匹配 on-policy

## 局限性

> [!warning] 边界条件
> 1. **小 mismatch 场景无优势**：当分布差距本身就小（如标准 PPO clipping、FP8 KV 量化），TIS 已经足够，JACKPOT 的额外复杂度没有回报
> 2. **仅验证了数学推理**：实验局限于 MATH 和 DeepScaleR 数据集，在 coding、agentic 等任务上的效果未知
> 3. **需要同时训练两个模型**：虽然 rollout 模型更小，但联合训练增加了系统复杂度
> 4. **$\lambda$ 的选择**：OBRS 的 rejection budget 参数需要调优，论文未给出自适应策略
> 5. **理论保证是 token 级别的**：OBRS 在每个 token 位置独立操作，没有考虑 sequence 级别的分布对齐

> [!comparison] 与相关方法的对比
>
> | 方法 | 思路 | 适用场景 | 与 JACKPOT 的关系 |
> |------|------|----------|-------------------|
> | TIS | Post-hoc IS 截断 | 小 mismatch（异步、量化） | 正交互补，可叠加 |
> | [[IcePop]] | 双边 masking | 中等 mismatch | 类似思路但无最优性保证 |
> | [[Rollout Routing Replay]] | 重放推理时路由 | MoE 训练-推理不一致 | 解决不同类型的 mismatch |
> | JACKPOT | OBRS + 联合训练 | 极端 mismatch（不同模型） | 理论最优的 rejection 规则 |

> [!interview] 面试视角
> **Q: JACKPOT 和 importance sampling 校正有什么区别？**
> A: IS 校正（如 TIS）是 post-hoc 的——先用 rollout 模型的分布采样，再用权重校正。JACKPOT 是 pre-hoc 的——通过 rejection sampling 直接修改参与训练的样本分布，从源头缩小差距。两者正交，可以叠加使用。
>
> **Q: 为什么不直接用经典 rejection sampling？**
> A: 经典 rejection sampling 要求 $\lambda \geq \max_i \frac{p_i}{q_i}$，在 100K+ 词表上这个值极大，acceptance rate 趋近于零。OBRS 放松了这个要求，允许用户指定 acceptance budget，在该 budget 下做到理论最优的分布对齐。
>
> **Q: JACKPOT 什么时候不该用？**
> A: 当 actor-policy 分布差距本身就小的时候（标准 on-policy 训练、FP8 量化 rollout），TIS 已经足够，JACKPOT 的额外复杂度没有收益。它专门为极端 mismatch 场景设计。

## 延伸阅读

**后续方向**：
- [[Decoupled RL Training]] — JACKPOT 推动的 rollout 与 policy 完全解耦的训练范式
- [[Speculative Decoding]] — 同样利用小模型加速大模型，但在推理阶段而非训练阶段

**参考资料**：
- [[Clippings/Paper/260206107v1/260206107v1|原始论文]] — 完整的理论推导和实验细节
