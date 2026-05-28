---
type: method
description: 基于 REINFORCE 的 RLHF 算法，用 greedy decoding reward 作为 baseline 替代 value model，大幅降低内存和复杂度
aliases:
  - ReMax 算法
prerequisites:
  - "[[REINFORCE]]"
  - "[[RLHF]]"
  - "[[Reward Model]]"
tags:
  - post-training
  - reinforcement-learning
  - rlhf
  - variance-reduction
  - alignment
created: 2026-02-26
updated: 2026-02-26
---

# ReMax

ReMax（REINFORCE + argMax）是一种专为 [[RLHF]] 设计的 RL 算法。它的核心洞察是：RLHF 任务有三个特殊性质——确定性转移、轨迹级奖励、快速模拟——而 [[PPO]] 完全没有利用这些性质，导致了不必要的复杂度。ReMax 回归到更基础的 [[REINFORCE]] 算法，去掉了 value model，用 greedy decoding 的 reward 作为 baseline 来降低方差，实现了更简单、更省内存、更快的训练。

## 动机：PPO 对 RLHF 来说太重了

> [!intuition] RLHF 不是一般的 RL 问题
> 一般的 RL 任务（如 StarCraft）有随机转移、密集奖励、慢模拟。PPO 的 value model 和 [[GAE]] 就是为应对这些困难设计的。但 RLHF 中的 LLM 文本生成完全不同：
>
> | 性质 | 一般 RL | RLHF |
> |------|---------|------|
> | 状态转移 | 随机 | **确定性**（下一个 state = 当前 state + 生成的 token） |
> | 奖励 | 每步都有（dense） | **只在生成完整回复后给一次**（trajectory-level） |
> | 模拟速度 | 慢（需要环境交互） | **快**（LLM 推理即可） |
>
> 既然问题更简单，为什么还要用为复杂问题设计的算法？

PPO 在 RLHF 中的代价是实实在在的：value model 及其训练组件（激活值、梯度、优化器状态）占用了约 46% 的 GPU 内存，训练时间是 SFT 的 4 倍以上，还引入了 4 个以上需要调优的超参数（importance sampling clipping、GAE 系数、value model 学习率、off-policy 训练轮数）。

## 核心机制

### 从 REINFORCE 出发

[[REINFORCE]] 的策略梯度估计器为：

$$\widehat{g}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t^i | x^i, a_{1:t-1}^i) \cdot r(x^i, a_{1:T}^i)$$

这个估计器是无偏的，且不需要 value model。但直接用 REINFORCE 有一个严重问题：**梯度方差太大**。原因在于 reward 的绝对值直接作为梯度的权重——不同 prompt 的 reward 分布差异很大（实验中观察到 reward 范围从 -14.25 到 7.25），导致梯度 norm 剧烈波动。

### ReMax 的关键改进：Greedy Baseline

ReMax 引入一个 baseline $b_\theta(x)$ 来归一化 reward：

$$\widetilde{g}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t^i | x^i, a_{1:t-1}^i) \cdot \big(r(x^i, a_{1:T}^i) - b_{\theta}(x^i)\big)$$

其中 baseline 的选择是 ReMax 的核心创新：

$$b_{\theta}(x) = r(x, \bar{a}_{1:T}), \quad \bar{a}_t \in \operatorname{argmax} \pi_{\theta}(\cdot | x, \bar{a}_{1:t-1})$$

> [!intuition] 为什么用 greedy decoding 的 reward？
> Greedy decoding（每步选概率最大的 token）代表了当前策略的"最佳确定性行为"。用它的 reward 作为 baseline，相当于问：**随机采样的回复比贪心回复好多少？**
>
> 这个 baseline 有两个关键优势：
> 1. **跨 prompt 自适应**：不同 prompt 的 greedy reward 不同，自动归一化了 reward 的量级差异
> 2. **跨训练过程自适应**：随着策略更新，greedy baseline 也随之变化，不会像固定 baseline 那样过时
>
> 对比 PPO 的 value model：PPO 用一个单独的神经网络来估计 baseline（即 value function），需要额外的内存和训练。ReMax 直接用一次 greedy decoding 就得到了 baseline，不需要任何额外的可训练参数。

### 算法伪代码

```python
# ReMax: 仅 6 行核心代码
for prompt in datasets:
    seq = lm.sample(prompt, greedy=False)       # 随机采样一个回复
    seq_max = lm.sample(prompt, greedy=True)     # 贪心解码一个回复
    rew = rm(prompt, seq) - rm(prompt, seq_max)  # reward - baseline
    logp = lm.inference(prompt, seq)             # 计算 log 概率
    loss = -(logp.sum(dim=-1) * rew).mean()      # 加权似然最大化
    lm.minimize(loss)
```

对比 PPO 的 30+ 行实现，ReMax 的简洁性是显而易见的。

### 理论保证

> [!math] 无偏性与方差界
> **无偏性**：减去 baseline 不影响梯度期望，因为 $b_\theta(x)$ 不依赖于采样的 action：
> $$\mathbb{E}[\widetilde{g}(\theta)] = \nabla_{\theta} \mathbb{E}_{x \sim \rho} \mathbb{E}_{a_{1:T} \sim \pi_{\theta}}[r(x, a_{1:T})]$$
>
> **方差界**：方差被 $c \cdot r_{\max}^2 \cdot T^2 \cdot S^2 / N$ 上界约束，其中 $S$ 是 score function 的上界，$r_{\max}$ 是最大 reward 绝对值。
>
> **方差降低的条件**：当最优 action 尚未主导时（如 $\pi_\theta(a_1|x) \leq 0.5$），ReMax 的 greedy baseline 有效降低方差。在策略已经过度优化的区域，方差可能略增——但这反而是好事，因为它自然地减缓了 reward over-optimization。

## 与其他方法的对比

> [!comparison] ReMax vs PPO vs DPO vs GRPO
>
> | 维度 | [[PPO]] | ReMax | [[DPO]] | [[GRPO]] |
> |------|---------|-------|---------|----------|
> | 需要 value model | 是 | **否** | 否 | 否 |
> | 需要 [[Reward Model]] | 是 | 是 | 否（隐式） | 可选 |
> | 在线学习 | 是 | 是 | **否**（离线） | 是 |
> | Baseline 来源 | Value function | Greedy reward | Reference model | Group mean |
> | 额外超参数 | 4+（clip, GAE λ 等） | 少 | β | Group size |
> | GPU 内存 | 高（+46% value model） | **低** | 低 | 低 |
> | 采样数/prompt | 1 | 2（random + greedy） | 0（离线数据） | K（group size） |
>
> **与 DPO 的本质区别**：DPO 是 offline 方法，依赖 [[Bradley-Terry Model]] 假设和固定的 reference model；ReMax 是 online RL 方法，可以搭配任意 reward model，不需要偏好数据。DPO 的 reference model 可以看作一种"固定 baseline"，而 ReMax 的 greedy baseline 是随训练动态变化的。
>
> **与 GRPO 的区别**：[[GRPO]] 对每个 prompt 采样一组回复（group），用组内均值作为 baseline。ReMax 只需要 2 个样本（1 random + 1 greedy），采样效率更高，但 GRPO 的 group 采样提供了更丰富的梯度信号。

## 效率分析

在 Llama-2-7B + 4×A800-80GB 的设置下：

| 指标 | PPO | ReMax | 改进 |
|------|-----|-------|------|
| 无 offload 能否训练 | 不能 | **能** | — |
| 最大 batch size（有 offload） | 112 | 152 | 1.4x |
| 单 epoch 训练时间 | 2.9h | 1.8h | **1.6x 加速** |

内存节省的来源很直接：去掉 value model 省下了约 46% 的 GPU 内存（对 7B 模型，reward model 推理仅占 4%，而 value model 训练占 46%）。省下的内存可以分配给更大的 batch size，进一步提升吞吐量。

训练速度的分析：PPO 每步需要 1 次生成 + 2 次反向传播（policy + value），ReMax 需要 2 次生成 + 1 次反向传播。由于生成通常比反向传播快，ReMax 的每步时间更短。

## 实验结果

- **Llama-2-7B + full-hh-rlhf**：ReMax 达到与 PPO 相当的 reward，训练更稳定（梯度 norm 更平滑）
- **DPO + ReMax 组合**：用 DPO 训练的模型作为初始化，再用 ReMax 微调，达到最高 84.7% win rate（vs SFT baseline），说明 DPO 是 RL 优化的好起点，ReMax 能修复 DPO 的 out-of-distribution 问题
- **Mistral-7B 排行榜**：AlpacaEval 94.78% win rate，MT-bench 7.739，当时 7B 开源模型 SOTA

## 局限性

> [!warning] 边界条件
> - **Greedy baseline 的理论弱点**：当策略已经高度优化（最优 action 概率 > 0.5）时，greedy baseline 可能增加方差而非降低。不过作者认为这反而有助于防止 reward over-optimization。
> - **需要 2 次前向传播**：每个 prompt 需要 1 次随机采样 + 1 次 greedy 解码，生成成本是 REINFORCE 的 2 倍（但仍比 PPO 的 value model 训练便宜得多）。
> - **单样本估计**：每个 prompt 只用 1 个随机样本估计梯度，相比 [[GRPO]] 的多样本 group 估计，信息量更少。
> - **仍需 reward model**：与 [[DPO]] 不同，ReMax 需要预训练的 reward model，增加了 pipeline 复杂度。

> [!interview] 面试要点
> **Q: ReMax 相比 PPO 的核心改进是什么？**
> A: ReMax 利用了 RLHF 的三个特殊性质（确定性转移、轨迹级奖励、快速模拟），回归到更简单的 REINFORCE 算法，用 greedy decoding 的 reward 替代 value model 作为 baseline。这省去了 value model 的训练（~46% GPU 内存），减少了 4+ 个超参数，训练速度提升约 1.6 倍。
>
> **Q: Greedy baseline 为什么能有效降低方差？**
> A: 两个原因：(1) 跨 prompt 自适应——不同 prompt 的 greedy reward 不同，自动归一化了 reward 量级；(2) 跨训练过程自适应——随策略更新，baseline 也随之变化。这比固定 baseline 或 PPO 的 value model 都更轻量。

> [!paper] 论文出处
> Li, Z., Xu, T., Zhang, Y., Lin, Z., Yu, Y., Sun, R., & Luo, Z.-Q. (2024). ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models. *ICML 2024*. arXiv: 2310.10505

## 延伸阅读

**后续发展与相关方法**：
- [[GRPO]] — 另一种去掉 value model 的方法，用 group 采样替代 greedy baseline
- [[RLOO]] — 基于 leave-one-out 的 REINFORCE 变体，也不需要 value model
