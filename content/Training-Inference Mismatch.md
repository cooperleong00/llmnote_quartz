---
description: RL 训练中训练引擎和推理引擎计算的概率分布存在差异，导致梯度噪声和训练不稳定，在 MoE 和 long CoT 场景下尤为严重
type: concept
aliases:
  - 训练-推理不一致
  - Training-Inference Discrepancy
  - Probability Discrepancy
  - Learner-Sampler Mismatch
prerequisites:
  - "[[GRPO]]"
  - "[[KL Divergence]]"
tags:
  - post-training
  - reinforcement-learning
  - optimization
  - stability
created: 2026-01-28
updated: 2026-02-11T22:55
---

# Training-Inference Mismatch

Training-Inference Mismatch（训练-推理不一致）是指在 RL 训练中，**训练引擎和推理引擎对同一 token 计算的概率分布存在差异**的问题。这种差异会引入梯度噪声，导致训练不稳定，在 [[Mixture of Experts|MoE]] 模型和 long CoT 场景下问题尤为严重。

> [!paper] 论文出处
> - Ring-1T Team, "Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model", 2025
>   - 首次系统分析了 training-inference mismatch 的理论性质
>   - 证明了概率差异会指数级累积（Compounding Probability Discrepancy）
>   - 提出了 [[IcePop]] 作为解决方案
> - [[Stabilizing Reinforcement Learning with LLMs (2025)|Qwen Team, "Stabilizing Reinforcement Learning with LLMs", 2025]]
>   - 提出一阶近似框架，将 IS 比率分解为 training-inference discrepancy × policy staleness
>   - 证明 token-level 目标有效的条件是同时最小化这两个因素

---

## 动机

> [!intuition] 为什么会出现这个问题？

现代大规模 RL 训练通常采用**分离架构**：
- **推理引擎**（如 SGLang、vLLM）：负责高效生成 rollout 样本
- **训练引擎**（如 Megatron、DeepSpeed）：负责梯度计算和参数更新

这种分离带来了效率优势，但也引入了一个隐患：**两个引擎对同一 token 计算的概率可能不同**。

差异来源包括：
1. **数值精度差异**：FP16 vs BF16、不同的 softmax 实现
2. **并行策略差异**：张量并行、流水线并行的实现细节
3. **动态路由**：[[Mixture of Experts|MoE]] 模型的路由器对输入敏感，微小差异可能导致不同的专家选择
4. **算子实现**：不同框架的 attention、LayerNorm 等算子实现可能有细微差异
5. **模型本身不同**：为降低 rollout 成本，用更小的模型（如 1.7B）为大模型（如 8B）生成 rollout，此时 mismatch 比数值精度差异大一个数量级以上（参见 [[JACKPOT]]）

> [!note] 与传统 Off-Policy RL 的区别
> 传统 off-policy RL 的分布差异来自**参数差异**（行为策略 vs 目标策略）。
> Learner-Sampler Mismatch 的分布差异来自**系统差异**，即使参数完全相同也会存在。

---

## 问题分析

### 概率差异的定义

设 $\pi_{\text{infer}}(\cdot;\theta)$ 和 $\pi_{\text{train}}(\cdot;\theta)$ 分别是推理引擎和训练引擎加载的策略模型。定义第 $t$ 步的概率差异为：

$$\delta_t = D_{KL}(\pi_{\text{infer}}(\cdot;\theta_t) \| \pi_{\text{train}}(\cdot;\theta_t))$$

### IS 比率的分解（一阶近似视角）

[[Stabilizing Reinforcement Learning with LLMs (2025)]] 提出了一个清晰的分解框架。对于 token $y_t$，其 IS 比率可以分解为：

$$\frac{\pi_\theta(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})}}_{\text{training-inference discrepancy}} \times \underbrace{\frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t})}}_{\text{policy staleness}}$$

其中：
- $\mu_{\theta_{\text{old}}}$：推理引擎计算的 rollout 策略
- $\pi_{\theta_{\text{old}}}$：训练引擎计算的同参数策略
- $\pi_\theta$：当前优化的目标策略

> [!intuition] 两个独立的差异来源
> - **Training-inference discrepancy**：同一参数在不同引擎上的概率差异（系统层面）
> - **Policy staleness**：参数更新导致的策略变化（优化层面）
>
> Token-level 目标是 sequence-level 目标的**一阶近似**，其有效性要求这两个因素都足够小。

### Compounding Probability Discrepancy

> [!math] Theorem 1（概率差异的指数累积）
>
> 在一定条件下，存在常数 $\eta > 0$ 使得：
> $$\delta_{t+1} \geq \left(1 + \frac{\eta}{2}\mu\right) \delta_t$$
>
> 其中 $\mu$ 是与学习率和梯度相关的常数。

> [!intuition] 直觉理解
> 这个定理说明：**概率差异会随训练迭代指数级增长**。
>
> 原因是一个恶性循环：
> 1. 推理引擎生成样本时使用的概率 $\pi_{\text{infer}}$
> 2. 训练引擎计算梯度时使用的概率 $\pi_{\text{train}}$ 与之不同
> 3. 梯度更新后，两个引擎的差异进一步放大
> 4. 下一轮迭代，差异更大...

### 梯度误差与序列长度的关系

> [!math] Theorem（梯度误差的二次增长）
>
> 设 $\nabla_\theta \mathcal{J}(\theta)$ 和 $\nabla_\theta \mathcal{J}_{actual}(\theta)$ 分别是理想梯度和实际计算的梯度，则：
> $$\|\nabla_\theta \mathcal{J}_{actual}(\theta) - \nabla_\theta \mathcal{J}(\theta)\|_2 \leq C \cdot T^2$$
>
> 其中 $C$ 是与单 token 数值误差和 score function 幅度相关的常数，$T$ 是序列长度。

> [!intuition] 为什么是二次关系？
> 二次增长来自两个因素的叠加：
> 1. **状态分布偏移**：时刻 $t$ 的状态分布偏差随时间线性累积
> 2. **误差求和**：每步分布偏差的求和
>
> 这与 [[TRPO]] 中 off-policy surrogate 的 penalty bound 形式一致。

### 在 MoE 中更严重

[[Mixture of Experts|MoE]] 模型的动态路由机制使问题更加严重：

- 路由器（Router）对输入非常敏感
- 微小的数值差异可能导致完全不同的专家选择
- 不同专家的输出差异远大于数值精度差异

[[Stabilizing Reinforcement Learning with LLMs (2025)]] 进一步分析了 MoE 中 IS 比率的分解：

$$\frac{\pi_\theta(y_t|x, y_{<t}, e_t^\pi)}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t}, e_{\text{old},t}^\mu)} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t}, e_{\text{old},t}^\pi)}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t}, e_{\text{old},t}^\mu)}}_{\text{training-inference discrepancy}} \times \underbrace{\frac{\pi_\theta(y_t|x, y_{<t}, e_t^\pi)}{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t}, e_{\text{old},t}^\pi)}}_{\text{policy staleness}}$$

其中 $e$ 表示路由选择的专家。关键问题是：
- **Training-inference discrepancy** 会导致 $e_{\text{old},t}^\pi \neq e_{\text{old},t}^\mu$（同参数不同路由）
- **Policy staleness** 不仅体现在参数变化，还体现在路由变化 $e_t^\pi \neq e_{\text{old},t}^\pi$

这使得一阶近似更容易失效，需要 [[Rollout Routing Replay]] 等方法来稳定训练。

### 在 Long CoT 中更严重

长链式思维（Long Chain-of-Thought）场景下：

- 序列长度可达数千甚至数万 token
- 每个 token 的概率差异会累积
- 长序列放大了 compounding 效应
- 梯度误差与序列长度 $T$ 呈**二次关系**（见上文 Theorem）

### Mismatch 是动态优化问题

> [!warning] 关键洞察
> Training-inference mismatch **不是静态的数值误差**，而是与优化过程动态耦合的。

实验证据：
1. **Mismatch 与梯度噪声同步增长**：训练后期，梯度范数上升的同时 mismatch 也在增大
2. **降低学习率可以抑制 mismatch**：将 LR 从 1e-6 降到 1e-7 可以显著延长稳定训练窗口
3. **与 epoch 数无关**：使用 25% 数据集训练，collapse 时间点与全量数据几乎相同

这表明 mismatch 与模型在权重空间的位置有关——训练后期模型可能进入了几何特性不好的区域（如高曲率区域），放大了数值差异。

---

## 影响

> [!warning] 训练不稳定的表现

1. **梯度范数剧烈波动**：概率差异导致梯度估计不准确
2. **训练 loss 震荡**：难以稳定收敛
3. **训练崩溃**：极端情况下可能导致 NaN 或训练完全失败
4. **性能下降**：即使不崩溃，最终模型性能也会受损

### 从 FP32 Optimizer 视角理解

混合精度训练中，优化器状态通常保持 FP32，而前向/反向传播使用 BF16：

- FP32 权重定义了一个"精确策略" $\pi^{fp32}$
- Learner 和 Sampler 都是 $\pi^{fp32}$ 的 BF16 近似
- 当 mismatch 小时，可以认为两者都是 $\pi^{fp32}$ 的合理近似
- 当 mismatch 大时，应该将它们视为**不同的策略**

### 与 Output Perturbation 的关联

实验表明，**mismatch 大的样本往往也对输入扰动敏感**：

- 平均 mismatch 与平均 output perturbation 强相关
- 这意味着 mismatch 大的样本是"病态"的（ill-conditioned）
- 过滤这些样本不仅减少噪声，还能稳定训练

---

## 解决方案

### 1. IcePop（Masking）

[[IcePop]] 通过**双边校准和 masking** 解决问题：

- 计算概率比率 $k = \frac{\pi_{\text{train}}}{\pi_{\text{infer}}}$
- 当 $k \notin [\alpha, \beta]$ 时，丢弃该 token 的梯度
- 典型值：$\alpha = 0.5$, $\beta = 5$

> [!intuition] 核心思想
> 与其试图校正差异，不如直接**丢弃不可靠的 token**。

### 2. TIS（Truncated Importance Sampling）

[[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training|TIS]] 通过**截断的重要性采样**校正概率差异：

$$\text{weight} = \min\left(\frac{\pi_{\text{train}}(y_t)}{\pi_{\text{infer}}(y_t)}, C\right)$$

其中 $C$ 是截断阈值（如 $C=2$）。截断操作防止当 $\pi_{\text{infer}}$ 很小时重要性比率过大导致梯度方差爆炸。

> [!note] 与 IcePop 的区别
> - **TIS**：保留所有 token，用截断的重要性比率加权梯度
> - **IcePop**：直接丢弃差异过大的 token（masking）
>
> 两种方法解决同一问题，TIS 在实验中表现良好，与 [[GSPO]] 等方法正交兼容。

### 3. Routing Replay（针对 MoE）

[[Rollout Routing Replay]]：在训练时**重放推理时的路由决策**：

- 推理时记录每个 token 的路由选择
- 训练时强制使用相同的路由
- 消除路由不一致带来的差异

### 4. JACKPOT（Rejection Sampling）

[[JACKPOT]] 针对**极端 mismatch**（rollout 模型与 policy 模型完全不同）提出了基于 Optimal Budget Rejection Sampling (OBRS) 的方案：

- 对 rollout 模型生成的 token，按概率比 $\frac{p_i}{\lambda q_i}$ 决定是否接受
- 被拒绝的 token 不参与梯度更新，等效于将 rollout 分布"削峰"向 policy 分布靠拢
- 同时用 reverse-KL loss 持续对齐 rollout 模型，防止训练过程中 gap 越拉越大

> [!intuition] 与 IcePop/TIS 的区别
> IcePop 和 TIS 是**事后校正**——接受所有 token，再通过 masking 或加权修正梯度。JACKPOT 是**事前过滤**——在 token 进入训练之前就拒绝分布差异大的样本，从源头缩小 mismatch。两者正交，可以叠加使用。

### 5. Length-decay LR Scheduler

基于 mismatch 是动态优化问题的洞察，通过**自适应学习率调度**来抑制 mismatch：

**核心思想**：
- 降低学习率可以有效抑制 mismatch（实验验证）
- 但全局降低 LR 会牺牲早期训练效率
- 需要一个**早期预警信号**来触发 LR decay

**Response Length Surge 作为预警信号**：
- 训练过程中，平均 response length 会出现突然激增（如从 1000 增到 3000-4000）
- 这种 surge 是不稳定性的前兆——更长的序列会放大数值差异（$O(T^2)$）
- 在 surge 后约 **1.8×** 的时间点开始 decay 是最佳选择

**调度策略**：
- 监测 response length surge 的发生时间点 $t_{surge}$
- 设置 decay period $T_{decay} \approx 1.8 \times t_{surge}$
- 每 $T_{decay}$ 步将 LR 减半，直到达到最小值（默认 10% $\eta_0$）

> [!note] 与 IS 方法的关系
> LR scheduler 和 Importance Sampling 在不同层面工作：
> - **IS**：校正单样本的梯度偏差
> - **LR scheduler**：调节整体更新幅度，抑制动态不稳定性
>
> 两者**正交兼容**——即使已使用 TIS，添加 LR scheduler 仍能进一步提升稳定性和性能。

> [!comparison] 方法对比

| 方法                         | 机制                | 优点         | 缺点       |
| -------------------------- | ----------------- | ---------- | -------- |
| [[IcePop]]                 | Masking 不可靠 token | 简单有效，无额外开销 | 丢失部分梯度信号 |
| [[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training\|TIS]]                        | 截断重要性采样校正           | 保留所有 token，与其他方法正交兼容      | 需要调节截断阈值  |
| [[Rollout Routing Replay]] | 重放路由决策            | 从根本上消除路由差异 | 需要存储路由信息 |
| [[JACKPOT]]                | Rejection sampling 过滤 token | 从源头缩小分布差距，支持极端 mismatch | 降低样本利用率，需额外内存 |
| Length-decay LR Scheduler | 自适应学习率调度 | 从优化层面解决，与 IS 正交兼容 | 需要监测 response length |

---

## 面试要点

> [!interview] 面试视角
>
> **Q: 什么是 Training-Inference Mismatch？**
> A: 在 RL 训练中，推理引擎和训练引擎对同一 token 计算的概率可能不同。这种差异来自数值精度、并行策略、算子实现等因素，在 MoE 模型中因动态路由而更严重。
>
> **Q: 为什么这个问题在 long CoT 场景下更严重？**
> A: 两个原因：(1) 概率差异会跨迭代指数级累积（Compounding Probability Discrepancy）；(2) 梯度误差与序列长度 $T$ 呈二次关系 $O(T^2)$。长序列意味着更多 token，累积效应更明显。
>
> **Q: 如何解决这个问题？**
> A: 主要有五种方法：(1) IcePop 通过 masking 丢弃差异过大的 token；(2) TIS 通过截断的重要性采样校正；(3) Routing Replay 重放推理时的路由决策；(4) JACKPOT 通过 rejection sampling 从源头过滤分布差异大的 token，适用于极端 mismatch（如用不同模型做 rollout）；(5) Length-decay LR Scheduler 通过自适应学习率调度从优化层面解决。前四种是样本级校正/过滤，第五种是优化级干预，它们正交兼容。
>
> **Q: 为什么说 mismatch 是动态优化问题而非静态数值误差？**
> A: 实验表明：(1) mismatch 与梯度噪声同步增长；(2) 降低学习率可以抑制 mismatch；(3) collapse 时间点与数据量无关。这说明 mismatch 与模型在权重空间的位置有关，训练后期可能进入高曲率区域放大了数值差异。

---

## 相关概念

- [[IcePop]] — 通过 masking 解决此问题的方法
- [[GRPO]] — 受此问题影响的 RL 算法
- [[Mixture of Experts]] — 问题在 MoE 中更严重
- [[Importance Sampling]] — TIS 的理论基础
- [[Rollout Routing Replay]] — 针对 MoE 的解决方案
- [[JACKPOT]] — 通过 rejection sampling 解决极端 mismatch（不同模型做 rollout）
- [[GSPO]] — 解决 MoE 专家激活波动的互补方法

---

## 参考资料

- [[Clippings/Paper/251018855v2/251018855v2|Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model (2025)]]
- [[Policy Gradient, Sequence, and Token— Part II Learner-Sampler Mismatch|Policy Gradient Part II]] — Learner-Sampler Mismatch 的理论分析和实验验证
- [[Clippings/Paper/260201826v1/260201826v1|Beyond Precision: Training-Inference Mismatch is an Optimization Problem (2026)]] — 提出 mismatch 是动态优化问题，通过 LR scheduler 解决
- [[Stabilizing Reinforcement Learning with LLMs (2025)]] — 一阶近似框架，IS 比率分解为 training-inference discrepancy × policy staleness
- [[Clippings/Paper/260206107v1/260206107v1|JACKPOT: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL (2025)]] — 用 OBRS 从源头缩小 rollout 与 policy 的分布差距
