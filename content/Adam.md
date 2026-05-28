---
type: method
description: 结合动量和自适应学习率的优化器，通过一阶和二阶矩估计实现参数自适应更新，是深度学习训练的事实标准
aliases:
  - Adam 优化器
  - Adaptive Moment Estimation
prerequisites:
  - "[[SGD]]"
  - "[[Momentum]]"
  - "[[RMSprop]]"
tags:
  - optimization
  - foundations
created: 2026-03-01
updated: 2026-03-01T22:44
---

# Adam

Adam (Adaptive Moment Estimation) 是深度学习中最广泛使用的优化器。它结合了动量（Momentum）的加速效果和 RMSprop 的自适应学习率，为每个参数维护独立的学习率，使训练既快速又稳定。

## 动机

> [!intuition] 为什么需要 Adam？
>
> **SGD 的问题**：
> - 所有参数共享同一个学习率，但不同参数的梯度尺度差异巨大
> - 在稀疏梯度场景（如 NLP）中，罕见特征更新缓慢
> - 在病态曲率（ill-conditioned）问题中，不同方向的最优步长差异很大
>
> **已有方案的局限**：
> - **Momentum**：加速收敛，但仍然用全局学习率
> - **RMSprop**：自适应学习率，但没有动量的加速效果
>
> **Adam 的核心洞察**：
> 同时维护梯度的**一阶矩**（均值，提供动量）和**二阶矩**（未中心化的方差，提供自适应学习率），让每个参数都有"记忆"和"自适应能力"。

## 核心机制


> [!intuition] 直觉理解
>
> 想象你在山谷中寻找最低点：
> - **一阶矩 (m)**：记住"最近往哪个方向走"，提供惯性，避免被噪声干扰
> - **二阶矩 (v)**：记住"这个方向的梯度有多陡"，陡的方向小步走，平缓的方向大步走
> - **偏差修正**：刚开始时记忆不足，需要修正估计偏差
>
> 每个参数都有自己的"记忆"和"步长"，不再一刀切。

> [!math] 数学形式
>
> 给定损失函数 $L(\theta)$，Adam 的更新规则：
>
> $$
> \begin{align}
> m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(一阶矩估计)} \\
> v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(二阶矩估计)} \\
> \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \quad \text{(偏差修正)} \\
> \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \quad \text{(偏差修正)} \\
> \theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
> \end{align}
> $$
>
> **符号说明**：
> - $g_t = \nabla_\theta L(\theta_{t-1})$：当前梯度
> - $m_t$：梯度的指数移动平均（一阶矩，类似动量）
> - $v_t$：梯度平方的指数移动平均（二阶矩，类似 RMSprop）
> - $\beta_1, \beta_2$：衰减率，控制历史信息的权重
> - $\alpha$：学习率（通常 0.001）
> - $\epsilon$：数值稳定项（通常 $10^{-8}$），防止除零
>
> **为什么这个公式 make sense**：
> - 分子 $\hat{m}_t$：带动量的梯度方向
> - 分母 $\sqrt{\hat{v}_t}$：梯度的标准差，用于归一化步长
> - 效果：梯度大的参数 → 分母大 → 步长小；梯度小的参数 → 分母小 → 步长大

### 偏差修正的必要性

> [!warning] 为什么需要偏差修正？
>
> 初始化 $m_0 = 0, v_0 = 0$ 会导致早期估计偏向零：
>
> $$
> \mathbb{E}[m_t] = \mathbb{E}[g] \cdot (1 - \beta_1^t)
> $$
>
> 在 $t$ 很小时，$(1 - \beta_1^t) \ll 1$，导致 $m_t$ 严重低估真实均值。
>
> 偏差修正通过除以 $(1 - \beta_1^t)$ 抵消这个偏差：
>
> $$
> \mathbb{E}[\hat{m}_t] = \mathbb{E}[g]
> $$
>
> 随着 $t$ 增大，$\beta_1^t \to 0$，修正项趋近 1，影响消失。

## 为什么 Adam 如此流行

1. **开箱即用**：默认超参数（$\beta_1=0.9, \beta_2=0.999, \alpha=0.001$）在大多数任务上表现良好
2. **鲁棒性强**：对学习率不敏感，不需要精细调参
3. **适应稀疏梯度**：在 NLP、推荐系统等稀疏场景中表现优异
4. **计算高效**：只需维护两个额外的向量（$m, v$），内存开销小

> [!example] 典型应用
> - **Transformer 训练**：BERT、GPT 等几乎都用 Adam（或 [[AdamW]]）
> - **生成模型**：GAN、VAE 的训练
> - **强化学习**：策略网络和价值网络的优化

## 局限性

> [!warning] 边界条件
>
> **1. 泛化性能争议**：
> - 在某些视觉任务（如 ResNet on ImageNet）中，SGD + Momentum 的泛化性能优于 Adam
> - 原因：Adam 的自适应学习率可能导致过拟合（sharp minima）
>
> **2. 权重衰减的交互问题**：
> - 标准 Adam 中，L2 正则化与自适应学习率交互，导致权重衰减效果不一致
> - 解决方案：[[AdamW]] 将权重衰减与梯度更新解耦
>
> **3. 学习率调度的复杂性**：
> - Adam 的自适应性与学习率调度器（如 cosine annealing）可能产生冲突
> - 需要仔细调整两者的配合
>
> **4. 非凸优化的理论保证较弱**：
> - Adam 的收敛性证明需要较强的假设
> - 在某些病态问题中可能不收敛（已有反例）

## 对比

> [!comparison] Adam vs 其他优化器
>
> | 优化器 | 动量 | 自适应学习率 | 适用场景 | 泛化性能 |
> |--------|------|--------------|----------|----------|
> | **SGD** | ❌ | ❌ | 需要精细调参的任务 | 优秀 |
> | **[[Momentum]]** | ✅ | ❌ | 凸优化、视觉任务 | 优秀 |
> | **[[RMSprop]]** | ❌ | ✅ | RNN 训练 | 良好 |
> | **Adam** | ✅ | ✅ | 通用、稀疏梯度 | 良好 |
> | **[[AdamW]]** | ✅ | ✅ | Transformer 训练 | 优秀 |
>
> **选择建议**：
> - **默认选择**：Adam 或 AdamW（尤其是 NLP 任务）
> - **追求极致泛化**：SGD + Momentum（需要调参）
> - **权重衰减重要**：AdamW（而非 Adam + L2）

## 速查

> [!example] 关键参数
>
> **默认超参数**（Kingma & Ba, 2015）：
> - $\alpha = 0.001$（学习率）
> - $\beta_1 = 0.9$（一阶矩衰减率）
> - $\beta_2 = 0.999$（二阶矩衰减率）
> - $\epsilon = 10^{-8}$（数值稳定项）
>
> **PyTorch 实现**：
> ```python
> optimizer = torch.optim.Adam(
>     model.parameters(),
>     lr=1e-3,
>     betas=(0.9, 0.999),
>     eps=1e-8
> )
> ```
>
> **调参建议**：
> - 学习率：通常在 $[10^{-4}, 10^{-3}]$ 范围内
> - $\beta_1$：很少调整，保持 0.9
> - $\beta_2$：稀疏梯度场景可以调高到 0.999 或 0.9999

> [!interview] 面试视角
>
> **Q: Adam 为什么比 SGD 更容易训练？**
> A: Adam 为每个参数维护独立的自适应学习率，梯度大的参数自动用小步长，梯度小的参数用大步长，不需要手动调整全局学习率。同时结合动量加速收敛，减少了超参数调优的负担。
>
> **Q: 为什么需要偏差修正？**
> A: 因为 $m_0 = 0, v_0 = 0$ 的初始化导致早期的矩估计偏向零。偏差修正通过除以 $(1 - \beta^t)$ 抵消这个偏差，确保估计的无偏性。随着训练进行，$\beta^t \to 0$，修正项的影响消失。
>
> **Q: Adam 和 AdamW 的区别？**
> A: Adam 将 L2 正则化加到损失函数中，导致权重衰减与自适应学习率交互，效果不一致。[[AdamW]] 将权重衰减直接作用于参数更新（$\theta \leftarrow (1-\lambda)\theta - \alpha \cdot \text{update}$），与梯度更新解耦，是 Transformer 训练的标准选择。
>
> **Q: 什么时候 SGD 比 Adam 更好？**
> A: 在某些视觉任务（如 ImageNet 分类）中，SGD + Momentum 的泛化性能优于 Adam。原因是 Adam 的自适应学习率可能导致收敛到 sharp minima，泛化能力较弱。但这需要精细的学习率调度，工程成本更高。

## 相关概念

**前置知识**：
- [[SGD]] — 最基础的梯度下降
- [[Momentum]] — Adam 的一阶矩来源
- [[RMSprop]] — Adam 的二阶矩来源

**改进变体**：
- [[AdamW]] — 修正权重衰减的 Adam，Transformer 训练的标准选择
- [[Adafactor]] — 内存高效的 Adam 变体，用于超大模型
- [[LAMB]] — 大 batch 训练的 Adam 变体

**延伸阅读**：
- Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization"
- Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization" (AdamW)
