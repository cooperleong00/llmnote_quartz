---
type: method
description: 通过 Trim-Elect-Merge 三步法解决 model merging 中的参数干扰问题，消除冗余参数和符号冲突
aliases:
  - TIES
  - TRIM, ELECT SIGN & MERGE
prerequisites:
  - "[[Model Merging]]"
tags:
  - fine-tuning
  - efficiency
  - multi-task
created: 2026-02-11
updated: 2026-02-11
---

# TIES-Merging

TIES-Merging（TRIM, ELECT SIGN & MERGE）是一种 [[Model Merging]] 方法，专门解决合并多个微调模型时参数之间的**干扰（interference）**问题。核心思路是：在合并前先清理噪声、统一方向，再有选择地合并——而非像 [[Model Merging#2. Task Arithmetic|Task Arithmetic]] 那样无差别地叠加所有 task vector。

> [!paper] 论文出处
> Yadav et al., "Resolving Interference When Merging Models", NeurIPS 2023
> GitHub: https://github.com/prateeky2806/ties-merging

## 动机：为什么直接合并会失败？

[[Model Merging]] 的基本思路是将多个任务的 task vector 叠加到预训练模型上。但随着合并的模型数量增加，性能下降越来越严重。TIES-Merging 的作者发现这背后有两个具体的干扰来源：

**干扰源 1：冗余参数的稀释**

微调过程中，大量参数虽然发生了变化，但对任务性能几乎没有影响——论文实验表明，只保留 top-20% 幅度最大的参数就能维持原始性能。问题在于：当某个参数对模型 A 很重要、但对模型 B/C 是冗余的，直接平均会让 B/C 的噪声值**稀释**掉 A 的有效信号。

**干扰源 2：符号冲突的抵消**

不同任务可能需要同一个参数往**相反方向**移动——模型 A 需要 +0.5，模型 B 需要 -0.3。直接相加得到 +0.2，两个任务的信号都被削弱了。论文发现即使只合并 2 个模型也存在符号冲突，且冲突比例随模型数量增加而上升。

> [!intuition] 核心洞察
> 这两种干扰的共同效果是**压缩合并后的参数幅度**，让有意义的更新被噪声和冲突淹没。TIES-Merging 的策略是：先去噪、再统一方向、最后只合并一致的信号。

## 核心方法：三步流程

TIES-Merging 基于 task vector 的概念：对于任务 $t$，task vector 定义为 $\tau_t = \theta_{\text{ft}}^t - \theta_{\text{init}}$，即微调参数相对于预训练参数的变化量。

### Step 1: Trim（修剪冗余参数）

对每个 task vector $\tau_t$，只保留幅度最大的 top-$k$% 参数，其余重置为零：

$$
\hat{\tau}_t = \text{keep\_topk\_reset\_rest\_to\_zero}(\tau_t, k)
$$

这一步的直觉：微调中大部分参数变化是噪声，只有少数参数承载了真正的任务信息。丢弃小幅度变化不会损害单任务性能，却能防止它们在合并时干扰其他模型的有效参数。

典型设置：$k = 20$，即只保留 20% 的参数。

### Step 2: Elect Sign（选举符号方向）

对修剪后的 task vector，为每个参数位置 $p$ 选择一个统一的符号方向：

$$
\gamma_m^p = \text{sgn}\left(\sum_{t=1}^{n} \hat{\tau}_t^p\right)
$$

做法是将所有模型在该参数上的值求和，取其符号——本质上是一种**加权多数投票**，幅度大的模型有更大的话语权。这确保了合并后每个参数都有一个明确的、一致的更新方向。

### Step 3: Disjoint Merge（不相交合并）

对每个参数 $p$，只保留符号与选举结果一致的模型值，取它们的均值：

$$
\mathcal{A}^p = \{t \in [n] \mid \hat{\gamma}_t^p = \gamma_m^p\}, \quad \tau_m^p = \frac{1}{|\mathcal{A}^p|} \sum_{t \in \mathcal{A}^p} \hat{\tau}_t^p
$$

关键细节：均值只在**符号一致的非零值**上计算，零值（被 Trim 掉的）和符号不一致的值都被排除。这避免了反方向的值相互抵消。

最终合并模型：

$$
\theta_m = \theta_{\text{init}} + \lambda \cdot \tau_m
$$

其中 $\lambda$ 是缩放超参数，与 Task Arithmetic 中的作用相同。

## 消融实验的关键发现

论文对各组件进行了消融（T5-base / (IA)$^3$ 设置）：

| 配置 | T5-base | (IA)$^3$ |
|------|---------|----------|
| 完整 TIES-Merging | 74.5 | 70.7 |
| - Trim | 73.0 | 70.6 |
| - Elect | 73.1 | 69.6 |
| - Disjoint Mean | 72.6 | 67.5 |
| - Scale ($\lambda$) | 72.0 | 65.5 |

每个组件都有贡献，其中 **Disjoint Mean** 和 **Scale** 的影响最大。这说明"只合并一致方向的参数"和"控制合并强度"是性能的关键。

另一个重要发现：使用 oracle 多任务模型的符号向量进行 TIES-Merging，性能可以达到 72%，接近多任务训练的 73.1%。这暗示**正确的符号方向**是 model merging 成功的核心因素。

## 局限性

> [!warning] 边界条件
> - **top-$k$ 的选择**：$k$ 是全局阈值，但不同层、不同任务的参数稀疏度可能不同，统一的 $k$ 未必最优
> - **多数投票的假设**：Elect Sign 假设"多数方向是正确方向"，但当任务数量少或任务差异大时，多数不一定代表正确
> - **仍需超参数调优**：$k$ 和 $\lambda$ 都需要验证集来选择，这在某些场景下可能不可用
> - **未考虑参数间的相关性**：每个参数独立处理，忽略了参数之间的结构性关系

> [!comparison] 与相关方法的对比
> - 相比 Task Arithmetic：TIES 额外处理了冗余和符号冲突，在所有实验设置中都优于 Task Arithmetic
> - 相比 Fisher Merging：不需要计算 Fisher Information Matrix（需要数据和梯度），计算成本更低
> - 相比 [[231103099v3|DARE]]：DARE 通过随机丢弃 + 重缩放来稀疏化，TIES 通过幅度排序来稀疏化；两者可以结合使用（DARE 作为预处理 + TIES 做合并）

> [!interview] 面试视角
> **Q: TIES-Merging 解决了 model merging 的什么问题？**
> A: 解决了两种参数干扰：(1) 冗余参数稀释有效信号，(2) 不同模型对同一参数的符号冲突导致相互抵消。通过 Trim-Elect-Merge 三步法，先去噪、再统一方向、最后选择性合并。
>
> **Q: 为什么 Trim 只保留 top-20% 就够了？**
> A: 微调产生的参数变化极度冗余——这与 [[Model Merging#核心概念：Delta Parameters|delta parameters 的稀疏性]]一致。大部分参数变化对任务性能没有实质影响，保留 top-20% 就能维持原始性能。

## 延伸阅读

**后续发展**：
- [[231103099v3|DARE]] — 另一种处理 delta parameter 冗余的方法，可与 TIES 结合使用
- [[Model Merging]] — Model Merging 领域的综合笔记，包含各方法对比
