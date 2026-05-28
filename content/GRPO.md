---
description: 用 group relative advantage 替代 value function 的高效 RL 算法，显著降低 PPO 的内存开销
type: method
aliases:
  - Group Relative Policy Optimization
  - 组相对策略优化
prerequisites:
  - "[[PPO]]"
  - "[[Reward Model]]"
tags:
  - post-training
  - reinforcement-learning
  - rlhf
created: 2026-01-27
updated: 2026-04-05T18:22
---

# GRPO (Group Relative Policy Optimization)

组相对策略优化（GRPO, Group Relative Policy Optimization）是 [[PPO]] 的高效变体，通过**用组内相对 reward 替代 value function** 来估计 advantage，从而省去了与 policy model 同等规模的 Critic 网络，大幅降低内存和计算开销。

> [!paper] 论文出处
> Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024
> - 在 MATH benchmark 上将 7B 模型从 46.8% 提升到 51.7%
> - 接近 GPT-4 (52.9%) 和 Gemini Ultra (53.2%) 的水平

---

## 动机

> [!intuition] 为什么需要 GRPO？

[[PPO]] 在 [[RLHF]] 中的核心问题：

1. **内存开销巨大**：PPO 需要训练一个与 policy model 同等规模的 **value function (Critic)**
   - 对于 7B 模型，Critic 也是 7B 参数
   - 训练时需要同时维护 4 个模型：policy、reference、reward model、value function

2. **Value function 训练困难**：在 LLM 场景下，通常只在序列末尾给一个 reward（sparse reward），这使得逐 token 的 value 估计变得困难

3. **[[Reward Model]] 的比较本质**：Reward model 本身就是在**同一问题的不同输出之间做比较**训练的，这与 GRPO 的组内比较思想天然契合

---

## 核心机制

### 从 PPO 到 GRPO

> [!comparison] PPO vs GRPO

| 方面 | PPO | GRPO |
|------|-----|------|
| **Advantage 来源** | Value function + [[GAE]] | 组内 reward 的相对值 |
| **额外模型** | 需要 Critic（与 policy 同规模） | 不需要 |
| **KL 正则化** | 在 reward 中加 KL penalty | 直接在 loss 中加 KL divergence |
| **内存占用** | 高 | 低 |

### Group Relative Advantage

> [!definition] 核心思想
> 对每个问题 $q$，采样一组输出 $\{o_1, o_2, \ldots, o_G\}$，用**组内 reward 的均值和标准差**来归一化，得到相对 advantage。

**具体步骤**：

1. **采样**：对问题 $q$，从旧策略 $\pi_{\theta_{old}}$ 采样 $G$ 个输出
2. **打分**：用 reward model 对每个输出打分，得到 $\mathbf{r} = \{r_1, r_2, \ldots, r_G\}$
3. **归一化**：计算组内相对 advantage

$$
\hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}
$$

> [!intuition] 直觉理解
> - 不需要学习一个 value function 来估计"平均水平"
> - 直接用**同一问题的其他输出**作为 baseline
> - 这与 reward model 的训练方式（比较同一问题的不同回答）天然一致

### GRPO 目标函数

> [!math] 数学形式

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^G \left( \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t} \hat{A}_i, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right) \right]
$$

其中：
- $r_{i,t} = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}$ 是概率比（与 PPO 相同）
- $\hat{A}_i = \tilde{r}_i$ 是归一化后的 group relative advantage
- $\epsilon$ 是 clip 范围
- $\beta$ 是 KL 系数

**关键区别**：KL 正则化直接加在 loss 中，而非像 PPO 那样加在 reward 中。

### KL Divergence 估计

GRPO 使用无偏估计器：

$$
D_{KL}[\pi_\theta \| \pi_{ref}] = \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1
$$

这个估计器保证非负。

---

## 两种监督模式

GRPO 支持两种 reward 监督方式：

### Outcome Supervision

> [!definition] 结果监督
> 只在输出末尾给一个 reward，所有 token 共享同一个 advantage。

$$
\hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}
$$

- 简单，只需要 outcome reward model
- 适合答案可验证的任务（如数学）

### Process Supervision

> [!definition] 过程监督
> 在每个推理步骤末尾给 reward，advantage 是后续步骤 reward 的累加。

设 $r_i^{index(j)}$ 是第 $i$ 个输出第 $j$ 步的 reward：

$$
\hat{A}_{i,t} = \sum_{index(j) \geq t} \tilde{r}_i^{index(j)}
$$

- 需要 process reward model (PRM)
- 提供更细粒度的监督信号
- 实验表明 process supervision 效果更好

---

## 训练配置

> [!example] DeepSeekMath 的 GRPO 训练设置

根据 DeepSeekMath 论文的实现：

| 参数 | 值 | 说明 |
|------|-----|------|
| **Batch size** | 1024 | 每个 batch 包含 1024 个问题 |
| **每问题采样数 (G)** | 64 | 每个问题采样 64 个输出 |
| **每 batch 总输出数** | 65,536 | 1024 × 64 个序列 |
| 学习率 | 1e-6 | |
| KL 系数 ($\beta$) | 0.04 | |
| 最大长度 | 1024 tokens | |
| 训练数据 | 144K 问题 | GSM8K + MATH 格式 |

### 关键权衡

**组大小 G 的选择**：
- **G 太小**（如 G=8）：组内 baseline 估计不准确，advantage 计算方差大
- **G 太大**（如 G=128）：推理成本线性增长，训练效率低
- **G=64**：在准确性和效率之间的经验最优值

**计算开销**：
- 每个训练步需要对 65,536 个序列进行前向传播
- 这是 GRPO 的主要计算瓶颈（采样开销）
- 但相比 PPO，省去了 value function 的训练，节省约 50% 内存

---

## 迭代 RL

随着训练进行，旧的 reward model 可能无法有效监督新的 policy。GRPO 支持**迭代训练**：

1. 用当前 policy 采样，生成新的 reward model 训练数据
2. 持续训练 reward model（加入 10% 历史数据防止遗忘）
3. 更新 reference model 为当前 policy
4. 继续 GRPO 训练

> [!example] 实验结果
> 迭代 RL 在第一轮迭代时带来显著提升。

---

## 实验结果

### DeepSeekMath-RL 7B

| Benchmark | Instruct | RL (GRPO) | 提升 |
|-----------|----------|-----------|------|
| GSM8K | 82.9% | 88.2% | +5.3% |
| MATH | 46.8% | 51.7% | +4.9% |
| CMATH | 84.6% | 88.8% | +4.2% |
| MGSM-zh | 73.2% | 79.6% | +6.4% |

> [!intuition] 关键发现
> - GRPO 只用 GSM8K 和 MATH 的 chain-of-thought 数据训练
> - 但在**所有** benchmark 上都有提升，包括 out-of-domain 任务
> - 说明 RL 提升的是模型的**整体推理能力**，而非过拟合特定数据

### RL 为什么有效？

论文的分析揭示了一个重要发现：

> [!warning] RL 的本质作用
> RL 提升了 **Maj@K**（多数投票准确率），但**没有提升 Pass@K**（至少一次正确的概率）。
>
> 这说明：RL 的改进来自于**让输出分布更稳定**，而非提升模型的基础能力。换句话说，RL 是在"对齐"模型已有的能力，而非创造新能力。

---

## 统一范式

论文提出了一个统一视角来理解不同的训练方法：

$$
\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{(q,o) \sim \mathcal{D}} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} GC(q, o, t, \pi_{rf}) \nabla_\theta \log \pi_\theta(o_t|q, o_{<t}) \right]
$$

三个关键组件：
1. **Data Source** $\mathcal{D}$：训练数据来源（online vs offline）
2. **Reward Function** $\pi_{rf}$：奖励信号来源（rule vs model）
3. **Gradient Coefficient** $GC$：梯度系数（算法决定）

| 方法 | Data Source | Reward | 特点 |
|------|-------------|--------|------|
| SFT | 人工标注 | - | $GC = 1$ |
| RFT | offline 采样 | Rule | 只强化正确答案 |
| [[DPO]] | offline 采样 | Rule | pair-wise loss |
| Online RFT | online 采样 | Rule | 实时采样 |
| PPO | online 采样 | Model | 需要 value function |
| **GRPO** | online 采样 | Model | group relative advantage |

> [!intuition] 关键洞察
> - **Online > Offline**：Online RFT 显著优于 RFT
> - **Model > Rule**：GRPO 优于 Online RFT，因为可以根据 reward 大小差异化强化/惩罚
> - **Process > Outcome**：细粒度监督效果更好

---

## 局限性

> [!warning] GRPO 的局限

1. **仍需 Reward Model**：虽然省去了 value function，但仍需要训练 reward model
2. **采样开销**：每个问题需要采样 $G$ 个输出（论文中 $G=64$），推理成本高
3. **组大小敏感**：$G$ 太小会导致 baseline 估计不准
4. **Reward Model 泛化**：如果 reward model 泛化能力差，RL 可能只是稳定分布而非提升能力
5. **Token 级重要性采样的理论缺陷**：GRPO 在每个 token 位置应用重要性权重，但基于单样本无法执行有效的分布校正，引入高方差噪声，在大规模 MoE 模型上可能导致训练崩溃。[[GSPO]] 通过序列级重要性采样解决了这一问题
6. **优化偏差**：GRPO 的长度归一化和标准差归一化引入优化偏差，导致错误响应变长、token 效率下降。[[Dr. GRPO]] 通过移除这些归一化项修正了偏差

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: GRPO 相比 PPO 的核心改进是什么？**
> A: 用 group relative advantage 替代 value function。对每个问题采样一组输出，用组内 reward 的均值作为 baseline，省去了与 policy 同规模的 Critic 网络。
>
> **Q2: GRPO 的 advantage 怎么计算？**
> A: $\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$，即用组内 reward 的均值和标准差归一化。
>
> **Q3: GRPO 的 KL 正则化与 PPO 有什么不同？**
> A: PPO 在 reward 中加 KL penalty，GRPO 直接在 loss 中加 KL divergence。这样避免了 KL penalty 影响 advantage 计算。
>
> **Q4: 为什么 group relative 的方式与 reward model 契合？**
> A: Reward model 本身就是在同一问题的不同输出之间做比较训练的（Bradley-Terry model），GRPO 的组内比较与这种训练方式天然一致。
>
> **Q5: GRPO 的实验发现 RL 提升了什么？**
> A: 提升了 Maj@K 但没提升 Pass@K，说明 RL 主要是让输出分布更稳定（对齐），而非提升基础能力。

---

## 后续发展

- [[DAPO]] — GRPO 的改进版本，解决 entropy collapse 等问题
- [[GSPO]] — GRPO 的改进版本，用序列级重要性采样解决大规模 MoE 训练的稳定性问题
- [[Dr. GRPO]] — GRPO 的无偏变体，修正长度和标准差归一化偏差
- [[SAPO]] — 用平滑软门替代硬裁剪，提升稳定性和样本效率
- [[DGPO]] — 从概率梯度角度重新思考 soft clipping，解决 CISPO/GPPO 的梯度发散问题

## 延伸阅读

- [[Credit Assignment]] — 组内相对 advantage 是一种隐式的 credit assignment
- [[Context-Folding (2025)]] — FoldGRPO 扩展：为 context-folding agent 引入 process rewards

---

## 参考资料

- [[240203300v3|DeepSeekMath (2024)]] — GRPO 的原始论文
- [[Bradley-Terry Model]] — Reward model 的理论基础
