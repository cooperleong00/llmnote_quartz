---
type: concept
description: RL 训练中策略熵急剧下降导致探索能力丧失的现象，是 LLM RL 训练的核心挑战之一
aliases:
  - entropy collapse
  - 熵坍塌
  - 熵崩溃
prerequisites:
  - "[[Policy Gradient]]"
  - "[[PPO]]"
  - "[[GRPO]]"
tags:
  - post-training
  - rlhf
  - training-dynamics
created: 2026-01-27
updated: 2026-01-31T22:42
---

# Entropy Collapse

Entropy Collapse 是 RL 训练中策略熵（policy entropy）急剧下降的现象，导致 token 分布趋近于 one-hot，模型丧失探索能力。这是 LLM RL 训练（如 [[GRPO]]、[[PPO]]）的核心挑战之一，直接影响训练效率和最终性能。

## 动机

> [!intuition] 为什么 Entropy Collapse 是个问题？
>
> RL 训练的本质是**探索-利用权衡**：
> - **探索**：尝试不同的 token 序列，发现更好的解法
> - **利用**：强化已知的好策略
>
> 当熵过低时，模型只会重复已知的高概率 token，无法探索新的可能性。这就像一个学生只会用一种方法解题，即使这种方法不是最优的。

## 核心现象

### 熵消耗的时间分布

来自 [[250522617v1|Entropy Mechanism of RL]] 论文的关键发现：

> [!warning] 惊人的不均衡
> - 训练**前 1/12 步**就消耗了 **73% 的熵**，获得 **76% 的性能提升**
> - 剩余 **2/3 训练步骤**的收益边际递减
> - 这意味着大部分训练时间可能是低效的

### 性能-熵的指数关系

> [!math] 量化关系
>
> 性能 $R$ 和熵 $H$ 呈指数关系：
> $$R = -a \cdot \exp(H) + b$$
>
> 其中：
> - $a$：模型将熵转换为性能的**效率**（entropy-to-performance efficiency）
> - $b - a$：熵耗尽时的**理论最大性能**
> - $a, b$ 与算法无关，反映模型和数据的内在属性
> - 系数与模型大小呈 **log-linear** 关系

## 数学机制

### 熵变化的本质

> [!math] 熵变化公式
>
> 熵的变化可以表示为：
> $$\Delta H \approx -\text{Cov}(\log \pi(a), \Delta z_a)$$
>
> 其中 $z_a$ 是 token $a$ 的 logit。
>
> **解读**：
> - 当**高概率 token 的 logit 增加**时，熵下降
> - 当**低概率 token 的 logit 减少**时，熵下降
> - 两者都会让分布更加"尖锐"

### Natural Policy Gradient 下的熵变化

> [!math] NPG 熵变化
>
> 在 Natural Policy Gradient 下：
> $$\Delta H \approx -\eta \cdot \text{Cov}(\log \pi(a), A(a))$$
>
> 其中 $A(a)$ 是 advantage function，$\eta$ 是学习率。
>
> **关键洞察**：当高概率 token 恰好有高 advantage 时，熵会快速下降。

## Forking Tokens：并非所有 token 都平等

来自 [[250601939v2|80/20 Rule RLVR]] 论文的重要发现：

> [!intuition] Token 的熵分布
>
> 大多数 token 的熵很低（模型很确定），只有约 **20%** 是高熵 token。
> 这些高熵 token 被称为 **Forking Tokens**，是推理的"分叉点"。

### Forking Tokens 的功能

| 类型 | 功能 | 示例 |
|------|------|------|
| **Logic Connectors** | 连接推理步骤 | "wait", "however", "therefore" |
| **Hypothesis Introducers** | 引入新假设 | "suppose", "let's try", "alternatively" |
| **Structure Completers** | 填充已确定的步骤 | 低熵 token，如数学符号、常见词 |

> [!warning] 为什么 Forking Tokens 重要？
>
> Forking Tokens 决定了推理的**方向**。如果这些 token 的熵过低，模型就会陷入固定的推理模式，无法探索不同的解题路径。

## 正负样本对熵的影响

来自 [[250601347v2|Negative Sample Reinforcement]] 论文的分析：

> [!intuition] 正负样本的不对称效应
>
> - **正样本**（$r=1$）：增加采样 token 的 logit，从其他 token "剥夺"概率质量 → **降低熵**
> - **负样本**（$r=-1$）：降低采样 token 的 logit，概率质量重新分配给其他 token → **增加熵**

> [!insight] 关键发现
>
> **负样本天然有助于增加熵**。这解释了为什么纯负样本训练（如 NegGRPO）也能有效——它在惩罚错误答案的同时，保持了探索能力。

## 缓解策略

### 1. Clip-Higher（[[DAPO]]）

> [!definition] 解耦 clip 上下界
>
> 标准 PPO/GRPO 使用对称的 $\varepsilon$（如 0.2）。Clip-Higher 解耦上下界：
> - $\varepsilon_{\text{low}} = 0.2$（保持不变）
> - $\varepsilon_{\text{high}} = 0.28$（放宽上界）
>
> **效果**：让低概率的"探索" token 有更大的增长空间。

### 2. [[CISPO]]（[[250613585v1|MiniMax-M1]]）

> [!definition] Clipped Importance ratio as Scaled advantage weight for Policy Optimization
>
> 将 advantage 与 clipping 解耦：clip 后的 ratio 作为 advantage 的**权重**，而非直接相乘。
>
> **效果**：避免 clip 机制过度抑制低概率 token 的更新。

### 3. Entropy-Based Advantage（[[250614758v4|Exploration in LLM Reasoning]]）

> [!definition] 高熵 token 的 advantage bonus
>
> 直接给高熵 token 增加 advantage bonus：
> $$A'(a) = A(a) + \beta \cdot H_{\text{token}}$$
>
> **效果**：显式鼓励在 forking tokens 上保持探索。

### 4. Clip-Cov / KL-Cov（Entropy Mechanism）

> [!definition] 限制高协方差 token 的更新
>
> 根据 $\text{Cov}(\log \pi(a), A(a))$ 限制更新幅度。高协方差意味着该 token 对熵下降贡献大。
>
> **效果**：从源头控制熵的消耗速度。

### 5. 增加负样本权重

> [!intuition] 利用负样本的熵增效应
>
> 由于负样本天然增加熵，可以通过增加负样本的权重来平衡熵的下降。

### 6. KL Penalty + Reference Policy Reset（[[250524864v1|ProRL]]）

> [!definition] 显式 KL 惩罚 + 周期性重置
>
> - 添加显式的 [[KL Divergence]] 惩罚项
> - 周期性重置 reference policy 到当前 policy
>
> **效果**：防止 policy 偏离太远，保持一定的探索能力。

### 7. 高采样温度

> [!warning] 只能延迟，不能阻止
>
> 提高采样温度可以增加输出的多样性，但这只是**延迟** entropy collapse，不能从根本上解决问题。

## 统一视角

> [!insight] 所有方法的本质
>
> 所有缓解策略本质上都在处理 **forking tokens 的熵**：
>
> | 策略类型 | 方法 | 机制 |
> |----------|------|------|
> | **增加高熵 token 更新** | DAPO, CISPO, Entropy-Based Advantage | 让 forking tokens 获得更大的更新幅度 |
> | **限制低熵 token 更新** | Clip-Cov, KL-Cov | 减少 structure completers 的更新幅度 |
>
> 这两类方法理论上可以**结合使用**，从两个方向同时控制熵的变化。

## 面试要点

> [!interview] 常见问题
>
> **Q: 什么是 Entropy Collapse？为什么它是个问题？**
> A: RL 训练中策略熵急剧下降，token 分布趋近 one-hot。问题在于模型丧失探索能力，只会重复已知的高概率 token，无法发现更好的解法。
>
> **Q: 如何检测 Entropy Collapse？**
> A: 监控训练过程中的 policy entropy。如果熵在训练早期快速下降（如前 1/12 步消耗 73% 的熵），说明正在发生 collapse。
>
> **Q: DAPO 的 Clip-Higher 如何缓解 Entropy Collapse？**
> A: 标准 clip 对低概率 token 的增长限制过严。Clip-Higher 放宽上界（如从 1.2 到 1.28），让低概率的"探索" token 有更大的增长空间。
>
> **Q: 为什么负样本有助于缓解 Entropy Collapse？**
> A: 正样本增加采样 token 的概率，从其他 token 剥夺概率质量，降低熵。负样本相反，降低采样 token 的概率，概率质量重新分配，增加熵。

## 相关概念

- [[Policy Gradient]] — 理论基础，理解梯度如何影响 policy
- [[PPO]] — Clip 机制的来源
- [[GRPO]] — LLM RL 的主流算法，同样面临 entropy collapse
- [[DAPO]] — Clip-Higher 的来源
- [[KL Divergence]] — KL penalty 的数学基础
- [[Reward Hacking]] — 另一个 RL 训练的核心挑战

## 参考资料

- [[Entropy Collapse and Mitigation Strategies]] — 综述博客
- [[Clippings/Paper/250522617v1/250522617v1]] — The Entropy Mechanism of RL for Reasoning LLMs
- [[Clippings/Paper/250601939v2/250601939v2]] — Beyond the 80/20 Rule (Forking Tokens)
- [[Clippings/Paper/250601347v2/250601347v2]] — Negative Reinforcement
- [[Clippings/Paper/250614758v4/250614758v4]] — Reasoning with Exploration
- [[Clippings/Paper/250524864v1/250524864v1]] — ProRL
- [[Clippings/Paper/250613585v1/250613585v1]] — MiniMax-M1 (CISPO)