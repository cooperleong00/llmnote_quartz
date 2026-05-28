---
description: 多教师在线蒸馏范式，通过领域专家教师提供 token-level KL 奖励，解决 post-training 的能力失衡和学习效率问题
type: method
aliases:
  - Multi-Teacher On-Policy Distillation
  - 多教师在线蒸馏
prerequisites:
  - "[[SFT]]"
  - "[[GRPO]]"
  - "[[KL Divergence]]"
  - "[[Importance Sampling]]"
tags:
  - post-training
  - reinforcement-learning
  - distillation
created: 2026-01-27
updated: 2026-02-01T01:06
---

# MOPD (Multi-Teacher On-Policy Distillation)

多教师在线蒸馏（MOPD, Multi-Teacher On-Policy Distillation）是一种新的 post-training 范式，是 [[Knowledge Distillation|知识蒸馏]] 在 LLM 对齐领域的创新应用。通过**让学生模型从自身分布采样，接收来自多个领域专家教师的 token-level KL 散度奖励**，MOPD 解决了传统 [[RLHF]] 中的**能力失衡**（see-saw 效应）和**学习效率低下**问题。

> [!paper] 论文出处
> LLM-Core Xiaomi, "MiMo-V2-Flash Technical Report", 2025
> - 309B MoE 模型（15B 激活参数）达到 SOTA 代码智能体性能
> - SWE-Bench Verified 73.4%，超越所有开源模型
> - 在数学、代码、推理等多领域同时保持峰值能力

---

## 动机

> [!intuition] 为什么需要 MOPD？

传统 post-training 面临两个核心挑战：

### 1. 能力失衡（Capability Imbalance）

**See-saw 效应**：提升一个能力往往导致其他能力退化。

- 训练数学推理 → 代码能力下降
- 训练代码智能体 → 通用对话能力退化
- 多任务联合训练 → 各领域都达不到峰值

> [!warning] 现有方法的局限
> - **参数合并**（Model Merging）：不同领域的参数更新可能相互冲突，合并后能力损失
> - **顺序训练**（Sequential Training）：后训练的任务会覆盖前面学到的能力（灾难性遗忘）
> - **多任务联合训练**：难以在所有领域同时达到峰值

### 2. 学习效率低下

- **离线蒸馏**：从静态数据集学习，存在 exposure bias 和 distribution mismatch
- **单教师蒸馏**：无法整合多个领域专家的知识
- **稀疏奖励**：只在序列末尾给 reward，credit assignment 困难

---

## 核心机制

### 三阶段框架

MOPD 采用三阶段框架，将专家知识整合与能力保持解耦：

```
Stage 1: SFT
    ↓
Stage 2: Domain-Specialized Training (多个独立 RL)
    ↓
Stage 3: Multi-Teacher On-Policy Distillation
```

#### Stage 1: [[SFT|Supervised Fine-Tuning]]

建立基础的指令遵循能力，为后续 RL 提供稳定的起点。

#### Stage 2: Domain-Specialized Training

训练多个**领域专家教师模型**，每个教师通过独立的 RL 优化达到其领域的峰值性能：

| 类型 | 领域 | 训练方式 |
|------|------|----------|
| **Agentic** | Search Agent | 大规模 RL |
| **Agentic** | Code Agent | 大规模 RL |
| **Agentic** | General Tool Use | 大规模 RL |
| **Non-agentic** | Math Reasoning | RL |
| **Non-agentic** | General Reasoning | RL |
| **Non-agentic** | Safety Alignment | RL/SFT |

> [!intuition] 关键洞察
> 每个教师只需要在**自己的领域**做到最好，不需要考虑其他领域的能力保持。这大大简化了单个教师的训练目标。

#### Stage 3: Multi-Teacher On-Policy Distillation

学生模型从**自身分布**采样，接收来自领域教师的 **token-level [[KL Divergence|KL 散度]]奖励**。

---

### 技术公式

> [!math] Reverse KL Divergence Loss

MOPD 使用 reverse KL divergence 作为蒸馏目标：

$$
\mathcal{L}_{\text{reverse-KL}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, y_t \sim \pi_{\theta}(\cdot | x, y_{< t})} \log \frac{\pi_{\text{domain}_x}(y_t | x, y_{< t})}{\pi_{\theta}(y_t | x, y_{< t})}
$$

其中：
- $\pi_\theta$：学生策略（训练目标）
- $\pi_{\text{domain}_x}$：根据 prompt $x$ 的领域选择的教师策略
- $\mathcal{D}$：prompt 分布

**梯度形式**：

$$
\nabla_{\theta} \mathcal{L}_{\text{reverse-KL}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, y_t \sim \pi_{\theta}} \left[ \log \frac{\pi_{\text{domain}_x}(y_t | x, y_{< t})}{\pi_{\theta}(y_t | x, y_{< t})} \nabla_{\theta} \log \pi_{\theta}(y_t | x, y_{< t}) \right]
$$

> [!intuition] 为什么用 Reverse KL？
> - **Forward KL** $D_{KL}(\pi_{\text{teacher}} \| \pi_{\text{student}})$：学生会 mode-covering，试图覆盖教师的所有模式
> - **Reverse KL** $D_{KL}(\pi_{\text{student}} \| \pi_{\text{teacher}})$：学生会 mode-seeking，专注于教师的高概率区域
>
> Reverse KL 更适合蒸馏场景，因为我们希望学生学习教师的核心能力，而非分散注意力。

### MOPD Surrogate Loss

结合 [[Importance Sampling]] 处理 training-inference 差异：

$$
\mathcal{L}_{\text{MOPD}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, y \sim \mu_{\theta}(\cdot|x)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} w_t \hat{A}_{\text{MOPD},t} \log \pi_{\theta}(y_t|x, y_{< t}) \right]
$$

其中：
- $\mu_\theta$：采样策略（inference engine）
- $\pi_\theta$：训练策略（training engine）
- $w_t$：重要性权重，带截断

**重要性权重**：

$$
w_{t}(\theta) = \begin{cases}
\text{sg}\left[\frac{\pi_{\theta}(y_{t}|x,y_{< t})}{\mu_{\theta}(y_{t}|x,y_{< t})}\right], & \epsilon_{\text{low}} \leq \frac{\pi_{\theta}}{\mu_{\theta}} \leq \epsilon_{\text{high}} \\
0, & \text{otherwise}
\end{cases}
$$

**Token-level Advantage**：

$$
\hat{A}_{\text{MOPD},t} = \text{sg}\left[\log\frac{\pi_{\text{domain}_x}(y_t|x,y_{< t})}{\pi_{\theta}(y_t|x,y_{< t})}\right]
$$

> [!intuition] 直觉理解
> - $\hat{A}_{\text{MOPD},t} > 0$：教师比学生更倾向于生成这个 token → 强化
> - $\hat{A}_{\text{MOPD},t} < 0$：学生比教师更倾向于生成这个 token → 惩罚
>
> 这是一个 **dense, token-level** 的监督信号，比只在序列末尾给 reward 的方式更高效。

### 结合 Outcome Reward

MOPD 可以与 ORM（Outcome [[Reward Model]]）结合，获得更全面的监督：

$$
\hat{A}_{\text{MOPD},t} = \text{sg}\left[\log\frac{\pi_{\text{domain}_x}(y_t|x,y_{< t})}{\pi_{\theta}(y_t|x,y_{< t})}\right] + \alpha \hat{A}_{\text{ORM}}
$$

其中 $\hat{A}_{\text{ORM}}$ 可以用 [[GRPO]] 的方式计算。

---

## 核心优势

> [!comparison] 与现有方法对比

### vs 参数合并（Model Merging）

| 方面 | 参数合并 | MOPD |
|------|----------|------|
| **能力保持** | 合并后能力损失 | 保留所有教师的峰值能力 |
| **冲突处理** | 参数冲突难以解决 | 通过 on-policy 学习自然整合 |
| **灵活性** | 合并后难以调整 | 可以动态选择教师 |

### vs 离线蒸馏（Offline Distillation）

| 方面 | 离线蒸馏 | MOPD |
|------|----------|------|
| **分布匹配** | Exposure bias | On-policy 避免 distribution mismatch |
| **数据效率** | 需要大量静态数据 | 从自身分布采样，更高效 |
| **适应性** | 固定数据集 | 随学生进化动态调整 |

### vs 单教师蒸馏

| 方面 | 单教师 | MOPD |
|------|--------|------|
| **能力覆盖** | 单一领域 | 多领域专家知识整合 |
| **扩展性** | 难以扩展 | 模块化，易于添加新教师 |

---

## 三大特性

### 1. Effective and Efficient

- **保留峰值能力**：学生可以达到或超过每个领域最强教师的水平
- **Dense Reward**：Token-level KL 奖励提供稳定的 credit assignment
- **On-policy**：从自身分布采样，避免 exposure bias

### 2. Modular and Scalable

- **教师灵活性**：教师可以是 RL 模型、SFT 模型，甚至学生自身
- **解耦设计**：新教师可以独立训练，无需重构整个 pipeline
- **与 ORM 兼容**：可以无缝结合现有的 outcome reward model

### 3. Iterative Co-Evolution

```
Student → Domain RL → Stronger Teachers → MOPD → Better Student → ...
```

- 蒸馏后的学生可以重新进入 Stage 2，训练更强的教师
- 更强的教师为下一代学生提供更高质量的监督
- 形成**自我强化的改进循环**

---

## 实验结果

> [!example] MiMo-V2-Flash MOPD 效果

| Benchmark | Student Before | Best Teacher | Student After | Δ |
|-----------|----------------|--------------|---------------|---|
| AIME 2025 | 89.3 | 93.9 (RL) | **94.1** | +0.2 |
| HMMT Feb. 2025 | 76.9 | 82.6 (RL) | **84.4** | +1.8 |
| LiveCodeBench | 77.5 | 82.6 (RL) | **83.2** | +0.6 |
| SWE-Bench Verified | 67.8 | 74.2 (RL) | 73.4 | -0.8 |
| τ²-Bench | 75.9 | 79.6 (RL) | **80.3** | +0.7 |

> [!intuition] 关键发现
> - 学生在多数领域**超过**最强教师
> - 即使教师是学生自身（Self），MOPD 也能带来提升
> - 说明 on-policy 蒸馏本身就有正则化效果

---

## 局限性

> [!warning] MOPD 的局限

1. **计算开销**：需要同时运行多个教师模型进行 logit 计算
2. **教师质量依赖**：学生的上限受限于教师的能力
3. **领域划分**：需要预先定义领域边界，prompt 路由可能不完美
4. **训练复杂度**：三阶段框架比端到端训练更复杂
5. **部分领域退化**：如 Creative Writing 在 MOPD 后略有下降（-3.9）

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: MOPD 解决了什么问题？**
> A: 解决了 post-training 中的两个核心问题：(1) 能力失衡（see-saw 效应），提升一个能力导致其他能力退化；(2) 学习效率低下，现有方法无法充分利用多个专家模型的知识。
>
> **Q2: MOPD 的三个阶段分别做什么？**
> A: Stage 1 是 SFT，建立基础指令遵循能力；Stage 2 是领域专家训练，通过独立 RL 训练多个领域教师；Stage 3 是多教师在线蒸馏，学生从自身分布采样，接收教师的 token-level KL 奖励。
>
> **Q3: 为什么用 reverse KL 而不是 forward KL？**
> A: Reverse KL 是 mode-seeking 的，让学生专注于教师的高概率区域，更适合蒸馏场景。Forward KL 是 mode-covering 的，会让学生分散注意力试图覆盖教师的所有模式。
>
> **Q4: MOPD 相比参数合并的优势是什么？**
> A: 参数合并会导致能力损失，因为不同领域的参数更新可能冲突。MOPD 通过 on-policy 学习自然整合多个教师的知识，保留所有教师的峰值能力。
>
> **Q5: MOPD 的 token-level advantage 是什么？**
> A: $\hat{A}_t = \log \frac{\pi_{\text{teacher}}}{\pi_{\text{student}}}$，表示教师和学生在该 token 上的概率比。正值表示教师更倾向于生成该 token，应该强化；负值表示应该惩罚。

---

## 参考资料

- [[260102780v2|MiMo-V2-Flash Technical Report (2025)]] — MOPD 的原始论文

