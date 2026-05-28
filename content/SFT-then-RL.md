---
type: method
description: SFT-then-RL 先用高质量推理轨迹建立可解题策略，再用可验证奖励强化已有推理能力，是训练 reasoning model 的高效两阶段 recipe
aliases:
  - SFT then RL
  - SFT→RL
  - SFT followed by RL
  - 先 SFT 后 RL
prerequisites:
  - "[[SFT]]"
  - "[[GRPO]]"
  - "[[On-Policy vs Off-Policy]]"
  - "[[Gradient Accumulation]]"
tags:
  - post-training
  - reasoning
  - optimization
created: 2026-05-01
updated: 2026-05-01T12:53
---

# SFT-then-RL

SFT-then-RL 是训练 LLM reasoning model 的两阶段方法：先用 [[SFT]] 在高质量推理轨迹上建立基本解题能力，再用 [[RLVR]] 或 [[GRPO]] 在可验证奖励上强化已经出现的正确推理策略。它的核心价值在于把“获得能力”和“强化能力”分成两个阶段，使 RL 从一开始就能看到足够多的正奖励样本。

> [!paper] 论文出处
> Limozin et al., "SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning", 2026。标准化 clipping 见 [2604.23747](Clippings/Paper/2604.23747/2604.23747.md)。
>
> 这篇论文的主要结论是：多篇 [[Mixed-Policy Methods]] 论文使用了被训练框架 bug 或弱超参数压低的 SFT baseline；修正后，标准 SFT→RL 在数学推理上强于被评估的 mixed-policy 方法，并且训练 FLOPs 更低。

## 动机

Reasoning RL 的难点来自稀疏奖励：模型在初始阶段很少生成完全正确的解答，verifier 给出的正奖励样本不足，RL 难以稳定改进。[[SFT]] 用 expert chain-of-thought demonstrations 把模型推到“能经常解出题”的区域；后续 RL 再根据 verifier reward 调整策略，把已有推理模式朝更高成功率压实。

> [!intuition] 直觉理解
> SFT 给模型提供可模仿的完整解题路径，RL 给模型提供可验证的选择压力。先完成 SFT，RL rollout 中正确答案的比例会显著提高，训练信号也随之变密。

[[Mixed-Policy Methods]] 试图在同一个训练阶段内混合 off-policy expert demonstrations 和 on-policy rollouts。这个方向的动机合理：expert traces 能覆盖当前模型还不会的问题，on-policy RL 能优化模型自己的采样分布。SFT-then-RL 的经验优势来自阶段分工更清晰：SFT 先集中处理知识注入和格式学习，RL 再集中处理策略改进和奖励优化。

## 核心机制

### 第一阶段：SFT 建立可解题策略

在 reasoning 场景中，SFT 数据通常是 `(problem, expert reasoning trace, answer)`。模型通过 [[Cross-Entropy Loss|cross-entropy loss]] 学习完整推理轨迹，包括问题拆解、计算步骤、验证答案和输出格式。

正确的 SFT loss 应该对所有有效 response tokens 做全局平均：

$$
\mathcal{L}_{\text{SFT}}
= - \frac{\sum_{i,t} m_{i,t}\log \pi_\theta(y_{i,t}\mid x_i, y_{i,<t})}{\sum_{i,t} m_{i,t}}
$$

其中 $m_{i,t}$ 是 response-token mask，用来排除 prompt tokens 和 padding tokens。这个写法强调一个工程细节：当不同样本的 response length 差异很大时，loss 需要先累加 token-level loss 和 token count，再做除法。

### 第二阶段：RL 强化可验证推理

SFT 之后，训练数据可以从“示范轨迹”切换成“题目 + 可验证答案”。模型对每个 prompt 生成多个 rollouts，verifier 判断答案是否正确，[[GRPO]] 这类方法再用组内相对 reward 更新 policy。

这一阶段的目标包括：

- 提高正确答案概率：让能得分的推理路径占更高概率质量。
- 修剪冗余推理：论文观察到 SFT 初始化后的回答较长，RL 会逐步压缩无效步骤。
- 保持策略稳定：SFT checkpoint 已经有较高初始 reward，RL 主要沿着已有能力继续优化。

## 为什么两阶段有效

SFT-then-RL 的效率来自 reward density。若模型一开始就能在相当比例的 rollouts 中得到正奖励，RL 的每一步更新都能基于有效差异进行比较；若模型很少答对，rollout reward 大量为零，RL 需要先等待模型偶然探索到正确解法。

在论文的 Llama-3.1-8B 实验中，这个差异很明显：经过 SFT 的模型进入 RL 时训练 reward 约为 60%，随后继续上升；LUFFY 和 ReLIFT 这类 mixed-policy 方法 500 步后仍低于 30%。对 Qwen2.5-Math-7B，SFT→RL 的 RL 阶段一开始 reward 已经超过 80%，mixed-policy 方法没有达到这个起点水平。

> [!comparison] 与 mixed-policy training 的区别
>
> | 维度 | SFT-then-RL | [[Mixed-Policy Methods]] |
> |---|---|---|
> | 训练组织 | SFT 后接 RL | 同一阶段混合或交替 SFT/RL 信号 |
> | 监督信号 | 先用 dense demonstration loss，再用 verifier reward | off-policy demonstrations 与 on-policy rollouts 同时参与 |
> | 早期 RL reward | SFT 后 reward 已经较高 | 初始 policy 弱时 reward 稀疏 |
> | 主要风险 | SFT baseline 实现错误会误导后续比较 | off-policy traces 和 on-policy rollouts 的分布差异会稀释学习信号 |
> | 适用判断 | 有高质量 demonstrations 和可靠 verifier 时优先考虑 | 需要动态补齐 hard problems 或在线 curriculum 时值得测试 |

## 工程陷阱：SFT baseline 很容易被低估

这篇论文最有价值的部分在于提醒：reasoning post-training 的方法比较非常依赖 SFT baseline 的正确性。一个被压低的 SFT baseline 会让后续方法看起来有额外收益。

### CPU-offloaded optimizer 与 gradient accumulation

论文发现 DeepSpeed [[ZeRO]] Stage 1/2 在 CPU-offloaded Adam 配置下存在一个 bug：使用 [[Gradient Accumulation|gradient accumulation]] 时，只有第一个 micro-batch 的梯度被复制到 CPU optimizer，后续 micro-batches 的梯度没有进入最终 optimizer step。结果是有效 batch 被缩小，梯度范数被压低，SFT 学习不足。

该问题影响使用相关 DeepSpeed 路径的 OpenRLHF、TRL 和 Llama-Factory 配置。论文使用 GPU-resident DeepSpeed optimizer 和 [[FSDP]]/verl baseline 做交叉验证，修复后 loss 曲线和 gradient norm 才与独立实现对齐。

### Loss aggregation

第二个问题来自 per-mini-batch mean 的平均。SFT 中每个样本的 response token 数不同，各 rank 和各 mini-batch 的有效 token count 也不同；直接平均多个 mini-batch 的 mean loss 会让短 response 的 mini-batch 得到过高权重。

正确做法是跨 mini-batches 和 data-parallel ranks 聚合 token-level loss sum 与有效 token count，再计算全局 mean。论文中这个修复带来的分数提升小于 optimizer bug，但能明显降低 loss variability。

## 实验结论

在 Qwen2.5-Math-7B 上，修正后的 SFT baseline 已经达到 52.2 Avg ID，超过 LUFFY、ReLIFT 和 Prefix-RFT；SFT→RL 达到 57.0 Avg ID，比最佳 mixed-policy baseline SRFT 高 3.8 分。OOD 任务上，SFT→RL 达到 59.9 Avg OOD，整体泛化仍然稳健。

在 Llama-3.1-8B 上，SFT→RL 的优势更大：修正后的 SFT baseline 为 33.9 平均分，SFT→RL 为 43.7；被比较的 mixed-policy 方法最高为 HPT 的 21.5。这个结果说明，当 base model 缺少数学预训练基础时，SFT 的 bootstrap 作用更关键。

论文还测试了短 RL schedule：SFT→RL 只跑 50 个 RL steps 时，Qwen2.5-Math-7B 仍达到 55.6 Avg ID 和 59.2 Avg OOD，训练 FLOPs 为 $3.63\times 10^{19}$，低于 LUFFY 的 $6.65\times 10^{19}$ 和 ReLIFT 的 $8.76\times 10^{19}$。

## 边界条件

> [!warning] 使用边界
> - 当前证据主要来自数学推理、Qwen2.5-Math-7B 和 Llama-3.1-8B；代码、通用推理和更大模型规模仍需要单独验证。
> - SFT-then-RL 依赖高质量 demonstrations 和可靠 verifier。若示范数据噪声大，SFT 会把错误推理模式写入初始 policy。
> - 论文推翻的是被评估 mixed-policy 方法相对弱 baseline 的公开比较；在正确 SFT checkpoint 之上继续使用 mixed-policy stage，仍可能带来增益，需要新的实验判断。
> - ReLIFT 和 HPT 的部分 SFT 配置不可完全复现，相关结论依赖论文作者能公开更多训练细节。

## 面试要点

> [!interview] 为什么 SFT-then-RL 在 reasoning 任务上高效？
> SFT 先把模型带到能产生正确 rollouts 的区域，RL 随后能从 verifier 得到足够多的正奖励信号。这样 RL 的更新主要用于强化和压缩已有推理策略，训练效率通常高于从弱 policy 同时学习知识和优化奖励。

> [!interview] 评估 mixed-policy 方法时为什么要重点审计 SFT baseline？
> Mixed-policy 方法的收益通常通过和 SFT 或 SFT→RL baseline 比较得到。若 baseline 因 framework bug、loss aggregation 或超参数选择被压低，方法增益会被系统性放大。

> [!interview] SFT loss aggregation 的正确实现是什么？
> 对所有有效 response tokens 累加 cross-entropy loss 和 token count，在所有 micro-batches 与 data-parallel ranks 聚合后做一次除法。逐 mini-batch 求 mean 再平均会在 response length 不均时改变样本权重。
