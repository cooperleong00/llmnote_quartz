---
type: paper
description: 首个 RL 缩放规律的系统研究，提出 sigmoid 缩放曲线和 ScaleRL 方法，实现 10 万 GPU 小时的可预测训练
aliases:
  - ScaleRL Paper
  - The Art of Scaling RL
  - ScaleRL
prerequisites:
  - "[[RLHF]]"
  - "[[PPO]]"
  - "[[GRPO]]"
tags:
  - rlhf
  - post-training
  - scaling
  - optimization
created: 2026-03-03
updated: 2026-03-29T16:48
---

# The Art of Scaling RL Compute for LLMs (2025)

> [!paper] 论文信息
> - **标题**: The Art of Scaling Reinforcement Learning Compute for LLMs
> - **作者**: Devvrit Khatri, Lovish Madaan, Rishabh Tiwari 等（Meta, UT Austin, UCL 等）
> - **年份**: 2025
> - **arXiv**: [2501.13786](https://arxiv.org/abs/2501.13786)

这篇论文用 40 万 GPU 小时做了 RL 缩放规律的系统实验，提出 sigmoid 缩放曲线和 ScaleRL 方法。实际验证中，从前 5 万小时拟合的曲线能准确外推到 10 万小时的性能（误差 ±0.02）。

## 为什么需要 RL 缩放规律？

Pre-training 已经有成熟的 [[Scaling Laws]]（power law），研究者可以在小规模实验中预测大规模训练的效果。但 RL 领域缺乏这样的框架，导致：

1. 算法评估成本高：不知道一个方法是否可扩展，只能跑满全部计算预算才能判断
2. 小规模结果不可信：小规模表现好的方法，大规模可能更差（"苦涩教训"）
3. 不同论文用不同规模实验，难以公平比较

这篇论文想为 RL 建立类似 pre-training 的可预测框架：在 8k GPU 小时拟合曲线，外推到 100k GPU 小时预测最终性能，通过比较曲线参数来评估算法改进。

## Sigmoid 缩放曲线

### 数学形式

$$R_C - R_0 = (A - R_0) \times \frac{1}{1 + (C_{\text{mid}}/C)^B}$$

其中：
- $R_C$：使用计算量 $C$ 后的性能（如 pass rate）
- $R_0$：初始性能（训练开始时）
- $C$：训练计算量（GPU 小时）
- **A**（Asymptotic Performance）：渐近性能上限
- **B**（Scaling Exponent）：计算效率指数
- **C_mid**：达到一半增益所需的计算量

> [!intuition] 为什么用 sigmoid 而非 power law？
>
> Pre-training 用 power law（$L \propto C^{-\alpha}$）拟合 loss，loss 是无界的。RL 的性能指标（如 pass rate）有界（0-1 之间），需要能捕捉饱和效应的曲线。
>
> Sigmoid 曲线的三个阶段：
> 1. 早期缓慢增长：模型还在学习基础能力，计算效率低
> 2. 中期快速提升：进入高效缩放区间
> 3. 后期趋于饱和：接近性能上限，边际收益递减
>
> 实验中 sigmoid 拟合比 power law 更稳定，尤其在大规模外推时。

### 三个参数的实践意义

| 参数 | 含义 | 影响因素 | 实践意义 |
|------|------|----------|----------|
| A | 性能天花板 | Loss 类型、batch size、FP32 precision | 决定方法的最终潜力 |
| B | 计算效率 | Loss aggregation、normalization、curriculum | 决定达到天花板的速度 |
| C_mid | 中点计算量 | 与 B 共同决定效率 | 预算规划的参考点 |

> [!warning] 常见误区
> - 误区：B 越大越好
> - 正解：B 和 A 需要综合考虑。高 B 但低 A 的方法可能早期表现好，但最终性能差（"苦涩教训"）
>
> 评估算法时先看 A（天花板够不够高），再看 B（达到天花板的效率）。根据计算预算选择：预算有限时优先 B，预算充足时优先 A。

## 实验发现

### 发现 1：性能上限不普适（Performance Ceilings are Not Universal）

不同方法的渐近性能（A）差异显著，不是所有方法都能达到相同的天花板。

实验中，DAPO 的 A = 0.52，而 CISPO/GSPO 达到 0.61（提升 17%）。仅仅在 LM head 使用 FP32 precision，就能把 A 从 0.52 拉到 0.61。

这意味着 loss 类型、batch size、数值精度等选择会直接决定性能上限，不能假设"只要训练够久，所有方法都会收敛到相同性能"。

> [!example] FP32 Precision 的影响
> 生成器和训练器使用不同的推理/训练 kernel，导致 token 概率的微小数值差异。这些差异直接影响 importance sampling ratio，在 RL 训练中被放大。
>
> 在 LM head 使用 FP32 精度后，A 从 0.52 跃升到 0.61。这不是优化技巧，是修复了根本性的数值问题。

### 发现 2：拥抱"苦涩教训"（Embracing the Bitter Lesson）

小规模表现好的方法，大规模可能更差。实验中某些方法在 4k GPU 小时时领先，但外推到 16k 时被反超。只有拟合 sigmoid 曲线，才能在早期判断方法的可扩展性。

传统的"在小规模上比较，选最好的"策略在 RL 中失效。正确做法是拟合前 50% 的数据，外推到 100%，验证预测准确性。这也让学术界能在有限预算下参与 RL 算法研究。

> [!comparison] 与 Pre-training 的对比
> Pre-training 的 scaling laws 相对温和，小规模优势通常能保持到大规模。
>
> RL 的"苦涩教训"更明显：探索-利用权衡让小规模和大规模的最优策略可能不同，性能指标有界导致饱和效应更强，对数值稳定性也更敏感。

### 发现 3：重新评估常见做法（Re-evaluating Common Wisdom）

Loss aggregation、normalization、curriculum 等常见优化主要影响计算效率（B），对渐近性能（A）影响不大。

实验中，prompt-level vs sample-level loss aggregation、batch-level vs prompt-level advantage normalization、zero-variance filtering、No-Positive-Resampling 都是 B 有差异而 A 相近。

这些优化不是可有可无的，它们加速达到天花板。计算预算有限时，高 B 的方法更实用。但追求极致性能，应该优先关注影响 A 的因素（loss 类型、precision）。

> [!intuition] 为什么这些优化主要影响 B？
>
> 这些优化本质上是改善训练信号的质量：
> - Loss aggregation：确保每个 prompt 贡献相等，避免 batch 不平衡
> - Advantage normalization：稳定梯度尺度，加速收敛
> - Zero-variance filtering：去除无信息的样本，提高有效 batch size
> - Curriculum：优先训练难度适中的样本，避免浪费计算
>
> 都是"如何更高效地学习"，而不是"能学到什么"，所以影响 B 而非 A。

## ScaleRL 方法

ScaleRL 不是新算法，而是整合已有最佳实践，形成可预测缩放的配方。

### 八大组件

| 组件 | 作用 | 影响参数 |
|------|------|----------|
| Asynchronous PipelineRL-8 | 生成器-训练器分离，减少 idle time | B（效率） |
| Forced length interruptions | 防止推理长度爆炸，稳定训练 | B（稳定性） |
| CISPO loss | 截断重要性采样，比 DAPO 更鲁棒 | A（天花板）+ B |
| Prompt-level loss averaging | 每个 prompt 贡献相等 | B（效率） |
| Batch-level advantage normalization | 跨 batch 标准化，稳定梯度 | B（稳定性） |
| FP32 precision at logits | 修复生成器-训练器数值差异 | A（天花板） |
| Zero-variance filtering | 去除无信息样本，提高有效 batch | B（效率） |
| No-Positive-Resampling | 移除已掌握的 prompt，聚焦难题 | B（效率） |

> [!intuition] 为什么需要这么多组件？
>
> Leave-one-out 实验发现：单独移除任何一个组件，对 A 影响不大，但对 B 有明显影响。这些组件的作用是累积的，每个解决一个特定的稳定性/效率问题，组合起来才能实现可预测缩放。
>
> 类比飞机的多个安全系统：单独看每个都不是必需的，但组合起来才能保证可靠飞行。

### 验证：10 万 GPU 小时的可预测训练

在前 5 万 GPU 小时拟合 sigmoid 曲线，外推到 10 万 GPU 小时，然后继续训练验证。外推曲线与实际性能高度吻合（误差 ±0.02）。在 17B×16 MoE 上同样可预测（前 16k 外推到 45k），downstream 任务（AIME-24）也遵循相同的缩放趋势。

> [!example] 可预测性的价值
>
> 假设你有 10 万 GPU 小时的预算，需要在两个算法中选择：
> - 算法 A：前 5k 小时表现更好
> - 算法 B：前 5k 小时稍差，但拟合曲线显示 A 更高
>
> 传统做法：选 A（基于当前表现）
> ScaleRL 框架：选 B（基于外推预测）
>
> 结果：B 在 10 万小时后性能更高，节省了重新训练的成本。

## 与其他方法对比

ScaleRL 在渐近性能和计算效率上均优于主流方法：

| 方法 | A（天花板） | B（效率） | 来源 |
|------|-------------|-----------|------|
| [[GRPO]] | ~0.55 | 较低 | DeepSeek |
| [[DAPO]] | ~0.52 | 中等 | Qwen-2.5 |
| Magistral | ~0.58 | 中等 | Magistral |
| MiniMax-M1 | ~0.59 | 较高 | MiniMax |
| ScaleRL | 0.61 | 最高 | 本文 |

A = 0.61（比 DAPO 高 17%），B 最大（更快达到天花板），外推误差 ±0.02。

> [!comparison] 为什么 ScaleRL 更好？
>
> 靠的是系统优化而非单点突破：40 万 GPU 小时的消融实验系统评估了所有设计选择，区分了影响 A 和 B 的因素，每个组件都经过 16k GPU 小时的 leave-one-out 验证。

## 方法论意义

这篇论文为 RL 研究提供了类似 pre-training scaling laws 的评估框架，但针对 RL 的特殊性（有界指标、探索-利用权衡）重新设计。

### 对研究者的价值

在 8k GPU 小时拟合曲线，外推到 100k GPU 小时预测最终性能，误差 ±0.02，足够用于算法选择。不需要跑满全部计算预算，通过比较 A 和 B 参数就能判断可扩展性。统一的缩放曲线框架也让不同论文的方法可以公平比较。

> [!intuition] 为什么这是"科学"而非"工程"？
>
> 科学的标志是可预测性和可重复性。在此之前，RL 算法评估更像炼金术，每个实验室用不同设置，结果难以比较。这篇论文给了 RL 研究统一的度量衡和预测框架。

### 对工业界的价值

根据 C_mid 估算达到目标性能所需的计算量，避免盲目投入。预算有限时优先高 B 方法，预算充足时优先高 A 方法。更重要的是能早期发现不可扩展的方法，及时止损。

> [!comparison] 与 Pre-training Scaling Laws 的对比
>
> 两者都用数学曲线拟合性能-计算关系，都能从小规模外推到大规模。
>
> 差异：
>
> | 维度 | Pre-training | RL |
> |------|--------------|-----|
> | 曲线类型 | Power law | Sigmoid |
> | 原因 | Loss 无界 | 指标有界（0-1） |
> | 关注指标 | Loss | Reward / Pass rate |
> | "苦涩教训" | 较温和 | 更明显 |
> | 稳定性 | 相对稳定 | 对数值精度敏感 |
>
> RL 更复杂的原因：探索-利用权衡让小规模和大规模的最优策略可能不同；性能有上限，必须用 sigmoid 捕捉；微小的数值差异会被 importance sampling 放大。

## 实验设置

- 规模: 40 万 GPU 小时（GB200）
- 模型: 8B dense（主要实验）+ 17B×16 MoE（验证泛化性）
- 任务: Math reasoning（Polaris-53K 数据集）
- 序列长度: 16,384 tokens（12,288 thinking + 2,048 solution + 2,048 prompt）
- Batch size: 768（48 prompts × 16 generations per prompt）
- 评估指标: Pass rate（mean@16，16 次采样的平均通过率）

> [!warning] 实验设置的局限性
> - 仅在数学推理任务上验证，其他任务（如代码、对话）的缩放规律可能不同
> - 使用 verifiable reward（数学题有标准答案），无 reward model 的场景可能不适用
> - 序列长度 16k，更长的 CoT（如 32k）虽然也验证了，但样本量较少

## 延伸阅读

原始论文：
- [[Clippings/Paper/251013786v1/251013786v1|The Art of Scaling RL Compute for LLMs (Full Paper)]] — 完整论文，包含详细消融实验

相关方法：
- [[GRPO]] — DeepSeek 使用的 RL 方法，ScaleRL 的对比基准之一
- [[DAPO]] — Qwen-2.5 使用的 RL 方法，引入 asymmetric clipping

相关概念：
- [[Scaling Laws]] — Pre-training 的缩放规律，本文的理论基础
- [[LLM RL Algorithm Evolution]] — RL 算法演进的全景图，本文是第六幕
