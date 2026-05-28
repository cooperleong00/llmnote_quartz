---
type: paper
description: Qwen 团队系统揭示 PRM 开发中 MC estimation 的噪声本质和 BoN 评估的三重偏差，提出 consensus filtering 机制显著提升数据效率
aliases:
  - Qwen PRM Lessons
  - PRM 开发经验
prerequisites:
  - "[[Reward Model]]"
  - "[[Process Reward Model]]"
tags:
  - reasoning
  - evaluation
  - math
  - reward-model
  - post-training
created: 2026-02-10
updated: 2026-02-10T23:49
---

# Lessons of Developing PRMs (2025)

> [!paper] 论文信息
> **The Lessons of Developing Process Reward Models in Mathematical Reasoning**
> Zhenru Zhang, Chujie Zheng 等 | Qwen Team, Alibaba Group | 2025
> ArXiv: 2501.07301
> 发布模型：Qwen2.5-Math-PRM-7B / 72B

这篇论文是 Qwen 团队在开发 [[Process Reward Model]] 过程中积累的实践经验总结。核心发现有两个：(1) 广泛使用的 Monte Carlo (MC) estimation 标注方法本质上是在用 Value Model 的方式训练 PRM，引入了大量噪声；(2) 常用的 Best-of-N (BoN) 评估存在系统性偏差，会高估 PRM 的真实能力。基于这些发现，论文提出了 consensus filtering 机制和更全面的评估框架。

---

## 背景：PRM 开发的两大挑战

[[Process Reward Model]] 为数学推理中的每个中间步骤提供正确性评估，比 Outcome RM 提供更细粒度的 [[Credit Assignment]] 信号。但 PRM 的开发面临两个核心挑战：

1. **数据标注**：人工标注（如 PRM800K）质量高但成本极高，自动化标注（MC estimation）成本低但质量存疑
2. **评估方法**：主流的 BoN 评估是否真正反映了 PRM 的过程验证能力？

论文通过大量实验系统回答了这两个问题。

---

## Lesson 1：MC Estimation 的根本局限

### PRM vs Value Model 的本质区别

这是论文最深刻的洞察之一：

> [!intuition] 核心区分
> - **PRM**：当前步骤是否正确？ -- 确定性评估器（deterministic evaluator）
> - **Value Model**：从当前步骤出发，未来能否解对？ -- 预测性估计器（predictive estimator）
>
> MC estimation 通过"从当前步骤出发做多次 completion，看有多少比例能得到正确答案"来估计步骤正确性。这本质上是在估计 **未来解题潜力**，而非 **当前步骤正确性**。

这个区分解释了 MC estimation 的两类噪声来源：

- **假阴性**：一个正确的步骤，但 completion model 能力不足，后续步骤全部做错 -- MC 估计为 0，标记为"错误步骤"
- **假阳性**：一个错误的步骤，但 completion model 恰好从错误中"歪打正着"得到正确答案 -- MC 估计为正，标记为"正确步骤"

### 三种标注方法的实验对比

论文在相同的 860K 数据上对比了三种标注方法：

| 标注方法 | 数据量 | BoN@8 Avg. | PROCESSBENCH Avg. F1 |
|----------|--------|------------|---------------------|
| MC Estimation (Math-Shepherd) | 440K | 64.3 | 28.9 |
| MC Estimation (Qwen 数据) | 860K | 65.9 | 40.1 |
| LLM-as-a-judge | 860K | 65.3 | 46.5 |
| Human Annotation (PRM800K) | 264K | 64.9 | **56.5** |

> [!warning] 关键发现
> MC estimation 在 BoN 上表现最好，但在 PROCESSBENCH（step-level 错误识别）上表现最差。人工标注仅用 264K 数据就在 PROCESSBENCH 上大幅领先。这种"BoN 和 PROCESSBENCH 表现倒挂"的现象正是 Lesson 2 要解释的。

### Hard Label vs Soft Label

在 MC estimation 场景下，论文发现：

- **过滤前**：soft label 和 hard label 差异不大（噪声掩盖了差异）
- **过滤后**：hard label 显著优于 soft label

原因在于：PRM 评估的是步骤正确性，这是一个**确定性判断**（对或错），而 soft label 引入了"未来可能性"的概率信息，与 PRM 的目标不一致。此外，仅 8 次 completion 的估计方差很大，soft label 的精度本身就不可靠。

**最优阈值**：如果必须使用 MC estimation，将阈值设为 0（即只要 8 次 completion 中有任何一次得到正确答案，就标记为正确步骤）效果最好。

---

## Lesson 2：BoN 评估的三重偏差

这是论文对 PRM 评估方法论的重要贡献。BoN 作为 PRM 的主流评估方式，存在三个系统性偏差：

### 偏差 1：BoN-PRM 目标不对齐

> [!intuition] 问题本质
> BoN 只关心"最终答案是否正确"，而 PRM 的目标是"每个推理步骤是否正确"。当 policy model 生成"答案正确但过程有误"的 response 时，两者的评判标准产生冲突。

论文对 Qwen2.5-Math-7B-Instruct 生成的正确答案 response 进行人工标注，发现：
- GSM8K（简单）：少量过程错误
- MATH（中等）：显著比例的过程错误
- OlympiadBench / Omni-MATH（困难）：大量过程错误

**越难的问题，"答案对但过程错"的比例越高**。这意味着一个真正好的 PRM 会给这些 response 低分，反而在 BoN 上表现更差。

### 偏差 2：BoN 分数膨胀

当 PRM 无法区分"答案正确但过程有误"的 response 并给予高分时，BoN 分数被虚假抬高。论文在 PROCESSBENCH 中提取了这类特殊样本进行测试：

- 除了 Qwen2.5-Math-PRM 系列，**所有开源 PRM 的检测准确率都低于 50%**
- MC estimation 训练的 PRM 在这类样本上尤其差，但 BoN 分数反而最高

这揭示了一个悖论：**BoN 分数高的 PRM 可能恰恰是过程验证能力差的 PRM**。

### 偏差 3：Process-to-Outcome Shift

论文分析了多个开源 PRM 的最低分分布，发现 EurusPRM、Math-Shepherd-PRM、Skywork-PRM 等模型超过 40% 的 response 中，最低分出现在最终答案步骤。这说明这些 PRM 实际上退化为了 [[Reward Model|ORM]]——它们主要在判断最终答案是否正确，而非中间步骤。

> [!warning] 实践启示
> 仅用 BoN 优化 PRM 会导致 PRM 退化为 ORM。必须同时使用 step-level 评估（如 PROCESSBENCH）来确保 PRM 真正具备过程验证能力。

### 不同 PRM 的最优评分策略

一个有趣的发现：PRM 的最优评分策略取决于训练数据的标注方式。

| 训练数据 | 最优评分策略 | 原因 |
|----------|-------------|------|
| MC estimation | **last score** | MC 估计的是"未来潜力"，最后一步自然整合了全过程信息 |
| LLM-as-a-judge / 人工标注 | **product / minimum** | 每步分数是独立的正确性判断，乘积/最小值反映整体过程质量 |

这进一步印证了 MC estimation 训练的 PRM 本质上更像 Value Model 而非真正的 PRM。

---

## Consensus Filtering：整合两种标注方法

论文提出的核心方法论贡献：

> [!definition] Consensus Filtering
> 同时使用 MC estimation 和 LLM-as-a-judge 对同一批数据进行标注，**只保留两者对错误步骤位置达成共识的样本**。

具体流程：
1. 用 MC estimation 标注 860K 样本（成本低，覆盖广）
2. 用 LLM-as-a-judge（Qwen2.5-72B-Instruct）对同一批样本进行步骤级验证
3. 只保留两者在错误步骤位置上一致的样本
4. 约 **40% 数据被保留**

> [!intuition] 为什么有效？
> MC estimation 的噪声主要来自 completion model 的不确定性，LLM-as-a-judge 的噪声来自 LLM 的推理能力局限。两者的噪声模式不同，取交集可以有效过滤掉各自的假阳性/假阴性。

效果：
- 在 PROCESSBENCH 上，过滤后的数据**显著优于 MC estimation**，接近 LLM-as-a-judge 的效果
- 在 BoN 上，三种方法差异不大（因为 BoN 本身有偏差）
- 数据效率大幅提升：用 40% 的数据达到接近甚至更好的效果

---

## 实验结果：Qwen2.5-Math-PRM

基于 consensus filtering 训练的最终模型：

### BoN@8 评估（policy: Qwen2.5-Math-7B-Instruct）

| 模型 | 参数量 | Avg. |
|------|--------|------|
| maj@8 (baseline) | - | 66.2 |
| 最佳开源 PRM | 7B | ~64.9 |
| **Qwen2.5-Math-PRM-7B** | 7B | **67.6** |
| Qwen2.5-Math-RM-72B (ORM) | 72B | 68.9 |
| **Qwen2.5-Math-PRM-72B** | 72B | **69.3** |

7B 模型在所有 7 个任务上超过 maj@8，平均提升 1.4%。72B 模型略优于同规模 ORM。

### PROCESSBENCH 评估（step-level 错误识别）

| 模型 | Avg. F1 |
|------|---------|
| 最佳开源 PRM (PRM800K-trained) | 56.5 |
| GPT-4o-0806 (LLM-as-judge) | 61.9 |
| **Qwen2.5-Math-PRM-7B** | **73.5** |
| **Qwen2.5-Math-PRM-72B** | **78.3** |
| o1-mini (LLM-as-judge) | 87.9 |

7B 模型超过所有开源 PRM 和 GPT-4o，72B 模型进一步拉开差距，但与 o1-mini 仍有差距。

---

## 实践指南总结

论文的"lessons"可以提炼为以下实践建议：

**数据标注**：
1. MC estimation 不是理想的 PRM 训练数据来源，优先考虑 LLM-as-a-judge 或人工标注
2. 如果必须使用 MC estimation，务必配合 consensus filtering
3. 过滤后使用 hard label（阈值 = 0），不要用 soft label

**评估方法**：
4. 不要仅依赖 BoN 评估，必须同时使用 step-level 评估（如 PROCESSBENCH）
5. 注意评分策略与训练数据的匹配：MC 训练用 last score，其他用 product/minimum

**模型设计**：
6. 警惕 PRM 退化为 ORM 的趋势（process-to-outcome shift）
7. 关注"答案正确但过程有误"的样本，这是检验 PRM 真实能力的试金石

---

## 局限性

论文自身指出的局限：
- PRM 与 BoN 上限（pass@8）之间仍有较大差距
- PRM 在强化学习中的最佳使用方式（如与 [[GRPO]] 结合）尚未探索
- 高质量人工标注数据的高效利用（如弱监督扩展）仍待研究

---

## 延伸阅读

**原始论文与数据**：
- [[Clippings/Paper/250107301v2/250107301v2|论文原文]] -- 完整实验细节和附录

**相关方法与概念**：
- [[Reward Model#Process RM vs Outcome RM]] -- PRM 与 ORM 的基本对比
- [[Reward Hacking]] -- PRM 评估偏差与 reward hacking 的关联
- [[GRPO#Process Supervision]] -- PRM 在 RL 训练中的应用场景
