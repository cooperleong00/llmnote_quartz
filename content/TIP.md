---
type: method
description: TIP 用 student entropy 与 teacher-student divergence 对 on-policy distillation 中的 token 进行分类和选择，以更少 token 保留高价值监督信号
aliases:
  - Token Importance in On-Policy Distillation
  - Token Importance in on-Policy distillation
  - token 重要性
  - token importance
prerequisites:
  - "[[On-Policy Distillation]]"
  - "[[Entropy]]"
  - "[[KL Divergence]]"
tags:
  - post-training
  - distillation
  - efficiency
  - optimization
  - reasoning
created: 2026-04-26
updated: 2026-04-26
---

# TIP

TIP（Token Importance in On-Policy Distillation）是一种用于 [[On-Policy Distillation|on-policy distillation]] 的 token-level 选择方法。它把每个 token 位置放到由 student entropy 和 teacher-student divergence 构成的二维平面中，优先训练高不确定性 token 与低不确定性高分歧 token，从而在降低显存和计算压力的同时保留关键监督信号。

## 动机

在 on-policy distillation 中，学生模型先从自己的分布采样生成 rollout，教师模型再对学生实际到达的每个上下文提供 token-level 分布监督。标准做法会对所有 token 位置计算蒸馏损失：

$$
\mathcal{L}=\frac{1}{m}\sum_{t=1}^{m}D_{\mathrm{KL}}(P_S(\cdot\mid c_t)\|P_T(\cdot\mid c_t)).
$$

这里的每个 token 都来自学生自己的生成轨迹，但它们的学习价值差异很大。已经被学生稳定掌握、且教师也认可的位置会贡献很小的修正信号；学生犹豫的位置提供了塑形空间；学生非常自信且与教师强烈分歧的位置往往暴露了系统性偏差。

> [!intuition] 为什么需要 TIP？
> [[On-Policy Distillation]] 的优势来自 dense token-level supervision，但 dense supervision 也会带来很多低价值 token。TIP 的核心判断是：token 重要性需要同时看学生自己的不确定性，以及教师和学生之间的分歧。只看不确定性可以找到一批关键位置，加入分歧轴后可以找到学生自信犯错的位置。

## 核心机制

TIP 对每个位置 $t$ 计算两个训练中已经可得的量。

第一个量是归一化 student entropy：

$$
h_t=\frac{H(P_S(\cdot\mid c_t))}{\log |V|}\in[0,1].
$$

[[Entropy|熵]] 衡量学生分布的扩散程度。$h_t$ 高说明学生还在多个候选 token 之间摇摆；$h_t$ 低说明学生分布已经集中到少数 token 上。

第二个量是 teacher-student divergence：

$$
\delta_t=D_{\mathrm{KL}}(P_S(\cdot\mid c_t)\|P_T(\cdot\mid c_t)).
$$

这里的 [[KL Divergence|KL divergence]] 就是每个 token 的 reverse KL 蒸馏损失。$\delta_t$ 高说明学生当前分布和教师分布差距大，教师会给出强修正信号。

### 四象限

把 $h_t$ 和 $\delta_t$ 交叉后，TIP 得到四类 token：

| 象限 | student entropy | teacher-student divergence | 学习作用 |
|---|---:|---:|---|
| Q1 | 高 | 高 | 修正明显错误，巩固脆弱知识 |
| Q2 | 高 | 低 | 稳定学生还不够自信的预测 |
| Q3 | 低 | 高 | 修正学生自信但偏离教师的系统性错误 |
| Q4 | 低 | 低 | 已经解决，监督信号很弱 |

这个 taxonomy 给出了一个很实用的读法：Q1/Q2 是 entropy 规则容易捕捉的高不确定性区域，Q3 是学生已经很自信但教师强烈不同意的区域，Q4 是预算紧张时优先丢弃的低信号区域。

> [!intuition] Q3 为什么重要？
> Q3 token 看起来很“容易”：学生分布很尖，entropy 很低。真正的问题在 divergence 轴上：教师分布把概率放在另一个方向，说明学生已经形成稳定偏差。对于数学推理或 agentic planning，一个自信的错误变量、算术步骤或约束选择会把后续轨迹带到错误分支，因此这类 token 的修正价值很高。

## Soft-OR 选择规则

实际训练时，TIP 先对 $h_t$ 与 $\delta_t$ 做 min-max normalization，得到 $\hat{h}_t,\hat{\delta}_t\in[0,1]$，然后计算 Soft-OR score：

$$
s_t=\hat{h}_t+\hat{\delta}_t-\hat{h}_t\hat{\delta}_t
=1-(1-\hat{h}_t)(1-\hat{\delta}_t).
$$

给定保留比例 $\rho$，TIP 保留 score 最高的 $\lfloor \rho m\rfloor$ 个 token：

$$
\mathcal{T}=\operatorname{TopK}(\{s_t\}_{t=1}^m,\lfloor \rho m\rfloor).
$$

训练损失只在这些 token 上平均：

$$
\mathcal{L}_{\mathrm{TIP}}
=\frac{1}{|\mathcal{T}|}\sum_{t\in\mathcal{T}}
D_{\mathrm{KL}}(P_S(\cdot\mid c_t)\|P_T(\cdot\mid c_t)).
$$

> [!math] 为什么 Soft-OR 合理？
> Soft-OR 的值会在任一轴激活时升高：高 entropy 的 Q1/Q2 得到保留，高 divergence 的 Q1/Q3 也得到保留。Q4 同时低 entropy 和低 divergence，score 接近 0。这样可以近似覆盖 oracle ordering：$Q1 > Q2 > Q3 \gg Q4$。

这个 score 没有额外权重系数。由于 $h_t$ 与 $\delta_t$ 已经在标准 OPD 前向与损失计算中得到，新增开销主要是 batch 内归一化与 Top-K 排序，论文给出的复杂度是每个 rollout $O(m\log m)$，相对模型 forward/backward 很小。

## 与 Entropy-Only 选择的关系

Entropy-only 选择把训练预算集中在 Q1/Q2，因此在保留 50% token 时已经能接近或超过全 token OPD。TIP 进一步加入 divergence 轴，把 Q3 也纳入预算。两者的关系可以这样理解：

- entropy 轴负责找到“学生还没定下来”的位置；
- divergence 轴负责找到“学生已经定下来，但教师强烈反对”的位置；
- Soft-OR 把两类监督信号放进同一个 Top-K 预算。

> [!comparison] 与 [[Process Reward Model|过程奖励模型]] 的关系
> PRM 通过外部监督或标注器给中间步骤打分，目标是定位推理过程中的好坏步骤。TIP 利用教师和学生的 token 分布差异，在 OPD 内部得到一种轻量的 token-level 信号；它更适合已有 teacher distribution 的蒸馏场景。

## 实验结论

论文在数学推理和 agentic planning 上验证了 TIP 的三层判断。

1. 高 entropy token 提供强一阶信号。保留 50% entropy-sampled token 往往能匹配或超过全 token OPD，并带来明显显存下降；Qwen3 实验中 peak memory 从 72.0GB 降到 38.1GB，Qwen2.5 从 35.8GB 降到 19.7GB。
2. Q3 token 的密度很小但修正信号集中。论文报告 Q3 只占约 3%-15% token；在 Qwen3 数学推理中，少于 10% 的 overconfident token 训练可以达到 76.1 MATH-500，而全 token baseline 是 76.7。
3. Soft-OR 在数学推理上稳定优于 entropy-only。比如 Qwen3-8B 到 4B 的 MATH-500 从全 token 76.7 提升到 Soft-OR 50% 的 79.1；AIME 2024 从 21.9 提升到 25.7。
4. Agentic planning 对 Q3 更敏感。在 DeepPlanning 中，Q3-only 20% 超过全 token OPD：14B teacher 下 12.6 vs. 11.7，32B teacher 下 13.6 vs. 12.8。计划任务中一个自信的错误承诺会影响后续多个约束，Q3 修正更集中。

## 边界条件

> [!warning] 使用 TIP 时要检查的条件
> - TIP 需要教师输出分布。只拿到 hard label 或最终答案时，无法直接计算 $\delta_t$。
> - Soft-OR 使用 batch 内 min-max normalization，极端 outlier token 可能影响同 batch 的相对排序。运行均值归一化或 robust normalization 是可探索替代方案。
> - 论文实验主要使用 reverse KL supervision。forward KL、JSD 或混合 divergence 下的象限排序还需要单独验证。
> - TIP 解决的是 token 预算分配问题。它依赖 teacher 的能力边界；当教师在某类任务上系统性错误时，高 divergence 也可能把学生推向教师偏差。

## 适用场景

TIP 适合训练成本由 token-level teacher supervision 主导的场景，尤其是：

- 大模型教师到小模型学生的 [[On-Policy Distillation|on-policy distillation]]；
- reasoning model 压缩，学生 rollout 中存在大量已掌握 token；
- agentic planning、tool-use 或长链任务，一个自信错误会影响后续多个决策；
- 需要降低显存峰值的 OPD 训练，尤其在全词表 KL 计算较重时。

TIP 的两轴视角也可以迁移到 [[RLHF]]、[[Process Reward Model|process reward fine-tuning]] 和 [[Speculative Decoding|speculative decoding]] 等 token-level 监督或验证场景：只要系统能同时观察学生不确定性与外部监督分歧，就可以用类似 taxonomy 判断哪些 token 更值得更新或验证。

## 面试要点

> [!interview] Q: TIP 的核心贡献是什么？
> A: TIP 把 OPD 中的 token 重要性拆成 student entropy 和 teacher-student divergence 两个轴，提出 Q1-Q4 taxonomy，并用 Soft-OR score 在固定 token 预算下保留高不确定性 token 和自信犯错 token。

> [!interview] Q: 为什么只看 entropy 会漏掉重要 token？
> A: entropy 只能反映学生分布是否分散。Q3 token 的学生分布很尖，entropy 很低，但教师分布和学生分布差距很大；这说明学生已经形成稳定偏差，需要 divergence 轴才能识别。

> [!interview] Q: Soft-OR 的工程优势是什么？
> A: 它不引入额外超参数，使用 OPD 已经计算出的 entropy 和 KL loss，只增加归一化与 Top-K 排序。训练时可以保留 20%-50% token，在实验中维持或提升准确率并降低显存。

## 论文出处

> [!paper] TIP: Token Importance in On-Policy Distillation
> Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, Alborz Geramifard. 2026. arXiv:2604.14084. 本笔记基于 [[Clippings/Paper/2604.14084/2604.14084|finalized paper clipping]] 整理，核心方法是 TIP 的 two-axis taxonomy 与 Soft-OR type-aware token selection。
