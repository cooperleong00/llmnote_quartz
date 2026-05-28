---
type: overview
description: 从采样分布、性能上限与梯度几何解释 post-training 中 SFT、RL 与 on-policy distillation 的阶段切换
aliases:
  - SFT-RL Phase Transition
  - Post-training 阶段切换
  - SFT 到 RL 的阶段切换
prerequisites:
  - "[[SFT]]"
  - "[[RLHF]]"
  - "[[Policy Gradient]]"
  - "[[KL Divergence]]"
tags:
  - post-training
  - reinforcement-learning
  - distillation
  - training-dynamics
created: 2026-05-01
updated: 2026-05-01T14:13
---

# Post-Training Phase Transition

Post-training phase transition 描述的是一个训练决策：什么时候继续用 [[SFT]] 从 teacher data 中吸收能力，什么时候转向 [[GRPO]] / [[PPO]] 这类 RL，让模型通过自己的 rollouts 继续提高，以及什么时候用 [[On-Policy Distillation|OPD]] 以更低成本接近 teacher。它解释了 [[SFT-then-RL]] 这类 recipe 背后的训练动力学，核心判断标准有三个：采样分布能否随模型变强而改善、当前能力距离 teacher 还有多远、每一步梯度信号是否稳定。

这篇笔记把 SFT、RL、OPD 与 self-distillation 放在同一个坐标系里理解。读完后应该能回答：为什么常见 pipeline 先 SFT 后 RL，为什么 OPD 在 same-family teacher 场景下很有吸引力，以及为什么 self-with-hint distillation 需要格外控制 KL。

## 为什么会有阶段切换

> [!intuition] 从固定数据到自我采样
> Teacher SFT 的数据分布在数据构造时已经固定。学生越接近 teacher，新增样本越像是在重复同一分布中的模式，边际信息量下降。
>
> RL 让学生从当前策略采样 rollouts，更新后的策略又决定下一批 rollouts。只要 verifier 能区分好坏，能力提升可以反馈到采样分布中，形成 compounding。

可以把训练早期和后期分开看。

训练早期，student 远低于 teacher。此时 teacher completion 中包含大量 student 还不会的能力，[[SFT]] 的每个 token 都能提供 dense supervision，单位算力收益很高。这个阶段继续做 RL 通常会把大量计算花在探索上，因为当前策略连基本解法空间都还没有覆盖好。

训练后期，student 已经接近 teacher。继续堆 teacher data 会逐渐遇到 teacher ceiling：数据能教给 student 的新东西变少，采样分布也不会随着 student 的改进自动变强。此时 [[Policy Gradient|policy gradient]] 的价值开始显现：student 自己生成 rollouts，偶然出现的新策略可以被 reward 或 verifier 抽取出来。

> [!definition] 阶段切换点
> SFT-RL phase transition 可以理解为：teacher data 的边际收益低于 student rollouts 的边际收益时，训练预算应该从静态监督转向 on-policy 采样。

这个切换点取决于 teacher 质量、student 当前能力、verifier 可靠性、rollout 成本和任务分布。数学推理任务里，如果 verifier 很强，RL 的后期收益会更明显；开放式写作任务里，verifier 本身难以定义，继续做偏好建模或蒸馏可能更稳。

## 三种分布决定三种上限

理解阶段切换，先分清训练时“样本从哪里来”。

| 方法 | 采样分布 | 梯度信号 | 实用上限 |
|------|----------|----------|----------|
| [[SFT]] | 固定 teacher / human data | 每个 token 的 one-hot label | 接近 teacher data 覆盖的分布 |
| Rejection-sampled SFT | teacher 或 student 样本经过 filter | 被筛选出的 token label | filter 饱和后受原采样分布限制 |
| [[On-Policy Distillation|OPD]] | student 当前策略 rollouts | teacher 对 student token 的 per-token KL signal | 接近 same-family teacher |
| [[GRPO]] / [[PPO]] | student 当前策略 rollouts | sequence-level reward broadcast 到 token | 受 verifier 能力限制 |

Rejection-sampled SFT 会把曲线抬高：错误样本被过滤，正确样本被保留，数据质量更好。它的采样分布仍然来自固定 teacher 或当前筛选流程；当 filter 能区分的样本已经被吃满，继续采样主要扩大同一类正确解的数量。

[[On-Policy Distillation|OPD]] 的关键优势来自 state coverage。SFT 在 teacher state distribution 上训练，评估时却要面对 student 自己生成的前缀；长序列里前缀稍微偏离，后续状态就可能落到训练分布外。OPD 直接在 student rollouts 上计算 teacher 的 per-token preference，因此训练分布和评估分布更一致。

RL 的上限由 verifier 决定。只要 verifier 可以评价某条 rollout 的结果，student 就有机会发现 teacher data 里没有出现过的策略。代价是 reward 稀疏，credit assignment 成本高，训练通常需要大 batch、低学习率和稳定的 baseline。

## Same-Family Teacher 为什么特殊

OPD 只有在 same-family teacher 场景下特别自然。same-family 指 teacher 和 student 共享 tokenizer、训练 recipe 接近，最好是同一模型家族的不同规模版本。

这带来两个好处。

第一，per-token logprob 可比较。OPD 的 reverse KL 需要在同一个 token 位置上比较 $\pi_T(y_t \mid y_{<t})$ 和 $\pi_\theta(y_t \mid y_{<t})$。tokenizer 不一致时，同一段文本会被切成不同 token，teacher 的 logprob 很难直接变成 student 的 token-level target。

第二，teacher 的偏好更像“能力差异”。如果 teacher 和 student 的 recipe 接近，teacher 更偏好的 token 通常对应更好的推理、格式或答案选择。跨家族蒸馏中，teacher 偏好的 token 可能大量反映风格、模板和 chain-of-thought 习惯，student 会把一部分容量花在模仿表面分布上。

> [!intuition] OPD 的训练位置
> 一个常见路线是：先用 SFT 把 student 带到 teacher 的附近，再用 OPD 在 student 自己的 rollouts 上对齐 teacher。这样既保留 on-policy state coverage，又避免 RL 早期探索成本。

这里的 ceiling 也很清楚。OPD 的目标是靠近 teacher distribution，因此它适合快速逼近 teacher；RL 的目标是提高 verifier reward，因此它更适合继续突破 teacher data 覆盖范围。

## 梯度几何：稀疏、密集、偏置、集中

阶段选择还可以从梯度形状理解。每种方法都在回答同一个问题：每个 token 的更新方向从哪里来，平均之后会留下什么。

### RL：稀疏信号，依靠大 batch 抵消噪声

在 [[GRPO]] 这样的 RLVR 训练中，一个序列的 reward 会被分配到整条 rollout 的 token 上。许多 token 对最终正确性没有因果贡献，却因为和好结果出现在同一条轨迹里拿到正 advantage，或者因为和坏结果出现在同一条轨迹里拿到负 advantage。

大 batch 和 group baseline 的作用，是让这些无关 token 的梯度在平均中互相抵消。留下来的，是和 reward 稳定相关的方向，例如更长思考、检查计算、选择更可靠的解法分支。这个估计接近 unbiased，但方差高，所以 RL 稳定且慢。

### SFT：密集信号，偏置分散在数据流形上

SFT 给每个 token 一个明确 label，信号非常 dense。它的梯度带有固定偏置：模型被拉向数据分布。SFT 之所以通常稳定，是因为高质量数据覆盖了很多任务、格式和表达方式，偏置被分散到许多方向上。

当数据多样且 teacher 与 student 距离适中时，SFT 像是在把模型推向一个宽的数据 manifold。若数据窄、重复度高或 teacher 风格很强，偏置会集中到表面形式上，能力迁移效率下降。

### OPD：密集信号，偏置校准到 student rollouts

OPD 的 token-level advantage 可以写成：

$$
A_t = \log \pi_T(y_t \mid y_{<t}) - \log \pi_\theta(y_t \mid y_{<t})
$$

其中 $y \sim \pi_\theta(\cdot \mid x)$ 来自 student 当前策略。直觉上，某个 token 如果 teacher 比 student 更喜欢，它就获得正向更新；如果 teacher 比 student 更不喜欢，它就被压低。

same-family teacher 让这个偏置通常更分散，因为 teacher 和 student 的分布形状接近。OPD 因此比 RL 信息更密集，又比跨家族蒸馏更容易把信号落在能力差异上。

### Self-with-hint distillation：密集偏置可能集中到少数 pivot token

Self-distillation 在没有外部 same-family teacher 时很有吸引力：让 student 自己生成 rollout，再让一个带 privileged information 的 self-teacher 对同一条 rollout 打 token-level 信号。例如 teacher 看到 expert demonstration、ground-truth answer 或 hint，student 采样时看不到这些信息。

风险来自 KL concentration。假设一个数学推理里有一个关键 pivot token，student 给它 $0.01$ 概率，带答案提示的 self-teacher 给它 $0.6$ 概率，那么这个 token 的 reverse-KL advantage 约为：

$$
\log \frac{0.6}{0.01} \approx 4.1
$$

普通 token 上 teacher 和 student 可能都给 $0.3$ 左右，贡献接近 $0$。于是少数 pivot token 主导梯度，模型会被快速推向原本低概率的区域。这解释了为什么 OPSD / SDFT 这类方法常需要 per-token KL clipping、固定 teacher 或其他 KL budget 机制。

## 一个统一视角：选择 teacher 与 KL budget

可以把这些方法看成 token-level policy gradient 的不同角落。两个旋钮最重要：

- $\alpha$：采样分布有多 on-policy，也就是样本来自固定数据、teacher，还是 student 当前策略。
- $\lambda$：每个 token 的 advantage 有多少来自 teacher KL，有多少来自 sequence-level outcome reward。

在这个视角下，[[SFT]] 接近“固定数据 + teacher label”，[[GRPO]] 接近“on-policy samples + outcome reward”，[[On-Policy Distillation|OPD]] 接近“on-policy samples + same-family teacher KL”。Self-distillation 与 OPD 的算法形状接近，teacher policy 换成了带 privileged information 的自己。

真正难的部分是选择 teacher 和 KL budget。好的 teacher 应该在当前 student 的 rollouts 上满足两个条件：

- 能显著提高 reward 或正确率。
- 与 student 的 token distribution 距离受控，让更新保持稳定。

可以写成一个 Lagrangian 目标：

$$
\max_{\pi_T} \; \mathbb{E}[\Delta R(\pi_\theta \rightarrow \pi_T)] - \beta \, D_{\mathrm{KL}}(\pi_T \,\|\, \pi_\theta)
$$

$\beta$ 控制激进程度。$\beta$ 大时，teacher 必须贴近 student，训练稳定但提升慢；$\beta$ 小时，teacher 可以给出更强建议，训练更容易被少数高 KL token 主导。

## 实践判断

> [!example] 训练预算应该花在哪里？
> - Student 明显低于可用 teacher：优先 SFT 或高质量 distillation，把基础能力补齐。
> - Student 接近 teacher，且 teacher 与 student same-family：优先考虑 OPD，用 on-policy state coverage 提高 distillation 效率。
> - Student 接近或超过 teacher data 覆盖，且 verifier 可靠：转向 RL，让采样分布随策略改进而 compounding。
> - 没有 same-family teacher，但有 answer、hint 或 demonstration：可以尝试 self-distillation，同时严格控制 per-token KL。

> [!warning] 边界条件
> - Verifier 不可靠时，RL 会把模型推向 reward hacking 或错误偏好。
> - Cross-family teacher 的 soft-target distillation 会遇到 tokenizer mismatch 和 recipe mismatch，dense signal 中会混入风格迁移成本。
> - Self-with-hint teacher 可能让少数 token 的 KL 过大，训练需要 clipping、teacher freezing 或更细的 KL budget。
> - SFT-RL 切换点没有固定公式，需要用 validation curve、rollout diversity、verifier quality 和 teacher gap 共同判断。

## 面试视角

> [!interview] Q: 为什么 post-training 常见顺序是先 SFT 后 RL？
> A: SFT 在早期提供便宜的 dense supervision，适合把 student 拉到 teacher 或人类数据覆盖的能力范围内。接近 teacher 后，固定数据的边际信息量下降；RL 通过 student 自己的 rollouts 采样，让新策略可以反馈进下一轮训练，因此更适合后期继续提高。

> [!interview] Q: OPD 为什么可能比 RL 更省算力？
> A: OPD 在 student rollouts 上训练，保留 on-policy state coverage，同时 teacher 对每个 token 提供 dense KL signal。RL 主要依赖 sequence-level reward，credit assignment 需要大量样本平均噪声。OPD 的 ceiling 通常受 teacher 限制，RL 的 ceiling 更依赖 verifier。

> [!interview] Q: Self-distillation 的主要风险是什么？
> A: 带 privileged information 的 self-teacher 可能强烈偏好少数 student 原本低概率的 pivot token，导致 KL 信号高度集中。训练中需要 KL clipping、固定 teacher、限制 hint 信息量，或者把 teacher 设计成高 reward、低 KL 的局部辅助策略。

## 参考资料

> [!paper] 来源
> 本笔记吸收自 [[Clippings/Article/On SFT, RL, and on-policy distillation|On SFT, RL, and on-policy distillation]]。原文提供了 SFT-RL compounding argument、same-family teacher 对 OPD 的重要性，以及 sparse/dense、biased/unbiased、concentrated/diffuse 的梯度几何分析。
