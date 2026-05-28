---
type: method
description: RLSD 用自我蒸馏得到 token-level 信用分配权重，并让环境奖励决定更新方向，从而把 RLVR 的稳定性与密集监督结合起来
aliases:
  - RLSD
  - Reinforcement Learning with Self-Distillation
  - Self-Distilled RLVR
  - 自蒸馏 RLVR
prerequisites:
  - "[[GRPO]]"
  - "[[PPO]]"
  - "[[On-Policy Distillation]]"
  - "[[Credit Assignment]]"
tags:
  - post-training
  - rlhf
  - self-distillation
  - credit-assignment
created: 2026-04-26
updated: 2026-04-26T22:34
---

# RLSD

RLSD（Reinforcement Learning with Self-Distillation）是一种用于 reasoning model 后训练的 [[RLVR]] 方法：模型仍然用 verifier 给出的序列级奖励决定策略更新方向，同时用自我蒸馏得到的 token-level 证据比率调节每个 token 的更新幅度。它可以看作 [[GRPO]] 的 advantage 重分配版本，把原本对整条回答共享的优势值拆成更细的 token-level 信用。

> [!paper] 论文出处
> Yang et al., "Self-Distilled RLVR", arXiv:2604.03128 (2026)。本文依据 [[Clippings/Paper/2604.03128/2604.03128|Self-Distilled RLVR]] 整理。

## 动机

在 [[GRPO]] 这类 [[RLVR]] 训练中，verifier 通常只返回一个结果级奖励，例如答案正确记为 1，错误记为 0。这个信号可靠且便宜，但它把同一条 rollout 里的所有 token 都赋予同一个 advantage：关键推理步骤、格式化文本、无关铺垫都会被同等强化或惩罚。长链推理由此变成 [[Credit Assignment|信用分配]] 问题：训练知道整条答案好坏，却缺少对每个局部推理动作贡献的判断。

[[On-Policy Distillation]] 提供了另一条路径。学生模型从自身策略采样 trajectory，教师模型沿着这些 trajectory 提供 token-level logits，密集信号能加速收敛。外部教师带来额外计算成本，并且要求教师与学生共享 vocabulary。OPSD（On-Policy Self-Distillation）进一步让同一个模型同时扮演教师和学生：学生只看到问题 $x$，教师额外看到特权信息 $r$，例如参考答案或 verified reasoning trace。

OPSD 的吸引力来自低成本密集监督，风险来自信息不对称。教师预测 $P_T(\cdot \mid x,r,y_{<t})$ 依赖学生推理时看不到的 $r$，学生预测 $P_S(\cdot \mid x,y_{<t})$ 只能依赖 $x$ 与已生成前缀。直接做分布匹配会推动学生吸收 $x \rightarrow r$ 的统计相关性，训练后期可能出现引用不可见 reference solution 的 privileged information leakage。

> [!intuition] RLSD 的核心定位
> RLSD 保留自我蒸馏中有价值的 token-level 差异信号，并改变它进入优化的方式：verifier reward 决定强化或惩罚的方向，自蒸馏信号只决定同一条 trajectory 内不同 token 分到多少 credit。

## OPSD 为什么会失效

OPSD 的教师分布与学生分布分别是：

$$
P_T(\cdot \mid x,r,y_{<t}) = \pi_\theta(\cdot \mid x,r,y_{<t}),
\qquad
P_S(\cdot \mid x,y_{<t}) = \pi_\theta(\cdot \mid x,y_{<t})
$$

如果学生只能访问 $x$，它可以逼近的是教师在 $r$ 上的边际分布：

$$
\bar{P}_T(y_t \mid x,y_{<t})
= \mathbb{E}_{r \sim P(r \mid x,y_{<t})}
\left[P_T(y_t \mid x,r,y_{<t})\right].
$$

OPSD 的训练目标要求每个具体 $(x,r)$ 上的学生分布匹配教师条件分布。论文给出的 KL 分解为：

$$
\mathcal{L}_{\text{OPSD}}
=
\mathcal{L}^*
+
I(Y_t;R \mid X,Y_{<t}).
$$

其中 $\mathcal{L}^*$ 是理想的边际匹配目标，$I(Y_t;R \mid X,Y_{<t})$ 衡量教师当前 token 预测对特权信息 $R$ 的依赖程度。只要教师确实利用了 $r$，这个条件互信息就是正的；学生的输入里没有 $r$，优化无法消除这部分残差。

在梯度层面，OPSD 的期望梯度可以对应到边际匹配，但单样本梯度含有 $r$ 特定偏差：

$$
g(\theta;r)=g^*(\theta)+\delta(\theta;r),
\qquad
\mathbb{E}_r[\delta(\theta;r)] = 0.
$$

偏差方差随教师分布对 $r$ 的敏感度增大。训练早期，边际匹配梯度 $g^*$ 仍然强，模型会快速提升；当学生接近边际分布后，$g^*$ 变弱，单样本偏差开始主导路径依赖的参数更新，于是模型逐步编码输入与特权答案之间的 spurious correlation。论文用 KL 停滞、leakage 计数上升、OPSD 后期性能下降支持这一解释。

## 核心机制

RLSD 把自我蒸馏教师从“生成目标”改成“幅度评估器”。对一条学生采样的 trajectory $y=(y_1,\ldots,y_T)$，它先计算同一 token 在普通上下文和特权上下文下的 log-probability 差：

$$
\Delta_t
=
\texttt{sg}\left(
\log P_T(y_t) - \log P_S(y_t)
\right).
$$

$\texttt{sg}$ 是 stop-gradient。这里的 $P_S(y_t)$ 表示模型只看问题与前缀时对 token $y_t$ 的概率，$P_T(y_t)$ 表示同一模型额外看到特权信息 $r$ 后的概率。$\Delta_t$ 只作为权重信号使用，避免额外的蒸馏损失路径。

接着，RLSD 根据序列级 advantage $A$ 的符号构造方向感知权重：

$$
w_t
=
\exp(\operatorname{sign}(A)\Delta_t)
=
\left(
\frac{P_T(y_t)}{P_S(y_t)}
\right)^{\operatorname{sign}(A)}.
$$

这个比率可以理解为 evidence ratio：特权信息让模型对当前 token 的信念提高了多少倍。对于正确 trajectory（$A>0$），$w_t=P_T/P_S$，被特权信息支持的 token 获得更大正信用；对于错误 trajectory（$A<0$），$w_t=P_S/P_T$，被特权信息反对的 token 承担更大惩罚。由于 $w_t$ 始终为正，$\hat{A}_t$ 的符号由环境 reward 继承，不会被教师信号翻转。

为了限制单个 token 的影响，RLSD 像 [[PPO]] 和 [[GRPO]] 一样使用 clipping：

$$
\hat{A}_t
=
A \cdot \operatorname{clip}(w_t,1-\epsilon_w,1+\epsilon_w).
$$

实际训练中还会用混合系数 $\lambda$ 在 uniform advantage 与 reweighted advantage 之间插值：

$$
\hat{A}_t
=
A \cdot
\left(
(1-\lambda)
+
\lambda \cdot \operatorname{clip}(w_t,1-\epsilon_w,1+\epsilon_w)
\right).
$$

这样可以在训练初期避免从 GRPO 突然切换到强 token-level 重加权。论文实验中 $\lambda$ 从 $0.5$ 线性衰减到 $0$，前 50 个 step 完成过渡，$\epsilon_w=0.2$。

## 与 GRPO 的关系

[[GRPO]] 对每条 response 采样 $G$ 个 rollout，用 group 内 reward 均值和标准差计算序列级 advantage：

$$
A^{(i)}
=
\frac{R(x,y^{(i)})-\mu_G}{\sigma_G}.
$$

标准 GRPO 将同一个 $A^{(i)}$ 复制到 response 的所有 token。RLSD 只替换这一项：每个 token 使用 $\hat{A}_t^{(i)}$，policy update 仍属于同一个 policy gradient 框架：

$$
\Delta\theta
\propto
\mathbb{E}_{y\sim P_S(\cdot \mid x)}
\left[
\sum_{t=1}^{|y|}
\hat{A}_t
\nabla_\theta \log P_S(y_t \mid x,y_{<t})
\right].
$$

因此 RLSD 可以作为 GRPO pipeline 的 drop-in replacement：rollout、verifier reward、group-relative advantage、policy ratio clipping 等主体流程保持一致，新增成本是一轮带特权上下文 $(x,r,y)$ 的 forward pass，用来得到 teacher logits。论文认为这部分成本相对于 rollout generation 较小。

> [!comparison] 三种 token-level advantage
> - GRPO：$\hat{A}_t=A$，方向可靠，token 间没有区分。
> - OPSD：$\hat{A}_t$ 来自 teacher-student 分布差异，信号密集，方向会随特权教师改变。
> - RLSD：$\hat{A}_t=A \cdot \text{positive weight}$，方向由 verifier reward 锚定，幅度由自蒸馏信号细分。

## 自我蒸馏改变了 RLVR 的什么

RLVR 的核心优势是 reward 来源可靠：数学题、代码题、多模态问答等任务可以通过 verifier 判断最终答案。但序列级 reward 对长推理链过于粗糙，容易把所有 token 一起推高或压低，造成 entropy collapse 或无差别惩罚。

RLSD 引入的自我蒸馏信号改变了 reward 的“分配形状”。对于正确答案，模型额外看到 ground-truth answer 后更支持的 token 会获得更大强化；对于错误答案，模型额外看到答案后更反对的 token 会受到更大惩罚。这样，RLVR 仍然由环境反馈决定训练目标，token-level 更新更接近“哪些局部决策真正影响结果”。

这个变化解释了论文中的两个现象：RLSD 在训练 reward 上比 GRPO 上升更快，收敛上限更高；同时它比 OPSD 稳定，因为特权信息没有作为可模仿的生成目标进入梯度方向。

## 实验结论

论文在 Qwen3-VL-8B-Instruct 上评估多模态推理任务，主要结果如下：

| 方法 | MMMU | MathVista | MathVision | ZeroBench | WeMath | Avg. |
|------|------|-----------|------------|-----------|--------|------|
| Base LLM | 62.44 | 73.80 | 47.37 | 19.76 | 54.10 | 51.49 |
| GRPO | 65.11 | 76.20 | 48.82 | 22.60 | 56.57 | 53.86 |
| OPSD | 63.82 | 75.10 | 47.53 | 21.06 | 54.95 | 52.49 |
| SDPO | 65.11 | 74.00 | 47.27 | 25.15 | 52.19 | 52.74 |
| GRPO+OPSD | 63.22 | 75.90 | 48.52 | 22.16 | 54.76 | 52.91 |
| RLSD | 67.22 | 78.10 | 52.73 | 24.85 | 58.00 | 56.18 |

几个信号值得保留：

- RLSD 平均准确率比 Base LLM 高 $4.69$，比 GRPO 高 $2.32$。
- MathVista 与 MathVision 的收益更明显，符合“细粒度推理步骤需要细粒度信用分配”的解释。
- 训练曲线中，RLSD 避开 OPSD 的后期 collapse，并保持比 GRPO 更高的 entropy。
- Case study 显示，正确 cube-counting 轨迹中关键计数与减法 token 得到更大 credit；错误 bar-model 轨迹中，误读关系式与错误答案得到更强惩罚。

## 实现要点

RLSD 的最小实现可以按以下流程理解：

```text
for each question x with privileged information r:
  sample G responses y^(1), ..., y^(G) from current policy
  compute verifier reward R(x, y^(i))
  compute group-relative advantage A^(i)

  for each response y^(i):
    run one teacher forward pass with (x, r, y^(i))
    for each token y_t:
      Delta_t = stop_gradient(log P_T(y_t) - log P_S(y_t))
      w_t = exp(sign(A^(i)) * Delta_t)
      A_hat_t = A^(i) * ((1 - lambda) + lambda * clip(w_t, 1-eps_w, 1+eps_w))

  update policy with A_hat_t inside the GRPO-style objective
```

论文实验中的工程设置包括：

- Base model：Qwen3-VL-8B-Instruct。
- Rollout group size：每个 prompt 采样 8 条 response。
- Learning rate：GRPO、GRPO+OPSD、RLSD 使用 $1\times10^{-6}$。
- Context：max prompt length 4096，max response length 4096。
- Teacher synchronization：每 10 个 training steps 同步一次教师参数，同步间隔内冻结。
- Privileged information：RLSD 只需要最终 ground-truth answer；OPSD 使用 verified reasoning traces。

## 边界与失效条件

> [!warning] 适用边界
> RLSD 依赖可验证 reward 与可用的特权信息。没有可靠 verifier 时，方向锚定会失去基础；ground-truth answer 本身噪声较大时，token-level 权重也会把噪声注入 credit 分配。

当前论文版本的实验证据主要来自多模态推理。作者提到已在纯文本推理、视频理解和更多模型族上做了初步验证，但正式结果仍待后续版本补充。对于开放式偏好对齐、creative writing、helpfulness 等 verifier 难以稳定定义的任务，RLSD 的适用性需要单独验证。

RLSD 也没有直接取代 [[Process Reward Model|PRM]] 或 step-level verifier。PRM 试图显式评估中间步骤质量，RLSD 用同一模型在特权上下文下的信念变化作为轻量代理。当前证据支持它能改善 token-level credit assignment，但它是否能覆盖过程监督在复杂推理中的全部作用，还需要更多对照实验。

## 知识连接

理解 RLSD 的第一条线是 [[GRPO]]：先掌握 group-relative advantage 如何把 verifier reward 转成 policy gradient，再看 RLSD 如何把同一个 advantage 重分配到 token。第二条线是 [[On-Policy Distillation]]：OPD 说明 on-policy trajectory 上的 teacher logits 可以提供密集监督，OPSD 暴露了信息不对称下的 leakage 风险。第三条线是 [[Credit Assignment]] 与 [[Process Reward Model]]：它们共同指向长程推理训练中“最终结果如何归因到局部动作”的问题。

RLSD 在方法谱系上靠近 [[VAPO (2025)]] 关心的 token-level credit assignment，但它避免训练 value model；它也靠近 self-distillation 方法，但优化目标仍由 verifier reward 主导。后续可以关注 TRRD、MOPD、SDPO 等把 distillation signal 注入 RL 的方法，比较它们让教师信号进入 trust region、advantage magnitude 或 auxiliary loss 的差异。
