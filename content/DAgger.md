---
type: method
description: DAgger 通过在学习者实际访问的状态上查询教师标签，缓解长序列模仿学习中的 covariate shift；在 LLM agent 中可实现为 teacher-interleaved rollouts。
aliases:
  - Dataset Aggregation
  - 数据集聚合
  - DAgger for LLM Agents
  - Teacher-Interleaved DAgger
prerequisites:
  - "[[SFT]]"
  - "[[On-Policy Distillation]]"
  - "[[SWE-Bench]]"
tags:
  - post-training
  - agentic
  - alignment
  - software-engineering
  - training-stability
created: 2026-05-24
updated: 2026-05-24
---

# DAgger

DAgger（Dataset Aggregation）是一种面向序列决策的 imitation learning 方法：让学习者进入自己会访问的状态，再请教师给出这些状态下的正确动作。放到 multi-turn LLM agents 里，它对应一种 teacher-interleaved post-training recipe：轨迹由学生和教师交替执行，训练标签始终来自教师。

这个方法的核心价值在于解决长 horizon 任务里的 [[Covariate Shift|covariate shift]]。[[SFT]] 或 behavior cloning 只看教师轨迹，部署时 agent 会被自己早期动作带到训练集中很少出现的状态；DAgger 把训练状态逐步移向学生分布，同时保留教师的 dense action supervision。

## 动机

> [!intuition] 为什么 long-horizon agent 需要 DAgger？
> 在单轮回答里，一个 token 级错误通常只影响后续局部文本；在软件工程 agent、web agent、tool-use agent 里，一个早期动作会改变环境状态。文件查错、命令失败、错误编辑都会改变后续 observation，后面的决策开始发生连锁偏移。
>
> [[SFT]] 给学生看的是教师会遇到的状态。部署时，学生访问的是自己动作诱导出的状态。两者分布错开后，模型在训练中学到的局部 imitation 能力会被放大成轨迹级失败。

[[On-Policy Distillation]] 已经把训练上下文移到学生访问状态，因此能缓解纯离线蒸馏的 exposure mismatch。它的早期训练仍然依赖学生 rollout：弱学生常在前几步进入无效轨迹，教师信号会集中在低质量前缀上。DAgger 的 teacher interleaving 让早期轨迹也有教师接管和恢复机会，训练数据更容易覆盖有用的长程行为。

## 经典形式

考虑有限 horizon 的序列决策问题，教师策略为 $\pi_e$，学生策略为 $\pi_\theta$，$d_\pi$ 表示由策略 $\pi$ 诱导出的平均状态分布。Behavior cloning 的目标可以写成：

$$
\min_\theta\ \mathbb{E}_{s\sim d_{\pi_e}, a_e\sim \pi_e(\cdot|s)}
\left[\ell(\pi_\theta(\cdot|s), a_e)\right].
$$

这里的状态来自 $d_{\pi_e}$。部署时学生访问 $d_{\pi_\theta}$，局部错误会沿着时间累积。经典模仿学习分析里，behavior cloning 的轨迹误差可能随 horizon 呈 $O(T^2\epsilon)$ 级增长；DAgger 通过训练在学生访问状态上的教师标签，把依赖关系改善到 $O(T\epsilon)$ 量级。

> [!math] DAgger 的数据聚合过程
> 第 $i$ 轮使用混合策略：
> $$
> \mu_i=\beta_i\pi_e + (1-\beta_i)\pi_{\theta_i},
> $$
> 其中 $\beta_i$ 随训练下降。执行 $\mu_i$ 得到一批访问状态后，在每个状态查询教师动作：
> $$
> \mathcal{D}_{i+1}
> =
> \mathcal{D}_i \cup
> \{(s,a_e): s\sim d_{\mu_i},\ a_e\sim\pi_e(\cdot|s)\}.
> $$
> 下一轮学生通过监督学习更新：
> $$
> \theta_{i+1}
> =
> \arg\min_\theta
> \mathbb{E}_{(s,a_e)\sim \mathcal{D}_{i+1}}
> \left[\ell(\pi_\theta(\cdot|s),a_e)\right].
> $$

直觉上，$\beta_i$ 控制训练状态从教师分布到学生分布的迁移速度。早期让教师多参与，轨迹更容易保持有效；后期让学生多参与，训练更贴近部署时的状态分布。

## 在 Multi-Turn LM Agents 中的版本

LLM agent 的状态 $s_t$ 可以看作当前任务、历史动作和环境 observation 的完整前缀：

$$
s_t = (x, a_{1:t-1}, o_{1:t-1}).
$$

动作 $a_t$ 通常包含 reasoning text、tool call、代码编辑、finish action 等。环境执行动作后返回 $o_t$，直到 agent 提交答案或达到 turn budget。

### Turn-Level DAgger-Style Rollout

在第 $i$ 轮训练里，每个 turn 独立决定由教师还是学生执行：

$$
b_t\sim \mathrm{Bernoulli}(\beta_i).
$$

执行动作来自：

$$
a_t\sim
\begin{cases}
\pi_e(\cdot|s_t), & b_t=1,\\
\pi_{\theta_i}(\cdot|s_t), & b_t=0.
\end{cases}
$$

无论当前 turn 由谁执行，训练数据都保存教师在同一状态下的标签 $\tilde a_t\sim \pi_e(\cdot|s_t)$：

$$
\mathcal{B}(\tau)=\{(s_t,\tilde a_t):t=1,\ldots,T\}.
$$

然后用 cross-entropy 训练学生模仿教师动作：

$$
\mathcal{L}_i(\theta)
=
\mathbb{E}_{(s_t,\tilde a_t)\sim \mathcal{D}_i}
\left[
-\sum_{j=1}^{m_t}
\log \pi_\theta(\tilde a_{t,j}\mid s_t,\tilde a_{t,<j})
\right].
$$

这里的关键设计是：状态来自 teacher-student mixture，标签来自 teacher。学生会看到自己动作导致的状态，也会学习这些状态下教师会怎样恢复。

### AggreVaTe-Style Rollout

AggreVaTe-style 采用 trajectory-level mixture。先采样一个学生前缀长度 $\kappa$，学生执行前 $\kappa$ 个 turn，之后教师接管完成轨迹：

$$
b_{1:\kappa}=0,\qquad b_{\kappa+1:T}=1.
$$

这种方式更像“让学生先试，再让教师完成”。它对长程 agent 有一个实用优点：学生前缀暴露部署状态，教师后缀把轨迹拉回可学习的成功路径附近。

## 与 SFT、RL、OPD 的统一视角

很多 post-training 算法都可以写成“采样状态和动作，再加权提高这些动作概率”的形式：

$$
\theta_{i+1}
=
\arg\max_\theta
\mathbb{E}_{s\sim p_s,\ a\sim p_a}
\left[
w(s,a)\log\pi_\theta(a|s)
\right]
-\lambda\Omega_i(\theta).
$$

| 方法 | 状态分布 $p_s$ | 动作/标签分布 $p_a$ | 权重 $w(s,a)$ | 核心效果 |
|---|---|---|---|---|
| [[SFT]] / behavior cloning | 教师状态 $d_{\pi_e}$ | 教师动作 $\pi_e$ | $1$ | dense supervision，状态覆盖偏离部署分布 |
| [[GRPO]] / policy gradient | 学生状态 $d_{\pi_\theta}$ | 学生动作 $\pi_\theta$ | advantage | on-policy coverage，依赖稀疏 reward 和 group sampling |
| [[On-Policy Distillation]] | 学生状态 $d_{\pi_\theta}$ | 学生动作 $\pi_\theta$ | teacher-based distillation weight | on-policy context，早期受 student rollout 质量限制 |
| DAgger-style | 混合状态 $d_i^{\text{turn}}$ | 教师动作 $\pi_e$ | $1$ | 部署状态覆盖 + 教师 dense labels |
| AggreVaTe-style | 学生前缀/教师后缀状态 $d_i^{\text{traj}}$ | 教师动作 $\pi_e$ | $1$ | 学生暴露状态 + 教师轨迹恢复 |

这个表给出一个很清晰的定位：DAgger 把“看哪些状态”和“学哪些动作”拆开。状态逐渐靠近学生部署分布，动作标签保持教师质量。对于黑盒教师模型，这也更实用，因为训练只需要教师生成动作，不需要访问 teacher logits。

## 为什么适合软件工程 Agent

软件工程 agent 在 [[SWE-Bench]] 这类任务上需要连续执行搜索、阅读、定位、编辑、测试和提交。一个早期错误会改变后续所有 observation：查错文件、误删代码、运行错误命令都会让后续上下文偏离 expert demonstration。

DAgger-style training 对这类任务特别贴合：

- 学生执行的 turn 暴露真实部署状态，让训练覆盖 agent 自己会制造的问题。
- 教师执行的 turn 提供 recovery path，让早期弱学生也能看到有效长轨迹。
- 每个 visited state 都有教师动作标签，学习信号比只看 terminal reward 的 [[GRPO]] 更密集。
- 只需教师动作，兼容 GPT、Gemini 等黑盒 teacher。

> [!paper] 论文出处
> Li et al. (2026), *Revisiting DAgger in the Era of LLM-Agents*，对应 clipping：[[Clippings/Paper/2605.12913/2605.12913|2605.12913]]。
>
> 论文在 SWE-Gym 上训练 Qwen3-4B/8B 学生，用 Qwen3-Coder-30B-A3B-Instruct 作为教师，并在 OpenHands scaffold 下评估。DAgger-style training 在 SWE-Bench Verified 上达到 4B $27.3\%$、8B $29.8\%$，相对最强非 DAgger post-training baseline 分别提升 $+3.9$ 和 $+3.6$ points。

## 实现细节与训练配方

> [!example] 论文中的一个具体配置
> - 每轮 rollout batch：$512$ 个 SWE task instances
> - 在线迭代：$5$ 轮 rollout-update
> - 学习率：$3\times 10^{-6}$
> - batch size：$16$
> - 每个 online batch 训练：$3$ epochs
> - rollout sampling：temperature $0.7$，top-$p=0.9$
> - evaluation：greedy decoding
> - context length：$64$K tokens
> - 最大环境交互：$100$ turns

论文中的 turn-level DAgger schedule 从强教师引导开始：

$$
\beta_i=\max(0.6, 1.0-0.2(i-1)).
$$

也就是说，前几轮逐步增加学生执行比例，但教师执行概率保持至少 $0.6$。这个设置偏保守，适合软件工程这种长程失败成本很高的环境。

## 局限性

> [!warning] 边界条件
> - 需要一个足够强的教师。教师质量差时，DAgger 会稳定地学习错误动作。
> - 需要可交互环境。只拥有离线 demonstration 数据时，DAgger 的 state-distribution correction 很难发挥作用。
> - 教师查询成本高。每个 visited state 都查询教师，长轨迹会显著增加 inference cost。
> - 当前证据集中在 SWE/OpenHands 场景。Web navigation、data analysis、scientific agent 等领域需要单独验证。
> - 如果主要瓶颈来自 context overflow、工具权限、环境不稳定或 benchmark 噪声，DAgger 只能改善策略学习部分。

## 面试要点

> [!interview] Q: DAgger 和 SFT 的核心区别是什么？
> SFT 在教师轨迹上训练，状态分布来自 $\pi_e$。DAgger 在混合策略访问到的状态上训练，随着训练推进逐渐覆盖学生会访问的状态。标签仍然由教师提供，所以它保留了监督学习的 dense feedback。

> [!interview] Q: DAgger 和 On-Policy Distillation 的差别在哪里？
> OPD 的上下文来自学生 rollout，动作样本也来自学生，再用教师信号加权或蒸馏。DAgger 的上下文来自 teacher-student mixture，动作标签直接来自教师。这个设计让早期训练更容易得到有效长轨迹，也降低了对 teacher logits 的依赖。

> [!interview] Q: 为什么 DAgger 能缓解 long-horizon covariate shift？
> 长程任务里，部署状态由学生动作递归产生。DAgger 在训练时主动采样这些学生诱导状态，并在这些状态上查询教师动作。模型学到的是“在自己会到达的位置如何恢复或继续”，因此局部 imitation error 对后续轨迹的放大程度更低。

> [!interview] Q: 什么时候 DAgger 不划算？
> 如果任务很短、学生初始策略已经稳定、教师查询很贵，或者环境交互成本远高于离线数据训练，DAgger 的额外 rollout 和 teacher labeling 可能抵消收益。对于 reward 信号密集且探索容易的任务，直接 RL 或 OPD 也可能更简单。
