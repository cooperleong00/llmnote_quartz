---
type: method
description: TCOD 通过按轨迹深度推进的时间课程稳定多轮智能体的 on-policy distillation，缓解跨轮错误累积导致的 KL 升高。
aliases:
  - Temporal Curriculum On-Policy Distillation
  - 时间课程在线蒸馏
  - 时间课程 On-Policy Distillation
prerequisites:
  - "[[On-Policy Distillation]]"
  - "[[KL Divergence]]"
  - "[[SFT]]"
tags:
  - post-training
  - alignment
  - efficiency
  - optimization
created: 2026-05-01
updated: 2026-05-01T12:53
---

# TCOD

TCOD（Temporal Curriculum On-Policy Distillation）是一种面向多轮 autonomous agents 的 [[On-Policy Distillation|on-policy distillation]] 方法。它把训练难度定义为 student 需要亲自执行的轨迹深度，从短轨迹逐步扩展到长轨迹，让 student 先学会低错误累积的局部决策，再学习完整任务执行。

TCOD 解决的核心问题是 [[Trajectory-Level KL Instability|轨迹级 KL 不稳定]]：在多轮环境中，student 的一次错误会改变后续 observation 和 history，使后续状态逐渐偏离 teacher 熟悉的分布，最终让 [[KL Divergence|KL 散度]] 升高、teacher supervision 变得不可靠。

## 动机

> [!intuition] 为什么多轮 OPD 会不稳定？
> 在单轮推理任务中，student 生成一个答案，teacher 在这个答案的上下文上给出 token-level 或 response-level 的蒸馏信号。多轮 agent 任务多了一层因果耦合：第 $t$ 轮 action 会改变第 $t+1$ 轮 observation，history $h_t$ 会不断吸收 student 自己造成的状态变化。
>
> 当早期 action 有偏差时，后续状态会越来越偏离 teacher 的有效支持域。teacher 在这些状态上仍然可以输出分布，但这个分布对 student 的训练价值下降，表现为 trajectory-level KL 升高和 success rate 下降。

多轮 agent 的 history 可以写成：

$$
h_t=(o_0,a_0,o_1,a_1,\ldots,o_{t-1},a_{t-1},o_t)
$$

完整轨迹为：

$$
\tau=(h_0,a_0,h_1,a_1,\ldots,h_{T-1},a_{T-1})
$$

vanilla OPD 在 student 自己采样的轨迹上对齐 teacher：

$$
\mathcal{L}_{\text{OPD}}(\theta)=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}
\mathcal{D}_{\text{KL}}
\left(
\pi_\phi(a_t\mid h_t)
\parallel
\pi_\theta(a_t\mid h_t)
\right)
\right]
$$

这里 $\pi_\phi$ 是 teacher policy，$\pi_\theta$ 是 student policy。这个目标的优势是 dense supervision：每一轮都能得到 teacher 分布作为训练信号；它在长 horizon agent 任务中的弱点来自完整轨迹一次性暴露给 student，早期错误会被后续轮次放大。

## 核心机制

TCOD 的关键操作很小：训练初期只让 student 面对深度为 $k$ 的短轨迹，之后用 pacing schedule 逐步增大 $k$，直到覆盖完整 horizon。论文采用线性增长：

$$
k = k_{\text{start}} + \left\lfloor n / \eta \right\rfloor
$$

其中 $n$ 是当前训练 step，$k_{\text{start}}$ 是初始轨迹深度，$\eta$ 控制课程增长速度。较小的 $\eta$ 让课程更快进入长轨迹，较大的 $\eta$ 让 student 在当前深度停留更久，通常带来更平稳的 KL 曲线。

> [!intuition] TCOD 的学习路径
> TCOD 把“能完成完整任务”拆成一串按时间展开的子能力：先在短 history 下稳定模仿 teacher，再逐步承受更长 history 带来的 partial observability、错误累积和长程 planning 压力。难度来自 trajectory depth 本身，因此不需要额外的 difficulty model。

## 两种课程方向

### TCOD-F2B

TCOD-F2B（Forward-to-Backward）从轨迹开头向后扩展。训练第 $n$ 步时，student 最多 rollout $k$ 步，并只在这些早期轮次上计算蒸馏损失：

$$
\mathcal{L}_{\text{TCOD-F2B}}(\theta)=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{k-1}
\mathcal{D}_{\text{KL}}
\left(
\pi_\phi(a_t\mid h_t)
\parallel
\pi_\theta(a_t\mid h_t)
\right)
\right]
$$

这种方式适合从任务初始状态开始学习 early-turn decision making。它的工程优势是实现简单，早期 rollout 更短，数据采集成本也更低。

### TCOD-B2F

TCOD-B2F（Backward-to-Forward）从接近成功的后段任务开始，再逐步向初始状态扩展。它先用 teacher 的 successful trajectory 前缀把环境推进到中间状态，然后让 student 接管剩余 $k$ 步；teacher 前缀只负责导航，不参与梯度更新。

$$
\mathcal{L}_{\text{TCOD-B2F}}(\theta)=
\mathbb{E}_{\tau\sim(\pi_\phi,\pi_\theta)}
\left[
\sum_{t=L-k}^{L}
\mathcal{D}_{\text{KL}}
\left(
\pi_\phi(a_t\mid h_t)
\parallel
\pi_\theta(a_t\mid h_t)
\right)
\right]
$$

其中 $L$ 是 teacher successful trajectory 的长度。训练初期 student 从 near-terminal state 开始，主要学习如何完成最后几步；随着 $k$ 增大，teacher 前缀缩短，student 最终从初始状态执行完整任务。

> [!comparison] F2B 与 B2F 的取舍
> F2B 更轻量，早期 trajectory 短，训练时间节省明显。B2F 借助 teacher successful prefix 控制早期状态质量，适合 student 容易在任务开头走偏、导致后续 supervision 失真的环境。

## 这里的 on-policy distillation 是什么

TCOD 继承 [[On-Policy Distillation]] 的核心设定：数据来自 student 当前 policy 的 rollout，teacher 在这些 student-induced states 上提供分布级监督。它和 [[SFT]] 的差别在于数据分布持续跟随 student 更新；它和 [[RLHF]] 或 [[GRPO]] 的差别在于训练信号主要来自 teacher distribution 的 KL，而非稀疏 reward 或 group-relative advantage。

这个设定对 multi-turn agents 很关键。student 自己 rollout 可以暴露真实 test-time distribution；teacher KL 可以提供比 binary task success 更密集的监督。TCOD 的时间课程负责调节二者之间的张力：保持 on-policy 数据，同时控制 student 进入高错误累积状态的速度。

## 工程实现

论文实现中还加入了异步 rollout 和训练，以提高 GPU 利用率。actor processes 持续采样 trajectories，central learner 从 shared buffer 中训练；每条 trajectory 带有采样时 policy version，过旧经验会被 staleness filter 丢弃。

多轮环境中一条完整 trajectory 可以拆成多个 prefix sub-trajectories 复用，但 replay 需要限制 stale policy 造成的 off-policy 偏移。论文经验设置为 $\Delta_{\max}=2$，在 sample efficiency 和 on-policy 约束之间取得较好平衡。

## 实验结果与意义

论文在 ALFWorld、WebShop、ScienceWorld 三个多轮 agent benchmark 上评估 TCOD，覆盖 embodied navigation、e-commerce navigation 和 scientific reasoning。主要结论如下：

- 在 ALFWorld 上，TCOD 相比 vanilla OPD 提升 success rate，Qwen2.5-3B student 的 valid unseen success rate 从 60.45 提升到 79.19，提升 18.74 points。
- 对小模型，TCOD 能缓解 vanilla OPD 中 success rate collapse 和 KL escalation 的同步出现。
- 对较大 student，TCOD 带来更平稳的 KL 曲线、更少 action rounds 和更快 convergence。
- 在 hard split 上，TCOD-B2F 的 student success rate 可超过 teacher，说明 student 通过课程训练获得了超出 teacher 单次策略表现的泛化能力。
- 课程增长率 $\eta\in\{2,4,6\}$ 的性能波动小于 2%，部署时对该超参数不太敏感。
- 训练成本方面，TCOD-F2B/B2F 相比 vanilla OPD 最多减少约 32% 总训练时间，主要收益来自早期短 rollout。

TCOD 和 [[GiGPO]] 都面向 multi-turn LLM agents，但关注点不同：GiGPO 处理 agentic RL 中的细粒度 credit assignment；TCOD 处理 distillation 训练中跨轮错误累积导致的 teacher supervision 失真。

## 局限性与边界

> [!warning] 边界条件
> TCOD 依赖 teacher 在目标环境中提供有用分布。论文显示 domain-specific teacher 的效果强于更大但领域适配较弱的 general teacher；teacher 的目标域能力会限制 TCOD 的上界。
>
> TCOD-B2F 需要预先收集 teacher successful trajectories。若 teacher 很难生成成功轨迹，B2F 的初始化成本会升高，训练覆盖也会受到限制。
>
> 课程深度主要按时间步增长，适合难度和 trajectory depth 强相关的任务。若任务难度更多来自隐藏约束、工具调用质量或环境随机性，仅调节 rollout depth 可能不足。
>
> TCOD 缓解 KL instability，但不会自动解决 reward hacking、工具安全、memory management、long-horizon credit assignment 等 agent training 问题。需要 RL 信号或精细 credit assignment 时，可以把它和 [[GRPO]]、[[GiGPO]] 等方法放在同一个训练设计空间中比较。

## 面试视角

> [!interview] Q: TCOD 为什么能稳定多轮 OPD？
> A: 多轮 OPD 的不稳定来自跨轮错误累积：student 早期 action 改变后续 observation，使 history 逐渐偏离 teacher 熟悉的状态分布，KL 升高后 supervision 质量下降。TCOD 用 trajectory depth curriculum 控制 student 暴露到长 horizon 的速度，让它先在短轨迹上稳定对齐 teacher，再逐步学习完整任务。

> [!interview] Q: TCOD-F2B 和 TCOD-B2F 怎么选？
> A: F2B 从初始状态开始，只限制 early rollout depth，工程最简单且节省训练时间。B2F 用 teacher successful prefix 把 student 放到接近成功的状态，再逐步减少 teacher prefix，适合早期错误会严重污染后续状态的任务。

> [!paper] 论文出处
> Jiaqi Wang, Wenhao Zhang, Weijie Shi, Yaliang Li, James Cheng. 2026. *TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents*. Clipping: [2604.24005](Clippings/Paper/2604.24005/2604.24005.md).
