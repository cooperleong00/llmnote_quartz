---
type: method
description: OpenClaw-RL 从智能体交互后的后继状态信号中提取评估与指令信号，用异步服务端-客户端架构和混合 RL 目标持续优化个人与通用 agent
aliases:
  - OpenClaw RL
  - OpenClaw 强化学习
  - 从后继状态信号学习的在线 Agent RL
prerequisites:
  - "[[Agentic RL]]"
  - "[[Process Reward Model]]"
  - "[[On-Policy Distillation]]"
  - "[[GRPO]]"
tags:
  - post-training
  - optimization
  - evaluation
  - alignment
created: 2026-06-01
updated: 2026-06-01
---

# OpenClaw-RL

OpenClaw-RL 是一种面向智能体（agent）的在线强化学习框架：每次 agent 行动之后，系统把用户回复、工具结果、终端输出、GUI 状态变化、测试结果等后继状态转成训练信号，让模型在真实使用中持续优化。它把 [[Agentic RL]] 的训练入口从预先收集的数据集扩展到部署时产生的交互流，并用混合目标同时利用标量评价和词元级（token-level）指令。

> [!paper] 论文出处
> Wang et al., 2026, *OpenClaw-RL: Train Any Agent Simply by Talking*. arXiv: [2603.10165](https://arxiv.org/abs/2603.10165), GitHub: [Gen-Verse/OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)。本笔记吸收自 [[Clippings/Paper/2603.10165/2603.10165|OpenClaw-RL: Train Any Agent Simply by Talking]]，重点整理为方法笔记。

## 动机

> [!intuition] 为什么需要从后继状态学习？
> Agent 的一次行动天然会留下“下一步发生了什么”：用户追问表示前一轮可能没满足需求，测试通过表示代码改动有效，stderr 或 traceback 指向失败原因，GUI 变化反映任务进度。OpenClaw-RL 把这些 [[Next-State Signals|后继状态信号（next-state signal）]] 视为在线监督来源，使个人 agent 能从日常使用中适应用户偏好，也使终端、GUI、SWE、工具调用等通用 agent 共享同一套训练入口。

传统 [[RLHF]] 链路通常先收集偏好数据，再训练 [[Reward Model]]，最后用 [[PPO]] 或相关 RL 方法优化策略。OpenClaw-RL 面向更动态的 agent 场景：数据在部署时持续产生，用户侧框架和工具环境会变化，训练过程还需要避免阻塞线上推理。因此它同时解决两个问题：

- **系统问题**：如何让个人设备、云端环境、工具调用和训练集群稳定接入同一个 RL 服务端。
- **算法问题**：如何把后继状态中的“做得好不好”和“应该怎样改”同时转成稳定梯度。

## 核心机制

OpenClaw-RL 的核心链路可以分成三层：

1. **交互流接入**：RL 服务端把策略 $\pi_\theta$ 暴露为无状态 completion API，个人设备或云端环境调用 API 并通过 HTTP 回传交互数据。
2. **信号抽取**：异步 [[Process Reward Model|PRM]] 服务读取动作 $a_t$ 和后继状态 $s_{t+1}$，产出评估信号（evaluative signal）与指令信号（directive signal）。
3. **策略更新**：训练器把标量奖励的 [[GRPO]] loss 与基于提示的 [[On-Policy Distillation|OPD]] loss 合成一个混合 RL 目标（hybrid RL objective）。

### 服务端-客户端在线架构

OpenClaw-RL 把 agent 使用环境和 RL 训练系统解耦。用户侧框架只需要会调用 completion API，就可以成为数据源；服务端通过会话编号（session id）区分不同用户和不同会话，通过请求类型区分主线轮次（main-line turn）与旁路轮次（side turn）。

| 组件 | 作用 | 为什么重要 |
|---|---|---|
| 策略服务（policy serving） | 对外提供推理 API | 用户侧 agent 可以持续工作 |
| 环境托管（environment hosting） | 承载个人设备或云端并行环境 | 同一架构覆盖个人 agent 和通用 agent |
| PRM / Judge | 从 $a_t, s_{t+1}$ 抽取奖励与提示（hint） | 信号抽取可以用更强 judge，且不增加推理延迟 |
| 策略训练（policy training） | 根据收集到的样本更新模型 | 权重在边界点推送到推理服务引擎，保持线上策略一致 |

这种架构让稀疏的个人对话流和密集的云端并行采样轨迹（rollout）都进入同一训练管线。论文覆盖的环境包括 OpenClaw 个人设备、终端沙箱、GUI 屏幕状态与可访问性树、SWE 仓库与测试套件、工具调用 API 返回值。

### 两类后继状态信号

> [!definition] 评估信号与指令信号
> **评估信号（evaluative signal）** 给前一动作一个标量评价，例如 $+1/-1/0$。它来自 PRM 对 $(a_t, s_{t+1})$ 的多次投票，多数票形成 $r_t$。
>
> **指令信号（directive signal）** 从后继状态中抽取可执行的纠正提示 $h$，再把 $h$ 拼到原始上下文 $s_t$，形成带提示的 prompt（hint-augmented prompt）$s_t^h=s_t\oplus h$，让教师分布 $\pi_T(\cdot\mid s_t^h)$ 提供词元级监督。

两类信号的价值互补。评估信号频率高：每个被评分轮次都能给出标量奖励；指令信号信息密度高：当后继状态包含明确纠正时，它能告诉模型哪些词元方向更合适。比如用户说“你应该先检查文件”，这既说明上一轮效果差，也给出了下一轮应该先做的动作；测试通过则主要提供评价，通常没有额外词元级指令。

[[RLVR]] 主要消费可验证的标量奖励，适合结果清晰的任务；OPD 能利用提示条件下的教师分布（hint-conditioned teacher distribution）。OpenClaw-RL 把两者放进同一个逐词元 loss：

> [!math] 混合 RL 目标
> 对词元 $i$，OpenClaw-RL 使用
> $$
> \mathcal{L}_i^{\mathrm{hybrid}}
> =
> w_{\mathrm{RL}}\mathcal{L}_i^{\mathrm{GRPO}}
> +
> w_{\mathrm{OPD}}\mathcal{L}_i^{\mathrm{OPD}}.
> $$
> 其中 $\mathcal{L}_i^{\mathrm{GRPO}}$ 由标量优势（advantage）驱动，$\mathcal{L}_i^{\mathrm{OPD}}$ 由提示条件下教师模型的词元分布驱动。论文默认 $w_{\mathrm{RL}}=w_{\mathrm{OPD}}=1$。

## 稳定训练的关键设计

指令信号的主要风险来自师生分布错配（teacher-student distribution mismatch）：教师在提示条件下可能把概率质量放到学生几乎不会采样的词元上，重要性比率（importance ratio）会变大，梯度也容易被低质量提示放大。OpenClaw-RL 用两个机制控制这个风险。

### 重合度引导的提示选择

给定学生已经生成的响应 $y$，在第 $i$ 个词元位置，设学生旧策略的 top-$k$ 词元集合为

$$
S_i^q=\mathrm{top}\text{-}k\{\pi_{\mathrm{old}}(\cdot\mid s_t,y_{<i})\},
$$

候选提示 $h$ 诱导的教师 top-$k$ 词元集合为

$$
S_{i,h}^p=\mathrm{top}\text{-}k\{\pi_T(\cdot\mid s_t^h,y_{<i})\}.
$$

OpenClaw-RL 用交集大小衡量提示与学生分布的支持集重合度：

$$
O[h,i]=|S_i^q\cap S_{i,h}^p|.
$$

选择方式有两种：

$$
h^\star(i)=
\begin{cases}
\arg\max_h \sum_i O[h,i], & \text{sequence-level},\\
\arg\max_h O[h,i], & \text{token-level}.
\end{cases}
$$

序列级（sequence-level）为整条轨迹选一个提示，词元级（token-level）为每个词元位置选提示。论文实验中两者效果接近，序列级在大批量 agentic RL 中更稳定。这个准则的直觉是：高重合度提示让教师在学生已有高概率区域内提供方向，蒸馏更新更容易落在学生当前策略能承接的位置。

### 对数概率差裁剪

选定 $h^\star$ 后，OPD loss 只在词表子集 $S_i$ 上计算，默认 $S_i=S_i^q$。对 $v\in S_i$：

$$
w_v=\mathrm{softmax}_{v\in S_i}\bigl(\ell_{\mathrm{old}}(v)\bigr),
$$

$$
\Delta_v=\mathrm{clip}\bigl(
\ell_{T,h^\star}(v)-\ell_{\mathrm{old}}(v),
-C,+C
\bigr),
$$

$$
A_v=\Delta_v\cdot w_v.
$$

这里 $\ell_{\mathrm{old}}(v)=\log\pi_{\mathrm{old}}(v\mid s_t,y_{<i})$，$\ell_{T,h^\star}(v)=\log\pi_T(v\mid s_t^{h^\star},y_{<i})$。$w_v$ 把权重集中到学生已经可能采样的词元，$\Delta_v$ 的裁剪把教师和学生的对数概率差限制在 $[-C,C]$。

再定义当前策略相对旧策略的逐词表项比率：

$$
\rho_v=\exp\bigl(\ell_{\mathrm{cur}}(v)-\ell_{\mathrm{old}}(v)\bigr).
$$

OPD loss 采用 PPO 风格的裁剪代理目标（clipped surrogate）：

$$
\mathcal{L}_i^{\mathrm{OPD}}
=
\sum_{v\in S_i}
\max\left(
-A_v\rho_v,\,
-A_v\mathrm{clip}\bigl(\rho_v,1-\varepsilon_{\mathrm{lo}},1+\varepsilon_{\mathrm{hi}}\bigr)
\right).
$$

论文沿用 $\varepsilon_{\mathrm{lo}}=0.2$、$\varepsilon_{\mathrm{hi}}=0.28$。重合度选择控制受监督词元的分布距离，$\Delta$ 裁剪控制单词元优势的幅度；两者一起让 OPD 的细粒度监督更适合在线 RL。

## 通用 Agent 的逐步奖励

长程 agent 任务的结果奖励常常只在终止时出现，例如 GUI 任务最终是否成功、SWE benchmark 的测试是否通过。OpenClaw-RL 用 PRM 基于实时后继状态给每一步打分，把过程奖励和最终结果相加：

$$
R_t=o+\frac{1}{m}\sum_{j=1}^{m}r_j,
$$

其中 $o$ 是可验证结果，$r_j$ 是 PRM 对 $(a_t,s_{t+1})$ 的独立判断。这样每个中间行动都能获得更密集的归因信号。论文还提到，真实终端或 GUI 状态难以稳定聚类，因此优势标准化按步骤编号分组，避免依赖状态聚类。

## 实验结论

在个人 agent 场景中，论文用 Qwen3-4B-Thinking-2507 作为策略模型和奖励模型，用 Qwen3-32B 模拟不同职业用户。三个用户偏好分别是：学生希望输出风格少一些 AI 痕迹，TA 希望批改具体详细，老师希望评语友好耐心。Hybrid RL 在联合优化三类用户时平均约 **10.3 个会话（session）** 达到目标效果，单独优化时约 **15.0 个会话**；GRPO、OPD、Mem0、Cognee 的效率整体更低。

在终端、GUI、SWE、工具调用等通用 agent 场景中，OpenClaw-RL 用同一套基础设施支持并行环境。论文报告结果奖励与过程奖励整合后，在工具调用与 GUI 设置中优于只用结果奖励的训练；提示选择的消融也显示，top-$k$ 重合度比随机提示选择更稳定，主实验选择 $k=4$。

> [!comparison] 和相关方法的关系
> OpenClaw-RL 继承了 [[GRPO]] / PPO 风格裁剪代理目标对稳定策略更新的关注，也借用了 [[On-Policy Distillation]] 对词元级引导的利用方式。它和 [[Process Reward Model]] 的连接在于：PRM 在这里既是评价器，也是从后继状态抽取提示的 judge。它和 [[Agentic RL]] 的连接在于：训练数据来自 agent 与环境的真实交互，环境可以是个人设备、shell、GUI、代码仓库或工具 API。

## 边界与风险

> [!warning] 使用 OpenClaw-RL 时要检查的前提
> - 后继状态需要能可靠反映上一动作质量。含糊用户反馈、偶然环境失败、恶意纠正会污染在线更新。
> - PRM / judge 的质量决定了评估投票和指令提示的上限。低质量提示会放大师生分布错配。
> - 异步在线训练需要额外工程成本，包括 PRM 服务、样本缓冲、权重发布边界、用户隐私保护和跨会话隔离。
> - 个性化优化会把用户偏好写入权重，可能带来隐私泄漏和偏好冲突风险；多用户共享模型时尤其需要数据过滤和隔离策略。
> - 论文的个人 agent 实验使用模拟用户，真实用户反馈更嘈杂，偏好也更会漂移；上线前需要更强的安全过滤和回滚机制。

## 面试视角

> [!interview] Q: OpenClaw-RL 的核心洞察是什么？
> A: Agent 每一步行动后的后继状态本身就包含训练信号。用户追问、测试结果、工具输出和 GUI 状态变化可以被 PRM 转成标量评价，也可以在包含纠正信息时转成提示，再通过混合 RL 目标更新策略。

> [!interview] Q: 为什么要同时用评估信号和指令信号？
> A: 评估信号覆盖率高，适合每个被评分轮次；指令信号覆盖率低，但能给词元级引导。混合目标用 GRPO loss 消费标量奖励，用 OPD loss 消费提示条件下的教师分布，从频率和信息密度两侧补足训练信号。

> [!interview] Q: 重合度引导的提示选择解决什么稳定性问题？
> A: 它用学生 top-$k$ 词元和提示条件下教师 top-$k$ 词元的重合度筛选提示。高重合度表示教师分布和学生当前分布在高概率区域相近，OPD 更新更容易保持稳定；再配合对数概率差裁剪，可以限制单词元优势过大。

## 速查

> [!example] 关键公式
> - 标量奖励：$r_t=\mathrm{majority}\{\mathrm{PRM}(a_t,s_{t+1})\}$，取值可为 $+1/-1/0$。
> - 混合目标：$\mathcal{L}_i^{\mathrm{hybrid}}=w_{\mathrm{RL}}\mathcal{L}_i^{\mathrm{GRPO}}+w_{\mathrm{OPD}}\mathcal{L}_i^{\mathrm{OPD}}$。
> - 提示重合度：$O[h,i]=|S_i^q\cap S_{i,h}^p|$。
> - 对数概率差裁剪：$\Delta_v=\mathrm{clip}(\ell_{T,h^\star}(v)-\ell_{\mathrm{old}}(v),-C,+C)$。
> - 逐步奖励：$R_t=o+\frac{1}{m}\sum_{j=1}^{m}r_j$。
