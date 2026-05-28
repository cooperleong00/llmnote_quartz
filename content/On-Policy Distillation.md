---
description: 学生从自身策略采样轨迹，并在这些轨迹上接收教师或自反馈信号的蒸馏范式，用于缓解自回归生成的 exposure bias
type: method
aliases:
  - 在线蒸馏
  - OPD
  - On-Policy KD
prerequisites:
  - "[[KL Divergence]]"
  - "[[SFT]]"
  - "[[RLHF]]"
tags:
  - post-training
  - distillation
  - alignment
created: 2026-01-28
updated: 2026-05-01T13:38
---

# On-Policy Distillation

On-Policy Distillation（OPD，在线蒸馏）是 [[Knowledge Distillation|知识蒸馏]] 在自回归 LLM 上的 on-policy 化：学生先按自身策略生成轨迹，教师随后在这些学生实际访问到的 prefixes 上提供 logit、reward、preference 或 self-play feedback。它把蒸馏监督放到学生自己的状态分布上，缓解 [[Training-Inference Mismatch|train-inference distribution mismatch]] 和 exposure bias；在 [[RLHF]] 语境中，OPD 也可以理解为用密集反馈降低 [[PPO]] / [[GRPO]] 这类 RL 训练的 credit assignment 难度。

> [!paper] 论文出处
> 本笔记吸收 [[Clippings/Paper/2604.00626/2604.00626|A Survey of On-Policy Distillation for Large Language Models (2026)]] 的统一 $f$-divergence 框架、三维 taxonomy 和工程边界，并结合 GKD、OPSD、SDFT、SDPO 等代表方法整理成 method note。
>
> [[Clippings/Article/On SFT, RL, and on-policy distillation|On SFT, RL, and on-policy distillation]] 补充了一个实用视角：OPD 是 same-family teacher 条件下，从 SFT 过渡到 RL 之间的高效能力转移方法；它的稳定性取决于 teacher-student 的 token-level KL 信号是否足够校准、足够分散。

---

## 核心直觉

> [!intuition] 三种学习方式的类比
> 想象你在学国际象棋：
> - **On-policy RL**：自己下棋，只知道输赢，不知道哪步错了
> - **Off-policy distillation**：看大师下棋，但大师的局面你可能永远遇不到
> - **On-policy distillation**：每一步都有教练评分，在**你自己的局面**上指导
>
> 这就是 on-policy distillation 的本质：在学生自己会遇到的状态上，获得教师的逐步指导。

---

## 动机

### 传统蒸馏的问题：Distribution Mismatch

传统 [[SFT]] 式知识蒸馏（Supervised KD）在**教师生成或静态收集的数据**上训练学生：

$$
\mathcal{L}_{\text{SFT-KD}} = -\mathbb{E}_{y \sim \pi_{\text{teacher}}} \left[ \sum_t \log \pi_{\text{student}}(y_t | y_{<t}, x) \right]
$$

这导致 **train-inference distribution mismatch**：
- **训练时**：学生看到的是教师生成的"完美"序列
- **推理时**：学生必须基于自己之前生成的 token 继续生成

> [!warning] Exposure Bias
> 学生从未在训练中见过自己的错误，一旦推理时犯错，就不知道如何恢复。错误会累积（compounding errors），导致生成质量急剧下降。
>
> 从 imitation learning 视角看，纯 behavior cloning 只在 expert distribution 上学习，学生推理时访问的是自己的 state distribution。DAgger 的分析给出一个重要锚点：在长度为 $T$ 的序列里，off-policy 训练的误差可能按 $O(\epsilon T^2)$ 累积；如果专家在 learner 自己访问的状态上提供监督，误差累积可降到 $O(\epsilon T)$。OPD 把这个思想迁移到自回归语言生成。

### On-Policy 的解决方案

On-Policy Distillation 让学生**从自己的策略分布采样**：

$$
\mathcal{L}_{\text{On-Policy}}=
\mathbb{E}_{y\sim\pi_\theta(\cdot|x)}
\left[
\sum_t D\big(p_T(\cdot|x,y_{<t}) \,\|\, p_\theta(\cdot|x,y_{<t})\big)
\right]
$$

**关键区别**：
- 序列 $y$ 来自学生自己的生成
- 教师在学生的序列上提供监督
- 学生学会在**自己的错误**上如何改进

### SFT、OPD 与 RL 的阶段切换

[[SFT]] 在固定数据分布上学习。若数据来自 teacher，学生越接近 teacher，新增样本提供的信息越少，因为采样分布已经在数据构造时固定。Rejection-sampled SFT 会通过筛选正确样本抬高数据质量，但它仍然受采样分布约束：filter 饱和后，继续增加样本主要是在同一分布上重复训练。

[[GRPO]] / [[PPO]] 这类 RL 让学生从自身策略采样，更新后的策略又影响下一批 rollouts，因此改进可以反馈到采样分布里。这个 compounding 能突破固定 teacher data 的实用上限，但代价是 sparse reward 带来的高方差和 credit assignment 成本。

> [!intuition] 训练阶段的判断
> 当学生还明显低于 teacher，teacher 数据通常是最便宜的能力来源；当学生接近 teacher，静态 SFT 的边际收益下降，学生自己的 rollouts 开始包含新的探索路径。OPD 位于这个转折附近：它保留学生采样带来的 state coverage，同时用 teacher logits 提供 dense signal。

---

## 本质理解：为什么比 RL 高效 50-100x？

> [!intuition] 信息论视角
> - **RL** 每个 episode 只传递 **O(1) bits**（对/错）
> - **Distillation** 每个 episode 传递 **O(N) bits**（N = token 数）
>
> 这是效率差异的根本原因：RL 的 sparse reward 需要大量样本来做 credit assignment，而 distillation 直接告诉你每个 token 该怎么改。

> [!intuition] 搜索 vs 学习视角
> RL 的大部分计算花在**搜索策略空间**——尝试不同策略，看哪个有效。一旦找到好策略，distillation 是**学习已知策略的捷径**——不需要重新搜索。
>
> **类比**：科学研究花大量时间探索，但一旦发现结论，用自然语言教给别人很快。这与运动技能不同——运动技能存在于肌肉记忆中，只能通过反复练习学习。
>
> RL 训练得到的策略位于"语义层面"（如何推理），具体权重只是承载方式，所以可以通过 distillation 高效传递。

---

## 核心机制

### GKD: Generalized Knowledge Distillation

GKD 是 On-Policy Distillation 的通用框架，支持多种 divergence 选择。

> [!definition] GKD 目标函数
> $$
> \mathcal{L}_{\text{OPD}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_{\text{mix}}(\cdot|x)}
> \left[
> \sum_{t=1}^{|y|}
> \mathcal{D}_f\big(p_T(\cdot|x,y_{<t}),p_\theta(\cdot|x,y_{<t})\big)
> \right]
> $$
>
> 其中 $\pi_{\text{mix}}$ 决定谁来探索训练 prefixes，可以是纯学生策略 $p_\theta$，也可以是数据分布、教师分布和学生分布的 mixture；$\mathcal{D}_f$ 来自 $f$-divergence 家族，可以实例化为 forward [[KL Divergence|KL]]、reverse KL、JSD 或 skewed KL。

GKD 的关键参数是 student data fraction $\lambda$：以概率 $\lambda$ 使用学生生成序列，以概率 $1-\lambda$ 使用数据集序列。$\lambda=1$ 时得到纯 on-policy 训练，$\lambda=0$ 时退回到传统 off-policy KD。

### 设计空间：三条轴

OPD 方法可以沿三条互相独立的轴理解：

| 轴 | 典型选择 | 影响 |
|----|----------|------|
| **反馈信号** | logit-based、outcome-based、self-play | 决定教师传递多少信息；logit 最密集，reward / preference 更接近 [[RLHF]] |
| **教师访问权限** | white-box、black-box、teacher-free | 决定能否使用完整词表 logits；API teacher 通常只能给文本、评分或偏好 |
| **损失粒度** | token-level、sequence-level、hybrid | 决定 credit assignment 的精度；token-level 稳定，sequence-level 给学生更多路径自由度 |

这个 taxonomy 有助于把 [[GRPO]]、[[PPO]] 这类 RL 训练和 OPD 放在同一个后训练设计空间里看：RL 侧重 outcome reward，白盒 OPD 侧重 dense teacher distribution，SDPO / reward-guided OPD 则位于两者之间。

### Same-Family Teacher：OPD 的高效条件

OPD 最适合 **same-family teacher**：teacher 与 student 使用相同 tokenizer、相近训练 recipe，通常是同一模型家族的不同规模版本。此时 teacher 在学生 rollouts 上给出的 per-token logprob 更像能力差距信号，较少被格式、风格或 tokenizer 边界污染。

不同 family 的 teacher 会引入两个成本。Tokenizer mismatch 会让 teacher completion 重新切分成 student tokens，per-token logprob 难以对应到同一个预测位置；recipe mismatch 会把 teacher 的格式习惯、推理风格和数据管线痕迹混进监督信号。学生一部分容量会花在拟合表面风格上，能力转移效率下降。

> [!warning] OPD 的 same-family 依赖
> Reverse KL 的 token-level advantage 是 $\log \pi_T(y_t)-\log \pi_\theta(y_t)$。这个值要成为有用的训练信号，需要 teacher 和 student 对“同一个 prefix 下的同一个 token”有可比的概率判断。Tokenizer 或 recipe 差异过大时，梯度会更关注“teacher 会怎么措辞”，能力相关信号被稀释。

### Divergence 选择

| Divergence | 公式 | 特点 |
|------------|------|------|
| **Forward KL** | $D_{KL}(\pi_T \| \pi_S)$ | Mode-covering，学生覆盖教师所有模式 |
| **Reverse KL** | $D_{KL}(\pi_S \| \pi_T)$ | Mode-seeking，学生专注高概率区域 |
| **JSD(β)** | $\beta D_{KL}(\pi_T \| M) + (1-\beta) D_{KL}(\pi_S \| M)$ | 平衡两者，$M = \beta \pi_T + (1-\beta) \pi_S$ |

> [!intuition] Divergence 选择是 Task-Dependent 的
> - **Instruction Tuning**：Reverse KL 效果更好（mode-seeking 适合专注核心行为）
> - **Summarization / Translation**：JSD 效果更好
> - **Self-distillation (OPSD)**：JSD_β=0.5 更稳定（教师学生分布接近，对称性有优势）
> - Reverse KL 的另一个优势：是"不可 hack"的——低 KL 总是对应教师认为好的行为

### Reverse KL Loss 的具体形式

> [!math] Reverse KL Loss
> $$
> \mathcal{L}_{\text{reverse-KL}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, y_t \sim \pi_{\theta}} \left[ \log \frac{\pi_{\text{teacher}}(y_t | x, y_{<t})}{\pi_{\theta}(y_t | x, y_{<t})} \right]
> $$
>
> 梯度形式：
> $$
> \nabla_\theta \mathcal{L} = -\mathbb{E}_{y_t \sim \pi_\theta} \left[ \underbrace{\log \frac{\pi_T(y_t)}{\pi_\theta(y_t)}}_{\text{advantage}} \cdot \nabla_\theta \log \pi_\theta(y_t) \right]
> $$

这个梯度形式与 [[Policy Gradient]] 非常相似：
- **Advantage** $\hat{A}_t = \log \frac{\pi_T(y_t)}{\pi_\theta(y_t)}$：教师与学生的 log probability ratio
- 正值表示教师更倾向于该 token → 强化
- 负值表示学生过度倾向于该 token → 惩罚

### Forking Tokens：关键分叉点

> [!intuition] 不同 token 的惩罚权重不同
> 在错误的推理链中，教师模型会**重点惩罚导致分叉的 token**——那些让推理走向错误方向的关键决策点。
>
> 最终答案虽然错误，但惩罚可能很小——因为给定前面的错误推理，这个答案是"可预测的"。
>
> 这意味着 on-policy distillation 自动学会了"哪里是关键决策点"，无需显式的 process reward model。

### 梯度几何：稀疏、密集、偏置与集中

从 token-level policy gradient 的角度看，不同 post-training 方法可以用两个主轴理解：信号是 sparse 还是 dense，梯度估计是更接近 unbiased 还是带有固定偏置。还需要再看第三个维度：偏置是否集中在少数 token 上。

| 方法 | 信号密度 | 偏置形态 | 稳定性来源 |
|------|----------|----------|------------|
| [[GRPO]] / RLVR | Sparse | 近似 unbiased | 大 batch 中无关 token 的梯度互相抵消，留下与 reward 相关的方向 |
| [[SFT]] | Dense | 偏向数据分布 | 偏置分散在多样化样本上，整体把模型拉向数据 manifold |
| Same-family OPD | Dense | 偏向 teacher 分布 | teacher 与 student 分布校准，KL 信号通常较分散 |
| OPSD | Dense | 偏向 self-with-hint teacher | ground-truth hint 可能让少数 pivot token 的 KL 过大，需要 clipping 或固定 teacher |

> [!intuition] 为什么 OPSD 更容易不稳定？
> 在一个错误数学推理中，student 可能几乎不会生成某个关键 pivot token，而 conditioned-on-answer 的 self-teacher 会强烈偏好它。若 student 给该 token 概率 $0.01$，teacher 给 $0.6$，reverse-KL advantage 约为 $\log(0.6/0.01)\approx 4.1$。这个 token 会主导梯度，把模型快速推向原本低概率的区域。OPSD 的 per-token KL clipping 和固定 teacher，正是在控制这种集中偏置。

---

## GKD 算法

> [!example] GKD 算法伪代码
> ```python
> def GKD_training(student, teacher, dataset, divergence='reverse_kl'):
>     for batch in dataset:
>         prompts = batch['prompts']
>
>         # Step 1: 学生生成序列 (on-policy sampling)
>         with torch.no_grad():
>             student_outputs = student.generate(prompts, do_sample=True)
>
>         # Step 2: 获取教师和学生的 logits
>         teacher_logits = teacher(prompts, student_outputs)
>         student_logits = student(prompts, student_outputs)
>
>         # Step 3: 计算 divergence loss
>         if divergence == 'reverse_kl':
>             # Reverse KL: E_student[log(student/teacher)]
>             loss = compute_reverse_kl(student_logits, teacher_logits)
>         elif divergence == 'forward_kl':
>             # Forward KL: E_teacher[log(teacher/student)]
>             loss = compute_forward_kl(student_logits, teacher_logits)
>
>         # Step 4: 更新学生
>         loss.backward()
>         optimizer.step()
> ```

### 与 RL Fine-tuning 结合

GKD 可以与 RL 结合，同时利用教师监督和 reward 信号：

$$
\mathcal{L}_{\text{GKD+RL}} = \mathcal{L}_{\text{GKD}} + \lambda \mathcal{L}_{\text{RL}}
$$

这种结合特别适合：
- 教师提供"如何生成"的监督
- Reward model 提供"生成得好不好"的反馈

---

## OPSD: On-Policy Self-Distillation

OPSD 是 On-Policy Distillation 的一个重要变体，**单个模型同时扮演教师和学生角色**。

> [!definition] OPSD 核心思想
> - **教师角色**：模型条件化在**特权信息**（如 ground-truth solution）上
> - **学生角色**：模型只看问题，生成 on-policy 序列
> - 教师在学生的序列上提供 token-level 监督

### 特权信息的作用

```
教师输入: [Question] + [Ground-truth Solution] → 生成 logits
学生输入: [Question] only → 生成序列
```

> [!intuition] 为什么这样有效？
> - 教师看到答案后，知道"正确的推理路径"应该是什么
> - 学生只看问题，必须自己探索
> - 教师的 logits 指导学生走向正确路径

### OPSD 的优势

1. **无需额外教师模型**：节省内存和计算
2. **自适应难度**：教师和学生能力同步提升
3. **高效利用数据**：每个样本都能提供监督

### 容量要求

> [!warning] OPSD 的前提条件
> 模型需要有**足够的容量**才能成功自蒸馏：
> - 太小的模型无法同时扮演好教师和学生
> - 实验表明 4B+ 模型效果较好，1.7B 模型自蒸馏效果有限
> - 这是因为"rationalization"（理解答案并解释）需要足够的模型能力

### 实现细节

**教师策略固定**：OPSD 通常把教师策略固定在**初始策略**，当前更新中的学生策略负责生成 rollouts。这样做有两个好处：
1. 稳定训练，避免 teacher 和 student 同时漂移
2. 隐式起到 KL 正则化的作用，防止过度偏离初始策略

**Full-vocabulary vs Sampled-token Distillation**：
- **Full-vocabulary**：计算完整词表上的 KL divergence，提供更丰富的监督
- **Sampled-token**：只在采样的 token 上计算，类似 policy gradient
- 实验表明 full-vocabulary 效果更好（AIME25: 84.1% vs 82.1%），但内存开销更大

---

## SDFT: Self-Distillation Fine-Tuning

SDFT 是 On-Policy Distillation 的另一个重要变体，专注于**从 demonstration 学习**的场景，特别适合 continual learning。

> [!definition] SDFT 核心思想
> - **Teacher**：模型条件化在 expert demonstration 上 $\pi(\cdot|x, c)$
> - **Student**：模型只看 prompt $\pi_\theta(\cdot|x)$
> - 使用 reverse KL divergence 训练，在学生自己的分布上采样

### 与 OPSD 的区别

| 方面 | OPSD | SDFT |
|------|------|------|
| **特权信息** | Ground-truth solution | Expert demonstration |
| **适用场景** | 有标准答案的任务（数学推理） | 有示例但无标准答案的任务 |
| **Teacher 构造** | 条件化在答案上 | 条件化在示例上 |
| **核心假设** | 模型能 rationalize 答案 | 模型有 in-context learning 能力 |

### IRL 视角：SDFT 作为隐式 Reward 最大化

> [!math] SDFT 的 IRL 解释
> SDFT 可以被解释为最大化隐式 reward function：
> $$r(y, x, c) = \log \pi(y|x, c) - \log \pi_k(y|x)$$
>
> 这个 reward 衡量的是：demonstration-conditioned 模型相比当前模型，对输出 $y$ 的偏好程度。
>
> Token-level reward：
> $$r_t(y_t | y_{<t}, x, c) = \log \frac{\pi(y_t | y_{<t}, x, c)}{\pi_k(y_t | y_{<t}, x)}$$

这个 IRL 视角揭示了 on-policy distillation 的本质：**从 demonstration 中提取隐式 reward，然后用 on-policy RL 优化**。

### ICL Assumption：为什么 Self-Distillation 有效

> [!intuition] In-Context Learning 假设
> SDFT 的核心假设是：
> $$\pi_{k+1}^*(y|x) \approx \pi(y|x, c)$$
>
> 即 demonstration-conditioned 模型近似最优策略。这需要两个条件：
> 1. **Optimality**：Teacher 的输出质量接近最优
> 2. **Minimal Deviation**：Teacher 与当前模型的 KL 距离足够小
>
> 实验表明，demonstration-conditioned teacher 与 base model 的 KL 距离（0.68 nats）远小于 SFT 模型（1.26 nats），这解释了为什么 SDFT 能减少 catastrophic forgetting。

### Continual Learning 应用

SDFT 的一个重要应用是 **continual learning**——让模型持续学习新技能而不遗忘旧能力。

> [!example] Sequential Skill Learning
> 在连续学习三个不同任务的实验中：
> - **SDFT**：能够学习每个新任务，同时保持之前任务的性能
> - **SFT**：每学习一个新任务，之前任务的性能就急剧下降
>
> 这是因为 on-policy 学习在学生自己的分布上训练，不会破坏模型对其他任务的表示。

### 训练 Reasoning Model 而无需 Reasoning Data

> [!example] Answer-Only Supervision
> 一个实际问题：如何用只有最终答案（无 chain-of-thought）的数据训练 reasoning model？
>
> - **SFT**：直接模仿短答案，导致 reasoning 能力退化（准确率从 31.2% 降到 23.5%）
> - **SDFT**：Teacher 保持模型的 reasoning 风格，准确率提升到 43.7%
>
> 这是因为 SDFT 的 teacher 是 demonstration-conditioned 的**同一个模型**，它会用自己的 reasoning 风格来解释 demonstration，避免直接复制短答案。

---

## SDPO: Self-Distillation Policy Optimization

SDPO 是 On-Policy Distillation 的一个 self-distillation 变体，核心创新是**使用环境反馈**（runtime errors、test failures、judge messages 等）作为特权信息，实现无需外部教师的自蒸馏。

> [!definition] SDPO 核心思想
> - **Teacher**：模型条件化在**环境反馈** $f$（如 runtime errors、test failures）上
> - **Student**：模型只看问题，生成 on-policy 序列
> - 通过 KL divergence 将 teacher 的"回顾性修正"蒸馏到 student

### 损失函数

$$\mathcal{L}_{\text{SDPO}}(\theta) := \sum_{t} \text{KL}(\pi_{\theta}(\cdot \mid x, y_{< t}) \| \text{stopgrad}(\pi_{\theta}(\cdot \mid x, f, y_{< t})))$$

其中 $f$ 是环境反馈，`stopgrad` 阻止梯度流过 teacher，防止 teacher 退化到忽略反馈。

### 与 GRPO 的关系

> [!math] SDPO 作为 GRPO 的扩展
> SDPO 的梯度可以写成 policy gradient 形式，只需替换 advantage 计算：
>
> **GRPO advantage**（sequence-level，基于 scalar reward）：
> $$A_{i,t}^{\text{GRPO}}(\hat{y}_{i,t}) := \mathbb{1}\{y_{i,t} = \hat{y}_{i,t}\} \left(r_i - \text{mean}\{r_i\}_{i=1}^G\right)$$
>
> **SDPO advantage**（logit-level，基于 teacher-student 差异）：
> $$A_{i,t}^{\text{SDPO}}(\hat{y}_{i,t}) = \log \frac{\pi_{\theta}(\hat{y}_{i,t} \mid x, f_i, y_{i,<t})}{\pi_{\theta}(\hat{y}_{i,t} \mid x, y_{i,<t})}$$
>
> SDPO advantage 在 teacher 和 student 完全一致时为零，对 teacher 更偏好的 token 为正，反之为负。

这意味着 SDPO 可以用**最小改动**集成到现有 RLVR pipeline——只需替换 advantage 计算。

### 与 OPSD/SDFT 的区别

| 方面       | OPSD                  | SDFT                 | SDPO                  |
| -------- | --------------------- | -------------------- | --------------------- |
| **特权信息** | Ground-truth solution | Expert demonstration | 环境反馈（errors, outputs） |
| **适用场景** | 有标准答案的任务              | 有示例的任务               | 有可验证环境的任务（代码、数学）      |
| **信用分配** | Token-level           | Token-level          | **Logit-level**（更密集）  |
| **核心假设** | 模型能 rationalize 答案    | 模型有 ICL 能力           | 模型能从反馈中回顾性修正          |

### RLRF：Reinforcement Learning with Rich Feedback

> [!intuition] 从 RLVR 到 RLRF
> 传统 RLVR（RL with Verifiable Rewards）只使用 scalar reward（对/错），浪费了丰富的环境信息。
>
> SDPO 形式化了 **RLRF** 范式：
> - 利用 tokenized feedback（runtime errors、test case failures、compiler messages）
> - 通过 self-teacher 的回顾性分析，将这些信息转化为 dense credit assignment
>
> 即使在标准 RLVR 环境中（只有 binary reward），SDPO 也能通过将同 batch 中的成功样本作为"反馈"来提供更丰富的监督。

### 密集信用分配

> [!example] 为什么 logit-level 优于 sequence-level
> 消融实验显示三种粒度的效果：
> - **Logit-level SDPO**：在每个位置的 top-100 tokens 上计算 advantage（最佳）
> - **Token-level SDPO**：只在生成的 token 上计算
> - **Sequence-level SDPO**：平均所有 token 的 advantage 得到单一 scalar（类似 GRPO）
>
> Logit-level 显著优于其他两种，说明**密集信用分配**是 SDPO 的关键优势。

### 实验结果

在 LiveCodeBench v6 上（Qwen3-8B）：
- **样本效率**：达到 GRPO 最终准确率只需 **4× 更少**的生成量
- **最终性能**：48.8% vs GRPO 的 41.2%
- **推理简洁性**：响应长度比 GRPO **短 3-7×**，同时准确率更高

> [!warning] 模型规模依赖
> SDPO 的效果与模型的 in-context learning 能力强相关：
> - 强模型（Qwen3-8B）：SDPO 显著优于 GRPO
> - 弱模型（Qwen3-0.6B）：SDPO+GRPO 混合效果更好
>
> 这是因为 self-teacher 的"回顾性修正"能力是随模型规模涌现的。

### Test-Time Self-Distillation

SDPO 的一个独特应用是 **test-time self-distillation**——在推理时对单个困难问题进行自蒸馏：

> [!example] 解决困难问题
> 对于 pass@64 < 0.03 的极难问题：
> - **Best-of-k sampling**：需要大量采样才能找到解
> - **SDPO**：通过迭代自蒸馏，逐步改进策略，更快发现解
>
> 关键洞察：即使从未解决过问题，环境反馈（如"哪个 test case 失败"）也能提供有用信号。

---

## 与其他方法的对比

### On-Policy vs Off-Policy Distillation

> [!comparison] 核心区别

| 方面                        | Off-Policy (SFT-KD) | On-Policy (GKD) |
| ------------------------- | ------------------- | --------------- |
| **采样分布**                  | 教师分布 $\pi_T$        | 学生分布 $\pi_S$    |
| **Distribution Mismatch** | 存在                  | 解决              |
| **学习内容**                  | 模仿教师的输出             | 学习在自己的错误上改进     |
| **数据效率**                  | 需要大量教师数据            | 动态生成，更高效        |
| **适应性**                   | 固定数据集               | 随学生进化调整         |

> [!warning] SFT 在自己样本上训练也会退化
> 一个反直觉的发现：即使在模型**自己的样本**上做 SFT（KL=0 in expectation），性能也会退化！
>
> 原因：每个 finite batch 的分布略有不同，训练会导致模型偏离原始状态，然后这个过程变成 off-policy，导致 compounding error。
>
> On-policy distillation 通过固定教师策略避免了这个问题——学生总是向固定目标收敛。

### On-Policy Distillation vs RL

> [!comparison] 效率对比

| 方面                    | RL (GRPO/PPO) | On-Policy Distillation |
| --------------------- | ------------- | ---------------------- |
| **监督信号**              | Sparse (序列末尾) | Dense (每个 token)       |
| **Credit Assignment** | 困难            | 直接                     |
| **训练效率**              | 1x            | **50-100x**            |
| **需要 Reward Model**   | 是             | 否（用教师 logits）          |
| **探索能力**              | 强             | 弱（受教师约束）               |

> [!intuition] 为什么 On-Policy Distillation 更高效？
> - **Dense supervision**：每个 token 都有监督信号，不需要 credit assignment
> - **无需 reward model**：教师 logits 直接提供"什么是好的"信息
> - **稳定训练**：没有 RL 的高方差问题

> [!comparison] 实用上限
> OPD 的实用上限通常接近 teacher，因为目标本身就是 teacher distribution；RL 的上限由 verifier 或 reward signal 决定，在可验证任务上可能超过 teacher。工程上常见的选择是：先用 SFT 接近 teacher 的分布，再用 OPD 高效吃掉 same-family teacher 的剩余能力；当目标需要发现 teacher 没有覆盖的新策略时，再投入 RL 的探索成本。

### 与 DPO 的关系

[[DPO]] 和 On-Policy Distillation 都试图绕过 RL 的复杂性：

| 方面 | DPO | On-Policy Distillation |
|------|-----|------------------------|
| **数据格式** | Pairwise preferences | 教师 logits |
| **监督粒度** | 序列级 | Token 级 |
| **需要教师** | 否 | 是 |
| **适用场景** | 偏好对齐 | 能力蒸馏 |

---

## 实验结果与效率分析

### GKD 实验结果

| 任务 | Supervised KD | GKD (On-Policy) | 提升 |
|------|---------------|-----------------|------|
| Summarization (ROUGE-L) | 32.1 | **35.8** | +3.7 |
| Translation (BLEU) | 28.4 | **31.2** | +2.8 |
| Reasoning (Accuracy) | 45.2 | **52.1** | +6.9 |

### OPSD 效率分析

> [!example] OPSD vs GRPO Token 效率

| 方法 | AIME'24 准确率 | Token 消耗 | 相对效率 |
|------|---------------|-----------|----------|
| GRPO | 50% | 8x | 1x |
| OPSD | 50% | **1x** | **8x** |
| OPSD | 60% | 2x | **4x** |

### Qwen3 工业应用

Qwen3 采用两阶段蒸馏策略：

```
Stage 1: Off-policy Distillation
    - 在教师生成的数据上 SFT
    - 建立基础能力

Stage 2: On-policy Distillation
    - 学生生成序列
    - 与教师 logits 对齐
    - 精细化调优
```

| 方法 | AIME'24 | GPU Hours |
|------|---------|-----------|
| RL (GRPO) | 72.1% | 17,920 |
| On-Policy Distillation | **74.4%** | **1,800** |

> [!intuition] 关键发现
> On-Policy Distillation 用 **1/10 的计算量**达到了**更好的效果**。

---

## 实现细节

### 关键超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| **Temperature** | 1.0 | 采样温度，影响探索程度 |
| **Top-p** | 0.9-1.0 | Nucleus sampling 参数 |
| **Learning Rate** | 1e-6 ~ 5e-6 | 比 SFT 更小 |
| **Batch Size** | 32-128 | 根据 GPU 内存调整 |
| **Divergence** | Reverse KL | 推理任务首选 |

### 实现技巧

**1. 采样策略**

```python
# 推荐：使用 temperature sampling
outputs = model.generate(
    prompts,
    do_sample=True,
    temperature=1.0,
    top_p=0.95
)
```

**2. Loss 计算**

```python
def compute_reverse_kl_loss(student_logits, teacher_logits, labels):
    # 获取概率分布
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # 只在生成的 token 上计算 loss
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Reverse KL: E_student[log(student/teacher)]
    # = E_student[log_student - log_teacher]
    kl = student_probs * (student_log_probs - teacher_log_probs)
    return kl.sum(dim=-1).mean()
```

**3. 梯度截断**

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 应用场景

### 1. Reasoning 能力提升

On-Policy Distillation 在数学推理任务上效果显著：
- 学生学会教师的推理模式
- 在自己的错误上获得纠正信号
- 比 RL 更高效地提升推理能力

### 2. Personalization

> [!example] 个性化应用
> - 教师：通用大模型
> - 学生：针对特定用户/场景的小模型
> - On-policy 采样确保学生在自己的分布上学习

### 3. Continual Learning

On-Policy Distillation 可用于持续学习：
- 旧模型作为教师
- 新模型在新数据上采样
- 保持旧能力的同时学习新能力

> [!example] 恢复被遗忘的能力
> 在 mid-training 后，模型可能遗忘 instruction-following 能力。
> 用原始 instruct 模型作为教师，在 Tulu3 等 chat prompts 上做 on-policy distillation，可以恢复 IF-eval 性能而不损失新学到的知识。
>
> 这使得 on-policy distillation 成为**持续学习**的有力工具——可以交替进行知识学习和行为恢复。

### 4. 数据效率：单 Prompt 多次采样

> [!example] 极端数据效率
> 实验表明，在**单个 prompt** 上多次采样（256 rollouts × 20 steps = 5120 sequences），也能达到接近教师的性能。
>
> 这是因为 on-policy distillation 学习的是教师的**完整分布**，单个答案记忆只覆盖极小的局部。RL 在同一 prompt 上多次训练会导致过拟合，但 distillation 可以维持分布级监督。

### 5. 模型压缩

将大模型的能力蒸馏到小模型：
- 比传统 SFT 蒸馏效果更好
- 小模型学会在自己的能力范围内最优表现

---

## 局限性

> [!warning] On-Policy Distillation 的局限

1. **需要教师模型**：训练时需要运行教师模型计算 logits，增加计算开销
2. **教师质量上限**：学生的能力受限于教师
3. **探索能力有限**：相比 RL，on-policy distillation 的探索受教师约束——无法发现教师不知道的策略
4. **采样开销**：每个训练步都需要学生生成完整序列
5. **不适合偏好对齐**：如果目标是学习人类偏好，[[DPO]] 或 [[RLHF]] 更合适
6. **Same-family 约束强**：tokenizer 或训练 recipe 差异过大时，per-token KL 信号会混入大量风格和切分噪声
7. **Self-distillation 的 KL 预算敏感**：带有 privileged information 的 self-teacher 可能产生集中梯度，需要 per-token clipping、teacher freezing 或更温和的 hint 设计

> [!intuition] 什么时候用 RL，什么时候用 Distillation？
> 当你需要**探索新策略**时用 RL；当你需要**高效学习已知策略**时用 Distillation。
>
> 实践中常见的做法是：先用 RL 探索找到好策略，再用 distillation 高效传播。

> [!intuition] Teacher 选择可以看成 KL 预算问题
> 理想 teacher 应该在每一步带来尽可能大的 reward improvement，同时让 student 与 teacher 的 per-token KL 保持在可训练范围内。Same-family teacher 通常位于稳定区域；conditioned-on-answer 的 self-teacher 可能给出更强信号，也更容易超出 KL 预算。

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: On-Policy Distillation 解决了什么问题？**
> A: 解决了传统蒸馏的 train-inference distribution mismatch 问题。传统蒸馏在教师数据上训练，学生从未见过自己的错误；On-Policy Distillation 让学生从自身分布采样，学会在自己的错误上改进。
>
> **Q2: On-Policy Distillation 和 RL 的主要区别是什么？**
> A: 监督信号的密度不同。RL 通常只在序列末尾给 sparse reward，需要 credit assignment；On-Policy Distillation 提供 token-level 的 dense supervision，训练效率高 50-100 倍。从信息论角度，RL 每 episode 传递 O(1) bits，distillation 传递 O(N) bits。
>
> **Q3: 为什么 Reverse KL 比 Forward KL 更常用？**
> A: Reverse KL 是 mode-seeking 的，让学生专注于教师的高概率区域，更适合容量有限的学生模型。Forward KL 是 mode-covering 的，要求学生覆盖教师所有模式，可能导致学生在低概率区域分配过多概率，产生 hallucination。
>
> **Q4: OPSD 的核心思想是什么？**
> A: 单模型同时扮演教师和学生。教师条件化在特权信息（如 ground-truth solution）上，学生只看问题。教师在学生生成的序列上提供监督，实现自蒸馏。关键是教师策略固定为初始策略，避免训练不稳定。
>
> **Q5: On-Policy Distillation 的梯度形式为什么类似 Policy Gradient？**
> A: Reverse KL 的梯度是 $\nabla_\theta \mathcal{L} = -\mathbb{E}[\log \frac{\pi_T}{\pi_\theta} \cdot \nabla_\theta \log \pi_\theta]$，其中 $\log \frac{\pi_T}{\pi_\theta}$ 扮演 advantage 的角色，正值强化、负值惩罚，与 policy gradient 形式一致。
>
> **Q6: 什么是 Forking Tokens？**
> A: 在错误推理链中，教师会把主要惩罚集中到导致推理分叉的关键 token；最终答案的惩罚通常较小，因为给定前面的错误推理，它可能已经是可预测延续。这意味着 on-policy distillation 自动学会了"哪里是关键决策点"，无需显式的 process reward model。
>
> **Q7: SDFT 与 OPSD 的主要区别是什么？**
> A: OPSD 使用 ground-truth solution 作为特权信息，适合有标准答案的任务；SDFT 使用 expert demonstration，适合有示例但无标准答案的场景。SDFT 还可以从 IRL 视角理解——它等价于最大化隐式 reward $r(y,x,c) = \log \pi(y|x,c) - \log \pi_k(y|x)$。
>
> **Q8: 为什么 SDFT 能减少 catastrophic forgetting？**
> A: 两个原因：(1) On-policy 学习在学生自己的分布上训练，不会破坏模型对其他任务的表示；(2) Demonstration-conditioned teacher 与 base model 的 KL 距离很小（实验中 0.68 vs SFT 的 1.26 nats），保持了与原始模型的接近性。

---

## 参考资料

- [[Clippings/Paper/2604.00626/2604.00626|A Survey of On-Policy Distillation for Large Language Models (2026)]] — 统一 $f$-divergence 框架、三维 taxonomy、工程取舍和开放问题
- [[Clippings/Paper/230613649v3/230613649v3|GKD (2023)]] — Generalized Knowledge Distillation 原始论文，提出 on-policy + 多种 divergence 的统一框架
- [[Clippings/Paper/260118734v1/260118734v1|OPSD (2025)]] — Self-Distilled Reasoner，单模型自蒸馏，4-8× token 效率
- [[Clippings/Paper/250509388v1/250509388v1|Qwen3 Technical Report]] — 工业级 On-Policy Distillation 应用，1/10 计算量达到更好效果
- [[On-Policy Distillation - TML|Thinking Machines Blog]] — 实践指南，含 Forking Tokens 分析、信息论视角、Continual Learning 应用
- [[Clippings/Paper/260119897v1/260119897v1|SDFT (2025)]] — Self-Distillation Fine-Tuning，利用 ICL 能力实现 continual learning，IRL 视角解释
- [[Clippings/Paper/260120802v1/260120802v1|SDPO (2026)]] — Self-Distillation Policy Optimization，利用环境反馈实现 RLRF，4× 样本效率提升
- [[Clippings/Article/On SFT, RL, and on-policy distillation|On SFT, RL, and on-policy distillation]] — SFT/RL 阶段切换、same-family teacher 约束、梯度几何与 optimal teacher 视角
