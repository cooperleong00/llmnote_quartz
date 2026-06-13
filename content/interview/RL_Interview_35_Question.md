---
type: interview
related: "[[06-MOC - Reinforcement Learning]]"
difficulty: advanced
description: 覆盖 LLM RL 算法与基础设施高频面试题的系统化问答笔记
aliases:
  - RL 面试 35 题
  - LLM RL 面试题
prerequisites:
  - "[[RLHF]]"
  - "[[PPO]]"
  - "[[GRPO]]"
  - "[[Mixture of Experts]]"
  - "[[FSDP]]"
tags:
  - interview
  - reinforcement-learning
  - post-training
  - llm-rl
created: 2026-06-13
updated: 2026-06-13
---

# RL 面试 Benchmark：35 题深度答案

算法 + 基础设施 · LLM RL / [[Agentic RL]] / 大规模训练系统

整理日期：2026-06-13｜版本：面试准备版

使用说明：本文采用面试回答的组织方式，区别于论文摘要式写法。每题先给可直接说出口的结论，再补公式、工程细节、常见陷阱和参考方向。部分 2025-2026 年方法仍在快速演化，尤其 SAPO、DPPO 等缩写在社区中存在重名；文中已在对应位置注明采用的解释。

阅读建议：算法岗至少掌握第 1-19 题的公式与 trade-off；系统岗至少掌握第 20-35 题的数据流、显存、[[KV Cache|KV cache]]、异步与确定性问题。真实面试中不要背诵固定答案，要能根据 dense/[[Mixture of Experts|MoE]]、[[RLHF]]/RLVR、single-turn/agent、多卡/单卡场景切换。

## 常用缩写速查

| **缩写/概念**   | **面试中应如何理解**                                                     |
|-----------------|--------------------------------------------------------------------------|
| **RLVR**        | Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习。     |
| **Actor**       | 待优化策略，LLM 中通常是生成模型本身。                                   |
| **Critic**      | 价值/优势估计器，[[PPO]] 中常为 value head；[[GRPO]] 通常不用独立 critic。       |
| **KL**          | 衡量策略相对 reference/old policy 的漂移，常用于 trust region。          |
| **old_logprob** | rollout 时行为策略对已采样 token 的 logprob，训练 ratio 必需。           |
| **staleness**   | 样本生成策略与当前训练策略之间的版本差或 KL/logprob 差。                 |
| **KV cache**    | [[Transformer]] 推理中缓存的 key/value，中断、复用、跨版本更新都需严格管理。 |

## 目录（按题号）

### Algorithm

1. 为什么 [[Actor-Critic]] 通常优于纯 Critic？

2. [[KL Divergence|KL 散度]]、[[Cross-Entropy Loss|交叉熵]]和 MLE 的关系是什么？

3. 不同 RL 场景下奖励应该怎么设计？

4. [[Importance Sampling|重要性采样]]、拒绝采样等 Monte Carlo 方法如何融入 RL？

5. PPO 和 GRPO 中 advantage 怎么算？为什么要减 baseline？标准差归一化必要吗？

6. RL 训练与 test-time scaling 的探索有什么不同？

7. PPO clipping 怎么工作？为什么取 min？没有 clipping 会怎样？[[CISPO]] 有什么不同？

8. GRPO 为什么包含 KL penalty？KL 怎么算？为什么 [[DAPO]]、[[GSPO]] 等方法会去掉或弱化它？

9. LLM 训练中如果 loss 被意外 AllReduce 多次会发生什么？

10. [[DPO]] 中的 reward function 是什么？会 [[Reward Hacking|reward hacking]] 吗？怎么缓解？

11. MoE 模型的 [[Training-Inference Mismatch|train-inference mismatch]] 如何处理？

12. RL 训练中 group size、learning rate、PPO epochs、generation length 怎么选？

13. 相比 GRPO，[[Dr. GRPO|Dr.GRPO]]、DAPO、GSPO、CISPO、[[SAPO]]、DPPO、MaxRL、SimKO 分别改进了什么？局限是什么？

14. [[TRPO]]、DPPO、AReaL 如何在 RL objective 上施加 trust-region 约束？

15. RL 能从根本上扩展 LLM 的能力边界吗？

16. 基于 ProRL 等工作，如何理解 RL 训练边界的 scaling？

17. [[On-Policy Distillation|OPD]] 相比传统 RL 和 SFT 改进了什么？有哪些应用？

18. LLM 的 reasoning ability 在训练哪个阶段 emergence？

19. 从 DeepSeek-R1 到 V3.2 及未来 V4，RL 相关改进是什么？MoE 中 RL 有何不同？

### Infrastructure

20. 忽略 CPU offload，GRPO 训练时显存中有多少份模型？各种优化能省多少？

21. 分布式推理中的 KV cache 传输优化和多 GPU 通信策略

22. INT8 vs FP8：取舍是什么？训练和推理偏好什么精度？

23. RL rollout 的 long-tail 问题是什么？如何处理？

24. [[Continuous Batching|Continuous batching]] 在 RL 训练中引入什么问题？vLLM 和 SGLang 有何差异？

25. 如何衡量 vLLM/SGLang 利用率？训练中如何评估 KV cache 利用率？

26. 大规模多节点 RL 训练中的反向传播如何实现？

27. 有哪些异步 RL 框架？它们解决什么同步瓶颈？

28. 在 AReaL 或 partial rollout 框架中，会保留旧策略的 KV cache 吗？

29. [[Expert Parallelism]] 如何影响 MoE 吞吐？

30. 长上下文训练中如何设计 compute-communication overlap？Megatron 与 [[FSDP]] 的并行策略有何不同？

31. 如何启用确定性执行？什么是 batch invariance？原因是什么？atomic add 参与吗？能解决吗？

32. AReaL 和 slime 对 RL rollout bottleneck 的理解有何不同？

33. 全异步 RL 中如何理解 staleness？实践中典型值是多少？

34. slime 中数据如何流动？如何与 Megatron 集成？loss 怎么算？

35. VeRL、TRL、Unsloth、AReaL、slime 之间怎么选？

## 一、Algorithm

### 1. 为什么 Actor-Critic 通常优于纯 Critic？

**面试结论：**Actor-Critic 的核心价值在于把“如何行动”和“如何评估”解耦，尤其适用于大动作空间、长序列、随机策略和约束更新；这比简单理解为“Actor 加一个 Critic”更准确。纯 Critic 适合动作空间小、可以枚举或近似 $max_a Q(s,a)$ 的场景；LLM 生成的动作是词表上的逐 token 选择，完整动作空间近似 $|V|^T$，纯 [[Q-Learning|Q-learning]]/critic-only 很难稳定地对整条序列做 argmax。

**核心机制：**Actor 直接参数化策略 $\pi_\theta(a|s)$，可用采样探索、KL 约束和 logprob 形式的 [[Policy Gradient|策略梯度]]；Critic 估计 $V(s)$ 或 $Q(s,a)$，主要用来降低方差、做 [[Credit Assignment|credit assignment]]。PPO 里的 value head 是 Critic，GRPO 则把“同一 prompt 的组内平均奖励”当作廉价 baseline，从而省掉独立 value 模型。

**为什么：**纯 Critic 在深度函数逼近下常常依赖 bootstrap 和 [[Experience Replay|off-policy replay]]，容易出现过估计、分布偏移和 Bellman 误差积累。Actor-Critic 的优势是可以在 on-policy 或近似 on-policy 数据上优化受限 surrogate objective，使训练目标直接对应生成分布，绕开先学习难以校准的全局 Q 函数这一步。

**工程要点：**LLM RL 中 Actor 通常就是待训练语言模型；Critic 可以是 value head、reward model、reference model 或 group baseline。若面试官追问“GRPO 没有 Critic 是否还是 Actor-Critic”，可以回答：严格说 GRPO 是 critic-free policy optimization，但它仍然使用 baseline 思想；关键区别在于 baseline 的来源：组内相对比较替代了可学习 Critic。

**局限：**Actor-Critic 也有明显局限。Critic 训练滞后会引入 bias，value loss 会占内存和算力，PPO 的 actor/critic/reference/[[Reward Model|reward model]] 多模型链路复杂；这也是 GRPO、DAPO、Dr.GRPO 等方法流行的原因。参考：[TRPO] [PPO] [GAE] [DeepSeekMath/GRPO]。

### 2. KL 散度、交叉熵和 MLE 的关系是什么？

**定义：**对真实分布 $P$ 和模型分布 $Q$，交叉熵 $CE(P,Q)=E_{x\sim P}[-log Q(x)]$；熵 $H(P)=E_{x\sim P}[-log P(x)]$；$KL(P||Q)=E_{x\sim P}[log P(x)-log Q(x)]$。因此 $CE(P,Q)=H(P)+KL(P||Q)$。当数据分布 $P$ 固定时，最小化交叉熵等价于最小化 $KL(P_{data}||Q_\theta)$，也就是最大似然估计 MLE。

**公式：**[[SFT]] 的 token-level loss 是 $-\sum_t log \pi_\theta(y_t|x,y_{<t})$，本质是对人工/教师轨迹做 MLE。DPO/偏好优化中的 logratio 项也是 $log \pi_\theta - log \pi_{ref}$，只是监督信号从“单条正确标签”换成“chosen 比 rejected 更好”的相对约束。

**为什么：**KL 的方向很重要。$KL(P_{data}||Q)$ 是“mode covering”：数据中出现的模式都希望给概率；$KL(Q||P_{ref})$ 或 RL 中常见的 $KL(\pi_\theta||\pi_{ref})$ 更像“mode seeking/保守约束”：让策略保持在参考模型的高概率区域附近。RLHF 里的 KL penalty 更适合理解为对新策略相对 reference 漂移的约束，作用机制不同于普通监督损失。

**工程要点：**PyTorch 的 cross_entropy 通常把 log_softmax 和 NLL 合并，默认 reduction 可能是 mean；分布式训练时要搞清楚按 token 平均、按 sample 平均还是按 global batch 平均。LLM post-training 中很多“loss 不对”的 bug 根源通常在 mask、长度归一化、all-reduce/reduction 约定不一致，公式本身反而常常没有问题。

**陷阱：**KL 具有方向性，$KL(P||Q)$ 与 $KL(Q||P)$ 对 mode collapse 的影响不同；cross entropy 在离散 token 上正是最大似然的实现，适用范围超出分类任务。参考：[DPO] [PPO]。

### 3. 不同 RL 场景下奖励应该怎么设计？

**面试结论：**奖励设计要先判断任务是 verifiable reward、human preference、tool/agent trajectory，还是安全/风格约束。奖励设计的第一原则应聚焦与真实目标一致、可校准、难被 hack；密集程度只是工具，并且要避免把长度、格式、模板、工具调用次数误当成能力。

**适用场景：**数学、代码、棋类等可验证任务适合 outcome reward：答案对/测试通过给正奖励，格式错、超长、运行错误给轻惩罚。对话与安全更适合 preference reward 或 pairwise/listwise ranking。Agentic RL 需要把最终成功、工具调用成本、异常恢复、环境状态变化拆开，既要终局奖励，也要少量过程约束。

**怎么做：**好 reward 往往分层：硬约束先过滤（非法输出、危险行为、无效 JSON），任务正确性给主奖励，长度和格式只做小权重 [[Reward Shaping|shaping]]。若要做 dense reward，优先选择可解释的检查点，例如单元测试、证明步骤 verifier、工具返回码，避免让奖励模型随意给每个 token 打分。

**陷阱：**常见 reward hacking 包括输出更长拿到更多“推理过程分”、复制题目或模板绕过格式检查、利用 verifier 漏洞、在代码题中硬编码测试、在 agent 任务中重复调用工具刷分。面试中要主动提监控：reward 与 human eval 的相关性、长度分布、pass@k、失败样例、OOD 集、格式错误率、KL/[[Entropy|entropy]]。

**工程要点：**reward scale 会影响 PPO/GRPO 的有效学习率。二值奖励可先用组内 baseline；连续 reward 需裁剪或标准化；多目标 reward 不要简单线性相加后就不看分项指标。潜势函数 shaping 要避免改变最优策略；在 LLM 中还要避免把“更会解释”误当成“更会解题”。参考：[DeepSeek-R1] [DAPO] [SimKO]。

### 4. 重要性采样、拒绝采样等 Monte Carlo 方法如何融入 RL？

**面试结论：**RL 本质上是在估计策略诱导分布下的期望，Monte Carlo 是基础工具。rollout 是 Monte Carlo，policy gradient 的 sample average 是 Monte Carlo，best-of-N/pass@K 评估也是 Monte Carlo。区别在于是否纠正采样分布和目标分布不一致。

**核心机制：**重要性采样用 $\rho=\pi_{target}(a|s)/\pi_{behavior}(a|s)$ 修正 [[On-Policy vs Off-Policy|off-policy]] 数据。在 LLM 中，$\rho$ 可以按 token 计算并相乘，也可以按 sequence 汇总。PPO/GRPO 的 ratio clipping 本质是截断重要性权重，牺牲无偏性换稳定性；GSPO、CTPO、DPPO 等近期方法都在重新思考“token ratio 是否是好的 trust-region 代理”。

拒绝采样：先从当前模型或强模型中采很多候选，然后只保留 verifier 通过或 reward 高的样本做 SFT/RFT。这会产生有偏数据，但实现简单、稳定、可并行，常用于把 RL 找到的好轨迹蒸馏回模型。DeepSeek-R1 这类 pipeline 中，RL、rejection sampling、SFT/蒸馏会交替使用。

其他 Monte Carlo：self-normalized IS、truncated IS、per-decision IS、control variate、baseline、pass@K 无偏估计、off-policy evaluation 都属于这套思想。LLM 工程里要特别关注 logprob 精度、mask、EOS、长度归一化，因为 sequence-level ratio 会随长度指数级变大或变小。

**陷阱：**重要性采样只能在分布偏移可控时复用旧数据。旧策略和新策略 KL 过大时，rho 方差会爆炸；裁剪后虽然稳定，但优化目标已改变。拒绝采样也会丢掉失败轨迹中的有用反例，导致模型只模仿成功表面格式。参考：[PPO] [GSPO] [CTPO] [MaxRL]。

### 5. PPO 和 GRPO 中 advantage 怎么算？为什么要减 baseline？标准差归一化必要吗？

**面试结论：**advantage 衡量“这个动作/回答比当前期望好多少”。减 baseline 不改变策略梯度期望，但能显著降低方差；标准差归一化主要用于工程尺度控制，理论上属于可选项，并且在 GRPO 中可能引入新的偏差。

PPO：经典 PPO 里 $A_t$ 常由 [[GAE]] 计算：$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$，$A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$。LLM RLHF 常把序列级 reward 分配到 response token，再由 value head 估计每个 token 的 $V$。优点是能做较细 credit assignment；代价是需要训练 Critic。

GRPO：对同一 prompt 生成 $G$ 个回答，得到奖励 $R_1...R_G$。常见实现使用 $A_i=(R_i-mean(R))/std(R)$ 并把 $A_i$ 广播到该回答的所有 response token。组内均值 baseline 自动扣掉“题目难度”：同一道题大家都低分，不代表每个回答都该被强惩罚；同一道题中相对更好的回答才应上调概率。

**为什么：**baseline 的数学原因是 $E_{a\sim\pi}[b(s)\nabla log \pi(a|s)]=b(s)\nabla\sum_a\pi(a|s)=0$。因此减去只依赖状态/prompt 的 baseline 不改变无偏性。实践上它降低同一 prompt 内奖励方差，让梯度关注“相对好坏”。

标准差归一化：好处是 reward scale 稳定，坏处是当组内 std 很小或全 0 时梯度消失/爆炸；不同 prompt 的 std 不同会改变题目权重；按回答长度再平均还会产生长度偏置。Dr.GRPO 的一个核心批评就是 GRPO 的长度归一化和 std 归一化会扭曲优化。可选策略包括只减均值不除 std、用全局 batch std、加 epsilon、增大 group size、提高采样温度或过滤全同奖组。参考：[GAE] [DeepSeekMath/GRPO] [Dr.GRPO] [DAPO]。

### 6. RL 训练与 test-time scaling 的探索有什么不同？

**面试结论：**test-time scaling 是固定模型下多采样、搜索、验证、反思；RL 训练是把探索结果写回参数。前者提高当前问题的求解概率，后者改变未来所有问题的采样分布。两者可以互补，但优化目标并不相同。

**核心机制：**test-time scaling 通过 temperature、top-p、beam/search、self-consistency、tool search、verifier [[Reranking|reranking]] 等方式增加尝试次数。若 base model 已经能以很低概率生成正确轨迹，pass@K 会随 K 增长。RL 则用 reward 把成功轨迹的概率提高，通常显著提升 pass@1 和低 K 表现。

探索差异：训练期探索必须在“可学习”与“稳定”之间取平衡。temperature 太低会 group collapse，太高会产生噪声；KL 太强会探索不足，太弱会语言能力/安全漂移。test-time 可以临时花更多 token 搜索，但不保证参数会学会；RL 可以降低未来推理成本，但可能牺牲多样性。

**陷阱：**很多 RLVR 方法会把概率质量集中到最常见成功路径，pass@1 上升但 pass@K 或新题探索下降。SimKO、MaxRL、PKPO 等工作正是围绕“训练目标要不要对齐 pass@K/最大似然成功概率”展开。面试回答要避免绝对化：RL 与 test-time scaling 蒸馏存在交集，但覆盖范围更广；“RL 一定创造全新能力”也属于过强表述。

**工程要点：**评估时至少同时看 pass@1、pass@K、majority vote、token 长度、样本多样性、错误类型和 verifier 成本。若目标是线上单次回答，RL 提升 pass@1 很重要；若线上允许 verifier+多采样，则训练目标应显式考虑 pass@K 或多样性。参考：[SimKO] [MaxRL] [ProRL]。

### 7. PPO clipping 怎么工作？为什么取 min？没有 clipping 会怎样？CISPO 有什么不同？

**公式：**PPO clipped objective 是 $E[min(r_t(\theta) A_t, clip(r_t(\theta),1-\epsilon,1+\epsilon) A_t)]$，其中 $r_t=\pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$。取 min 是为了构造保守下界：当更新已经让正 advantage 动作概率增加太多，clip 后不再给额外收益；当负 advantage 动作概率降低太多，也不再继续鼓励。

**为什么：**min 同时处理 $A$ 正负两种符号。$A>0$ 时，希望 $r$ 增大，但最多到 $1+\epsilon$；$A<0$ 时，希望 $r$ 变小，但最多到 $1-\epsilon$。没有 clipping，多 epoch 复用同一批 rollout 时策略可能迅速远离 behavior policy，重要性权重爆炸，KL 失控，reward hacking 和语言退化都会更常见。

**工程要点：**clip 只能提供逐样本 ratio 的启发式约束，离完整 trust region 还有差距。因此还要监控 approx_kl、clip_fraction、entropy、reward、response length。eps 太小会学不动，太大会不稳定；长序列 token-level ratio 的噪声会累积，所以 LLM RL 中经常出现 sequence-level clipping、KL penalty、early stopping 或动态 clip。

CISPO：CISPO 的思路是直接裁剪重要性采样权重，并用 detach 后的裁剪权重乘 logprob objective；这一区别于 PPO 在两个 surrogate 之间取 min 的写法。这样做的动机是避免某些 token 因 hard clip 完全失去梯度，让所有 token 仍有更新信号，同时抑制过大的 IS 权重。

**局限：**CISPO 更像一种稳定化的 biased estimator，效果依赖 clip 上下界、优势尺度和任务 reward。若 reward 噪声很大或 behavior/target 差太远，保留梯度也可能把错误方向更新得更稳定。参考：[PPO] [CISPO/MiniMax-M1]。

### 8. GRPO 为什么包含 KL penalty？KL 怎么算？为什么 DAPO、GSPO 等方法会去掉或弱化它？

**面试结论：**GRPO 的 KL penalty 是为了防止模型为了短期 reward 远离 reference policy，保住语言质量、安全边界和分布稳定性。它的角色是 trust-region/regularization，不能替代 reward 本身。是否去掉 KL，取决于 reward 是否可靠、clip 是否足够、训练是否需要强探索。

**怎么做：**LLM 中通常保存 generated tokens 上的 $log \pi_\theta$ 和 $log \pi_{ref}$，只在 response mask 上计算。常见近似包括逐 token 的 logprob 差、$KL(\pi_\theta||\pi_{ref})$ 的采样估计，以及 Schulman k3 estimator：若 $r=pi_{ref}/token\_prob\_old$? 实现里常写成 $exp(logp_{ref}-logp) - (logp_{ref}-logp) - 1$，用当前策略样本估计相对 reference 的 KL。最终可以按 token 平均、按 sequence 平均或作为 reward penalty 加到 advantage 中。

**为什么：**verifiable reward 常常稀疏且漏洞多。没有 KL 时，模型可能学会奇怪格式、过长推理、重复 token、利用评测漏洞，或者牺牲通用对话能力换某一类题的分数。KL 也让多轮 PPO/GRPO 的 policy update 更接近 on-policy 假设。

DAPO/GSPO 去 KL 的动机：DAPO 在若干配置中把 KL beta 设为 0，转而依赖 decoupled clipping、dynamic sampling、token-level loss、overlong reward shaping 等机制提升探索和稳定性。GSPO 把优化单元从 token-level ratio 提到 sequence-level ratio，认为 token-level IS 对长序列和 MoE 路由不稳定；更合理的 sequence-level trust region 可以减少对额外 KL penalty 的依赖。

**陷阱：**去掉 KL 不等于“KL 没用”。更准确的说法是：在强 verifier、足够好的 clip/采样/长度控制和在线监控下，显式 reference KL 可以减弱甚至关闭，以换取探索；但在偏好对齐、安全、开放问答和弱 reward 场景，KL 仍很重要。参考：[DeepSeekMath/GRPO] [DAPO] [GSPO]。

### 9. LLM 训练中如果 loss 被意外 AllReduce 多次会发生什么？

**面试结论：**最常见后果是有效梯度尺度错了，等价于学习率被乘以或除以 world_size 的若干次方。它可能表现为训练几乎不动、reward 不涨、grad norm 异常小；也可能在使用 sum all-reduce 时表现为梯度爆炸、loss spike、NaN。

**核心机制：**[[Data Parallelism|DDP]]/FSDP 通常会在 backward 时对梯度做一次跨 rank reduce/average。若你在 backward 前又对 loss 做 all_reduce(mean)，本 rank 的计算图只包含本地 loss，但数值被除以 world_size；随后 DDP 再平均梯度，等于额外缩小一次。若重复 $k$ 次 mean all-reduce，有效梯度约多除以 $world\_size^k$。反过来，如果用 sum 而不除，可能多乘 world_size。

**工程要点：**RL 里更容易出错，因为 loss 可能先按 token、sample、microbatch、DP rank、rollout group 做多层归一化。比如先对 response token mean，再对 samples mean，再对 DP all_reduce mean，最后框架又自动 average gradients；任何一层重复都会改变优势尺度和 PPO clip 的有效强度。

怎么排查：对比单卡和多卡的同一 batch 梯度范数；打印 pre/post all_reduce 的 loss、token count、non-pad token count；确认最终除数到底是 local_tokens 还是 global_tokens；用一个固定小模型做 world_size=1/2/4 的等价性测试。若 global token average 是目标，应 all_reduce numerator 和 denominator，避免先做本地 mean 再做一次 mean。

**陷阱：**loss 数值“看起来更平滑”不代表正确。多次平均会让 [[Adam]] 的自适应项部分掩盖问题，但 weight decay、gradient clipping、KL coefficient、entropy bonus 的相对尺度都会变。参考：[PyTorch FSDP] [MegatronCore]。

### 10. DPO 中的 reward function 是什么？会 reward hacking 吗？怎么缓解？

**公式：**DPO 从 KL-regularized RLHF 的最优解反推隐式 reward：$r_\beta(x,y)=\beta[log \pi_\theta(y|x)-log \pi_{ref}(y|x)] + c(x)$。训练时无需显式拟合 reward model，目标改为最大化 chosen 相对 rejected 的 logratio margin：$log \sigma(\beta[(log \pi_\theta(y_w)-log \pi_{ref}(y_w))-(log \pi_\theta(y_l)-log \pi_{ref}(y_l))])$。

**面试结论：**DPO 的“reward”就是当前策略相对 reference 的对数概率提升，乘以 beta 后只在同一 prompt 内有意义。跨 prompt 直接比较它会失去严格意义；c(x) 是任意 prompt 常数，会在 pairwise 差分中抵消。

会不会 hacking：会，只是形式不同。DPO 可能过拟合 [[Preference Data|偏好数据]]中的长度、礼貌模板、拒答模板、格式伪特征；beta 太大时 margin 会爆炸，模型把 chosen 风格推得过强；reference 太弱或数据覆盖不足时，模型容易把“偏好数据捷径”学成主信号，弱化真正的任务能力。离线 DPO 还可能把错误 rejected/ chosen 标签放大。

缓解：清洗偏好数据，做长度控制和 prompt 分层采样；加入 label smoothing、conservative DPO、[[IPO]]/[[KTO]]/[[ORPO]] 等变体或小 KL/regularization；监控 win-rate 之外的 factuality、安全、长度、拒答率、OOD；把 DPO 作为 warmup，再用在线 PPO/GRPO/RLVR 验证；对 reward margin 做 early stopping。

**陷阱：**DPO 无需 rollout，所以便宜稳定；但它也因此不能发现当前策略的新失败模式。它的优化目标是数据集偏好，环境交互成功并未被直接优化。参考：[DPO]。

### 11. MoE 模型的 train-inference mismatch 如何处理？

**面试结论：**MoE 的 mismatch 涵盖训练 BF16、推理 FP8 等精度差异，也涵盖路由、专家容量、batch shape、EP/TP 拓扑、kernel、负载均衡策略和 KV/cache 行为不一致。RL 中 rollout engine 和 training engine 常常不同，这会破坏 on-policy logprob 假设。

主要来源：训练时可能用 Megatron/FSDP，rollout 用 vLLM/SGLang；训练 batch 和推理 continuous batching 的 token 顺序不同；router top-k、capacity factor、token dropping、expert combine 的实现不同；MoE all-to-all 在不同 EP group 上的数值归约顺序不同；推理[[Quantization|量化]]或 FP8 KV 改变 logits；动态 batch 会导致 batch invariance 问题。

处理方法：第一，尽量让 rollout 和 training 使用相同权重、[[Tokenization|tokenizer]]、position id、chat template、router 配置和精度策略。第二，启用 deterministic/batch-invariant inference，或在训练侧复用 serving kernel。第三，对 rollout 样本保存 old_logprobs、router 相关元数据、policy version，并在训练时用 importance ratio/clip 校正。第四，MoE 大模型可使用 sequence-level objective（如 GSPO）降低 token-level routing 噪声。

MoE 特有措施：使用 [[Loss-Free Load Balancing|loss-free/auxiliary-loss-free load balancing]]、node-limited routing、capacity-free 或合理 capacity 策略、DeepEP/分层 all-to-all、expert parallel 与 sequence parallel 配合。若使用 routing replay，要明确它主要减少训练/推理路由差异；真实部署路径下的性能仍要单独暴露和评估。

**陷阱：**只比较 final answer accuracy 可能看不出 mismatch。要比较同一 prompt 同一 seed 下的 token logprob、first divergent token、router load、expert hotness、KV hit、[[Tensor Parallelism|TP]] size 改变后的 logits 差异。参考：[DeepSeek-V3] [GSPO] [vLLM Batch Invariance] [SGLang RL Systems]。

### 12. RL 训练中 group size、learning rate、PPO epochs、generation length 怎么选？

**面试结论：**这些超参相互耦合。group size 决定 advantage 方差和探索覆盖，learning rate 决定单次策略漂移，PPO epochs 决定数据复用和 off-policyness，generation length 决定能否解题以及长尾/长度偏置。调参目标是 reward 上升、KL/clip_fraction 可控、entropy 不塌、长度不过度膨胀。

Group size：GRPO 至少需要 $G>1$；$G=4/8$ 常用于小规模试验，$G=8/16/32$ 更稳定但 rollout 成本线性增加。若大量 prompt 组内奖励全 0 或全 1，advantage 没信号，应提高采样温度、增大 $G$、提高题目难度混合、使用 dynamic sampling 过滤无效组，这些措施优先于盲目加学习率。

Learning rate：全参 RL 通常比 SFT 更小；[[LoRA]]/QLoRA 可稍大。经验上不要只看 loss，要看 approx_kl、ratio、clip_fraction、entropy、reward、grad_norm。若 reward 涨但 KL 爆、长度暴涨，LR/clip/KL/长度惩罚要收紧；若 KL 接近 0 且 reward 不动，可能 LR 太小、KL 太强或 advantage 全 0。

PPO epochs：1-4 是常见范围。epochs 越多，样本利用率越高，但 pi_theta 与 pi_old 差距越大，clip 更频繁，旧 logprob 更不可靠。LLM RL 尤其长序列场景常用较少 epochs、较大 token batch，通过更多 fresh rollouts 换稳定性。

Generation length：上限要覆盖正确解的必要推理，同时要防止模型通过“写很长”获得隐性优势。数学/代码可用 curriculum：先较短，再放长；同时监控 overlong ratio、EOS 率、截断后 reward 分布。DAPO 的 overlong reward shaping、token-level loss 和 dynamic sampling 就是围绕这类问题设计的。参考：[DAPO] [Dr.GRPO] [verl Best Practices] [Unsloth RL]。

### 13. 相比 GRPO，Dr.GRPO、DAPO、GSPO、CISPO、SAPO、DPPO、MaxRL、SimKO 分别改进了什么？局限是什么？

**面试结论：**这些方法大多是在 GRPO 框架上做局部修补，重点处理长度偏置、组内方差、token-level ratio、MoE 稳定性、探索/利用和 trust-region 代理等问题。面试中最好先给分类，再讲每个方法的适用场景。

Dr.GRPO：核心是指出 GRPO 的 response-length normalization 和 group std normalization 会引入偏置，尤其可能奖励更长但错误的输出。它倾向于移除或修正这些归一化，使梯度按问题/回答更公平。局限是 reward scale 需要重新校准，少了 std 归一化后不同任务混合时尺度更敏感。

DAPO：Decoupled Clip + Dynamic Sampling。常见四件套是 Clip-Higher、Dynamic Sampling、Token-level Policy Gradient Loss、Overlong Reward Shaping。它提高探索、减少无信息样本、处理长输出偏置，并开源了大规模复现实验。局限是 pipeline 更复杂，dynamic sampling 可能改变数据分布，超参依赖任务。

GSPO：把 token-level policy optimization 改成 sequence-level ratio/clip/reward，目标是减少长序列 token ratio 高方差，尤其稳定 MoE RL。局限是 sequence-level credit assignment 更粗，单个关键 token 的学习信号可能被整条序列平均掉。

CISPO：裁剪重要性采样权重并保留 logprob 梯度，避免 PPO/GRPO hard clipping 后部分 token 梯度消失。局限是目标更偏工程启发，clip 权重、advantage scale 和任务 reward 噪声都会影响稳定性。

SAPO：缩写在社区中不唯一。若指 Soft Adaptive Policy Optimization，核心是用软门控/温度化 clipping 替代硬裁剪，让梯度从“突然截断”变为平滑衰减；若指 Segment-Aligned Policy Optimization，则把更新单元对齐到语义段/推理步骤。前者局限是温度/门控超参，后者局限是需要可靠分段或步骤边界。

DPPO：本文按 Divergence Proximal Policy Optimization 理解。它认为 token ratio 是大词表下很噪的 divergence 代理，转而用 KL/TV 等直接策略分布差异的近似 mask 或约束。局限是全词表 divergence 成本高，需要 binary/top-k 近似；硬 mask 也可能丢梯度。

MaxRL：把 binary correctness 场景看成“最大化正确 rollout 的隐式似然”，构造 compute-indexed objectives，在更多采样预算下逼近 MLE，改善 test-time scaling 效率。局限是依赖可验证成功信号和多样本采样成本。

SimKO：针对 RLVR 容易 pass@1 上升、pass@K 下降的过度集中问题，在高熵 fork token 上对正确轨迹提升 top-K 候选，对错误轨迹更强惩罚 top-1 候选。局限是需要 token 分布分析和额外超参；若线上只追求 deterministic pass@1，收益未必最大。参考：[Dr.GRPO] [DAPO] [GSPO] [CISPO/MiniMax-M1] [SAPO] [DPPO] [MaxRL] [SimKO]。

### 14. TRPO、DPPO、AReaL 如何在 RL objective 上施加 trust-region 约束？

**面试结论：**TRPO 是显式 KL 约束，PPO/GRPO 是 ratio clipping 近似，DPPO 是直接用策略分布 divergence 约束，AReaL 则是在异步系统中把 trust-region 与 staleness 控制结合起来。它们解决的是同一件事：样本来自旧策略，更新需要控制与旧策略的距离。

TRPO：优化 surrogate objective，同时约束 $E_s[KL(\pi_{old}(.|s)||\pi_\theta(.|s))] <= \delta$。实现上用自然梯度/共轭梯度和 line search。优点是理论清晰、更新保守；缺点是大模型上二阶近似和全分布 KL 成本高，不适合直接套在 LLM 全参 RL 上。

PPO/GRPO：PPO/GRPO 省去显式约束优化，改在 objective 中裁剪 ratio。它便宜、可 minibatch、多 epoch，但 ratio 只是 sampled action 上的一维近似，与完整策略分布距离有差异。长序列和大词表下，这个代理会更噪。

DPPO：Divergence PPO 的动机是“别用 sampled token ratio 当完整 divergence”。它直接估计 sampled state 上新旧策略分布的 KL/TV，并用 binary/top-k 近似降低全词表成本；超界 token 的更新会被 mask 或调节。优点是更贴近 trust-region 几何；缺点是近似和硬 mask 带来新的偏差/实现复杂度。

AReaL：完全异步时，rollout policy 可能落后 learner 好几个版本，传统 on-policy 假设更弱。AReaL 用 workload balance 控制 staleness，并采用 staleness-enhanced PPO/相关校正，让旧样本在可控边界内仍可训练。这里的 trust-region 既包含算法公式，也包含系统层面的 policy version、最大 off-policyness、队列长度和权重同步频率。

**陷阱：**KL penalty、ratio clip、divergence mask、staleness bound 都是 trust-region 手段，但没有一个能替代 reward 质量。约束太强会学不动，太弱会 drift。参考：[TRPO] [PPO] [DPPO] [AReaL]。

### 15. RL 能从根本上扩展 LLM 的能力边界吗？

**面试结论：**取决于你怎样定义“能力边界”。若指 pass@1、稳定遵循格式、在已有知识/推理模式中更可靠地找到答案，RL 明显可以扩展。若指创造预训练中完全不存在的知识或无法由环境反馈验证的能力，单纯 RL 很难凭空产生；它更像把潜在能力从低概率区域拉到高概率区域，并通过交互反馈学习策略。

支持观点：DeepSeek-R1/R1-Zero 展示了纯 RL 可以诱导 self-reflection、verification、long [[Chain-of-Thought|CoT]] 等推理行为；ProRL 类工作显示，延长 RL、控制 KL、重置 reference、使用多样任务，可以继续推动边界样本和更难题表现。Agentic RL 还能通过工具环境提供新信息，使模型学会搜索、调用、修复和规划。

保守观点：许多研究发现，RL 往往提升低 K 表现，但高 K 下 base model 可能已能偶尔解出，说明 RL 主要在重分配概率质量。若 reward 稀疏、verifier 狭窄或探索不足，RL 会收缩多样性，甚至降低 pass@K 和开放问题能力。

**回答框架：**RL 扩边界有三个来源：第一，重加权，把已存在但低概率的正确轨迹变常见；第二，策略学习，把“尝试、验证、修正、工具调用”的过程内化；第三，环境交互，通过新数据和反馈获得预训练外的信息。前两者不等价于新增世界知识，第三者才更接近持续学习。

工程含义：评估时需要超越单点 benchmark。若 RL 后 pass@1 涨、pass@256 跌，说明 exploit 强而 explore 弱；若难题集合、OOD、工具任务和高 K 都改善，才更有资格说边界扩大。参考：[DeepSeek-R1] [ProRL] [SimKO] [MaxRL]。

### 16. 基于 ProRL 等工作，如何理解 RL 训练边界的 scaling？

**面试结论：**Prolonged RL 的核心在于长期训练中防止 collapse、保持探索、扩大任务分布，并让 reference/KL/长度/数据难度随训练动态调整；简单延长训练步数远远不够。RL scaling 的边界同时受 base model、reward、rollout compute、训练稳定性和评估指标约束。

**核心机制：**ProRL 类方法强调 KL 控制、reference policy resetting、多样任务混合和长时间训练。早期 RL 往往快速提升容易题 pass@1，随后进入平台期；继续 scaling 需要更多新颖问题、更强 verifier、更大 group/search、更合理的长度预算，否则只会过拟合同类模板。

边界解释：base model 的“可达区域”很重要。若模型在高 K 下从不成功，RL 几乎没有正样本可学；若偶尔成功，RL 可以通过奖励把这些轨迹放大。ProRL 的边界扩展应理解为把低概率能力系统性放大，并在某些任务上通过持续探索发现新的策略组合。

**工程要点：**长训要监控 KL、entropy、response length、success diversity、prompt-level learning curve、pass@K、失败簇。reference reset 能防止 KL 约束过早锁死；长度惩罚要避免把必要推理截断；任务采样要避免所有 batch 都是全 0 或全 1。

**局限：**scaling RL 的瓶颈远超 GPU 数量。reward bottleneck、verifier 漏洞、长尾 rollout、staleness、MoE 路由不稳、数据泄漏都会成为新边界。参考：[ProRL] [BroRL] [MaxRL]。

### 17. OPD 相比传统 RL 和 SFT 改进了什么？有哪些应用？

**面试结论：**这里按 On-Policy Distillation / On-Policy Guided Distillation 理解 OPD。它让学生模型在自己的分布上采样，再由教师、verifier 或带特权信息的同模型提供密集 token/step 监督。它介于 SFT 和 RL 之间：比 SFT 更少 train-test mismatch，比稀疏 RL 方差更低。

对比 SFT：SFT 学的是离线专家轨迹，学生一旦偏离专家分布，就可能遇到训练集中没有的状态。OPD 采样来自学生当前策略，教师纠正的是学生真的会走到的位置，因此更像 dataset aggregation，能减少 exposure bias。

对比 RL：传统 RLVR 可能只有最终 0/1 reward，credit assignment 粗、样本效率低。OPD 可把教师的 next-token 分布、错误解释、工具反馈、privileged context 转成密集监督，让失败轨迹也有学习信号。它通常更稳定，适合 reward 稀疏但教师可用的场景。

应用：推理模型自蒸馏、语音/多模态 LLM 对齐、自动驾驶规划、小模型从大模型/同模型特权模式学习、工具/agent 轨迹纠错、代码错误反馈蒸馏。X-OPD/OPSD 这类方法特别强调“学生生成自己的轨迹，教师只负责在这些轨迹上给指导”。

**局限：**OPD 受教师上限约束，教师成本高；若教师偏差强，学生会稳定地学错；若只蒸馏成功样式，仍可能缺少探索。最稳妥的 pipeline 往往是 SFT warmup + OPD/RL 混合 + verifier eval。参考：[OPSD/OPD] [X-OPD]。

### 18. LLM 的 reasoning ability 在训练哪个阶段 emergence？

**面试结论：**推理能力通常由预训练、SFT、RL、蒸馏共同塑造，呈现跨阶段累积的特征。预训练提供知识、算法片段和潜在推理模式；SFT 教模型把推理过程写成可读格式；RL 把可验证成功的推理策略概率提高；蒸馏把大模型或 RL 模型的过程迁移给小模型。

预训练阶段：模型通过大规模文本、代码、数学、证明、问答学习到大量隐式结构。很多 base model 在高 temperature/high K 下已经能偶尔解难题，这说明 reasoning 的一部分潜能早已存在，只是 pass@1 不稳定。

SFT 阶段：SFT 强化格式、指令遵循、CoT 风格和题型模板。它能让模型“知道应该怎么回答”，但若 SFT 数据过窄，也会让模型停留在表面推理模仿，缺少真正验证。

RL 阶段：DeepSeek-R1-Zero 类型结果说明，在强 verifiable reward 下，RL 可以诱发自我反思、回溯、验证、动态调整策略等行为。更准确地说，RL 放大了能带来 reward 的内部策略，并把低概率成功路径变得更常见。

面试追问：如果问“aha moment 是 RL 产生的吗”，建议回答：公开结果显示 RL 训练中会出现明显行为转变，但后续分析也发现一些 base model 已有类似 bias 或雏形。因此 emergence 是预训练能力、采样探索、reward 选择和优化动力学共同作用的结果。参考：[DeepSeek-R1] [Dr.GRPO]。

### 19. 从 DeepSeek-R1 到 V3.2 及未来 V4，RL 相关改进是什么？MoE 中 RL 有何不同？

**面试结论：**DeepSeek-R1 的核心叙事是用 RLVR/GRPO 激发推理；[[DeepSeek-V3 (2024)|DeepSeek-V3]] 提供高效 MoE base、MLA、DeepSeekMoE、auxiliary-loss-free load balancing 和 [[Multi-Token Prediction|MTP]] 等基础；DeepSeek-V3.2 公开资料强调 DSA [[Sparse Attention|稀疏注意力]]、可扩展 RL 协议、post-training compute scaling 和大规模 agentic task synthesis。未来 [[DeepSeek-V4 (2026)|V4]] 适合作为技术趋势推断，避免当作事实陈述。

R1：R1-Zero 展示不依赖人工 CoT SFT 的纯 RL 推理增强；R1 则加入 cold-start SFT、多阶段 RL、rejection sampling、蒸馏等，使输出更可读、更稳定。它的关键点是 verifiable reward、组内相对优势、长 CoT 和自我验证行为。

V3/V3.2：V3 是强 MoE 基座，训练后经过 SFT 和 RL 释放能力；V3.2 在公开技术报告中把长上下文效率和 agentic/reasoning post-training 放到核心位置，DSA 解决长上下文成本，scalable RL 与合成 agent 任务解决“会推理且会用工具”。

MoE 中 RL 的不同：第一，模型副本和 expert states 巨大，rollout/training/ref/reference 同时放置更难；第二，routing 会随 batch、精度和 policy update 改变，导致 train-inference mismatch；第三，EP all-to-all、expert load imbalance 和长尾序列会放大系统瓶颈；第四，token-level objective 可能与 expert routing 交互不稳定，因此 GSPO、路由一致性、aux-loss-free balancing、batch invariance 更重要。

未来趋势：更强稀疏注意力/长上下文、更大的可验证和 agentic 环境、更异步的 RL 系统、更精细的 trust-region/staleness 控制、更低精度 rollout 与更确定的训练-推理一致性。对“V4 会怎样”的回答应说明这是合理方向，并避免编造未公开细节。参考：[DeepSeek-R1] [DeepSeek-V3] [DeepSeek-V3.2] [GSPO]。

## 二、Infrastructure

### 20. 忽略 CPU offload，GRPO 训练时显存中有多少份模型？各种优化能省多少？

**面试结论：**naive GRPO 至少要考虑训练 actor、rollout/inference actor、reference model 三类权重副本；没有 critic 是 GRPO 相比 PPO 的主要省内存点。若不用 KL/reference，可少一份 reference；若 rollout 与训练权重共享或分离到不同 GPU，单卡视角会不同。面试时要明确“逻辑副本”和“同一 GPU 上的物理副本”区别。

逻辑副本：1）训练 actor：含参数、梯度、优化器状态、激活；2）rollout actor：推理引擎权重和 KV cache；3）reference model：计算 KL/logprob，通常 no_grad；4）reward model/verifier：若是模型式 reward，还要额外一份；5）old policy 通常以 rollout 时保存的 old_logprobs 表示，通常无需额外保留完整模型副本。PPO 还常有 critic/value model，GRPO 省掉这部分。

显存粗算：BF16 参数约 $2P$ bytes，梯度约 $2P$，Adam 一阶/二阶 FP32 状态约 $8P$，若有 FP32 master weights 还会更多；激活与 sequence length、microbatch、checkpointing 有关；KV cache 与 $batch\_size * seq\_len * layers * kv\_heads * head\_dim * dtype$ 成正比。RL rollout 长上下文时，KV 往往和权重一样成为瓶颈。

优化：FSDP/[[ZeRO|ZeRO-3]] 可把参数、梯度、优化器状态按 DP shard，理论上训练态大头接近除以 world_size；GRPO 无 critic 省掉 value model 及其优化器；LoRA/QLoRA 只训练 adapter，可大幅省 optimizer/grad；[[Activation Checkpointing|gradient checkpointing]] 省激活但加重算；FP8/INT8 KV 降 KV cache；weight sharing/Unsloth Standby/SGLang sleep-wake 可避免训练与推理权重双常驻；sequence packing/token-level batching 提升有效利用。

**陷阱：**回答时需要超越“GRPO 两份模型”这种粗略说法。在真实系统里，reference、reward、rollout engine、optimizer states、KV cache、CUDA graph workspace、fragmentation 都占显存。参考：[DeepSeekMath/GRPO] [Unsloth RL] [SGLang RL Systems] [verl]。

### 21. 分布式推理中的 KV cache 传输优化和多 GPU 通信策略

**面试结论：**分布式推理的核心瓶颈常同时集中在 KV cache、跨 GPU 通信和 GEMM；在 RL rollout 中，前两者经常更决定吞吐。prefill 计算密集、decode 内存带宽/KV 访问密集；长上下文和 RL rollout 让 KV 的分配、迁移、复用、淘汰决定吞吐。

KV 优化：[[Paged Attention|PagedAttention]]/vLLM 用分页管理减少 KV 内存碎片；SGLang 的 [[Radix Attention|RadixAttention]] 更强调前缀树复用；LMCache/HiCache/层级 cache 把 KV 在 GPU、CPU、存储、网络之间移动；PD disaggregation 把 prefill 和 decode 分离，需要高效传输 KV。优化手段包括 pinned memory、RDMA/NVLink/NCCL、批量 KV 迁移、异步 I/O 与 compute overlap、FP8 KV、压缩、prefix-aware routing 和 cache pinning。

多 GPU 通信：TP 需要每层 all-reduce/all-gather/reduce-scatter；PP 需要 stage 间传 activation/KV；DP 做请求级或 replica 级负载均衡；EP/MoE 需要 token dispatch/combine 的 all-to-all；PD 分离需要跨 engine 传 KV。策略要按硬件拓扑决定：同节点 NVLink 可用 TP/EP，跨节点尽量降低细粒度通信，把 DP/PP/PD 边界放到网络较慢处。

RL 特殊点：rollout 权重会频繁更新，KV cache 必须与 weight version 绑定。更新权重后继续使用旧 KV 会产生错误 logits；同一 policy version 内可以复用 prompt prefix。多轮 agent rollout 要支持 pause/resume、cache-aware routing、慢请求迁移和故障重试。

指标：TTFT、TPOT、tokens/s、KV hit rate、GPU KV usage、eviction rate、跨节点带宽、all-to-all time、prefill/decode 分离后的队列长度。参考：[vLLM/PagedAttention] [SGLang] [LMCache] [SGLang RL Systems]。

### 22. INT8 vs FP8：取舍是什么？训练和推理偏好什么精度？

**面试结论：**INT8 是定点量化，依赖 scale/zero point；FP8 是浮点量化，有指数位，动态范围更适合训练中的激活/梯度。训练通常以 BF16/FP16 为基准，在 Hopper/Blackwell 上用 FP8 GEMM 加 BF16/FP32 master/optimizer；推理可选 INT8/INT4 权重量化或 FP8 权重/KV，取决于硬件和精度要求。

INT8：优点是生态成熟、老 GPU 支持广、权重量化压缩稳定，AWQ/GPTQ/SmoothQuant 等方案常用于推理。缺点是 scale 管理复杂，对异常值敏感，训练反传和激活动态范围不如 FP8 自然；INT8 KV 也需要 scale 元数据，实际节省不总是理想。

FP8：常见 E4M3 精度较高但范围小，E5M2 范围大但 mantissa 少。NVIDIA Transformer Engine 常用混合方案：前向激活/权重用 E4M3，反向梯度相关用 E5M2 或其他策略，并保留高精度 master weights。FP8 的优势是硬件张量核吞吐高、训练可用；缺点是需要支持 FP8 的硬件、缩放策略和数值监控。

推理选择：若目标是最大兼容和低显存，INT8/INT4 权重量化常见；若目标是 H100/B200 上高吞吐、较小精度损失和大 KV cache，FP8 weights/KV 很有吸引力。安全/数学推理模型要特别评估 logprob、pass@K 和长上下文退化，同时参考 perplexity 之外的指标。

**陷阱：**低精度不仅影响答案，还影响 RL 的 old_logprob、KL、ratio 和 reward。训练用 BF16、rollout 用 FP8/INT8 时，可能出现训练-推理 mismatch。参考：[NVIDIA Transformer Engine FP8] [vLLM FP8 KV] [SGLang Quantization]。

### 23. RL rollout 的 long-tail 问题是什么？如何处理？

**面试结论：**long-tail 指同一批 rollout 中少数样本生成特别长、工具调用特别慢、环境交互卡住或 reward/verifier 慢，导致同步训练必须等最慢样本，GPU 大量空转。推理越像 agent、多轮、长 CoT，long-tail 越严重。

来源：问题难度差异、模型陷入重复、没有 EOS、工具 API 超时、代码沙箱编译慢、verifier 队列拥塞、连续 batching 中短请求不断插入、MoE expert imbalance。RL 中还会因 group sampling 放大：一个 prompt 生成 G 条，某一条很长就拖住整个组。

处理：按 token 粒度做 batching，避免以样本粒度粗糙分批；长度 bucket 和 max_new_tokens curriculum；过长惩罚与 EOS 奖励；动态 sampling，丢弃全对/全错或过慢组；异步 rollout/training 解耦；SGLang pause/resume 或 abort/retract 慢请求；达到足够有效样本后提前开训；沙箱和 reward 服务独立扩缩容；对工具调用设超时和重试预算。

算法影响：丢弃慢样本会改变数据分布，可能偏向短答案；截断会把潜在正确长推理当失败；异步会引入 staleness。因此需要记录每种 status：finished、truncated、timeout、tool_error、aborted，并在 advantage/reward 中区分处理。

指标：p50/p95/p99 生成长度、rollout step wall time、GPU idle、waiting queue、timeout rate、truncated reward、每个 prompt 的有效样本数。参考：[AReaL] [SGLang RL Systems] [slime Fully-Async Rollout]。

### 24. Continuous batching 在 RL 训练中引入什么问题？vLLM 和 SGLang 有何差异？

**面试结论：**continuous batching 提高推理吞吐，但会让 RL 面临 policy version、KV cache、batch invariance、请求排序、权重更新时机和可复现性问题。服务系统追求吞吐，RL 训练追求“这批样本确实来自可记录的旧策略”。

问题一：权重更新与请求并发。若 rollout engine 正在生成，训练侧更新了权重，后续 token 可能来自新 policy，前半段来自旧 policy，old_logprob 不再对应单一 behavior policy。解决方式是暂停/清空请求、version tagging、flush KV，或允许 partial rollout 但把版本和 staleness 写进样本。

问题二：batch invariance。相同 prompt 在不同 batch size/order 下 logits 可能不同，原因包括浮点非结合性、kernel reduction order、TP all-reduce、custom kernel。RL 中这会造成 rollout logprob 与 training recompute logprob 对不上，ratio 噪声增大。

问题三：调度偏差。continuous batching 会优先填充短请求，提高吞吐但可能改变样本完成顺序；长请求被反复推迟会影响训练数据分布。多轮 agent 还会要求环境状态和请求状态一致，普通 text generation 的调度假设不够。

vLLM vs SGLang：vLLM 的核心优势是 PagedAttention、成熟 OpenAI serving、广泛集成和详细 metrics；TRL/verl 常用它做 rollout。SGLang 强调 RadixAttention 前缀复用、程序化前端、SGLang Model Gateway、PD/EPD、RL 专用 sleep/wake、refit、pause/continue 和 deterministic inference 接口，更适合复杂 agent/RL 生命周期。两者都在快速演化，选择要看模型、硬件、框架和团队熟悉度。参考：[vLLM] [SGLang] [TRL GRPO] [SGLang RL Systems]。

### 25. 如何衡量 vLLM/SGLang 利用率？训练中如何评估 KV cache 利用率？

**面试结论：**nvidia-smi GPU utilization 只提供很粗的视角。LLM serving 的瓶颈可能是 prefill compute、decode memory bandwidth、KV cache、调度队列、网络、reward server 或沙箱。需要同时看吞吐、延迟、显存、cache、队列和 MFU。

通用指标：tokens/s（prefill 和 decode 分开）、TTFT、TPOT、E2E latency、running/waiting/swapped requests、GPU SM 利用、HBM 带宽、显存占用、每卡 batch tokens、request p99、错误/timeout。MFU 可用估算 FLOPs/理论峰值，但 decode 阶段更应看带宽利用和 KV 命中。

vLLM：Prometheus /metrics 暴露 running/swapped/waiting、GPU/CPU KV cache usage、prompt/generation tokens、TTFT、TPOT、latency、prefill/generation throughput 等。训练时重点看 GPU KV usage 是否接近上限、是否频繁 swap/evict、waiting queue 是否长。

SGLang：开启 --enable-metrics 和 --enable-mfu-metrics 后，可看 prompt/generation tokens、token_usage、cache_hit_rate、num_running_reqs、num_used_tokens、gen_throughput、TTFT/TPOT/E2E，以及估算 per-GPU flops/read/write bytes。cache_hit_rate 和 token_usage 对前缀复用/长上下文很关键。

KV 利用率：看 allocated pages/blocks、used tokens/max tokens、内部碎片、prefix hit、eviction、recompute、per-rank imbalance、FP8/BF16 KV 比例、KV transfer 带宽。RL 还要按 policy version 统计：更新权重后是否 flush cache，旧 cache 是否错误复用。参考：[vLLM Metrics] [SGLang Production Metrics]。

### 26. 大规模多节点 RL 训练中的反向传播如何实现？

**面试结论：**LLM RL 的 backward 本质仍是 teacher-forcing 风格的 token loss 反传：rollout 阶段生成 tokens 并保存 old_logprobs/rewards/masks，训练阶段把 prompt+response 喂回 actor，重新计算 response token logprob，构造 PPO/GRPO loss，然后通过 FSDP/Megatron 等并行策略反传。

数据流：样本包含 input_ids、attention_mask、response_mask/action_mask、old_logprobs、ref_logprobs、reward、advantage、policy_version、status。训练 forward 只对 response token 的 logprob 进 loss；prompt token 通常只作为上下文不计入 RL loss。若有 KL，reference model no_grad forward；若 PPO，有 critic forward/value loss。

FSDP/ZeRO：参数、梯度、optimizer state 按 DP shard。forward/backward 前 all-gather 当前 layer 参数，完成后 reshard；梯度 reduce-scatter；optimizer 在 shard 上更新。优点是代码侵入低、适合 HuggingFace 模型；缺点是长上下文和 TP/EP 组合需额外设计 overlap。

Megatron：通过 TP 切分层内矩阵，[[Pipeline Parallelism|PP]] 切分层深度，CP 切分序列，EP 切分专家，DP 复制/分片数据。backward 中会有 TP all-reduce/reduce-scatter、PP activation/gradient 传递、CP attention 通信、EP all-to-all。优点是性能强、适合超大 MoE；缺点是接入成本高。

**工程要点：**activation checkpointing、sequence packing、microbatch、[[Gradient Accumulation|gradient accumulation]]、loss scaling、global token count all-reduce 必须一致。RL 的 advantage 是外部张量，通常不需要梯度；ratio 中 old_logprob 要 detach；reference/old policy 不能被误反传。参考：[PyTorch FSDP] [MegatronCore] [verl] [slime]。

### 27. 有哪些异步 RL 框架？它们解决什么同步瓶颈？

**面试结论：**异步 RL 框架的目标是打破“生成一整批 -> 等最慢 -> 训练 -> 同步权重 -> 再生成”的串行循环。它们把 rollout、reward、training、weight sync、environment 变成 producer-consumer 管线，以更高 GPU 利用率换取可控 staleness。

代表系统：AReaL 是完全异步 RL 系统，rollout worker 持续生成，trainer 收够 batch 就更新，并用 staleness-enhanced PPO 稳定旧样本。LlamaRL/AsyncFlow 也强调异步 off-policy、权重同步和流水化。slime 提供 fully-async rollout 示例，保持固定数量 in-flight generation，避免训练 step 等待最慢样本。ProRL Agent/Polar 更偏 agent harness 和 Rollout-as-a-Service。

解决的瓶颈：1）长尾 rollout 导致训练 GPU 空等；2）训练时 rollout GPU 空等，rollout 时训练 GPU 空等；3）多轮 agent 环境/沙箱慢；4）reward model/verifier 成为中心队列；5）大模型权重同步导致推理服务反复重启；6）异构 GPU 上不同阶段资源需求不匹配。

新问题：staleness、off-policy correction、样本版本管理、权重广播、KV cache invalidation、队列背压、失败重试、动态负载均衡、评估可复现性。异步提速依赖 rollout、training 和队列背压的平衡；若 rollout 远慢于 training，会训练旧数据；若 training 远慢于 rollout，队列膨胀且样本过期。

面试答法：先说同步瓶颈，再说异步架构，再说 staleness 控制。回答时要超越框架名罗列。参考：[AReaL] [LlamaRL] [AsyncFlow] [slime Fully-Async Rollout] [ProRL Agent]。

### 28. 在 AReaL 或 partial rollout 框架中，会保留旧策略的 KV cache 吗？

**面试结论：**KV cache 是由“权重 + token + position + kernel/precision”共同决定的中间状态。权重更新后，旧 KV 只适配旧权重。partial rollout 可以在同一 policy version 内保留 KV；若跨 policy version 继续生成，就必须把样本标记为旧策略/stale，并保留真实来源。

正确区分：1）同一个 rollout 请求暂停后继续，且权重未变、KV 未 flush，可以保留 KV；2）权重更新后继续旧请求，可以选择用旧权重完成该 trajectory，作为 stale sample 训练；3）若要用新权重继续，必须重算 prefix KV；4）静态 prompt prefix cache 也应按 weight version 失效，否则 logits 错。

SGLang 语义：pause_generation 有 abort/retract/in_place 等模式。in_place 依赖 KV cache 继续，因此后续 flush_cache 不可行；update weights API 默认常带 flush_cache，正是为了避免旧 KV 污染新权重。partial rollout 主要用于控制长尾和请求生命周期；跨版本免费复用 KV 会破坏语义。

AReaL 语义：AReaL 的重点是接受 rollout policy 与 learner policy 的版本差，并通过 staleness-aware objective 训练旧样本；旧 KV 迁移成新 policy KV 这条路径会破坏语义。训练时仍需用当前 actor 重新 forward 计算 logprob，old_logprob 只作为 behavior policy 记录。

**陷阱：**KV cache 复用错误常表现为偶发乱码、logprob 对不上、reward 波动，直接 crash 反而不常见。RL 系统应把 cache key 包含 model version、tokenizer/chat template、position scheme、precision、TP/EP 配置。参考：[AReaL] [SGLang RL Systems] [slime]。

### 29. Expert Parallelism 如何影响 MoE 吞吐？

**面试结论：**Expert Parallelism（EP）把不同 experts 放到不同 GPU 上，让 MoE 可以扩展总参数和 expert 数，但吞吐取决于 token dispatch/combine 的 all-to-all、expert load balance、grouped GEMM 效率和网络拓扑。EP 可以提升可训练规模，也可能把瓶颈从 GEMM 变成通信。

正向作用：每个 token 只激活少数专家，EP 让每张卡只保存/计算部分 experts，降低单卡内存和计算压力。专家足够大、batch tokens 足够多、路由均衡时，grouped GEMM 能很好利用 GPU，MoE 的 activated params 性价比高。

瓶颈：top-k routing 会把 token 分发到远端专家，产生 all-to-all；热门专家会排队，冷专家空闲；短序列/小 batch 导致每个 expert token 太少，GEMM 不饱和；跨节点 EP 通信延迟高；RL rollout 的长度和 prompt 难度差异会让每 step expert load 波动更大。

优化：node-limited routing 把专家限制在拓扑友好的范围；auxiliary-loss-free/loss-free load balancing 用动态 bias 控制负载且减少干扰梯度；DeepEP/高性能 dispatcher 降低 all-to-all；expert tensor parallel 处理超大 expert；sequence parallel 与 EP/TP 配合；batch-level overlap 隐藏 EP-A2A 通信。

面试要点：EP 不一定越大越好。要用 profiler 看 all-to-all time、expert token histogram、dropless capacity、GEMM occupancy、跨节点带宽。MoE RL 还要看 routing determinism 和训练-推理一致性。参考：[MegatronCore MoE] [DeepSeek-V3]。

### 30. 长上下文训练中如何设计 compute-communication overlap？Megatron 与 FSDP 的并行策略有何不同？

**面试结论：**长上下文训练的瓶颈是 attention/KV/activation 内存和跨设备通信。overlap 的目标是让 all-gather、reduce-scatter、all-to-all、P2P activation 传输尽量藏在 attention/MLP 计算后面。Megatron 更偏显式组合 TP/PP/CP/EP；FSDP 更偏数据并行维度的参数/梯度/优化器分片。

长上下文策略：[[Flash Attention|FlashAttention]]/块状 attention 降低显存；activation checkpointing 用重算换内存；sequence/context parallel 把序列维切到多卡；ring attention 或 CP 让每个 rank 只保留部分序列激活，并在 attention 处通信 KV；pipeline microbatch 和 interleaved schedule 减少 bubble；chunked prefill/sequence packing 提高有效 token 密度。

Overlap 设计：TP 的 all-reduce 可与下一层计算 overlap；FSDP 的参数 all-gather 可 prefetch 下一层，backward reduce-scatter 可与前一层梯度计算 overlap；PP 用 1F1B/interleaving overlap 前后向；EP all-to-all 可与 expert GEMM 或其他 batch 的计算 overlap；CP 的 KV all-gather/reduce-scatter 要围绕 attention kernel 分块。

Megatron：提供 TP、PP、DP、SP、CP、EP、distributed optimizer、MoE dispatcher 等强组合，适合千卡/万卡、超大 dense/MoE、长上下文极致性能。缺点是模型代码和配置复杂，对团队系统能力要求高。

FSDP：包装式分片参数/梯度/optimizer，易接 HuggingFace，弹性好，适合中大模型和研究迭代。缺点是单靠 FSDP 不解决层内矩阵太大、长序列 attention 和 MoE expert all-to-all；常需与 TP/CP/activation checkpointing 混用。参考：[PyTorch FSDP] [MegatronCore] [Megatron Context Parallel]。

### 31. 如何启用确定性执行？什么是 batch invariance？原因是什么？atomic add 参与吗？能解决吗？

**定义：**确定性执行通常指相同代码、相同硬件/软件、相同 seed 和相同输入下输出一致。Batch invariance 更强：同一个请求的输出不应受 batch size、batch order、同批其他请求影响。LLM RL 更关心后者，因为 rollout engine 会 continuous batching，而 training engine 可能用不同 batch 形状重算 logprob。

原因：GPU 浮点加法非结合，parallel reduction 顺序不同会导致 bit-level 差异；dynamic batching 改变 kernel shape；TP all-reduce/custom all-reduce 的归约顺序不同；MoE capacity/routing 可能依赖 batch；随机采样、dropout、uninitialized memory、KV cache 重用、量化 scale、CUDA graph 也会引入差异。

atomic add：atomic 操作常是非确定性的来源之一，因为多个线程写同一地址的顺序不固定，浮点求和顺序不同结果不同。使用 atomic add 本身不能“解决确定性”，除非你能保证固定顺序或使用确定性替代算法。要解决需要 deterministic kernel、固定 reduction tree、固定调度、或在硬件/软件层面保证相同归约顺序。

怎么启用：固定 random/NumPy/PyTorch seed；关闭 dropout；torch.use_deterministic_algorithms(True)；设置 CUBLAS_WORKSPACE_CONFIG；关闭 cuDNN benchmark；固定 tokenizer/chat template/position id；固定 TP size 和 kernel；使用 vLLM batch invariance 或 SGLang deterministic inference；禁用会引入非确定性的 custom all-reduce；对 MoE 固定 capacity/routing 策略。

**陷阱：**确定性通常有性能代价，跨 PyTorch/CUDA/驱动/GPU 架构一致性也需要单独验证。temperature=0 仍不足以保证 batch invariance；greedy decoding 也会因 logits 微小差异在近 tie token 处分叉。参考：[PyTorch Reproducibility] [vLLM Batch Invariance] [SGLang RL Systems] [TP-invariant inference]。

### 32. AReaL 和 slime 对 RL rollout bottleneck 的理解有何不同？

**面试结论：**AReaL 把核心瓶颈定义为同步 RL 的阶段串行和长尾等待，因此用 fully asynchronous generation/training 和 staleness-aware PPO 解耦。slime 更强调把 Megatron training、SGLang rollout、custom data/reward/environment、Data Buffer 串成高性能 post-training 系统，瓶颈在数据生成链路、引擎切换、长尾控制和工程可扩展性。

AReaL：rollout worker 不等 trainer，trainer 收够 batch 就更新；系统通过 workload balance 控制 staleness，并在算法上接受旧策略样本。它的关键词是 decouple、staleness、throughput、fully async。适合面试回答“为什么同步框架 GPU 利用率低”。

slime：官方定位是连接 Megatron 与 SGLang 的 RL post-training 框架，强调高性能训练、灵活数据生成、custom rollout/reward/verifier/environment、SGLang server group、weight sync、partial/fully async rollout。它的关键词是 Megatron+SGLang、Data Buffer、custom rollout、GLM 风格大规模工程。

差异：AReaL 更像“异步 RL 系统和算法论文”，重点是异步带来的 off-policyness 如何被约束；slime 更像“可落地的训练-推理-数据生成框架”，重点是各种 rollout 形态都能进入统一 buffer 和 Megatron loss。两者都处理长尾，但 AReaL 从训练/生成解耦入手，slime 从 rollout driver、队列和 SGLang 控制接口入手。

选择：若研究异步 RL 理论和 staleness，讲 AReaL；若面试岗位强调 Megatron、SGLang、MoE、custom agent env 和生产训练链路，讲 slime。参考：[AReaL] [slime] [slime Fully-Async Rollout]。

### 33. 全异步 RL 中如何理解 staleness？实践中典型值是多少？

**定义：**staleness 是 rollout 行为策略与当前训练策略的版本差。可按 learner update step 计，也可按 wall-clock、token 数、KL/logprob drift、队列等待时间计。LLM RL 中最有用的观测对象是 staleness distribution 及其导致的 ratio/KL 分布，单个整数信息量有限。

为什么重要：policy gradient 假设样本来自当前策略或可用重要性采样纠正的近邻策略。staleness 越大，old_logprob 与 current logprob 差越大，ratio 方差越高，clip 更频繁，甚至方向错误。异步系统的吞吐收益正是用 staleness 风险换来的。

实践范围：同步训练 staleness=0。保守异步通常把 head off-policyness 控制在很小的单 digit 版本差，例如 1-4 个 learner update；更激进系统可能允许更大队列或阶段式高 staleness，但必须配合 relaxed clipping、negative-advantage veto、proximal policy、样本丢弃或 KL gating。不同模型、LR、batch token 和任务差异很大，固定数值缺乏普适性。

怎么控制：限制 rollout queue 长度；训练/rollout throughput 配平；对超过阈值的样本降权或丢弃；用 policy_version 和 old_logprob 做精确 ratio；监控 per-sample KL、clip fraction by staleness、reward by staleness；周期性同步权重；必要时 reference/prox policy 插值。

面试答法：先定义，再说 trade-off，再说指标。典型工程回答是“从 staleness=0 跑通等价性，再逐步放到 1、2、4，观察 KL/clip/reward；上线同时看分布与均值”。参考：[AReaL] [Mu-GRPO] [A-3PO]。

### 34. slime 中数据如何流动？如何与 Megatron 集成？loss 怎么算？

**面试结论：**slime 的核心数据流是：数据源/环境产生 prompt -> SGLang rollout 生成 Sample -> reward/verifier 填 reward/status/loss_mask -> Data Buffer 组 batch -> Megatron actor forward/backward 计算 RL loss -> 参数同步回 SGLang。它把“数据生成”抽象得很灵活，把“训练”交给 Megatron。

数据入口：常见是 JSONL prompt/label/metadata，也可以是 custom rollout function、reward model、verifier、agent environment。rollout 返回的 Sample 至少要包含 tokens、response_length、reward、status、loss_mask 等字段。dynamic sampling 和 partial rollout 都是在进入 buffer 前或进入 buffer 时处理样本有效性。

Megatron 集成：slime 直接传 Megatron 参数配置训练后端，例如 TP、SP、PP、CP/ring attention、EP/ETP、recompute、checkpoint。训练 actor 由 Megatron 管；rollout 由 SGLang server/server group 管；训练前或训练后通过权重同步让 SGLang 更新 actor 权重，也可配置 co-locate 或 disaggregated。

Loss 计算：slime 支持 GRPO、GSPO、[[REINFORCE++]]、PPO 等 advantage estimator。训练时对 response token 计算 logprob，与 old_logprob 构造 ratio/clip，乘 advantage 和 loss_mask。默认可按 per-sample mean：$mean_i(sum\ token\_loss_i / len_i)$，也可按 per-token：$sum\ losses / sum\ lengths$。PPO 还需要 critic/value loss；OPD 可作为正交选项加入蒸馏信号。

**工程要点：**loss_mask 决定哪些 token 学；status 决定截断/超时样本如何处理；per-sample vs per-token 会改变长度偏置；SGLang context length 可与训练不同但要保证 logprob 重算一致；权重更新后 cache flush/version 管理非常重要。参考：[slime Usage Guide] [slime]。

### 35. VeRL、TRL、Unsloth、AReaL、slime 之间怎么选？

**选择建议：**若只能给一个默认答案，我会选 verl 作为通用大规模 LLM RL 研究/工程起点；它支持 PPO/GRPO/DAPO/DrGRPO/GSPO 等多算法，能接 FSDP/Megatron、vLLM/SGLang，社区和复现实验丰富。但真正选择取决于模型规模、硬件、是否需要异步、是否需要 Megatron+SGLang、是否只是 LoRA 小实验。

TRL：适合 HuggingFace 生态、快速原型、单机/小多机、教学和算法验证。GRPO/DPO/PPO 接口清晰，能用 vLLM rollout。局限是超大规模、多引擎、复杂 MoE/agent pipeline 需要大量自定义。

Unsloth：适合消费卡/单机/LoRA/QLoRA、低显存、长上下文小中模型 RL。它在 memory-efficient GRPO、权重共享、gradient checkpointing、低显存 notebook 体验上很强。局限是如果你要几百卡全参 MoE、复杂异步 rollout 或深度 Megatron 集成，Unsloth 的优先级会下降。

verl：适合工业级多节点 RL post-training，算法覆盖广、HybridFlow 抽象清楚，能集成 FSDP/Megatron/vLLM/SGLang，DAPO 等复现生态强。局限是系统复杂度高，调度和配置学习成本高。

AReaL：适合重点解决 long-tail、异步 rollout/training、staleness 和 agent/reasoning 大吞吐问题。若面试岗位强调 async RL 系统，它是重点。局限是异步引入 off-policy/staleness 调参，不如同步框架容易 debug。

slime：适合 Megatron+SGLang、MoE、大规模 post-training、custom rollout/reward/environment、GLM/DeepSeek/Qwen 类模型工程。局限是栈更“重”，对 Megatron/SGLang 经验要求高；若只是小模型实验会过度复杂。

面试答法：先问约束：GPU 数、模型大小、全参还是 LoRA、算法、rollout engine、是否 agent、多轮/工具、是否 MoE。然后给决策：原型 TRL；低显存 Unsloth；通用生产 verl；异步 AReaL；Megatron+SGLang 大模型 slime。参考：[TRL] [Unsloth RL] [verl] [AReaL] [slime]。

## 三、参考文献与延伸阅读

以下参考用于定位原始论文、官方文档或方法来源。面试准备时建议优先读官方论文/文档中的 objective、system diagram、ablation 和 limitations；二手解读只适合作为辅助。

**[TRPO]** Schulman et al., Trust Region Policy Optimization, arXiv:1502.05477, https://arxiv.org/abs/1502.05477

**[PPO]** Schulman et al., Proximal Policy Optimization Algorithms, arXiv:1707.06347, https://arxiv.org/abs/1707.06347

**[GAE]** Schulman et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation, arXiv:1506.02438, https://arxiv.org/abs/1506.02438

**[DPO]** Rafailov et al., Direct Preference Optimization: Your Language Model is Secretly a Reward Model, arXiv:2305.18290, https://arxiv.org/abs/2305.18290

**[DeepSeekMath/GRPO]** Shao et al., DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, arXiv:2402.03300, https://arxiv.org/abs/2402.03300

**[DeepSeek-R1]** DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, arXiv:2501.12948, https://arxiv.org/abs/2501.12948

**[DeepSeek-V3]** DeepSeek-AI, DeepSeek-V3 Technical Report, arXiv:2412.19437, https://arxiv.org/abs/2412.19437

**[DeepSeek-V3.2]** DeepSeek-AI, DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models, arXiv:2512.02556 / Hugging Face model card, https://arxiv.org/abs/2512.02556

**[Dr.GRPO]** Understanding R1-Zero-Like Training: A Critical Perspective / Dr.GRPO, arXiv:2503.20783, https://arxiv.org/abs/2503.20783

**[DAPO]** Yu et al., DAPO: An Open-Source LLM Reinforcement Learning System at Scale, arXiv:2503.14476, https://arxiv.org/abs/2503.14476

**[GSPO]** Group Sequence Policy Optimization, arXiv:2507.18071, https://arxiv.org/abs/2507.18071

**[CISPO/MiniMax-M1]** MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention / CISPO, arXiv:2506.13585, https://arxiv.org/abs/2506.13585

**[SAPO]** Segment-Aligned Policy Optimization and Soft Adaptive Policy Optimization related materials; note that SAPO is an overloaded acronym in 2025-2026 literature.

**[DPPO]** Qi et al., Rethinking the Trust Region in LLM Reinforcement Learning / Divergence Proximal Policy Optimization, arXiv:2602.04879, https://arxiv.org/abs/2602.04879

**[MaxRL]** Tajwar et al., Maximum Likelihood Reinforcement Learning, arXiv:2602.02710, https://arxiv.org/abs/2602.02710

**[SimKO]** Peng et al., SimKO: Simple Pass@K Policy Optimization, arXiv:2510.14807, https://arxiv.org/abs/2510.14807

**[ProRL]** ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in LLMs, arXiv:2505.24864, https://arxiv.org/abs/2505.24864

**[BroRL]** BroRL: Broadening Exploration for Prolonged Reinforcement Learning in LLMs, related ProRL follow-up, 2026.

**[OPSD/OPD]** Self-Distilled Reasoner / On-Policy Self-Distillation, arXiv:2601.18734, https://arxiv.org/abs/2601.18734

**[X-OPD]** Cross-modal On-Policy Distillation for speech/multimodal LLM post-training, 2026 literature.

**[CTPO]** Cumulative Token Policy Optimization, arXiv:2503/2025 literature on prefix-level/token-level IS correction.

**[HybridFlow/verl]** Sheng et al., HybridFlow: A Flexible and Efficient RLHF Framework, arXiv:2409.19256; verl docs: https://verl.readthedocs.io/

**[TRL]** Hugging Face TRL documentation, GRPOTrainer/DPO/PPO, https://huggingface.co/docs/trl/

**[Unsloth RL]** Unsloth Reinforcement Learning Guide and Memory Efficient RL documentation, https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide

**[AReaL]** Fu et al., AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning, arXiv:2505.24298; docs: https://inclusionai.github.io/AReaL/

**[slime]** slime documentation: Megatron + SGLang post-training RL framework, https://slime.readthedocs.io/

**[vLLM/PagedAttention]** Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, arXiv:2309.06180; vLLM docs: https://docs.vllm.ai/

**[SGLang]** Zheng et al., SGLang: Efficient Execution of Structured Language Model Programs, arXiv:2312.07104; docs: https://docs.sglang.ai/

**[LMCache]** LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference, arXiv:2510.09665, https://arxiv.org/abs/2510.09665

**[SGLang RL Systems]** SGLang for RL Systems documentation, sleep/wake, refit, pause/resume, deterministic inference, https://docs.sglang.io/docs/advanced_features/sglang_for_rl

**[vLLM Metrics]** vLLM production metrics documentation, https://docs.vllm.ai/en/latest/usage/metrics.html

**[SGLang Production Metrics]** SGLang Production Metrics documentation, https://docs.sglang.io/docs/references/production_metrics

**[vLLM Batch Invariance]** vLLM Batch Invariance documentation, https://docs.vllm.ai/en/latest/features/batch_invariance/

**[PyTorch FSDP]** PyTorch Fully Sharded Data Parallel documentation, https://docs.pytorch.org/docs/stable/fsdp.html

**[MegatronCore]** NVIDIA Megatron Core Parallelism Strategies Guide, https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html

**[MegatronCore MoE]** NVIDIA Megatron Core MoE documentation, https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html

**[NVIDIA Transformer Engine FP8]** NVIDIA Transformer Engine FP8 documentation, https://docs.nvidia.com/deeplearning/transformer-engine/

**[PyTorch Reproducibility]** PyTorch Reproducibility documentation, https://docs.pytorch.org/docs/stable/notes/randomness.html

**[TP-invariant inference]** Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch, arXiv:2511.17826, https://arxiv.org/abs/2511.17826

**[Mu-GRPO]** How Off-Policy Can GRPO Be? Mu-GRPO for Efficient LLM Reinforcement Learning, arXiv:2605.17570, https://arxiv.org/abs/2605.17570

**[A-3PO]** A-3PO: Accelerating Asynchronous LLM Training with Staleness-aware Proximal Policy Approximation, arXiv:2512.06547, https://arxiv.org/abs/2512.06547
