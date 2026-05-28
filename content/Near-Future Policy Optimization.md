---
type: method
description: NPO 在 RLVR 中用同一训练轨迹的近未来 checkpoint 提供 verifier-filtered 正确轨迹，在信号质量和 off-policy 方差之间取得更好的折中
aliases:
  - NPO
  - AutoNPO
  - Near-Future Policy Optimization
  - 近未来策略优化
prerequisites:
  - "[[GRPO]]"
  - "[[RLVR]]"
  - "[[On-Policy vs Off-Policy]]"
  - "[[Importance Sampling]]"
  - "[[KL Divergence]]"
  - "[[Entropy Collapse]]"
tags:
  - post-training
  - reinforcement-learning
  - reasoning
created: 2026-04-26
updated: 2026-04-26T22:14
---

# Near-Future Policy Optimization

Near-Future Policy Optimization（NPO）是一种用于 [[RLVR]] 的 mixed-policy 训练方法：当当前策略在某个 prompt 上还解不出来时，用**同一条训练轨迹上稍晚一点的 checkpoint** 生成一条 verifier 认可的正确轨迹，替换 rollout group 中的一个位置。它的核心判断是：好的辅助轨迹需要同时满足强度和距离约束；轨迹质量要足够高，分布也要离当前策略足够近，否则 [[Importance Sampling|重要性采样]] 带来的方差会吃掉高质量信号。

> [!paper] 论文出处
> Qin et al., "Near-Future Policy Optimization", arXiv:2604.20733 (2026)
> - 本地 clipping：[[Clippings/Paper/2604.20733/2604.20733|Near-Future Policy Optimization]]
> - arXiv: https://arxiv.org/abs/2604.20733

---

## 动机：为什么需要 NPO？

[[GRPO]] 这类 on-policy RLVR 的瓶颈通常来自 rollout 分布本身：当前策略采不到足够好的学习样本，目标函数再复杂也缺少可用的正信号。

训练早期，很多 prompt 的 $n$ 条 rollout 全错，group relative advantage 几乎没有可用的正样本信号；训练后期，策略分布变窄，容易出现 [[Entropy Collapse|entropy collapse]] 或平台期，模型反复采样已有的推理模板，难以探索出新的正确路径。

自然的补救是引入 off-policy 轨迹，但不同来源各有缺陷：

- 外部 teacher 轨迹质量高，但分布可能离当前策略很远，更新方差大，模型未必吸收得了。
- 历史 replay 轨迹离当前策略相对近，但质量被过去 checkpoint 的能力上限卡住，难以突破当前平台。
- 纯 on-policy 轨迹最稳定，却只能学习当前策略自己已经能采到的东西。

> [!intuition] 核心问题
> mixed-policy RLVR 真正要优化的是"当前策略能吸收多少有效信号"。NPO 把这个问题转化为选择一个合适的未来 checkpoint：它要足够未来，能解决当前不会的 prompt；又不能太未来，否则分布差距会让 off-policy 更新变得高方差。

---

## 信号质量与方差成本

论文用两个量刻画辅助轨迹来源是否值得用：

> [!definition] 有效学习信号
> 对距离当前策略 $\Delta$ 个优化步的辅助策略，定义
> $$
> \mathcal{S}(\Delta)=\frac{Q(\Delta)}{V(\Delta)}
> $$
> 其中 $Q(\Delta)$ 是信号质量：在当前策略失败的 prompt 中，辅助策略能生成 verifier-correct 轨迹的比例；$V(\Delta)$ 是方差成本：把这个辅助策略的轨迹放进当前策略更新时，由分布差异引入的梯度方差。

这个公式背后的学习逻辑很直接：

- $\Delta$ 太小：future policy 几乎和当前策略一样，$Q(\Delta)$ 低，给不出新知识。
- $\Delta$ 太大：future policy 虽然更强，但离当前策略更远，$V(\Delta)$ 快速升高。
- 合适的 $\Delta^*$：$Q$ 已经明显提升，而 $V$ 还没有爆炸，$\mathcal{S}$ 达到峰值。

这也是 NPO 与普通 off-policy teacher 的关键区别：NPO 在**同一优化路径**上寻找一个近未来 self；普通 off-policy teacher 通常直接引入一个外部强模型。因为初始化、数据、优化历史都相同，近未来 checkpoint 的推理分布通常比外部 teacher 更接近当前策略。

---

## 核心机制

### 1. 先得到一个近未来 guide

假设当前训练到 step $t$，策略是 $\pi^{(t)}$。NPO 先继续训练 $\Delta$ 步得到 $\pi^{(t+\Delta)}$，然后回滚到 step $t$，让 $\pi^{(t+\Delta)}$ 作为当前策略的 guide。

对每个 prompt $x$，guide 离线生成若干回答，经过 verifier 筛选后，只保留一条正确轨迹 $o'_x$。如果 guide 也解不出这个 prompt，就不缓存该 prompt。

这里的缓存很重要：NPO segment 内不需要每步都调用 future policy，只是在 replay 期间复用这些 verifier-filtered trajectories。

### 2. 只替换困难 prompt 的一个 rollout slot

标准 GRPO 会对每个 prompt 采样 $n$ 条 on-policy trajectories：

$$
o_i \sim \pi^{(t)}(\cdot \mid x), \quad i=1,\dots,n
$$

NPO 先计算当前 group 的 on-policy pass rate：

$$
\hat{p}(x)=\frac{1}{n}\sum_{i=1}^n r(x,o_i)
$$

当 $\hat{p}(x)\le \tau_{\text{gate}}$ 且缓存中存在 $o'_x$ 时，NPO 用 $o'_x$ 替换第 $n$ 个 rollout；否则保持原 group 不变：

$$
\mathcal{G}_{\text{NPO}}(x)=\{o_1,\dots,o_{n-1},\tilde{o}_n\}
$$

$$
\tilde{o}_n=
\begin{cases}
o'_x, & \hat{p}(x)\le \tau_{\text{gate}} \text{ 且 } o'_x \text{ 存在}\\
o_n, & \text{otherwise}
\end{cases}
$$

> [!intuition] 为什么只替换一个位置？
> NPO 的目标是在原本的 [[GRPO]] group 中注入一条"当前策略暂时采不到、但又很接近当前策略"的正样本，同时避免把训练推成 imitation learning。只替换一个 slot 可以保留大部分 on-policy 分布，让 group-relative advantage 仍然主要由当前策略自己的 rollouts 决定。

### 3. 目标函数基本不变

替换后的 group 仍然用 group-relative advantage：

$$
A_i=\frac{r_i-\mathrm{mean}(\{r_j\}_{j=1}^n)}{\mathrm{std}(\{r_j\}_{j=1}^n)}
$$

再放入原来的 clipped policy objective。严格地说，guide slot 的行为策略是 $\pi^{(t+\Delta)}$，on-policy slots 的行为策略是 $\pi^{(t)}$，所以可以用重要性比率修正：

$$
\rho_{i,k}^q(\theta)=
\frac{\pi_\theta(o_{i,k}\mid x,o_{i,<k})}
{q_i(o_{i,k}\mid x,o_{i,<k})}
$$

其中 $q_i=\pi^{(t)}$ 对应普通 rollout，$q_i=\pi^{(t+\Delta)}$ 对应 guide rollout。

> [!warning] IS correction 对 NPO 的收益有限
> 论文的 ablation 显示，对 NPO 来说精确 IS correction 和把 guide trajectory 近似当成 on-policy 的效果接近。NPO 刻意选择 near-policy guide，使 $\pi_\theta / \pi^{(t+\Delta)}$ 本身接近 1，因此 IS correction 的边际收益有限。这个结论不能直接外推到外部 teacher：guide 离当前策略很远时，去掉 IS correction 可能导致训练崩溃。

---

## 两种人工干预

NPO 可以出现在训练的两个不同阶段，解决的问题也不同。

### Early-stage bootstrapping

训练刚开始时，正确 rollout 稀少，很多 group 没有正样本。NPO 可以先跑一个短 scout segment 得到稍微更强的 checkpoint，再从 base weights 重新开始训练，用 scout checkpoint 给早期窗口提供正确轨迹。

这相当于给稀疏奖励阶段加了一个"近未来热启动"：guide 足够近，所以不会把模型拉向陌生分布；guide 又比初始策略稍强，所以能在困难 prompt 上提供可学习的正样本。

### Late-stage plateau breakthrough

训练后期，策略可能在一批稳定失败的 prompt 上停住。NPO 可以继续训练到稍强的 checkpoint，再回滚到平台期附近，用未来 checkpoint 在同一批 prompt 上提供正确轨迹。

这里 NPO 的作用是**突破 on-policy ceiling**：当前策略自己采不到的正确路径，被同源近未来策略重新带回 replay 窗口。冷启动依赖早期窗口，平台期干预则依赖已经变强的 near-future checkpoint。

---

## AutoNPO

人工 NPO 需要人看训练曲线来决定什么时候介入、回滚多远。AutoNPO 把这两件事自动化。

它维护一个 mistake pool $\mathcal{B}$，记录最近 batch 中 group accuracy 很低的 prompt 以及失败发生的 step。然后用三个阶段控制干预：

1. **Trigger**：当 reward 的 EMA 停滞，同时 entropy 继续下降时，认为可能出现探索坍塌；随后用当前策略在 mistake pool 的小子集上 probe，确认它现在是否能解决过去失败的问题。
2. **Rollback distance**：在保存的 checkpoint 距离集合 $\mathcal{D}$ 中选择
   $$
   \Delta^*=\arg\max_{\Delta\in\mathcal{D}}\frac{\hat{Q}(\Delta)}{\hat{V}(\Delta)}
   $$
   其中 $\hat{Q}$ 来自当前策略对对应 mistake slice 的 pass rate，$\hat{V}$ 用当前 checkpoint 与回滚 checkpoint 的 per-token [[KL Divergence|KL]] 估计。
3. **Execution**：用当前策略作为 guide，为 $\mathcal{B}_{\Delta^*}$ 中的 prompt 生成并缓存正确轨迹；训练回滚到 $t-\Delta^*$，在 replay 这个 segment 时只对 mistake pool 中的 prompt 触发 NPO 替换。

> [!intuition] AutoNPO 的本质
> AutoNPO 把"当前策略是过去策略的 near-future self"这件事反过来用：当现在的模型已经会做过去不会做的题，就让现在的模型回去教过去的自己。

---

## 为什么这个设计有效？

### 近未来 self 同时满足 strong enough 与 close enough

外部 teacher 通常 strong enough 但不 close enough；历史 replay 通常 close enough 但不 strong enough。NPO 的近未来 checkpoint 位于同一训练轨迹上，因此它和当前策略共享架构、初始化、数据分布和优化历史。

这让 $\Delta$ 成为一个可调旋钮：增加 $\Delta$ 会提高 guide 的解题能力，也会增加分布差距。NPO 的目标就是找到一个让 $\mathcal{S}(\Delta)$ 最大的中间点。

### Verifier-filtered trajectory 降低噪声

NPO 只缓存 verifier 认可的正确轨迹，不把 future policy 的所有输出都当成监督。这样做避免了一个常见陷阱：future checkpoint 更强，不代表它每条 reasoning trace 都值得学习。

### Gate 保护容易 prompt

NPO 只在 $\hat{p}(x)\le \tau_{\text{gate}}$ 时注入 guide trajectory。对于当前策略已经会做的 prompt，继续使用纯 on-policy group，避免无意义地把训练推向 off-policy。

### 保留原 RL objective

NPO 不改 verifier、不改 reward，也不引入额外 distillation loss。它更像是改变 trajectory source 的插件，因此可以接在 GRPO-style RLVR backbone 上。

---

## 实验信号

论文在 Qwen3-VL-8B-Instruct 上以 GRPO-style RLVR 为 backbone，训练数据为 MMFineReason-123K，并在 8 个多模态推理 benchmark 上评估。

关键结果：

- Base model 平均分 57.88，纯 GRPO 到 60.25。
- NPO early-stage only 达到 62.12。
- NPO early + late-stage 达到 62.84。
- AutoNPO 达到 63.15，高于 GRPO 的 60.25，也高于 RLEP 的 61.48。

训练动态上，AutoNPO 的提升主要来自几个干预窗口带来的阶梯式改善。它还能让 entropy 停止持续下降甚至重新扩张，说明 near-future guidance 可能帮助恢复探索，同时也会提供可模仿的正确答案。

---

## 边界与失效条件

> [!warning] 什么时候 NPO 可能不工作？
> 
> 1. **没有可靠 verifier**：NPO 依赖 verifier 过滤正确轨迹。如果 reward/verifier 噪声大，缓存的 guide trajectory 可能会放大奖励漏洞。
> 2. **future checkpoint 不够强**：如果 $\Delta$ 太小，guide 解不出当前失败的 prompt，$Q(\Delta)$ 太低，NPO 退化为几乎纯 on-policy。
> 3. **future checkpoint 太远**：如果 $\Delta$ 太大，guide 变成 far-future policy，分布差距和 IS 方差升高，NPO 会接近外部 teacher 或远未来 replay 的问题。
> 4. **训练轨迹不可回滚或 checkpoint 太稀疏**：NPO 需要保存 checkpoint，并能 replay 某个训练 segment；工程系统如果不支持回滚，AutoNPO 的收益会打折。
> 5. **prompt 分布快速漂移**：如果 replay segment 的 prompt 分布和 guide 缓存时的 prompt 集不一致，缓存轨迹的覆盖率会下降。
> 6. **把 NPO 误用成 SFT**：如果替换太多 rollout slots 或取消 gate，训练会从"注入近未来信号"变成强制模仿 future policy，损失 on-policy RLVR 的探索与校正能力。

---

## 与相关方法的关系

NPO 继承了 [[On-Policy vs Off-Policy|off-policy]] 轨迹能提高样本效率的想法，同时把普通 replay 的"轨迹来源"问题改成 checkpoint distance 的选择问题。ExGRPO 这类历史 replay 用的是过去成功轨迹，近但能力受限；RLEP 这类 far-future replay 用的是更强策略，强但可能过远；NPO 则把 checkpoint distance 当成可控变量，在质量与方差之间找峰值。

[[DAPO]]、GSPO、SAPO 等 GRPO-family 方法主要改 clip 形式、advantage 估计或 token weighting；NPO 的主要改动发生在**rollout group 的来源**上。因此它可以和某些 objective-level 改进组合，但组合时仍要重新检查 $\mathcal{S}(\Delta)$ 是否成立。

NPO 也和 [[On-Policy Distillation]]、[[RLSD]] 共享"让模型从更强的 self 学习"的直觉。NPO 的 teacher strength 来自优化时间；[[On-Policy Distillation]] 或 [[RLSD]] 往往依赖 privileged context、token-level teacher logits 或额外蒸馏损失。NPO 注入的是 sequence-level verifier-correct trajectory。

---

## 面试要点

> [!interview] Q: NPO 为什么不用最强的 future checkpoint？
> 因为辅助轨迹的有效性取决于 $\mathcal{S}=Q/V$，同时受 $Q$ 和 $V$ 影响。最强 checkpoint 可能解题率高，但与当前策略分布差距大，importance ratio 方差高，当前策略未必能稳定吸收。NPO 选择 near-future checkpoint，用来在"更强"和"更近"之间找最佳折中。

> [!interview] Q: NPO 和经验回放有什么本质区别？
> 经验回放通常复用过去 checkpoint 已经生成的成功轨迹，优势是接近当前训练分布，但质量被历史策略上限限制。NPO 使用同一训练轨迹上的未来 checkpoint 主动重新生成当前 prompt 的正确轨迹，因此比历史 replay 更强，同时又比外部 teacher 或 far-future policy 更接近当前策略。

> [!interview] Q: AutoNPO 自动化了什么？
> 它自动决定两个问题：什么时候介入，以及回滚多远。触发依据是 reward stagnation 加 entropy drop 这类探索坍塌信号；回滚距离通过最大化经验估计的 $\hat{S}(\Delta)=\hat{Q}(\Delta)/\hat{V}(\Delta)$ 来选择。
