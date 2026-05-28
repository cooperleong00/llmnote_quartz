---
type: method
description: DORA 通过多版本流式 rollout、动态资源编排和 KV-cache 迁移，在保持异步 RL 收敛约束的同时消除 long-tail generation bubble
aliases:
  - Dynamic ORchestration for Asynchronous Rollout
  - DORA 异步 RL 系统
prerequisites:
  - "[[GRPO]]"
  - "[[PPO]]"
  - "[[KV Cache]]"
  - "[[RLHF]]"
tags:
  - post-training
  - rlhf
  - reinforcement-learning
  - system
  - efficiency
created: 2026-05-01
updated: 2026-05-01T12:29
---

# DORA

DORA（Dynamic ORchestration for Asynchronous Rollout）是面向大模型 RL post-training 的异步 rollout 系统。它把长尾响应继续留在原策略版本中生成，同时让已完成样本立即进入训练，从而在保留 [[GRPO]] / [[PPO]] 这类标准 RL 目标约束的前提下提升 rollout 吞吐。

> [!paper] 论文出处
> **DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training**  
> Tianhao Hu et al., 2026, arXiv:2604.26256  
> Clipping: [[Clippings/Paper/2604.26256/2604.26256|DORA Paper]]

## 动机

大模型 RL 训练通常循环执行 rollout、reward/reference 计算和 policy training。对长 CoT 推理、代码生成、agentic 任务来说，rollout 往往占单步时间的 50%-80%，并且输出长度呈长尾分布：一批样本会被少数极长响应拖住。

同步训练的 batch barrier 会制造两类空泡：

- **intra-node bubble**：同一设备上短请求已结束，slot 空闲等待长请求。
- **inter-node bubble**：快节点完成整批生成后，等待最慢节点。

异步训练可以让 rollout 和 training 重叠，但会引入 stale trajectory。DORA 的核心问题因此变成：怎样提升 rollout 利用率，同时保留 RL 训练能解释清楚的行为策略分布。

## 三个约束

DORA 将异步 RL 的正确性压缩成三个系统必须维护的约束。

> [!definition] C1: intra-trajectory policy consistency
> 一条完整 trajectory 的所有 token 都由同一个策略版本 $w_j$ 生成：
> $$
> \forall t,\quad y_{i,t}^{(j)} \sim \pi_{w_j}(\cdot \mid x, y_{i,<t}^{(j)})
> $$
> 这样 importance ratio 中的 behavior policy 是明确的，[[GRPO]] 或 [[PPO]] 的策略更新仍然有清晰语义。

> [!definition] C2: data integrity
> 所有已经分配的响应都要保留，尤其是长尾 CoT trajectory。丢弃长响应会改变样本长度分布，也会破坏 group-relative advantage 的统计基础。

> [!definition] C3: bounded staleness
> 训练策略和生成策略之间的版本差要有上界：
> $$
> v(\theta) - v(w_j) \le K
> $$
> $K$ 是吞吐和新鲜度之间的控制旋钮。较大的 $K$ 允许更多并行版本，较小的 $K$ 让样本更接近 on-policy。

这三个约束解释了 DORA 和 [[ROLL Flash (2025)]] 等异步系统的差别。ROLL Flash 侧重 rollout-train 解耦和 sample-level pipeline，并通过 off-policy 机制处理 staleness；DORA 进一步把同一 trajectory 的策略版本一致性作为系统调度边界，让标准 RL 目标和系统迁移优化共用同一个不变量。

## 核心机制

### 多版本流式训练

DORA 在 rollout worker 上同时维护多个 policy version。每个请求派发时带上版本标签，随后完整 trajectory 都在该版本下生成。只要训练端收集到 $TBS$ 个样本，就可以开始下一次训练；尚未完成的长尾请求继续在旧版本中运行，并在后续步骤流入训练。

这个设计解决了同步 barrier 的根源：长尾请求继续占用它需要的版本资源，短请求释放出来的吞吐可以服务新版本请求。系统层面保留多个 behavior policy，算法层面仍能为每条 trajectory 找到唯一的行为策略。

### 滑动窗口 staleness 控制

活跃版本集合写作：

$$
W = \{w_j, \dots, w_{j-K+1}\}
$$

窗口只有在最旧版本的所有 trajectory 都完成并进入训练后才向前滑动。这个协议让 staleness 上界变成一个确定的系统性质，避免把新鲜度交给队列状态的偶然性。论文实验中 $K=3$ 能显著提升吞吐，同时 reward 曲线仍接近同步训练；$K=1$ 更新更新鲜，但系统并行度较低。

### 动态资源编排

多版本并行会带来资源碎片：旧版本的剩余请求逐渐变少，固定占用的 DP group 会越来越空。DORA 用集中式 Load-Balancing Orchestrator 按版本负载重分配资源：

$$
\mathcal{DP}_w =
\mathrm{Round}\left(
\mathcal{DP}_{total}
\times
\frac{R_w}{\sum_{w'} R_{w'}}
\right)
$$

其中 $R_w$ 是版本 $w$ 的活跃请求数。调度器基于三类事件触发重平衡：训练步完成后推广新版本、KV-cache pressure 超过阈值、以及周期性清理 legacy version 的残余请求。调度器还会优先给最新 policy version 补充新 prompt，让大多数新增样本保持较高 freshness；旧版本只补足残余空闲 slot，用来维持硬件利用率。

### KV-cache 迁移

DORA 最有价值的系统洞察来自 C1：同一 trajectory 在同一 policy version 下生成时，任意承载该版本的物理实例都会得到等价的 [[KV Cache]] 状态。因此，请求跨 DP group 迁移时可以转移已有 KV-cache，避免对长上下文重新 prefill。

实际迁移分成轻量 metadata forwarding 和重型 KV-cache data transfer。metadata 包含 request ID、生成状态、已解码 token 数和版本标签；KV-cache 用高性能通信原语传输。调度器会优先把请求留在原 rank，只有版本重分配确实需要时才发生物理迁移；长上下文压力较高时，DORA 还会把暂时等待的 KV-cache offload 到 host memory。

这点在长 CoT 和 [[Mixture of Experts|MoE]] 模型上尤其重要。长上下文 re-prefill 会随着上下文长度增长而变贵；MoE 的 expert parallelism 还会放大非 MoE 层的负载不均。DORA 把算法约束转化成系统优化条件：保持策略版本一致性后，KV-cache reuse 才有数学基础。

## 与其他异步方案的关系

> [!comparison] 设计取舍
> | 方案 | bubble 处理 | trajectory 一致性 | 数据完整性 | staleness |
> |---|---|---|---|---|
> | 同步 RL | 等待最长响应 | 保持 | 保持 | 无 stale |
> | one-step off-policy | rollout 与 training 重叠 | 保持 | 保持 | 通常 $K=1$ |
> | replication / oversampling | 多生成、早收够 | 保持 | 可能丢长样本 | 可控 |
> | partial rollout | 将长响应切段 | 放宽 | 保持 | 需额外修正 |
> | DORA | 多版本流式 + 动态迁移 | 保持 | 保持 | 滑动窗口有界 |

[[LongCat-Flash-Thinking (2025)]] 中已经把 DORA 作为训练基础设施使用，用多版本 Actor 流式生成和弹性调度提升大规模 reasoning RL 的训练效率。这篇 DORA 论文把该机制独立展开，补足了形式化约束、调度算法和 KV-cache reuse 的系统论证。

## 效果

在 64/128 H800 GPU 的 dense 32B 实验中，DORA 的主要收益来自 rollout 阶段压缩：

- 64 GPUs：rollout 从同步的 14.9 min 降到 1.8 min，rollout speedup 达到 $8.2\times$。
- 64 GPUs：平均 step time 从同步的 22.91 min 降到 14.67 min，端到端吞吐达到 23,327 tokens/s。
- 128 GPUs：rollout-only fraction 约为 24%，rollout speedup 相比同步训练达到 $5.9\times$。
- 128 GPUs：端到端吞吐达到 34,135 tokens/s，相比同步训练提升 $2.12\times$。
- 生产 MoE 场景：在 4,096 accelerators、最大 64K response length 的 LongCat-Flash 训练中，rollout 在数学/TIR 场景约 $3.6\times$ 加速，在 agentic training 场景约 $6.2\times$ 加速。

收敛实验显示，$K=1$ 和 $K=3$ 的 DORA reward 曲线与同步训练趋势接近。$K=3$ 的 per-step 收敛略慢，但 wall-clock 训练速度更快，体现了 staleness bound 的实际权衡。

系统开销主要来自资源重平衡和请求迁移。论文报告的 Load-Balancing 开销在 64 GPUs 为 0.414%，128 GPUs 为 1.519%；Request Transfer 开销分别为 3.627% 和 2.123%；Free Cache 开销约 0.02%。这些数字说明 DORA 的收益主要来自 rollout bubble 压缩，调度和迁移成本没有吞掉主要增益。

## 局限性

> [!warning] 适用边界
> - DORA 适合 rollout 明显成为瓶颈、输出长度长尾严重的 RL 训练；如果训练阶段占主导，收益会下降。
> - 系统需要同时维护多个 policy version，会增加权重同步、状态迁移和调度复杂度。
> - $K$ 需要结合任务和模型调节。过大的 staleness window 会提高吞吐，但可能放慢每步收敛；论文当前依赖 PPO clipping 缓解 stale trajectory 的 off-policy bias，尚未加入 adaptive staleness control 或 explicit delay compensation。
> - KV-cache reuse 依赖同版本实例之间的状态等价；实现层面要保证推理 kernel、采样状态、位置编码和缓存布局的一致性。
> - 论文的公开对比主要在同一内部 RL framework 中完成，缺少与 veRL、AReaL 等公开系统的直接 benchmark；MoE 大规模实验也主要来自生产数据，开源 MoE 上的系统性验证还不充分。

## 面试视角

> [!interview] Q: DORA 为什么能同时提升吞吐并保持标准 RL 目标？
> A: 它把异步的粒度放在 trajectory level：完成的 trajectory 立即进入训练，未完成的长 trajectory 继续由原 policy version 生成。这样 rollout 不再被最长响应阻塞，同时每条 trajectory 仍有唯一 behavior policy，importance ratio 和 staleness bound 都能被清楚定义。

> [!interview] Q: DORA 和 partial rollout 的核心差别是什么？
> A: partial rollout 会把长响应跨版本切段，因此需要额外算法修正，并且每次权重更新后容易触发 re-prefill。DORA 保留同一 trajectory 的版本一致性，让长响应在旧版本中完整生成，再用多版本调度和 KV-cache 迁移解决资源碎片。
