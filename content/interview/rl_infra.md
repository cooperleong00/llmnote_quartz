# 面试答题文档：slime / GRPO-PPO-CISPO / SGLang / Determinism / Fully Async

下面这份可以当作面试答题手册使用。整体思路是：**先讲系统数据流，再讲 loss 和算法，再讲工程观测与异步训练取舍**。面试时不要只背公式，要把“为什么这么做”和“出了问题怎么看”讲清楚。

---

## 1. slime 里 data 怎么流动，Megatron 怎么结合，经过哪些函数，loss 怎么计算

### 1.1 slime 的整体架构

slime 可以理解成三块：

1. **Rollout 侧**：SGLang engine + sglang-router，负责根据 prompt 生成 response，记录 rollout logprob、tokens、reward、metadata。
2. **Data buffer / data source**：在 prompt 初始化、动态采样、partial rollout、fully async 队列之间做桥接。
3. **Training 侧**：Megatron backend，负责 actor/critic/ref forward、loss 计算、backward、optimizer step，然后把新权重同步给 rollout 侧。

官方文档把 slime 描述为把 **Megatron 训练** 和 **SGLang 数据生成** 连接起来的 RL scaling framework；README 里也明确说 training 从 data buffer 读数据并把参数同步到 rollout，rollout 用 SGLang/router 生成数据和 reward 后写回 buffer。([GitHub][1])

---

### 1.2 同步训练主链路：从 prompt 到 Megatron loss

面试时可以按下面这条路径讲：

```text
train.py::train
  ├─ create_placement_groups / create_rollout_manager
  │    └─ 启动 SGLang engines + sglang-router
  ├─ create_training_models
  │    └─ 创建 actor / critic / reference，训练 backend 是 Megatron
  ├─ actor_model.update_weights
  │    └─ 初始把 Megatron actor 权重同步到 SGLang
  └─ for rollout_id:
       ├─ rollout_manager.generate
       │    └─ sglang_rollout.py::generate_rollout_async
       │         ├─ data_source.get_samples
       │         ├─ submit_generate_tasks
       │         ├─ generate_and_rm_group
       │         │    ├─ generate_and_rm
       │         │    │    ├─ generate: HTTP POST /generate 到 router
       │         │    │    └─ async_rm / batched_async_rm 计算 reward
       │         │    └─ group-level reward / dynamic filtering
       │         └─ 返回 List[Sample]
       ├─ convert_samples_to_train_data
       │    └─ Sample -> tokens / response_lengths / rewards / loss_masks / logprobs
       ├─ actor_model.async_train / critic_model.async_train
       │    └─ Megatron forward + loss + backward + optimizer
       └─ actor_model.update_weights
            └─ 同步新权重到 rollout engine
```

代码上，`train.py` 会创建 rollout manager、actor/critic model，并在训练开始前调用 `actor_model.update_weights`；同步训练 loop 中，每个 rollout 会先 `rollout_manager.generate`，再训练 actor/critic，然后在需要时 onload rollout 并 update weights。([GitHub][2])

在 rollout 侧，`sglang_rollout.py::generate()` 会向 router 的 `/generate` 发请求，payload 里包含 sampling params，并要求 `return_logprob=True`；返回后解析 `output_token_logprobs`，把新 token、response length、response 文本、loss mask、rollout logprob 等写入 `Sample`。([GitHub][3])

---

### 1.3 数据对象怎么变形

slime 的数据格式大概是：

```text
JSONL prompt / label / metadata
  -> Sample
      tokens
      response_length
      reward
      loss_mask
      rollout_log_probs
      rollout_routed_experts
      weight_versions
      metadata
  -> train_data dict
      tokens
      response_lengths
      rewards
      raw_reward
      loss_masks
      sample_indices
      rollout_log_probs
      metadata
  -> Megatron rollout_data tensors
      log_probs
      ref_log_probs
      rewards
      values
      response_lengths
      loss_masks
      total_lengths
```

`Sample` dataclass 中包含 `tokens`、`response_length`、`reward`、`loss_mask`、`weight_versions`、`rollout_log_probs`、`metadata` 等字段；自定义 `convert_samples_to_train_data` 也要求返回 `tokens`、`response_lengths`、`rewards`、`loss_masks` 等训练字段。([GitHub][4])

multi-turn / agent 场景里，常见做法是：**模型输出 token 的 loss mask 为 1，tool/env observation 的 loss mask 为 0**，这样训练 loss 只打在模型自己生成的 token 上，不会让模型去拟合环境返回内容。slime 文档的 agent 生成示例也是“model action → parse/execute tool → append observation → fill tokens/loss_masks/reward”。([ThudM][5])

---

### 1.4 Megatron 怎么结合

slime 默认 training backend 是 Megatron。Megatron 侧负责模型并行、forward/backward、optimizer step；SGLang 侧负责 rollout inference。要注意一个工程细节：Megatron 不能直接读 HF checkpoint，SGLang 可以用 HF checkpoint 加载 rollout model；训练前 slime 会把 Megatron 的参数同步到 SGLang，所以 HF checkpoint 不需要是最新权重。([ThudM][6])

Megatron 侧支持常见并行配置，例如 tensor parallel、sequence parallel、pipeline parallel、context parallel、expert parallel、expert tensor parallel，以及 recomputation。训练时 slime 还可以用 dynamic batch，把变长样本 pack 起来，并通过 `max-tokens-per-gpu` 控制每张卡上的 token 数，保证 loss 正确性。([ThudM][6])

---

### 1.5 loss 怎么计算：GRPO / PPO 的主公式

slime 训练侧会先得到当前 policy 的 logprob，必要时得到 reference logprob、critic value，然后算 advantage / returns。代码里 GRPO/GSPO 路径会把 reward 转成 per-token return；PPO 路径会把 KL penalty 加到 token reward 上，然后用 value 和 GAE 算 advantage/return。([GitHub][7])

核心 policy loss 可以这样讲：

```text
old_log_probs = rollout 阶段记录的 log π_old(a_t | s_t)
log_probs     = 当前 actor forward 得到的 log π_θ(a_t | s_t)

ppo_kl = old_log_probs - log_probs
ratio  = exp(-ppo_kl) = exp(log_probs - old_log_probs)

pg_loss_1 = - ratio * advantage
pg_loss_2 = - clamp(ratio, 1 - eps, 1 + eps_high) * advantage

policy_loss = max(pg_loss_1, pg_loss_2)
loss = policy_loss - entropy_coef * entropy + kl_loss_coef * ref_kl
```

这和 slime 代码一致：`compute_policy_loss` 里先算 `ratio = exp(-ppo_kl)`，再取 unclipped loss 和 clipped loss 的最大值；actor loss 里还会加 entropy 项和可选 reference KL loss。([GitHub][8])

slime 的 KL 支持多种近似，比如 `k1`、`k2`、`k3/low_var_kl`；`low_var_kl` 对应一种低方差、非负的 KL 估计。Quick Start 中 GRPO 示例也会配置 `--use-kl-loss`、`--kl-loss-coef`、`--kl-loss-type low_var_kl`、`--eps-clip` 等参数。([GitHub][8])

---

### 1.6 面试中的一句话总结

可以这样答：

> slime 的 data flow 是 prompt 先进入 data source / buffer，rollout manager 调 SGLang router 生成 response，同时拿到 rollout logprob、reward、loss mask 和 metadata；这些 `Sample` 会被转换成 Megatron 能吃的 tensor batch。Megatron actor forward 得到当前 logprob，和 rollout logprob 算 ratio，再结合 GRPO/PPO advantage 算 clipped policy loss；训练完后通过 `update_weights` 把新 actor 权重同步回 SGLang。整个链路本质是 SGLang 负责采样，Megatron 负责训练，中间用 Sample/train_data/rollout_data 做协议层。

---

## 2. GRPO：advantage 怎么算；不减 baseline 会怎样；PPO 为什么双边 clip；clip 后梯度怎样；不 clip 会怎样；CISPO 怎么做、为什么可能不好

### 2.1 GRPO advantage 怎么算

GRPO 的核心是：**不用 critic，而是对同一个 prompt 采样一组 response，用组内 reward 做相对比较**。DeepSeekMath 论文提出 GRPO 时就强调，它省掉 critic model，用 group scores 估计 baseline，从而降低 PPO 的资源消耗。([ar5iv][9])

最常见的 outcome-level GRPO advantage 是：

```text
对同一个 prompt 采样 G 个回答：
R_1, R_2, ..., R_G

mean_R = mean(R_1, ..., R_G)
std_R  = std(R_1, ..., R_G)

A_i = (R_i - mean_R) / (std_R + ε)
```

然后把 `A_i` 分配给该 response 的所有生成 token。文档中也明确说，outcome supervision 下会用组内 reward 减均值、除标准差，归一化后的 reward 作为该 response 所有 token 的 advantage；process supervision 下，可以把未来 step 的归一化 reward 累加成 token-level advantage。([ar5iv][9])

在 slime 里，GRPO 路径的 `get_grpo_returns` 会把每个样本的 scalar reward broadcast 到 response token 上，形成 per-token return / advantage。([GitHub][8])

---

### 2.2 如果不减 baseline 会怎么样

从 policy gradient 的数学上讲，只要 baseline 不依赖当前 action，减 baseline 不改变期望梯度，但会显著降低方差。GRPO 里组内 baseline 的意义更具体：它把“绝对 reward”变成“同一 prompt 下哪个 answer 更好”。DeepSeekMath/GRPO 的解释就是用 group average reward 替代 learned value baseline。([ar5iv][9])

不减 baseline 的问题主要有四个：

1. **方差变大**：同一 prompt 难度不同，reward 标尺不同，不减均值会让不同 prompt 的 reward scale 混在一起。
2. **不能表达相对优劣**：如果一个 prompt 下所有回答 reward 都是正的，不减 baseline 会把差回答也往上推；减 baseline 后，低于组均值的回答会得到负 advantage。
3. **容易被 reward bias 带偏**：某些 prompt 天然更容易拿高分，会对整体梯度贡献过大。
4. **训练不稳定**：advantage 均值不接近 0 时，更新方向更容易出现整体性偏移，KL 和 ratio 更容易变大。

面试中可以说：

> 不减 baseline 并不是一定 biased，但 variance 会明显增大，而且 GRPO 失去“组内相对比较”的核心优势。减 group mean 后，同一个 prompt 内好样本被增强、差样本被抑制；再除 std 是为了让不同 prompt 的 reward scale 更可比。

---

### 2.3 PPO 为什么是双边 clip

PPO 的 clipping 是为了限制新旧 policy ratio：

```text
r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)

L_clip = min(
  r_t(θ) * A_t,
  clip(r_t(θ), 1 - ε, 1 + ε) * A_t
)
```

如果写成 loss 最小化，就是 slime 代码里的：

```text
pg_loss = max(
  -r_t * A_t,
  -clip(r_t, 1 - ε, 1 + ε_high) * A_t
)
```

PPO 论文的核心思想是用 clipped surrogate objective 允许多轮优化，但避免 policy update 过大；slime 的实现也支持 `eps_clip` 和 `eps_clip_high`，因此可以做对称或非对称 clip。([arXiv][10])

“双边 clip”不是说所有情况下上下界都会让梯度为 0，而是说 ratio 的信任区间有上下两侧。真正激活哪一侧取决于 advantage 的符号：

| 情况                   |        ratio 变化 | clipping 的作用                  |
| -------------------- | --------------: | ----------------------------- |
| `A > 0`，动作是好动作       | `ratio > 1 + ε` | 不让好动作概率继续暴涨，policy loss 梯度被截断 |
| `A < 0`，动作是坏动作       | `ratio < 1 - ε` | 不让坏动作概率继续暴跌，policy loss 梯度被截断 |
| `A > 0` 但 `ratio` 变小 |            未必截断 | 因为这是往坏方向走，目标会把它拉回来            |
| `A < 0` 但 `ratio` 变大 |            未必截断 | 因为这是往坏方向走，目标会把它压下去            |

所以 PPO clip 的本质是：**只截断“让 surrogate objective 改善过多”的方向，而不是截断所有越界方向**。

---

### 2.4 clip 以后梯度怎么样

对 policy-gradient loss 来说：

```text
未 clip 分支：
L = -r_t * A_t
∂L / ∂logπθ = -A_t * r_t

clip 分支：
L = -clip(r_t, lower, upper) * A_t
如果 clamp 后是常数，则对 logπθ 的 policy-gradient 梯度为 0
```

也就是说，某些 token 的 **policy-gradient 项** 会被截断到 0，但整个 loss 里可能还有 entropy loss、reference KL loss、aux loss 等，所以“这个 token 完全没有梯度”要看具体 loss 组成。slime 的 actor loss 就是在 policy loss 之外还可以加 entropy 和 KL loss。([GitHub][7])

面试时可以这样讲：

> PPO clip 后，被选中 clipped branch 的 token 在 PG 项上不再推动 ratio 继续越界；但如果还有 KL regularization 或 entropy regularization，它仍可能通过这些项产生梯度。clip 的目的不是让模型不学，而是阻止某些样本把 policy 一步推太远。

---

### 2.5 如果 PPO 不 clip 会怎么样

不 clip 时，loss 变成普通 importance-ratio policy gradient：

```text
L = -r_t * A_t
```

问题是当 `r_t` 很大或很小时，少数 token 会主导梯度。尤其在 LLM RL 里，长序列、旧 rollout logprob、stale policy、reward 噪声都会放大 ratio mismatch。结果通常是：

1. KL spike，模型突然远离 old policy / reference policy；
2. entropy collapse，输出分布过早变尖；
3. reward hacking 或格式崩坏；
4. off-policy 样本权重爆炸；
5. 训练曲线出现大幅震荡。

如果面试官追问“有 KL penalty 还要不要 clip”，可以回答：

> KL penalty 是软约束，clip 是 sample-level 的硬截断。KL penalty 控制整体分布距离，但单个高 advantage token 仍可能产生很大梯度；clip 是防止 ratio 局部爆炸，两者通常是互补的。

---

### 2.6 CISPO 是怎么做的

CISPO 来自 MiniMax-M1，核心区别是：**PPO clip ratio 后会让一些 clipped token 的梯度被 suppress；CISPO 改成 clip importance weight 本身，并 detach 这个 clipped weight，让 clipped token 仍然通过 logprob 项贡献梯度**。SWIFT 文档把 CISPO 的 loss 写成类似下面的形式：([Swift 文档][11])

```text
w_t = exp(logπθ - logπold)
w_t_clipped = clamp(w_t, max=ε_high).detach()

L_CISPO = - w_t_clipped * A_t * logπθ
```

所以 CISPO 的梯度近似是：

```text
∂L / ∂logπθ = - w_t_clipped * A_t
```

和 PPO clipped branch 的区别是：PPO 中被截断的 token 在 PG 项上可能梯度为 0；CISPO 中它们仍有梯度，只是梯度权重被上界限制。

---

### 2.7 CISPO 为什么可能不好

这类问题不要回答成“CISPO 一定不好”。更准确的说法是：**CISPO 解决了 PPO 可能过度 suppress 关键 token 的问题，但也削弱了 PPO 的保守更新特性**。

主要风险有三类：

1. **trust-region 约束变弱**
   PPO 的 clipped objective 是 pessimistic bound：当 ratio 已经过大时，不再继续奖励这个方向。CISPO 仍然给高-ratio token 梯度，只是把权重 capped；如果 reward/advantage 有噪声，policy 仍可能被继续推远。

2. **token-level credit assignment 更敏感**
   在 outcome-level RL 里，一个 response 的所有 token 经常共享同一个 advantage。ASPO 论文也指出，在 outcome-supervised RL 中，所有 token 共用 response-level advantage 可能让 token-level advantage 不准确甚至误导。CISPO 保留 clipped token 梯度，会把这种 credit assignment 错误放大。([arXiv][12])

3. **对 stale/off-policy 样本更敏感**
   如果 rollout policy 和当前 policy 差距已经很大，CISPO 的 detached clipped weight 只能限制权重大小，但不能让样本重新变成 on-policy。高 staleness 下继续训练这些 token，可能带来 bias 和不稳定。

一句话总结：

> PPO clip 是“宁可不更新，也不要让 ratio 继续越界”；CISPO 是“ratio 过大也继续更新，但把权重截住”。它对长推理里关键低概率 token 可能有帮助，但在 reward noisy、advantage 粗糙、staleness 大时，可能比 PPO 更激进。

---

## 3. SGLang 如何看利用率；KV cache 在训练中的利用率；训练时间占比

### 3.1 SGLang 利用率怎么看

可以分三层看。

第一层是机器层：GPU utilization、显存、SM occupancy、HBM bandwidth、网络带宽。这一层通常用 `nvidia-smi`、DCGM、Nsight 或集群监控。

第二层是 SGLang engine 层。SGLang 支持 Prometheus metrics，启动时加 `--enable-metrics`，需要 MFU 相关指标时加 `--enable-mfu-metrics`，然后查 `/metrics`。官方 metrics 包括 prompt/generation token 计数、token usage、cache hit rate、running requests、queued requests、generation throughput、TTFT、time per output token、E2E latency 等。([SGLang 文档][13])

第三层是 router / slime RL 层。sglang-router 也有 Prometheus endpoint，默认端口是 `29000`；slime 训练脚本里会拿 router metrics 地址，并把 open metrics 接入 tracking。([PyPI][14])

面试时可以按下面这些指标解释：

| 指标                                | 说明                   | 怎么判断                                        |
| --------------------------------- | -------------------- | ------------------------------------------- |
| `gen_throughput`                  | decode token/s       | 低说明生成侧没吃满或 decode 被瓶颈卡住                     |
| `num_running_reqs`                | 正在跑的请求数              | 低且 GPU 低，通常是请求供给不足                          |
| `num_queue_reqs`                  | 排队请求数                | 高说明 engine 饱和或 prefill/decode 跟不上           |
| `num_used_tokens` / `token_usage` | token pool / KV 压力相关 | 高说明 KV cache 压力大，可能影响调度                     |
| `cache_hit_rate`                  | prefix cache 命中率     | 低说明 prefix 复用差或 routing 不稳定                 |
| TTFT                              | 首 token 延迟           | 高通常和 prefill、排队、prefix cache miss 相关        |
| TPOT                              | 每 output token 时间    | 高通常是 decode 性能、batching、memory bandwidth 问题 |

---

### 3.2 KV cache 在“训练中”的利用率怎么样

这里要先澄清：**Megatron 训练 forward/backward 通常不用 autoregressive KV cache；训练侧主要存的是 activation，或者用 recomputation 降低 activation memory。KV cache 主要在 rollout inference 侧使用。**

所以面试时不要说“训练的 KV cache 利用率很高”。更准确的是：

> RL 训练系统里，KV cache 利用率是 SGLang rollout 侧的指标，不是 Megatron backward 的指标。训练阶段消耗的是 activation memory、optimizer state、gradient memory；rollout 阶段消耗的是 KV cache、prefix cache 和 decode batch memory。

在 slime 里，SGLang 通过 router 做 rollout；文档还提醒 cache-aware routing 会影响 request 分布，如果强制 balance，可能降低 multi-turn 的 prefix cache hit。([ThudM][6])

不同任务下 KV / prefix cache 价值不同：

| 任务          | KV / prefix cache 特点                                                                     |
| ----------- | ---------------------------------------------------------------------------------------- |
| 单轮 math     | prompt 相似度有限，prefix cache 主要命中 system prompt / template；长 CoT 会让 KV 占用高，但 cache hit 不一定高 |
| 多轮 agent    | 同一个 session 反复追加历史，prefix cache 命中更重要；routing 不稳定会损失 cache                               |
| 多样本 GRPO    | 同 prompt 采样 G 个 response，如果 batching/routing 做得好，prefill prefix 可复用                      |
| fully async | 请求完成顺序不固定，cache hit 和调度策略强相关                                                             |

---

### 3.3 训练时间占比怎么说

不能给一个所有任务通用的固定比例。比较稳妥的回答是：

> 在长推理 RL 里，wall-clock 往往主要花在 rollout generation，而不是 Megatron backward。同步训练会被最长 response 或最慢 tool call 卡住；fully async 的目的就是减少这部分 straggler idle。

异步 RL 相关文档也强调，同步 loop 里 generation 是 autoregressive 的，常常主导 wall-clock；对于 reasoning，输出可能达到 8K–64K tokens；agentic 任务还会叠加 tool/sandbox/env 的长尾延迟。([Hugging Face][15])

面试里可以给“经验区间”，但要说明是 workload-dependent：

| 场景                     | rollout / reward 占比 | train forward/backward 占比 |
| ---------------------- | ------------------: | ------------------------: |
| 短回答 RLHF               |             30%–60% |                   40%–70% |
| 长 CoT math             |             50%–80% |                   15%–40% |
| agentic tool / sandbox |             70%–95% |                    5%–30% |

这不是理论值，是工程上常见的量级。真实回答应该补一句：

> 我会看每个 rollout step 的 generation time、reward time、train time、weight sync time，以及 SGLang 的 gen throughput / queue / TTFT 来判断瓶颈。

---

## 4. 确定性模式怎么开；什么是 batch invariance；是什么导致的；有没有 atomic add；atomic add 能解决吗；开了效率怎么样

### 4.1 slime / SGLang 确定性模式怎么开

slime 的 Reproducibility 文档给了一套配置：

```bash
pip uninstall flash_attn_3 -y

export NCCL_ALGO=Ring
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

训练参数里加：

```bash
--sglang-enable-deterministic-inference
--sglang-attention-backend flashinfer
--deterministic-mode
```

SGLang 侧也可以通过 `--enable-deterministic-inference` 开启 deterministic inference。SGLang/slime 的 reproducibility 文档还提到，为了 full training reproducibility，训练 backward 更推荐 FA2 而不是 FA3，并配合 Megatron deterministic mode、NCCL ring、CUBLAS workspace、`torch.use_deterministic_algorithms(True)` 等。([ThudM][16])

PyTorch 官方也提醒，确定性算法可能更慢，并且跨 release、跨平台的完全复现不能保证；`torch.use_deterministic_algorithms(True)` 会让部分算子走确定性实现，或在没有确定性实现时直接报错。([PyTorch Docs][17])

---

### 4.2 什么是 batch invariance

**Batch invariance** 指的是：同一个 request，在相同模型、相同输入、相同 seed 下，不应该因为和哪些其他 request 被 batch 在一起、batch size 多大、请求顺序怎样而改变输出或 logprob。

这比普通 deterministic 更严格：

| 概念              | 要求                                |
| --------------- | --------------------------------- |
| deterministic   | 同一批输入重复跑，结果一致                     |
| batch invariant | 单个输入在不同 batch composition 下，结果仍一致 |

SGLang 的 deterministic inference 工作就是在解决 dynamic batching、chunked prefill、radix cache、non-greedy sampling 下的 batch-invariant 问题。官方博客明确说，动态 batching / radix cache 会改变 kernel 的 reduction splitting，而浮点加法非结合律会导致结果变化；SGLang 用 batch-invariant kernels、fixed split-KV 和 per-request seed 来处理。([LMSYS Org][18])

---

### 4.3 是什么导致不确定性

常见来源包括：

1. **浮点加法非结合律**：`(a+b)+c` 和 `a+(b+c)` 在 FP16/BF16/FP32 下可能不同。
2. **reduction 分块不同**：batch size、sequence length、split-KV 策略变了，reduce 顺序就变了。
3. **dynamic batching**：同一个 request 每次和不同 request 拼在一起。
4. **prefix/radix cache**：cache hit/miss 改变 prefill 路径。
5. **sampling RNG**：非 greedy sampling 如果 seed 不按 request 固定，会受 batch order 影响。
6. **并行通信**：NCCL all-reduce、Megatron TP/PP/EP 下的通信顺序可能不同。
7. **某些 CUDA 算子**：scatter、index_add、atomic accumulation、MoE routing 统计等都可能引入非确定性。

SGLang 博客中特别点名了 dynamic batching / radix cache 改变 reduction splitting，以及浮点非结合律；PyTorch 文档也说明确定性模式会替换或禁止某些非确定性算子。([LMSYS Org][18])

---

### 4.4 atomic add 能解决吗

不能简单说 atomic add 能解决。更准确的是：

> atomic add 只能保证并发写入不会丢更新，但不能保证多个线程加法的顺序固定。由于浮点加法非结合，顺序不同结果就可能不同。因此 atomic add 经常是 nondeterminism 的来源，而不是 batch invariance 的万能解法。

如果某个 nondeterminism 来自 data race，那么 atomic add 可以修正“正确性问题”；但如果问题来自 reduction order 不固定，atomic add 仍然不保证 bitwise deterministic。要解决 batch invariance，通常需要 fixed-order reduction、固定 split 策略、batch-invariant kernel、per-request RNG，以及通信顺序控制。

面试时可以这样说：

> atomic add 可以避免 lost update，但不能保证 deterministic accumulation order。真正要做 batch invariance，需要固定 reduce 的数学路径，而不是只把写操作变成 atomic。

---

### 4.5 开确定性后效率怎么样

SGLang 博客给出的 deterministic inference 性能结论是：在 FlashInfer/FA3 等设置下，deterministic mode 的 slowdown 多数在 25%–45%，平均 slowdown 约 34.35%。换算成吞吐，大约是原来的 `1 / 1.3435 ≈ 74%`。([LMSYS Org][18])

面试时可以给这样的量级：

> 只开 inference deterministic，吞吐大概可能是原来的 70%–80%；如果训练侧也强制 deterministic，包括 Megatron、NCCL、CUBLAS、FA backward 等，端到端可能更低。实际要看模型、并行策略、sequence length 和是否用了 MoE。

---

## 5. fully async staleness 怎么看；训练时大概是多少

### 5.1 staleness 是什么

fully async 里，rollout 和 training 解耦。某个样本生成时用的是 policy version `v_old`，等它进入训练时，当前 actor 可能已经更新到 `v_cur`。这个差值就是 staleness：

```text
staleness = current_train_policy_version - behavior_policy_version
```

如果 trajectory 中途被新权重接管，那么还可能是 per-token staleness，因为不同 token 可能来自不同 policy version。AReaL 论文也强调，fully async 会让一个 batch 里包含不同模型版本生成的样本，所以需要控制 staleness 并修改 PPO 目标来处理这种混合版本数据。([arXiv][19])

---

### 5.2 staleness 怎么看

工程上不要只看一个数字，至少看四类：

1. **version lag**

   ```text
   sample.weight_version
   current_actor_version
   lag = current_actor_version - sample.weight_version
   ```

   看 mean / p95 / max。

2. **queue age**

   ```text
   now - sample.finish_generation_time
   now - sample.start_generation_time
   ```

   对 agentic 任务尤其重要，因为 tool call 可能很慢。

3. **policy mismatch**

   ```text
   | current_logprob - rollout_logprob |
   KL(old || current)
   ratio = exp(current_logprob - rollout_logprob)
   ```

   slime 训练代码里也会记录类似 `train_rollout_logprob_abs_diff` 的 mismatch 指标，并支持 TIS / OIS 相关处理。([GitHub][7])

4. **训练稳定性指标**

   ```text
   approx_kl
   clipfrac
   entropy
   reward mean/std
   length distribution
   ratio max/p95
   ```

面试时可以说：

> 我不会只说 staleness 是几，而是会同时看 version lag、样本排队时间、rollout logprob 和 train logprob 的差、PPO clipfrac/KL。如果 version lag 不大但 logprob mismatch 很大，仍然说明 off-policy 问题严重。

---

### 5.3 “你训练的时候大概是多少”怎么答

不要编一个绝对值。可以用下面这个模板，替换成自己的真实日志：

> 我们通常会把 staleness 控在 0–1 个 weight version 或 1 个 rollout step 内；为了吞吐开 fully async 时，会允许 bounded staleness，比如 p95 在 1–3 个 optimizer step，max 超过阈值就 drop、降权或用 importance correction。超过 4–8 个版本我会比较谨慎，因为 ratio/KL mismatch 往往会明显变大。

AReaL 的设计也是 bounded staleness：rollout workers 持续生成，trainer 拿到 batch 就更新；系统通过最大 staleness、throttling、优先训练老 trajectory、拒绝超过约束的新请求来控制 stale data。论文页面还指出 naive PPO 在 staleness 下会退化，而有界 staleness 配合 decoupled PPO 可以缓解，无界 staleness 效果更差。([arXiv][19])

---

## 6. fully async 的实现原理；AReaL 和 slime 有什么不同；fully async 什么时候同步；off-policy 样本怎么处理

### 6.1 slime 里的 async / fully async

slime 有几种层级：

1. **同步 loop**：generate → train → update weights。
2. **`train_async.py` pipeline async**：当前 batch 训练时，提前启动下一个 rollout，但 weight update 前会等待 in-flight generation，避免生成中途更新权重。
3. **fully async rollout example**：后台 `AsyncRolloutWorker` 用线程 + asyncio event loop 维护固定数量 in-flight generation task；完成的 group 进入 output queue，训练侧从 queue 里 drain 到 `rollout_batch_size` 就开始训练。([GitHub][20])

slime fully async 文档说，第一次调用会创建 process-wide `AsyncRolloutWorker`，保持最多 `sglang_server_concurrency` 个任务在飞；每个任务调用 `generate_and_rm_group`，完成的 group 进入输出队列；被 abort 的 group 会重新放回 data buffer，当前文档还说明 partial-rollout-style resume 尚未接好，abort 后会 requeue 并 start over。([ThudM][21])

README 里还提到 slime Relax 会用 TransferQueue 解耦 Actor、Rollout、ActorFwd、Reference、Advantage，并通过 DCS weight-sync 支持 full async 和 configurable staleness，这更接近专门的全异步 RL 系统设计。([GitHub][1])

---

### 6.2 AReaL 的 fully async 原理

AReaL 的核心是 streaming generation + asynchronous training：

```text
rollout workers:
  不断生成 trajectories，放入 rollout buffer

training workers:
  buffer 中够一个 batch 就开始更新

weight sync:
  actor 更新后，把新权重同步给 rollout workers

staleness control:
  控制最大 staleness
  throttle generation
  优先消费老 trajectory
  拒绝超过 staleness 约束的新 generation
```

AReaL 论文页面明确说，rollout workers continuously generate，training workers 在 batch collected 后更新；系统控制 staleness，并使用 staleness-enhanced PPO，报告最高 2.77× speedup。([arXiv][19])

AReaL 还做了算法层面的处理：它认为 async batch 里可能混合多个 policy version，因此 decoupled PPO 要把 behavior policy 和 proximal policy 分开；如果直接用 behavior policy 当 proximal，可能把最新 policy 拉回旧的、低质量的 policy。([arXiv][19])

---

### 6.3 AReaL 和 slime 的区别

可以这样对比：

| 维度            | slime                                                                                    | AReaL                                                                       |
| ------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 主要定位          | Megatron + SGLang 的 RL scaling framework，支持同步、pipeline async、fully async/Relax           | 专门围绕 fully asynchronous RL 设计的系统                                            |
| rollout       | SGLang/router 生成，Sample 进入 buffer                                                        | rollout workers streaming generation                                        |
| training      | Megatron actor/critic/ref                                                                | ReaLHF/Megatron-Core training backend                                       |
| async 方式      | `train_async.py` 可 overlap rollout 和 train；fully async worker 维护 in-flight task          | rollout 和 training 完全解耦，batch 满就训                                           |
| staleness     | 通过 weight sync interval、队列、mask/TIS/logprob mismatch 等控制；Relax 支持 configurable staleness | 显式 max staleness、throttle、prioritize old trajectories、reject stale requests |
| 算法处理          | GRPO/PPO、TIS、rollout logprob、partial rollout mask 等                                      | staleness-enhanced / decoupled PPO，处理 mixed policy versions                 |
| partial/abort | fully async example 文档说 abort 后 requeue/start over，partial resume 尚未完全接好                 | 论文系统围绕 continuous streaming 和 staleness 控制设计                                |

AReaL 的实现页面还说明它基于 ReaLHF，generation 用 SGLang，training backend 用 Megatron-Core。([arXiv][19])

---

### 6.4 fully async 什么时候同步权重

同步时机取决于系统策略，常见有三种：

1. **每个 optimizer step 后同步**

   * staleness 最小；
   * sync 开销大；
   * rollout 可能被频繁打断。

2. **每 N 个 optimizer step 后同步**

   * 吞吐更好；
   * staleness 更大；
   * 需要 off-policy correction 或 drop stale samples。

3. **rollout worker 空闲 / batch boundary 时同步**

   * 避免生成中途权重改变；
   * 对长 agentic trajectory 会产生老权重长时间占用的问题。

在 slime `train_async.py` 的 pipeline async 里，到了 `update_weights_interval` 时，会先等待当前 generation future 完成，再 update weights，避免在 generation 中途更新权重。([GitHub][20])

面试时可以说：

> 严格 on-policy 的做法是 rollout 完一批、train 一步、sync 一次；fully async 为了吞吐会让 rollout worker 持续生成，只在设定的 sync interval 或安全点同步。同步越频繁越 on-policy，但吞吐越低；同步越少 staleness 越高，需要 bounded staleness 和 correction。

---

### 6.5 off-policy 样本怎么处理

有五种常见策略：

1. **直接丢弃**

   * 如果 `staleness > threshold`，drop sample。
   * 最稳，但浪费 generation 成本。

2. **importance sampling / TIS**

   * 用 rollout logprob 和 current logprob 算 ratio。
   * token-level ratio 过大时截断或 mask。
   * slime 支持 `use_rollout_logprobs` 和 TIS 相关处理。([GitHub][7])

3. **只训练 on-policy token，mask off-policy token**

   * partial rollout 场景下，旧权重生成的 prefix token 不训练，新权重继续生成的 token 才训练。
   * slime 代码里 partial rollout 开 `mask_offpolicy_in_partial_rollout` 时，会把之前生成 token 的 `loss_mask` 设为 0。([GitHub][3])

4. **重新 prefill / restart**

   * 如果要严格 on-policy，就丢弃旧 KV cache，用新权重对已有 token 重新 prefill，或者整个 trajectory 重采样。

5. **降权训练**

   * 不完全丢弃，但按 staleness、ratio、KL mismatch 降低样本权重。

面试时推荐的回答是：

> off-policy 样本不能无脑用。短 staleness 可以用 rollout logprob 做 ratio correction；ratio 或 KL mismatch 太大的样本要 drop 或 mask。partial rollout 里旧 policy 生成的 token 可以保留作 context，但 loss_mask 置 0；严格 on-policy 则需要重新 prefill 或重新采样。

---

## 7. 为什么 agentic 任务适合 fully async + on-policy；math 需不需要；fully async 时要不要丢弃 KV cache 重新 prefill

### 7.1 为什么 agentic 任务需要 fully async

agentic 任务有几个特点：

1. **trajectory 长度差异大**：有的任务一步完成，有的要几十轮 tool call。
2. **环境延迟长尾明显**：sandbox、browser、代码执行、外部 API 可能秒级到分钟级。
3. **同步 barrier 浪费严重**：一批里最慢的几个 trajectory 会让所有训练 GPU 等着。
4. **reward 计算也可能慢**：需要 verifier、unit test、environment final state。

异步 RL 文档也指出，agentic RL 的 tool/sandbox/env 多轮 trajectory 会有高 variance latency，straggler 会让 GPU idle；disaggregated async rollout 可以让 inference/training 并行，减少等待。([Hugging Face][15])

所以 agentic 任务用 fully async 的核心原因是：

> 不是因为算法上必须 fully async，而是因为同步采样的工程效率太差。fully async 让快完成的 trajectory 先进入训练，慢任务继续跑，避免被长尾卡住。

---

### 7.2 为什么 agentic 任务又强调 on-policy

agentic 任务的 on-policy 要求比普通单轮 math 更敏感，因为 action 会改变环境状态：

```text
policy 选择 tool A
  -> 环境返回 observation A
  -> 下一步 state 改变
  -> 后续 action 分布也改变
```

如果 trajectory 是旧 policy 生成的，新 policy 可能根本不会走同样的 tool path。此时仅靠 token-level importance ratio 很难完全修正，因为 state distribution 已经变了。这就是 agentic 任务中 off-policy correction 更困难的地方。

面试时可以说：

> LLM 单轮 response 的 off-policy 主要是 token distribution mismatch；agentic 任务的 off-policy 还包括 environment state distribution mismatch。旧 policy 做了哪个 tool call、拿到什么 observation，会决定后面的状态。这个 mismatch 不是简单 token ratio 能完全补回来的，所以更希望 bounded staleness，甚至尽量 on-policy。

---

### 7.3 math 任务需要 fully async + on-policy 吗

math 任务也需要尽量 on-policy，但不一定必须 fully async。

对 math 来说：

1. **on-policy 仍然重要**：GRPO/PPO 都默认 rollout 来自当前或近当前 policy；stale 太大会导致 ratio/KL mismatch。
2. **fully async 的收益取决于长度分布**：如果都是短题，sync 或 pipeline async 足够；如果是长 CoT、8K–64K token reasoning、多 sample per prompt，rollout generation 会成为瓶颈，fully async 才更有价值。
3. **math 的环境长尾小于 agentic**：reward/verifier 通常比 tool/sandbox 稳定，因此 straggler 问题没 agentic 那么严重。

可以这样答：

> math 我会优先保证低 staleness/on-policy；是否 fully async 看 rollout 占比和 length variance。如果是短 CoT math，fully async 的复杂度可能不值得；如果是长 CoT、多样本 GRPO、generation 占 70% 以上，那 fully async 或至少 pipeline async 会有明显收益。

---

### 7.4 fully async 时需要丢弃 KV cache 重新 prefill 吗

严格说：**如果权重变了，旧 KV cache 就不再对应新模型的 hidden states，因此严格 on-policy / 严格一致性下应该丢弃旧 KV cache，并用新权重重新 prefill。**

但是工程上有不同取舍：

| 做法                             | 说明                              | 适用场景                              |
| ------------------------------ | ------------------------------- | --------------------------------- |
| 旧权重继续完成 trajectory             | KV 一致，trajectory 内部自洽，但样本 stale | bounded staleness，允许轻微 off-policy |
| 权重更新后 abort/requeue            | 丢弃旧生成，重新开始                      | 严格 on-policy，成本较高                 |
| 保留旧 token 作 context，重新 prefill | 用新权重重算 prefix KV，再继续生成          | 想减少浪费但保持新权重一致                     |
| 继续生成但 mask 旧 token             | 旧 token 不训练，只训练新 token          | partial rollout / mixed-policy 近似 |
| 不处理直接复用旧 KV                    | 不推荐                             | hidden states 和新权重不匹配             |

slime 的 partial rollout 文档说，未完成样本可以 cache 半生成结果，在下一个 rollout 继续；代码里也会在 partial rollout abort 时收集 partial samples，并记录 `start_rollout_id`。但 fully async rollout example 的当前文档说明，partial-rollout-style resume 尚未接好，被 abort 的 group 会重新放回 buffer 并 start over。([ThudM][5])

面试中建议这样答：

> 如果严格 on-policy，我会丢弃旧 KV 或重新 prefill，因为 KV 是由旧权重算出来的。工程上为了吞吐，可以让旧权重把 trajectory 跑完，然后把样本标记为 stale，用 ratio/TIS/drop 控制；partial rollout 可以保留旧 token 作为 context，但旧 token loss mask 置 0，新权重重新 prefill 后继续生成更干净。

---

# 最后：一版可直接背的总回答

> slime 里 SGLang 负责 rollout，Megatron 负责训练。prompt 从 data source 进入 rollout manager，SGLang router 返回 response token、rollout logprob、reward、loss mask 和 metadata，形成 Sample；Sample 再被 convert 成 Megatron 的 train_data。训练时 Megatron actor forward 得到当前 logprob，和 rollout logprob 算 ratio，再结合 GRPO 或 PPO advantage 计算 clipped policy loss；如果有 reference model，还会加 KL loss；训练完通过 update_weights 把 actor 权重同步回 SGLang。
>
> GRPO 的 advantage 是同 prompt 多个 response 的组内相对 reward，常见形式是减 group mean 再除 group std。不减 baseline 虽然不一定引入 bias，但 variance 会大，且失去组内相对比较。PPO 做双边 clip 是为了限制 ratio 的上下界，但真正被截断的是“让目标改善过多”的方向；clip 后该 token 的 PG 项梯度可能为 0，但 KL/entropy 仍可能有梯度。不 clip 会导致 ratio 爆炸、KL spike、entropy collapse。CISPO 改成 clip importance weight 并 detach，让 clipped token 仍有梯度；它可能帮助关键低概率 token，但也削弱 PPO 的保守更新，在 reward noisy、advantage 粗糙、staleness 大时更激进。
>
> SGLang 利用率要看 GPU 层、engine metrics 和 RL pipeline 时间。SGLang 可以通过 Prometheus 看 gen throughput、running/queued requests、token usage、cache hit rate、TTFT、TPOT 等。KV cache 是 rollout inference 侧的资源，Megatron 训练侧主要是 activation/optimizer/gradient memory。长 CoT 和 agentic 任务通常 rollout 占比很高，agentic 还会被 tool/env 长尾拖慢，所以 fully async 很有价值。
>
> 确定性模式需要同时控制 SGLang deterministic inference、Megatron deterministic mode、NCCL/CUBLAS/NVTE 等环境。batch invariance 指同一个 request 不因 batch composition 改变结果；根因是 dynamic batching、prefix cache、reduction split 和浮点非结合律。atomic add 不能保证顺序固定，所以不是万能解法。开 deterministic 后吞吐通常会下降，SGLang 报告过平均约 34% slowdown，也就是大约剩 70%–80% 吞吐量。
>
> fully async 的 staleness 是样本生成时 policy version 和训练时 policy version 的差。要看 version lag、queue age、train/rollout logprob mismatch、ratio、KL、clipfrac。slime 的 fully async worker 用后台 asyncio 保持 in-flight generation，训练侧从完成队列取 batch；AReaL 更系统地做 streaming rollout、trainer batch-ready update、显式最大 staleness 和 decoupled PPO。off-policy 样本可以 drop、importance correction、TIS、mask stale token、重新 prefill 或降权。agentic 任务适合 fully async 是因为工具调用和环境交互长尾严重，但又需要 on-policy 是因为旧 policy 的 tool path 会改变 state distribution，token ratio 很难完全修正。对于 math，是否 fully async 取决于 rollout 占比和长度方差；on-policy/低 staleness 仍然重要。

[1]: https://github.com/THUDM/slime "GitHub - THUDM/slime: slime is an LLM post-training framework for RL Scaling. · GitHub"
[2]: https://github.com/THUDM/slime/blob/main/train.py "https://github.com/THUDM/slime/blob/main/train.py"
[3]: https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py "https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py"
[4]: https://raw.githubusercontent.com/THUDM/slime/main/slime/utils/types.py "raw.githubusercontent.com"
[5]: https://thudm.github.io/slime/get_started/quick_start.html "Quick Start — slime"
[6]: https://thudm.github.io/slime/get_started/usage.html "Usage Guide — slime"
[7]: https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/loss.py "https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/loss.py"
[8]: https://github.com/THUDM/slime/blob/main/slime/utils/ppo_utils.py "https://github.com/THUDM/slime/blob/main/slime/utils/ppo_utils.py"
[9]: https://ar5iv.org/html/2402.03300v3 "https://ar5iv.org/html/2402.03300v3"
[10]: https://arxiv.org/abs/1707.06347 "https://arxiv.org/abs/1707.06347"
[11]: https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/CISPO.html "https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/CISPO.html"
[12]: https://arxiv.org/html/2510.06062v1 "https://arxiv.org/html/2510.06062v1"
[13]: https://docs.sglang.ai/references/production_metrics.html "https://docs.sglang.ai/references/production_metrics.html"
[14]: https://pypi.org/project/sglang-router/0.1.5/ "https://pypi.org/project/sglang-router/0.1.5/"
[15]: https://huggingface.co/blog/async-rl-training-landscape "Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries"
[16]: https://thudm.github.io/slime/advanced/reproducibility.html "https://thudm.github.io/slime/advanced/reproducibility.html"
[17]: https://docs.pytorch.org/docs/2.12/notes/randomness.html "https://docs.pytorch.org/docs/2.12/notes/randomness.html"
[18]: https://lmsys.org/blog/2025-09-22-sglang-deterministic/ "https://lmsys.org/blog/2025-09-22-sglang-deterministic/"
[19]: https://arxiv.org/html/2505.24298v2 "https://arxiv.org/html/2505.24298v2"
[20]: https://github.com/THUDM/slime/blob/main/train_async.py "https://github.com/THUDM/slime/blob/main/train_async.py"
[21]: https://thudm.github.io/slime/_examples_synced/fully_async/README.html "Fully-Async Rollout Example — slime"

