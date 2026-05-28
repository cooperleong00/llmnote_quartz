---
title: Your Efficient RL Framework Secretly Brings You Off-Policy RL Training | Notion
author:
  - "[[fengyao]]"
url: https://fengyao.notion.site/off-policy-rl
created: 2026-01-27
tags:
category: clippings/article
aliases:
  - TIS
updated: 2026-01-27T23:26
---
> [!abstract] 摘要
> Feng Yao*     Liyuan Liu*     Dinghuai Zhang     Chengyu Dong     Jingbo Shang     Jianfeng Gao

## Your Efficient RL Framework Secretly Brings You Off-Policy RL Training

[Feng Yao\*](https://yaof20.github.io/) [Liyuan Liu\*](https://liyuanlucasliu.github.io/) [Dinghuai Zhang](https://zdhnarsil.github.io/) [Chengyu Dong](https://www.chengyu-dong.me/) [Jingbo Shang](https://shangjingbo1226.github.io/) [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/?from=https://research.microsoft.com/en-us/um/people/jfgao/&type=exact)

\*: Equal Contributions (Work in Progress)

Last Updated on October 13, 2025 | First Published on August 5, 2025 | \[ : [Github](https://github.com/yaof20/verl/tree/flash-rl/recipe/flash_rl)\]

TL;DR

In modern RL training frameworks (e.g., VeRL), different implementations are used for rollout generation (e.g., vLLM) and model training (e.g., FSDP). Here, we show the implementation gap implicitly turns the on-policy RL to be off-policy, and discuss a simple yet effective importance sampling technique for handling such discrepancy.

![](https://fengyao.notion.site/image/attachment%3Aa61c1c32-1e6a-40c2-8e19-a49b84972248%3Adapo_32b.png?table=block&id=245721e3-f6c4-80ea-a892-fe53b080c05d&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure 1.Left: Token probability differences brought by the mismatch problem. Right: Performance comparison between normal RL training and training after fixing the mismatch problem. Experiments are conducted on Qwen2.5-32B dense model using 4 nodes of 8 H100 GPUs. \[[wandb log](https://wandb.ai/llychinalz/Flash-DAPO/?nw=w2j18d5w12)\]

💡

\[News\] Added [Policy Gradient, Sequence, and Token](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#28b721e3f6c480c3a756f8fb319e860d) — [Part I](https://fengyao.notion.site/28b721e3f6c480b88b5be1d89512ac3a?pvs=25) & [Part II](https://fengyao.notion.site/28b721e3f6c480f8a4b0e1f8301d90ac?pvs=25). 2025/10/13

\[News\] Blog updated with [Rollout-Training Mismatch Analysis](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#279721e3f6c48092bbe2fcfe0e9c6b33) section. 2025/09/24

\[News\] Blog updated with [TIS Analysis](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#257721e3f6c48013b361fbe1ea664623) section. 2025/08/22

\[News\] [slime](https://github.com/THUDM/slime) has integrated TIS. \[[GitHub](https://github.com/THUDM/slime/pull/179)\] \[News\] [VeRL](https://github.com/volcengine/verl/) has integrated TIS. \[[GitHub](https://github.com/volcengine/verl/pull/2953)\]\[[Example](https://github.com/volcengine/verl/blob/b8dc5377c6484f5873102e02f6a63829528ab8c9/recipe/dapo/run_dapo_qwen2.5_32b_tis.sh#L66)\]

\[News\] [OAT](https://github.com/sail-sg/oat) has [verified](https://x.com/zzlccc/status/1958915933743063070) and implemented TIS. \[[GitHub](https://github.com/sail-sg/oat/pull/62/files#diff-2500089b3585b36b9676cda9798d315cf00f2dfeeb0c4ad7c9844f23fbb1c652)\] \[[Tweet from OAT](https://x.com/zzlccc/status/1958915933743063070)\]

\[News\] [SkyRL](https://github.com/NovaSky-AI/SkyRL) has integrated TIS. \[[GitHub](https://github.com/NovaSky-AI/SkyRL/pull/145)\] \[[Tweet from SkyRL](https://x.com/sumanthrh/status/1958960763898622268)\]

\[News\] [REINFORCE++](https://medium.com/@janhu9527/reinforce-baseline-is-all-you-need-in-rlvr-f5406930aa85) verified TIS in Tool-Integrated-Reasoning (TIR) setting. \[[Blog](https://medium.com/@janhu9527/reinforce-baseline-is-all-you-need-in-rlvr-f5406930aa85)\]

\[News\] [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) has integrated TIS. \[\]

## The Mismatch Problem

For simplicity, we use the [[REINFORCE]] algorithm as an example, which supposedly updates the policy $\pi$ — an LLM parameterized by $\theta$ — via:

$$
\theta \gets \theta + \mu \cdot  \mathbb{E}_{\underbrace{a \sim{\pi}(\theta)}_{rollout}} [R(a)\cdot \underbrace{\nabla_\theta \log {\pi}(a, \theta)}_{\tiny{training}}].
$$

In practice, rollout generation is expensive and modern RL frameworks (e.g., [VeRL](https://github.com/volcengine/verl)) typically employ highly optimized inference engines (e.g., vLLM, SGLang) to boost throughput, while using a separate backend (e.g., FSDP, Megatron) for model training. Such hybrid design makes the updating:

$$
\theta \gets \theta + \mu \cdot  \mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} [R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)].
$$

Here, we use $\pi_{\rm sampler}$ to represent the model loaded with the inference engine (e.g., vLLM, SGLang) and $\pi_{\rm learner}$ to denote the same model instantiated with the training backend (e.g., FSDP, Megatron). Unless unspecified, our experiments use vLLM and FSDP as sampler and learner backends.

There is unexpected rollout-training mismatch observed.As shown in [Figure](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#245721e3f6c480eaa892fe53b080c05d) [1](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#245721e3f6c480eaa892fe53b080c05d), despite $\textcolor{blue}{\pi_{\text{fsdp}}}$ and $\textcolor{red}{\pi_{\text{vllm}}}$ sharing the same model parameters $\theta$, they can produce significantly different token probabilities. For certain tokens $a$, they even yield contradictory predictions, i.e., $\textcolor{red}{\pi_{\text{vllm}}}(a, \theta)\!=\!1$ and $\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta)\!=\!0$. This unexpected behavior implicitly breaks the on-policy assumption, secretly making the RL training become off-policy.

## How to Fix It?

### Mitigate the system-level mismatch

Does higher-precision vLLM help? We first hypothesized that vLLM is the root cause, and thus we patched vLLM to address two commonly suspected contributors to the mismatch problem.

Inaccessible true sampling probabilities: vLLM v1 engine [does not support](https://docs.vllm.ai/en/v0.10.0/usage/v1_guide.html?h=immediately#semantic-changes-to-logprobs) directly returning the adjusted probabilities used for sampling, introducing an additional gap.

→ Our patch forces vLLM to return the actual probabilities used for sampling [\[upstreamed\]](https://github.com/vllm-project/vllm/pull/22387).

Backend numerical differences: vLLM lm\_head’s precision [does not match](https://discuss.vllm.ai/t/numerical-difference-between-vllm-logprobs-and-huggingface-logprobs/151) that of HuggingFace transformers, which is also denoted in the MiniMax-M1 [technical report](https://arxiv.org/pdf/2506.13585#page=8).

→ Our patch provides the option to force vLLM casting lm\_head to fp32.

However, as shown in [Figure](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#245721e3f6c4802fa3ffce687eef2d0a) [1](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#245721e3f6c4802fa3ffce687eef2d0a), the mismatch problem still exists after applying both patches.

### Embrace the mismatch — Apply algorithm-level fix

Instead of mitigating the distribution mismatch at the system level, we propose to adapt the model update such that it’s aware of this mismatch. A simple way is via importance-sampling correction. Specifically, we handle the mismatch between $\textcolor{blue}{\pi_{\text{learner}}}$ and $\textcolor{red}{\pi_{\text{sampler}}}$ by adding the importance ratio to the model update, i.e., changing the current gradient computation from

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} [R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)],
$$

to

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \Bigl[\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)} \cdot R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)\Bigr].
$$

While there has been extensive study on how to design a stable and effective importance sampling, in practice we find it usually sufficient to use a classical technique, [truncated importance sampling](https://ionides.github.io/pubs/ionides08-jcgs.pdf):

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \Bigl[\underbrace{\min\Bigl(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)}, C\Bigr)}_{\text{truncated importance ratio}} \cdot R(a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)\Bigr],
$$

where C is a hyper parameter.

#### Extension to Other Algorithms

It is straightforward to extend the above analyses to other algorithms, as one can switch the exact form for gradient computation from REINFORCE $R(a) \cdot\nabla \log {\pi}(a, \theta)$ to any form. Here, we provide similar analyses to the commonly used PPO algorithm as an additional example.

The policy gradient of PPO $\nabla_\theta L^{\mathrm{CLIP}}(\theta)$ is defined as:

$$
\small{ \mathbb{E}_{a\sim\pi_{\theta_{\mathrm{old}}}}
\Bigl[
\nabla_\theta \min\Bigl(
\frac{\pi_\theta(a)}{\pi_{\theta_{\mathrm{old}}}(a)}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\pi_\theta(a)}{\pi_{\theta_{\mathrm{old}}}(a)},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]}.
$$

To improve throughput, hybrid RL systems adopt vLLM engine for rollout generation —sampling tokens $a$ from $\pi_{\theta_{\text{old}}}$, while using FSDP backend both to sample from $\pi_\theta$ and to [recompute](https://github.com/volcengine/verl/blob/3e2bceb1afcaa77ebc40106a64f7b440509b67e1/verl/workers/fsdp_workers.py#L782) the token probabilities for $\pi_{\theta_{\mathrm{old}}}$ for gradient computation:

$$
\small{
\mathbb{E}_{a\sim\textcolor{red}{\pi_{\text{sampler}}}(\theta_{\mathrm{old}})}
\Bigl[
\nabla_\theta \min\Bigl(
\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\mathrm{old}})}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\mathrm{old}})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]
}.
$$

Similar to the analysis above, the gap between $\textcolor{blue}{\pi_{\text{learner}}}$ and $\textcolor{red}{\pi_{\text{sampler}}}$ shows up again, and we fix it with truncated importance sampling:

$\small{\mathbb{E}_{a\sim\textcolor{red}{\pi_{\mathrm{sampler}}}(\theta_{\mathrm{old}})}\Bigl[\underbrace{\min\Bigl(  \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})},  C\Bigr)}_{\text{truncated importance ratio}}\cdot\nabla_{\theta}\,\min\Bigl(  \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}\,\hat{A},  \mathrm{clip}\Bigl(    \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})},    1-\epsilon,\;1+\epsilon  \Bigr)\,\hat{A}\Bigr)\Bigr]}$ where C is a hyper-parameter.

Additional Discussion on PG, Sequence, and Token

Our discussion above does not touch the specific formulation of the state and the action. We discuss policy gradient at both token and sequence levels, how they connect to each other, and the impact of learner-sampler mismatch:[💬](https://fengyao.notion.site/pg-seq-token-part1-basics?pvs=25)

[Policy Gradient, Sequence, and Token— Part I: Basic Concepts](https://fengyao.notion.site/pg-seq-token-part1-basics?pvs=25)[🔍](https://fengyao.notion.site/pg-seq-token-part2-mismatch?pvs=25)

[Policy Gradient, Sequence, and Token— Part II: Learner-Sampler Mismatch](https://fengyao.notion.site/pg-seq-token-part2-mismatch?pvs=25)

#### Connection to Classical Wisdom

[[Importance Sampling]]

When direct Monte Carlo estimation of the expected value under a target distribution is difficult, importance sampling allows us to sample from an alternative distribution instead. In our case, the target distribution is $\textcolor{blue}{\pi_{\text{learner}}}$, but it is extremely slow to sample from. Using a separate backend (e.g., vLLM) for rollout generation means that we are sampling from $\textcolor{red}{\pi_{\text{sampler}}}$ instead.The discrepancy is then corrected by weighting each sample with an importance ratio:

$$
\mathbb{E}_{a \sim \textcolor{blue}{\pi_{\text{learner}}}(\theta)} [R(a)] 
= \mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \left[ 
\underbrace{\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)}}_{\tiny\text{importance ratio}} \cdot R(a) 
\right].
$$

Decoupled PPO

[Decoupled PPO](https://arxiv.org/pdf/2110.00641) is a special case of using importance sampling to bridge the gap between the rollout generation and gradient computation, which has been adopted in async-RL frameworks like [AReaL](https://arxiv.org/pdf/2505.24298#page=6). It is worth mentioning that AReaL didn’t implement the truncated importance ratio as we discussed here. Instead, AReaL [will drop the training sample entirely](https://github.com/inclusionAI/AReaL/blob/main/realhf/impl/model/utils/ppo_functional.py#L127), if the importance ratio exceeds a pre-defined threshold.

## Experiments

We further conducted empirical analyses to elaborate on the impact of the distribution gap and the effectiveness of the proposed Truncated Importance Sampling (TIS) fix.

### Does the gap matter a lot?

Due to the resource constraints, we only finished the first 250 steps of the training, yet the gap-aware fix TIS has already helped boost the performance significantly. As the only difference between these two runs is the introduced term, i.e., $\min(\frac{\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta)}{\textcolor{red}{\pi_{\text{vllm}}}(a, \theta)}, C)$, the improvement showcased the potential impact of the distribution gap.

### How well can TIS fix it?

![](https://fengyao.notion.site/image/attachment%3A766b9627-d7c4-4f0d-ba10-6eda045390a1%3Agsm8k_int8.png?table=block&id=246721e3-f6c4-803f-b9f1-c1e707b64b02&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure 2.Left: Token-level probability differences. Right: Performance comparison for normal RL training on GSM8K and RL training with INT8 quantized rollouts. Experiments are conducted on Qwen2.5-0.5B dense model using one node of 4 A6000 GPU.

We designed a controlled experiment to measure how well TIS fixes the issue. We conduct RL training following [GSM8K example in the verl tutorial](https://verl.readthedocs.io/en/latest/start/quickstart.html) and use two different settings:

Normal RL training: the maximum token probability difference is considerably smaller (~0.4) than the previous setting (1.0 on DAPO on Qwen-2.5-32B dense),

: the maximum token probability difference is considerably larger (1.0) than normal RL training.

We conduct regular PPO training in setting 1, which is “almost” on-policy, and both regular PPO training and PPO training with truncated importance sampling in setting 2, whose generation rollout and gradient computation have a larger gap.

As visualized in [Figure](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#246721e3f6c48059b664cddaabfa2448) [2](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#246721e3f6c48059b664cddaabfa2448), performing PPO in setting 2 leads to significant performance degradation, comparing to the PPO in setting 1. At the same time, applying truncated importance sampling manages to mitigate the gap greatly, effectively allowing the setting 2 run achieves a similar performance with setting 1.

More analysis are provided in [Section Analysis](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#257721e3f6c48013b361fbe1ea664623) below.

### Does TIS always help?

![](https://fengyao.notion.site/image/attachment%3A40ff1a9f-0eda-4aab-924e-fc99bae3d95c%3Adapo_1.5B.png?table=block&id=246721e3-f6c4-8040-9b75-ecd01f2e6681&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure 3.Left: Token probability differences brought by the mismatch problem. Right: Performance comparison between normal RL training and after fixing the mismatch problem. The experiment is conducted on the DeepSeek-R1-Distill-Qwen-1.5B model using 4 nodes of 8 H100 GPUs.In this case, the mismatch is not huge because we use a standard bfloat16 rollout in both runs and the model is relatively small. \[[wandb log](https://wandb.ai/llychinalz/Flash-DAPO/?nw=xv3gvup0lw8)\]

We also observed that, in cases where the probability difference is relatively small, introducing the additional Truncated Importance Sampling term cannot bring performance gain. Meanwhile, it is worth mentioning that, the importance sampling ratio term would have a value of 1.0, in the strict on-policy RL setting.

## TIS Analysis

### Analysis about different TIS-Variants

We also summarize two alternative solutions for mitigating the distribution gap.

PPO Importance Sampling (PPO-IS)

Vanilla Importance Sampling (vanilla-IS)

To assess the effectiveness of TIS and understand the impact of its design choices, we conducted experiments comparing TIS with the two variants above. TIS outperforms both variants consistently, especially in cases where the gap is large (e.g., FP8/INT8).

![](https://fengyao.notion.site/image/attachment%3Ac27fcce8-f231-44d1-b7fc-15f88292e419%3Agsm8k_int8_fp8_TIS_variant_only.png?table=block&id=257721e3-f6c4-80c6-8d12-d6f55866ca3f&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure 4. We ablate different rollout-training mismatch mitigation strategies on Qwen2.5-0.5B with GSM8k. Note PPO-IS and Vanilla-IS achieves near 0 accuracy for INT8 rollouts thus being highly overlapped. We also plot the KL divergence between vLLM sampled distribution and the FSDP distribution on the right. \[[int8 wandb](https://wandb.ai/llychinalz/Flash-GSM8K?nw=2yfyyqo0fm)\]\[[fp8 wandb](https://wandb.ai/llychinalz/Flash-GSM8K?nw=cih3nmuhn8p)\]\[[bf16wandb](https://wandb.ai/llychinalz/Flash-GSM8K?nw=8zvq7u05spp)\]

💡

Why the two variants (PPO-IS and vanilla-IS) here gives unstable training?

Vanilla-IS v.s. TIS

Regarding vanilla-IS, the instability is mainly from the cases when the rollout $a\sim\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})$ is sampled with low probability and thus the importance ratio is large, amplifying gradient variance by $(\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})})^2$. Therefore, we use a clamp operation in our truncated importance sampling (TIS) for training stabilization. For example, when the ratio $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})}$ reaches 16 for one token, the gradient noise for that token would be amplified by 256 times via Vanilla-IS, 4 times via TIS-2, or 64 times via TIS-8.

PPO-IS v.s. TIS

Since the release of our blog, a lot of people have asked why we do not directly incorporate the importance sampling into PPO (i.e., the PPO-IS variant above). Frankly speaking, we start by changing the ppo clip directly as in PPO-IS, but it doesn't perform well in our experiment setting.

As to the underlying rationale, by doing the PPO-IS, the gradient is actually still biased from the on-policy version of PPO. In other words, although it may still optimize towards the unbiased objective, it may be less effective compared to PPO.

Additionally, we remark that the PPO trust region technique is proposed to constrain the probability ratio between rollout $\theta_{\rm old}$ and current model $\theta$ to be close to 1 to approximate on-policy REINFORCE gradient. However in PPO-IS, even when $\theta=\theta_{\rm old}$, the probability ratio $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\rm old})}$ is already not equal to 1 due to the mismatch — this makes the clipping happen with high possibility and the training much less informative. Furthermore, in our TIS method, we separately clip $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})}$ and $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}$ and thus much more mild; notice $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}$ equals to 1 when $\theta=\theta_{\rm old}$ which is suitable for the trust region constraint.

### From Ill-conditioned to Benign

Beyond rollout acceleration, rollout quantization serves as an effective testbed for examining the impact of distribution gaps between rollout generation and gradient computation. We demonstrate that RL training with quantized rollouts exhibits characteristic instabilities commonly observed in other scenarios when this gap is not addressed. Additionally, introducing the TIS term making RL training stable and benign.

#### Entropy Collapse and Abnormal Response Length

A number of have shown that RL training in LLM will lead to entropy collapse — the categorical distribution on the token level becomes close to one-hot distribution, restricting RL training from exploration effectively.

Our INT8 rollout experiments revealed severe entropy collapse. [Figure](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#257721e3f6c4804ba5d6dde838e718e8) [5](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#257721e3f6c4804ba5d6dde838e718e8) shows entropy dropping below 0.2 and continuing to decrease throughout training. We also observed abnormally long response generation—another failure mode in RL training. Incorporating the TIS term reverses this trend, enabling the model to be trained in a stable and benign manner.

![](https://fengyao.notion.site/image/attachment%3A36b8262b-a709-40c2-b93e-b2b23d80e4a0%3Adapo_32b_int8_casestudy.png?table=block&id=257721e3-f6c4-807e-9649-fafdc903d95d&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure 5. DAPO-Qwen2.5-32B INT8 training exhibits various instability and is successfully stabilized via introducing Truncated-Importance-Sampling

In contrast, BF16 rollout experiments showed no severe entropy collapse. Nevertheless, the TIS term still increased entropy values. With a smaller distribution gap compared to INT8 rollouts, response lengths remained within reasonable bounds.

![](https://fengyao.notion.site/image/attachment%3A6ba377e7-0ebc-4955-ac9f-573f561bbb5b%3Adapo_32b_bf16_casestudy.png?table=block&id=257721e3-f6c4-805d-b703-ce95494d0c4c&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure 6. DAPO-Qwen2.5-32B BF16 training exhibits various instability and can be successfully stabilized via introduced Truncated-Importance-Sampling.

#### On the Impact of Distribution Gap: A Case Study on KL Estimation

One unbiased kl estimator for $\rm{KL}(\textcolor{blue}{\pi_{\rm old}^{\rm fsdp}} \Vert \textcolor{blue}{\pi^{\rm fsdp}})$ is the $k_1$ [estimator](http://joschu.net/blog/kl-approx.html): $\log \textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}(a) - \log \textcolor{blue}{\pi^{\rm fsdp}} (a)$ where $a\sim \textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}(a)$. However, modern RL training framework generate rollouts from $\textcolor{red}{\pi_{\rm old}^{\rm vllm}}$ instead of $\textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}$, introducing bias to the kl estimation similar to the gradient estimation bias as discussed earlier.

Accordingly, we can use KL estimation as a case study to explore the impact of the mismatch between $\textcolor{red}{\pi_{\rm old}^{\rm vllm}}$ and $\textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}$. Without any bias, the KL divergence is a non-negative by definition. However, the substantial distribution mismatch in INT8 rollouts causes the biased $k_1$ estimator to frequently yield negative values, as shown in [Figure](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#257721e3f6c4804ba5d6dde838e718e8) [5](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#257721e3f6c4804ba5d6dde838e718e8). These negative KL estimates signal ill-conditioned training dynamics.

#### Biased Reward in Training Log

An interesting phenomenon of integrating TIS is that it may lead to worse reward logging while bringing better downstream performance. This is because the gap between $\textcolor{red}{\pi_{\text{sampler}}}$ and $\textcolor{blue}{\pi_{\text{learner}}}$ introduce biases in not only gradient estimation but also the reward estimation in logging. Particularly, the logged rewards is from the rollout policy, i.e., $E_{\textcolor{red}{\pi_{\text{sampler}}}}[{\rm R}]$ instead of $E_{\textcolor{blue}{\pi_{\text{learner}}}}[{\rm R}]$. Specifically, as shown in [Figure](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#257721e3f6c48090beccfb75c5f5d3ad) [6](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#257721e3f6c48090beccfb75c5f5d3ad) (right two subplots), the logged reward metric shows that BF16-Rollout is better than BF16-Rollout w. TIS. However, if we look at the downstream performance of AIME accuracy, BF16-Rollout w. TIS significantly outperforms the vanilla BF16-Rollout.

#### Intuitions of TIS’s Working Mechanism

While the exact mechanism of TIS remains an open question, we provide high-level intuitions on how TIS mitigating the distribution gap.

Particularly, neglecting the bias on rollouts having $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\rm old})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\rm old})} < 1$ could lead to entropy collapse through the following mechanism: For rollouts with negative advantages, policy gradients tend to reduce $\textcolor{blue}{\pi_{\mathrm{learner}}}$. When a large distribution gap exists after parameter updates, the reduction in $\textcolor{blue}{\pi_{\mathrm{learner}}}$ may not be reflected in $\textcolor{red}{\pi_{\mathrm{sampler}}}$. Consequently, policy gradient continues pointing towards further reductions in $\textcolor{blue}{\pi_{\mathrm{learner}}}$. Intuitively, such penalization may force the model to overly commit to a output distribution with a small entropy.

At the same time, TIS sticks to the un-truncated importance ratio for $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\rm old})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\rm old})} < 1$, thereby eliminating bias for this subset of rollouts and breaks this mechanism.

## Rollout-Training Mismatch Analysis

We conduct a series of controlled experiments to identify the factors that introduce or amplify the discrepancy between rollout generation and gradient computation. Specifically, we find that differences in parallelism strategies and long response length contribute to the mismatch, and the choice of the sampler backends alone have only marginal impacts.

### Analysis Setup

Model & Data. We experiment with two representative models — [DAPO-32B](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B) and [Polaris-7B](https://huggingface.co/POLARIS-Project/Polaris-7B-Preview) trained with the [DAPO](https://arxiv.org/pdf/2503.14476) and [POLARIS](https://fengyao.notion.site/1dfa954ff7c38094923ec7772bf447a1?pvs=25) RL recipes. For evaluation, we use the first 512 prompts from [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset to evaluate the discrepancy between sampler and learner outputs.

Metric. We measure response-level mismatches using two metrics:

Max Mismatch per response: $\max_{a\,\in\, \text{response}} |p_{\tiny\text{sampler}}(a) - p_{\tiny\text{learner}}(a)|$

Mean Mismatch per response: $\frac{1}{|\text{response}|}\sum _{a\,\in\, \text{response}} |p_{\tiny\text{sampler}}(a) - p_{\tiny\text{learner}}(a)|$

### Larger Parallelism Difference, Larger Max Gap

We observe that parallelism discrepancies between the sampler and learner contribute nontrivially to the Max Mismatch metric.

Simplest Setting

Using the DAPO-32B model, we begin with the simplest configuration: the sampler runs on vLLM with TP1 and the learner uses FSDP with SP1. Since the sampler and learner have the same parallelism setting, we refer to this as Same Parallelism and its distribution gap attributes to factors other than parallelism difference.

Adding Tensor Parallelism

Adding Sequence Parallelism

To study the impact of [Ulysses Sequence Parallel](https://arxiv.org/abs/2309.14509) difference, we then change the learner from SP1 to SP8 (Different TP & SP). As shown in Figure 7 Middle, additional SP difference increases the number of high maximum mismatch from two to double digits.

![](https://fengyao.notion.site/image/attachment%3A57c04038-75d8-4e33-ace1-91c2c36d5399%3Avllm_fsdp_parallelism.png?table=block&id=279721e3-f6c4-8005-be3f-e86dfa0df133&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure 7. Max Mismatch of the same set of prompts under different parallelisms. [\[how-to-read\]](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#279721e3f6c48052aa7fdcbe7950c95f)

Disentangling Parallelism and Sharding

As shown in Figure 8 Left below, for a similar distributed world size (e.g., 8 devices), using Tensor Parallelism (TP8) in the learner leads to a smaller mismatch with the TP2 sampler than using Sequence Parallelism (SP8). We hypothesize this is because the implementation differences between a TP8 learner and a TP2 sampler are less pronounced than those between an SP8 learner and a TP2 sampler. This reinforces the finding that minimizing parallelism differences between the sampler and learner consistently reduces the gap.

![](https://fengyao.notion.site/image/attachment%3A82d124b2-e301-497d-8e8d-5c8b08c12a72%3Avllm_megatron_parallelism.png?table=block&id=279721e3-f6c4-806e-9215-f3811bd6544e&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure 8. Max Mismatch of the same set of prompts under various parallelisms. [\[how-to-read\]](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#279721e3f6c48052aa7fdcbe7950c95f)

Mean Mismatch and KL

Desipte we observe consistent patterns for Max Mismatch, it is worth mentioning that we didn’t observe any significant difference on the Mean Mismatch/KL divergence of those configurations.

### Longer Response, Larger Max Gap

Our experiments consistently show that longer generation sequences lead to a larger maximum mismatch, while the mean mismatch is much less affected. We ablate the effect of sequence length using both DAPO-32B and Polaris-7B models.

![](https://fengyao.notion.site/image/attachment%3Ac030a2b2-299e-4449-8438-d01a6145dc8b%3Amax_mean_20_4.png?table=block&id=279721e3-f6c4-80ce-9b09-ee4a9355a7b8&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure 9. Left: Max Mismatch of different response length. Right: Mean Mismatch of different response lengths. Colors indicate results for DAPO-32B and Polaris-7B, respectively. [\[how-to-read\]](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#279721e3f6c48052aa7fdcbe7950c95f)

![](https://fengyao.notion.site/image/attachment%3A93f062c7-3a35-4727-8974-4cdcbf166d2c%3Amax_4n5_4first.png?table=block&id=279721e3-f6c4-80a4-8a1e-f51cdfa2b47b&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Interestingly, we observe the mismatch accumulates as generation progresses: the maximum mismatch within just the first 4K tokens of a 20K-token response often exceeds the maximum mismatch of an independent 4K-token response. This indicates that the internal states of sampler and learner diverge increasingly over long generation contexts.

### Altering Sampler Alone, Gap Still There

Finally, we investigate whether the choice of the sampler backend itself is a major contributor to the mismatch. We compared three configurations for the sampler: 1) vLLM; 2) SGLang; and 3) [SGLang](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) with [deterministic kernel](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) enabled.

The results show that the sampler backend alone does not have a decisive impact. With the DAPO-32B model, SGLang yields a smaller mean mismatch, whereas with the Polaris-7B model, vLLM performs better (i.e., vLLM has a smaller mean mismatch). Thus, no single sampler backend consistently dominates across different settings.

![](https://fengyao.notion.site/image/attachment%3A1f820845-a43e-4ed6-8061-98af976d6b8f%3Asglang_vllm_dapo.png?table=block&id=279721e3-f6c4-80f0-ab62-c1cf298e8c71&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2) ![](https://fengyao.notion.site/image/attachment%3A65d1350d-89b3-44f7-8305-bd6e0c2d8672%3Asglang_vllm_pol.png?table=block&id=279721e3-f6c4-8030-95d6-f5ad53d497b9&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure 11.Left: Max Mismatch of different sampler backends. Right: Mean Mismatch of different sampler backends. Colors indicate results for DAPO-32B and Polaris-7B, respectively. [\[how-to-read\]](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#279721e3f6c48052aa7fdcbe7950c95f)

Notably, enabling deterministic sampling in SGLang, without aligning the training configuration, does not noticeably reduce the gap. This suggests the mismatch originates primarily from deeper implementation differences (e.g., parallelism or numerical precision), rather than from stochastic sampling alone.

### What’s More

There are other dimensions may influence the rollout–training mismatch, including model type (e.g., dense vs. MoE, base vs. post-trained), prompt data characteristics (e.g., difficulty, domain), GPU hardware, and the choice of training backend. For example, we find it relatively consistent that dense and MoE models of comparable scales (32B and 30B) exhibit different levels of mismatch, and base models have smaller rollout–training mismatch than their post-trained counterparts. We are continuously working towards a deeper understanding and better utilization of rollout-training mismatch for practical LLM post-training. Stay tuned!

## Discussion

We discuss the potential impact of our fix, Truncated Importance Sampling (TIS), on the RL of MoE architecture specifically. We also highlight TIS’s connection to recent works (e.g., [GSPO](https://arxiv.org/pdf/2507.18071), [GMPO](https://arxiv.org/pdf/2507.20673)) that aim to improve the importance-sampling ratio in policy update.

#### The gap can be amplified in

While our current experiments and analyses focus on dense models, we believe this distribution gap also exists in MoE RL and can be even more severe. There are two main reasons:

Dynamic Routing: Different from dense models, MoE utilize a router to dynamically activate specific experts for This routing mechanism is inherently precision-sensitive; even slight numerical discrepancies can result in substantially different expert activations.

Specially Optimized Kernels: MoE models are usually large-scale and modern inference engines (e.g., vLLM) have unique optimization for MoE models compared to dense models, which makes the backend numerical inconsistencies even larger.

Together, these characteristics can significantly amplify the distribution mismatch, making solutions like TIS particularly valuable in MoE RL.

#### TIS is orthogonal and compatible with existing GxPOs

Recent works improve the stability of policy update by renovating the calculation of the importance-sampling ratio. For example, [[GSPO]] calculates the ratio at the sequence level instead of the token level, while [GMPO](https://arxiv.org/pdf/2507.20673) computes the geometric mean instead of the arithmetic mean.

Orthogonal to these works, our TIS fix addresses the distribution mismatch problem rooted in the system level, which is brought by the different compute kernels used in rollout generation and model training. Such a problem widely exists in RL training frameworks that adopt a hybrid computation design. Thus, our fix can be applied irrespective of the specific RL algorithms used.

