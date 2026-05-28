---
title: "Learning from Mixed Rollouts: Logit Fusion as a Bridge Between Imitation and Exploration | Notion"
author:
  - "[[juzhengz]]"
url: https://juzhengz.notion.site/logit-fusion
created: 2026-03-05
tags:
category: clippings/article
updated: 2026-03-05T15:23
---
> [!abstract] 摘要
> Logit fusion for learning from mixed teacher and student rollouts, bridging imitation and exploration.


## Notes

![](https://juzhengz.notion.site/images/page-cover/nasa_orion_nebula.jpg)

🔬

## Learning from Mixed Rollouts: Logit Fusion as a Bridge Between Imitation and Exploration

🧠

TL;DR: We propose a hybrid post-training method that blends teacher and student logits at every decoding step, producing rollouts that are neither fully on-policy nor fully off-policy. Combined with importance-sampling correction and a decaying mixing coefficient, this logit fusion strategy lets smaller models absorb expert reasoning while retaining their own exploratory capabilities.

👥

[Juzheng Zhang](https://juzhengz.github.io/) $^1$ · [Abhimanyu Hans](https://ahans30.github.io/) $^1$ · [John Kirchenbauer](https://www.cs.umd.edu/people/jkirchen) $^1$ [Micah Goldblum](https://goldblum.github.io/) $^2$ · [Ashwinee Panda](https://kiddyboots216.github.io/) $^1$ · [Tom Goldstein](https://www.cs.umd.edu/~tomg/) $^1$

$^1$ University of Maryland · $^2$ Columbia University

: [Github link](https://github.com/juzhengz/logit-fusion) | Feb 25, 2026

![](https://juzhengz.notion.site/image/attachment%3Abe2335c5-f71d-40fe-a57b-c0aacb8e030c%3AScreenshot_2026-02-20_at_6.03.56_AM.png?table=block&id=30d4b5c2-cba9-80b7-ba58-ec664e26b146&spaceId=e044b5c2-cba9-81af-946c-0003b6109f05&width=1420&userId=&cache=v2)

### Motivation

Large language models (LLMs) have made remarkable strides in complex reasoning, yet training them effectively remains a persistent challenge. The two dominant paradigms each carry significant drawbacks:

Supervised Fine-Tuning (SFT) tends to lock models into [imitative, rigid behavior](https://arxiv.org/abs/2501.17161) and is prone to [catastrophic forgetting](https://arxiv.org/abs/2510.18874).

On-policy Reinforcement Learning (RL) encourages active exploration, but primarily [reinforces what the base model already knows](https://arxiv.org/abs/2504.13837) rather than instilling new reasoning abilities. It [optimizes existing knowledge](https://arxiv.org/abs/2506.14245) and [struggles on hard problems](https://blog.ml.cmu.edu/2025/11/26/how-to-explore-to-scale-rl-training-of-llms-on-hard-problems/) where all rollouts receive zero reward.

⚠️

The SFT-then-RL Dilemma

The dominant post-training pipeline applies SFT before RL in a [two-stage fashion](https://arxiv.org/abs/2411.15124). Yet this paradigm does [not consistently outperform pure RL](https://arxiv.org/abs/2506.19767). SFT can [disrupt established patterns and induce overfitting](https://arxiv.org/abs/2508.11408), undermining the benefits of subsequent RL.

### The Landscape of Hybrid Approaches

Incorporating off-policy expert data into the on-policy RL loop is a promising strategy: it preserves RL's exploratory drive while injecting expert knowledge to unlock new reasoning abilities. Several families of approaches have emerged:

📥 Incorporating Off-Policy Data into On-Policy Training

🏫 On-Policy Distillation

### Our Approach: Logit Fusion at the Data Level

💡

Key Insight. None of the existing methods fundamentally modify the data distribution. We address the on-policy / off-policy mixing challenge directly at the data level.

By constructing a mixed behavior policy that blends student and teacher distributions and sampling from this mixture, we learn from data that reflects both the student's exploration and the teacher's expertise.

❌

Why not fully off-policy?

That would essentially become SFT with a reward function, preventing the target policy from bootstrapping itself and actively exploring.

❌

Why not fully on-policy?

That would restrict the model to its base capabilities and prevent it from learning new abilities. It also gets stuck on hard problems where all rollouts receive zero reward.

We propose a hybrid post-training approach that incorporates offline expert knowledge into RL training through a logit fusion technique:

> We form a fused-logit behavior policy by linearly interpolating the teacher and student logits at each decoding step, then sample the next token from the mixed distribution.

This generates coherent sequences while distilling knowledge from the teacher without straying too far off-policy, maintaining semi-on-policy characteristics and leveraging the student's exploratory capabilities. Our method unifies external supervision with self-exploration, enabling smooth transitions from supervised tuning to fully autonomous reinforcement learning.

Off-policy training is inherently challenging, and [previous](https://arxiv.org/abs/2504.14945) [works](https://arxiv.org/abs/2507.01679) have noted optimization difficulties, often limiting off-policy traces to just one in a group of eight.

> We stabilize training through importance sampling correction and employ a decaying coefficient to smoothly guide the transition from off-policy imitation to on-policy exploration.

### Logit Fusion: Mathematical Formulation

At the core of our approach is a simple yet powerful technique: token-by-token logit fusion. At each decoding step during rollout generation, we blend the logits from the teacher model $\pi_T$ and the student model $\pi_S$ before sampling the next token.

Consider generating a completion $y = (y_1, y_2, \ldots, y_T)$ given prompt $x$. At decoding step $i$, we have already generated previous tokens $y_{<i} = (y_1, \ldots, y_{i-1})$ and want to sample the next token $y_i$.

Both models produce distributions over the vocabulary. We construct the fused distribution by linearly interpolating their logits:

$$
\begin{gathered}
\ell_{\text{mix}}(y_i \mid x, y_{<i})
= \alpha \,\ell_T(y_i \mid x, y_{<i}) + (1-\alpha)\, \ell_S(y_i \mid x, y_{<i}), \\
\pi_{\text{mix}}(y_i \mid x, y_{<i})
= \text{softmax}(\ell_{\text{mix}}(y_i \mid x, y_{<i})).
\end{gathered}
$$

where $\alpha \in [0,1]$ controls the interpolation strength. Linearly interpolating in logit space is equivalent (up to an additive constant) to interpolating in log-probability space, as commonly done in methods such as [Contrastive Decoding](https://arxiv.org/abs/2210.15097), [MiniLLM](https://arxiv.org/abs/2306.08543), [proxy-tuning](https://arxiv.org/abs/2401.08565), and [ThinkMerge](https://arxiv.org/abs/2512.02874). We then sample $y_i \sim \pi_{\text{mix}}(\cdot \mid x, y_{<i})$, append it to the context, and repeat for subsequent positions until the full sequence is generated. This construction is equivalent to taking a geometric mean of the teacher and student token distributions:

$$
\pi_{\text{mix}}(y_i \mid x, y_{<i}) \propto \pi_T(y_i \mid x, y_{<i})^\alpha \cdot \pi_S(y_i \mid x, y_{<i})^{1-\alpha}.
$$

❓

Why geometric mean? The geometric mean yields sharper and more consistent results than direct probability weighting $\alpha \pi_T + (1 - \alpha) \pi_S$, which can produce overly smooth distributions that dilute both models' signal.

### Example: Fused Generation

To illustrate how logit fusion works in practice, consider mixing Qwen2.5-0.5B-Instruct (student) with Qwen2.5-3B-Instruct (teacher) on a simple counting task:

🔍

Prompt: How many 'r' characters are in the word 'character'?

Student alone ($\alpha = 0$): "The word 'character' contains 4 'r' characters."

Teacher alone ($\alpha = 1$): "There are two 'r' characters in the word 'character'."

Fused ($\alpha = 0.1$): "The word 'character' contains 4 'r' characters."

Fused ($\alpha = 0.4$): "The word 'character' contains 3 'r' characters."

Fused ($\alpha = 0.7$): "There are 2 'r' characters in the word 'character'."

As $\alpha$ increases, the fused model remains fluent while being progressively steered toward the teacher's preferred token choices. The teacher provides the correct answer (2 'r' characters), and higher $\alpha$ values make the fused generation increasingly align with that answer, illustrating how logit fusion can transfer useful knowledge without fully collapsing into pure teacher imitation.

### Implementation Details

#### 📐 Vocab Size Mismatch

Our approach requires that the teacher and student come from the same model family (or are at least tokenizer-compatible). Without this premise, logit fusion is ill-defined: the same token ID could map to different strings, or vice versa.

#### 🛑 EOS Token Mismatch

EOS token IDs can also differ between teacher and student, especially when mixing base and instruct variants. Our fix is straightforward: treat EOS as a set of valid stopping tokens. Let $e_S$ and $e_T$ be the student and teacher EOS token IDs respectively. During fused rollout generation, we stop when the sampled token matches either one. In HuggingFace, this simply means setting

eos\_token\_id=\[e\_S, e\_T\]

.

#### ⚙️ Rollout Generation and Inference Framework Compatibility

Rollout generation is the main engineering bottleneck for logit-fusion RL: the behavior policy depends on both models at every decoding step. Each step requires computing logits from two models, fusing them, and sampling the next token, doubling forward-pass cost. In our implementation, we keep both models on GPU, maintain KV-caches for both, and decode in lockstep so that cache updates stay aligned across the fused trajectory.

The deeper challenge is incompatibility with modern high-throughput inference frameworks. Frameworks like vLLM are designed around a single active model that owns the KV-cache and drives sampling. While vLLM supports custom logit processors, the interface only exposes logits from the currently running model and does not natively support fetching and combining logits from a second model.

In practice, this leaves three options:

Implement fused rollout in a custom generation loop outside vLLM.

Fork or extend vLLM internals to support two-model sampling.

Use an alternative inference stack where synchronized multi-model decoding is easier to express.

We chose (3) and implement fused rollout directly with HuggingFace. Empirically, running vLLM with a custom logit processor that queries an external teacher model, as in (1), is actually slower than running both models in HuggingFace with lockstep decoding.

### Importance Sampling Ratio and Clipping

Our rollouts are sampled from a fused-logit behavior policy. We denote the behavior policy that actually generated the training data as $\pi_{{\text{mix}}_{\text{old}}}$. The model we are updating is the student/target policy $\pi_S$. Since $\pi_S \neq \pi_{{\text{mix}}_{\text{old}}}$ in general, training is off-policy.

To correct for this distribution mismatch, we apply importance sampling (IS). At token position $i$, the per-token IS ratio is:

$$
r_i(\theta)=\frac{\pi_S(y_i \mid x, y_{<i};\theta)}{\pi_{{\text{mix}}_{\text{old}}}(y_i \mid x, y_{<i})}.
$$

It is tempting to reuse PPO-style ratio clipping, i.e. clip $r_i(\theta)$ symmetrically around 1. However, in an off-policy setting, this can behave poorly because $r_i(\theta)$ is not expected to concentrate near 1, even in the perfectly on-policy case where $\theta = \theta_{\rm old}$, so clipping triggers frequently and reduces the informativeness of the update.

A practical symptom is unbalanced clipping. In Figure 1 below, the fraction of tokens clipped on the low side is substantially larger than the fraction clipped on the high side, indicating that the ratio distribution is heavily skewed rather than concentrated near 1. This occurs because tokens sampled under the mixed policy often have low probability under the learner.

![](https://juzhengz.notion.site/image/attachment%3Af0074d00-17d2-4782-8c18-0b18822b5d1e%3Appo_clipping_three_panels_ema.png?table=block&id=30d4b5c2-cba9-8006-92b8-c7bf376184c1&spaceId=e044b5c2-cba9-81af-946c-0003b6109f05&width=1420&userId=&cache=v2)

Figure 1: Unbalanced PPO-style ratio clipping under fused-logit rollouts. The student is Qwen2.5-0.5B and the teacher is Qwen2.5-7B. The baseline is student-only training with GSPO.

This skew also explains why PPO-style clipping can make training less informative. Tokens with high learner probability but low rollout probability tend to dominate the gradient magnitude. However, the most informative signal, tokens that surprise the learner, often comes from tokens the learner assigns low probability to but the behavior policy (guided by the teacher) assigns high probability to. Excessively aggressive clipping can therefore impair learning when the rollout distribution is intentionally shifted by teacher guidance.

For this reason, we avoid PPO clipping and instead use an IS-corrected REINFORCE-style objective, combined with a simple safeguard that clips only extreme ratios that would otherwise create unstable, high-variance gradients:

✂️

Our solution: Cap overly large ratios with $\tilde{r}_t=\min(r_t(\theta), C_\text{cap})$, where $C_{\text{cap}}$ is a hyperparameter (e.g., 3). This preserves informative gradient signal while preventing high-variance updates.

### Bias-Variance Trade-offs in Importance Sampling

There are multiple reasonable choices for the importance ratio, each reflecting a different point on the bias-variance spectrum.

##### Option 1: On-policy ratio (low variance, high bias)

Treat each token as if it were sampled by the student behavior policy. The ratio reduces to the standard on-policy PPO ratio:

$$
r_i^{\text{on}}(\theta)= \frac{\pi_S(y_i \mid x, y_{<i};\theta)}{\pi_{S_{\text{old}}}(y_i \mid x, y_{<i})}.
$$

This ratio is typically well-behaved because $\pi_{S_{\text{old}}}$ is close to $\pi_S$. The downside is that it ignores the fact that data were actually generated by the fused policy, introducing bias whenever the fused behavior distribution differs meaningfully from the student distribution.

##### Option 2: Mixed ratio (lower bias, higher variance)

A more faithful correction uses the actual behavior policy, giving the ratio $r_i(\theta)$ defined above. This is less biased with respect to the true behavior distribution. It is [not fully unbiased](https://www.notion.so/271211a558b7808d8b12d403fd15edda) because we apply the correction at the token level rather than the sequence level, but it is a much closer approximation than ignoring the rollout policy entirely. The trade-off is higher variance: $\pi_{{\text{mix}}_{\text{old}}}$ can differ substantially from $\pi_S$, especially early in training or on hard prompts where teacher guidance is strong.

##### Option 3: Shaped ratio (biased, amplified signal)

A third strategy, used in [prior work](https://arxiv.org/abs/2504.14945), applies a policy shaping transform to amplify updates from low-probability tokens. We replace $r_i$ with $f(r_i) = \frac{r_i}{r_i+\gamma}$ using $\gamma = 0.8$. Note that we use a different $\gamma$ than [LUFFY](https://arxiv.org/abs/2504.14945) ($\gamma = 0.1$) because the GSPO importance ratio operates on a different scale than the standard PPO ratio. This transform emphasizes informative low-probability tokens, but because $f(r_i) \neq r_i$, it introduces bias relative to the $r_i(\theta)$ estimator.

In our experiments, we directly compare these three ratio choices while keeping all other training details fixed on DeepMath-103K. Figure 2 reports both the training reward (left) and evaluation reward (right). An important distinction: training reward is measured on trajectories sampled from the fused teacher+student behavior policy $\pi_{\text{mix}}$, whereas evaluation reward is measured on trajectories sampled by the student policy $\pi_S$ alone. The evaluation reward is therefore the more meaningful metric—it reflects whether the target policy has genuinely internalized what it learned from the mixed trajectories and can perform well on its own, without teacher assistance at evaluation time.

![](https://juzhengz.notion.site/image/attachment%3Ad1223131-f771-49cc-a4e6-d7a4c3b734de%3Appo_reward_is_variants.png?table=block&id=3174b5c2-cba9-80de-b4ef-fe969be567b6&spaceId=e044b5c2-cba9-81af-946c-0003b6109f05&width=1420&userId=&cache=v2)

Figure 2: Reward curves for different importance sampling ratios on DeepMath-103K. Left: training reward. Right: evaluation reward. The mixed ratio $\pi_S/\pi_{{\text{mix}}_{\text{old}}}$ achieves the best evaluation performance; the on-policy ratio lags behind; the shaped ratio $f(r_i)$ exhibits unstable training dynamics. The student is Qwen2.5-0.5B and the teacher is Qwen2.5-7B. The baseline is student-only training with GSPO.

🏆

Result: The mixed ratio $r_i(\theta)=\pi_S/\pi_{{\text{mix}}_{\text{old}}}$ performs best. It yields a clear and sustained improvement in evaluation reward throughout training. The on-policy ratio is stable but consistently underperforms, reflecting the bias from ignoring the true behavior policy. The shaped ratio reaches high reward early but is highly unstable: amplifying low-probability tokens may help optimization initially, yet the resulting updates do not translate into consistent gains for the target policy.

### Separating IS Correction from Trust Region

A natural question is whether we can get the best of both worlds: correct for off-policy data and enforce a trust region, by separating these two roles in the IS ratio. A well-known method is decoupled PPO as used in [blog1](https://fengyao.notion.site/off-policy-rl#279721e3f6c48052aa7fdcbe7950c95f) and [blog2](https://www.notion.so/271211a558b7808d8b12d403fd15edda), which observes that the full off-policy ratio $r_i(\theta) = \pi_S(y_i;\theta) / \pi_{{\text{mix}}_{\text{old}}}(y_i)$ conflates two distinct roles, importance sampling correction and trust region enforcement, and separates them. The decomposition is:

$$
r_i(\theta) 
= \underbrace{\frac{\pi_{S_{\text{old}}}(y_i \mid x, y_{<i})}{\pi_{{\text{mix}}_{\text{old}}}(y_i \mid x, y_{<i})}}_{w_i}
\cdot \underbrace{\frac{\pi_S(y_i \mid x, y_{<i};\theta)}{\pi_{S_{\text{old}}}(y_i \mid x, y_{<i})}}_{r_i^{\text{on}}(\theta)},
$$

where $w_i = \pi_{S_{\text{old}}} / \pi_{{\text{mix}}_{\text{old}}}$ is a static IS correction weight (fixed, not a function of $\theta$), and $r_i^{\text{on}}(\theta) = \pi_S / \pi_{S_{\text{old}}}$ is the standard PPO trust-region ratio, which equals 1 when $\theta = \theta_{\text{old}}$ and is well-suited for symmetric clipping around 1.

In principle, this decomposition lets each component operate in its intended regime: PPO clipping acts on $r_i^{\text{on}}$ (centered around 1), while $w_i$ handles off-policy correction and is separately truncated. Within the decoupled framework, the IS weight $w_i$ can be truncated at two granularities:

Sequence-level: compute $w_{\text{seq}} = \min(\prod_i w_i,\, C_{\text{seq}})$ and assign this single truncated weight to all tokens in the trajectory.

Token-level: truncate each token independently via $w_{\text{tok}} = \min(w_i,\, C_{\text{tok}})$, giving finer-grained control.

Despite its theoretical appeal, decoupled PPO does not outperform the simpler single-ratio approach in our experiments. Figure 3 compares decoupled PPO (with both sequence-level and token-level truncation) against our default single ratio $r_i(\theta) = \pi_S / \pi_{{\text{mix}}_{\text{old}}}$ with token-level capping. For the decoupled variants, we set $C_{\text{seq}} = 5$ and $C_{\text{tok}} = 3$, and clip $r_i^{\text{on}}$ with the default GSPO lower threshold $\varepsilon_l = 3\text{e-4}$ and upper threshold $\varepsilon_h = 4\text{e-4}$; for the single-ratio $r_i(\theta)$ baseline, we use $C_{\text{cap}} = 3$. The single-ratio objective achieves higher training reward than both decoupled PPO variants (left) and pulls ahead in evaluation reward after roughly 3500 steps, reaching the highest final performance (right). The two decoupled PPO variants track each other closely on evaluation reward and plateau below the single-ratio curve. We attribute this gap to additional approximation error introduced by the decomposition: separately truncating $w_i$ and clipping $r_i^{\text{on}}$ does not faithfully recover the behavior of the original joint ratio under substantial off-policy shift and distribution mismatch.

![](https://juzhengz.notion.site/image/attachment%3Ac35bda47-6008-4c58-936f-2c0f7410f226%3Appo_reward_truncation_methods.png?table=block&id=30d4b5c2-cba9-80ea-a6cf-d33e4dcb54e0&spaceId=e044b5c2-cba9-81af-946c-0003b6109f05&width=1420&userId=&cache=v2)

Figure 3: Decoupled PPO vs. single-ratio comparison on DeepMath-103K. Left: training reward. Right: evaluation reward. The single ratio $\pi_S/\pi_{{\text{mix}}_{\text{old}}}$ with token-level capping achieves the best evaluation performance, while both decoupled PPO variants (sequence-level and token-level IS truncation) plateau at a lower level. The student is Qwen2.5-0.5B and the teacher is Qwen2.5-7B.

✅

Verdict: We adopt the simpler single-ratio formulation $r_i(\theta) = \pi_S / \pi_{{\text{mix}}_{\text{old}}}$ with token-level capping as our default. It is easier to implement, requires fewer hyperparameters ($C_{\text{cap}}$ only), and performs best empirically.

Our final RL objective is:

$$
\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\text{mix}}(\cdot \mid x)} \left[\sum_{i=1}^{|y|} \min \left(\frac{\pi_S(y_i \mid x, y_{<i};\theta)}{\pi_{{\text{mix}}_{\text{old}}}(y_i \mid x, y_{<i})},\, C_{\text{cap}}\right) \cdot A_i \right],
$$

where $\mathcal{D}$ is the prompt set and the $\min$ operator caps the per-token IS ratio at $C_{\text{cap}}$ to prevent high-variance gradient updates. The token-level advantage $A_i$ is computed via group normalization over the sequence reward $R(x, y)$:

$$
A_i = R(x, y) - \text{mean}(\{R(x, y^{(j)})\}_{j=1}^{G}), \quad \text{for } i = 1, \ldots, |y|,
$$

where $G$ is the number of rollouts per prompt and $R(x, y^{(j)})$ is the reward for the $j$ -th rollout.

### The Role of α\\alpha: Controlling Teacher Guidance

The mixing coefficient $\alpha \in [0,1]$ controls how strongly the teacher influences rollout generation. In our fused-logit behavior policy, $\alpha$ interpolates teacher and student logits at every decoding step. When $\alpha=0$, rollouts are purely on-policy under the student. When $\alpha=1$, rollouts follow the teacher distribution. Intermediate values yield a hybrid behavior policy that preserves student exploration while increasing the probability of sampling successful trajectories on hard prompts.

There is an inherent trade-off associated with $\alpha$. If $\alpha$ is too large, rollouts become strongly teacher-shaped and training can resemble off-policy imitation, with reduced opportunity for the student to discover new strategies on its own. If $\alpha$ is too small, the behavior policy may fail to produce any positive traces on hard prompts, and RL receives little learning signal. This is why a fixed, global $\alpha$ is often suboptimal: the ideal amount of guidance depends on the difficulty of the prompt and the current competence of the student. We therefore use a prompt-specific guidance coefficient that combines:

A global annealing schedule that decays over training

A per-prompt difficulty scaling based on a difficulty label provided by the dataset

We first define a step-dependent base coefficient $\alpha_{\text{base}}$ that decays linearly from an initial value $\alpha_{\text{init}}$ to 0 over the first $K$ training steps:

$$
\alpha_{\text{base}}=\alpha_{\text{init}} \cdot \max\big(0, 1 - \frac{k}{K}\big).
$$

In our experiments, $\alpha_{\text{init}}$ is a hyperparameter (e.g., 0.5) and we use $K = 5000$ steps. Next, for each prompt we compute a normalized difficulty scale. Let $d \in [d_{\min}, d_{\max}]$ be a scalar difficulty label from the dataset (e.g., an integer in $[1,10]$). We map it to $[0,1]$ via:

$$
s(d(x))= \frac{d - d_{\min}}{d_{\max} - d_{\min}}.
$$

Finally, the prompt-specific coefficient is $\alpha(x) =\alpha_{\text{base}} \cdot s(d(x))$, where $d(x)$ is the difficulty score associated with prompt $x$. This design has two intended effects. First, early in training, harder prompts receive larger $\alpha$, which increases the probability of sampling a positive trajectory when the student is still weak. Second, as training progresses, $\alpha_{\text{base}}$ decays to 0, so the rollout policy gradually becomes fully student-driven, reducing off-policy mismatch and enabling the final policy to stand on its own at evaluation time.

To understand the interaction between the $\alpha$ schedule and ratio clipping, we compare four configurations in Figure 4: (i) scheduled $\alpha$ with clipping threshold $\varepsilon_l = 2\text{e-3}$, (ii) scheduled $\alpha$ with no clipping, (iii) fixed $\alpha = 0.5$ with clipping threshold $\varepsilon_l = 2\text{e-3}$, and (iv) fixed $\alpha = 0.5$ with no clipping. In all scheduled $\alpha$ runs, $\alpha$ decays linearly from 0.5 to 0 over 5000 steps following the prompt-specific schedule described above. The clipping thresholds are defined in terms of the GSPO importance ratio, which operates on a different scale than the standard PPO ratio. Starting from the default GSPO lower threshold $\varepsilon_l = 3\text{e-4}$ and upper threshold $\varepsilon_h = 4\text{e-4}$, we raise $\varepsilon_l$ from $3\text{e-4}$ to $2\text{e-3}$ while keeping $\varepsilon_h$ unchanged, which lowers the lower clipping bound $1 - \varepsilon_l$ and widens the clipping range on the low side to admit more low-probability tokens under the learner.

![](https://juzhengz.notion.site/image/attachment%3Af9127cac-b546-41e6-948b-45205f4325c9%3Appo_reward_alpha_clipping_ablation.png?table=block&id=30d4b5c2-cba9-804f-867b-ed230950648b&spaceId=e044b5c2-cba9-81af-946c-0003b6109f05&width=1420&userId=&cache=v2)

Figure 4: Effect of $\alpha$ schedule and GSPO clipping threshold on DeepMath-103K. Left: training reward. Right: evaluation reward. Scheduled $\alpha$ decays linearly from 0.5 to 0 over 5000 steps; fixed $\alpha = 0.5$ remains constant throughout training. "Clip lower" raises $\varepsilon_l$ from the default $3\text{e-4}$ to $2\text{e-3}$, lowering the lower clipping bound $1 - \varepsilon_l$, while keeping $\varepsilon_h$ at $4\text{e-4}$. The student is Qwen2.5-0.5B and the teacher is Qwen2.5-7B. The baseline is student-only training with GSPO.

The most striking pattern is that removing clipping is only viable when combined with the $\alpha$ schedule. Without clipping, the behavior policy introduces many tokens that the student assigns low probability to. When $\alpha$ is fixed, this influx of off-policy tokens is sustained throughout training and destabilizes optimization: the fixed $\alpha$ + no clipping run achieves high training reward early on but quickly deteriorates, exhibiting large oscillations and failing to transfer gains to evaluation reward. This indicates that the student does not internalize the teacher's reasoning abilities.

Lowering the lower clipping bound stabilizes training under fixed $\alpha$: the fixed $\alpha$ + clip lower run avoids the reward collapse and achieves higher evaluation reward than its no-clipping counterpart, confirming that some form of ratio constraint is necessary when the teacher influence is persistent. However, under the scheduled $\alpha$ regime, clipping falls behind the no-clipping variant. When the $\alpha$ schedule already provides a natural mechanism for tapering off-policy exposure, the additional filtering from clipping becomes overly conservative and discards useful gradient signal.

🏆

Best configuration: Scheduled $\alpha$ \+ no clipping. Annealing $\alpha$ enables a smooth transition from imitation to reward optimization. The scheduled runs don't produce dramatically higher training reward than the baseline (expected, since $\alpha$ decays toward 0), but the key benefit is visible in evaluation reward: the student retains reasoning capabilities gained during the early teacher-guided phase.

📋

Practical recipe. Set $\alpha_{\text{init}} = 0.5$ and decay linearly to 0 over ~5000 steps. Scale per-prompt by normalized difficulty. Skip ratio clipping entirely. The schedule itself acts as a natural regularizer.

### Limitations

⏱️ Inference overhead during rollout generation. Every decoding step requires a forward pass through both the teacher and the student. The lockstep two-model decoding is incompatible with high-throughput engines like vLLM, forcing a fallback to a slower HuggingFace-based loop. This increases wall-clock training cost and lowers throughput.

📊 Single dataset and model scale. All experiments use DeepMath-103K with a Qwen2.5-0.5B student and Qwen2.5-7B teacher. It remains to be verified whether the findings generalize to larger students, different model families, and more diverse training datasets.

📉 Linear $\alpha$ schedule. We adopt a simple linear decay. More sophisticated schedules (adaptive decay based on student performance or per-prompt success rate) could yield better results but are not explored here.

🔄 Training-inference mismatch. During training, rollouts come from $\pi_{\text{mix}}$, but at evaluation time only $\pi_S$ is used. The student must internalize teacher-guided reasoning patterns well enough to reproduce them independently. Our experiments show this works in practice, but the degree to which it holds under larger distribution shifts remains an open question.

### Conclusion

🎯

Logit fusion offers a principled middle ground between imitation and exploration. By blending teacher and student logits at every decoding step, we generate rollouts that carry expert signal without abandoning self-exploration. Combined with a simple importance-sampling correction and a decaying $\alpha$ schedule, this approach lets small models steadily absorb stronger reasoning capabilities and retain them when the teacher is removed at evaluation time.

### Citation

@article { zhang2026logitfusion, title \= { Learning from Mixed Rollouts: Logit Fusion as a Bridge Between Imitation and Exploration }, url \= { https:/ / juzhengz.notion.site / logit \- fusion }, author \= { Zhang, Juzheng and Hans, Abhimanyu and Kirchenbauer, John and Goldblum, Micah and Panda, Ashwinee and Goldstein, Tom }, journal \= { Notion Blog }, year \= { 2026 } }