---
title: "Policy Gradient, Sequence, and Token— Part I: Basic Concepts"
author:
  - "[[fengyao]]"
url: https://fengyao.notion.site/pg-seq-token-part1-basics
created: 2026-01-29
tags:
category: clippings/article
updated: 2026-01-29T23:13
---
Part I is designed as a run-through introduction to policy gradient at both token and sequence levels, and how they connect to each other.

When applying Reinforcement Learning (RL) to Large Language Models (LLMs), a fundamental design choice is whether to apply the policy gradient updates at token level or sequence level.

For REINFORCE algorithm, they are identical. For more advanced methods (e.g., TRPO, PPO, GRPO, GSPO), they diverge in ways that have significant practical implications, while both token-level and sequence-level formulations remain reasonable.

### On-Policy REINFORCE: Where They’re All the Same

The classic entry point to policy gradients is the REINFORCE algorithm. The goal is to maximize the expected reward by adjusting the policy's parameters, $\theta$. For a language model, the policy $\pi_\theta$ generates a sequence $y$ of $T$ tokens (actions): $y = (y_1, y_2, ..., y_T)$.

The objective function $J(\theta)$ for a sequence-level reward $R(y)$ is:

$$
\begin{equation}
J(\theta) = \mathbb{E}_{y \sim \pi_\theta} [R(y)] = \sum_y R(y) \cdot \pi_\theta(y)
\end{equation}
$$

The gradient of this sequence-level objective is:

$$
\begin{equation}
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta} [R(y)\cdot \frac{\nabla_\theta  \pi_\theta(y)}{\pi_\theta(y)}]=\mathbb{E}_{y \sim \pi_\theta} [R(y)\cdot \nabla_\theta \log \pi_\theta(y)]
\end{equation}
$$

Since the probability of a sequence is the product of the probabilities of all its tokens, $\pi_\theta(y) = \prod_{t=1}^T \pi_\theta(y_t | y_{<t})$, its log-probability can be decomposed as $\log \pi_\theta(y) = \sum_{t=1}^T \log \pi_\theta(y_t| y_{<t}).$Then, the sequence-level objective ([Equation 2](https://fengyao.notion.site/28b721e3f6c480b88b5be1d89512ac3a?pvs=25#28b721e3f6c4818ea591fbd385714b39)) can be rewritten as:

$$
\begin{align}\nabla_\theta J(\theta) = \mathbb{E}_{y_t \sim \pi_\theta} \left[ \sum_{t=1}^T R(y) \cdot \nabla_\theta \log \pi_\theta(y_t\mid y_{<t}) \right].\end{align}
$$

This form is exactly the gradient you would get, if you defined a token-level objective where the reward $R(y)$ is applied to each token in the sequence.

💡

Key Takeaway: For the on-policy REINFORCE algorithm, the sequence-level and token-level formulations are mathematically equivalent.

### The Complication: Trust Region and Multi-step MDP

While the fundamental REINFORCE algorithm directly optimizes the expected reward, trust region methods (e.g., TRPO, PPO, …) optimize a different :

$$
\small\begin{equation}
L(\theta) = \mathbb{E}_{s \sim d^{\pi_{\theta_{\rm old}}},\;a \sim \pi_{\theta_{\rm old}}(a|s)} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\rm old}}(a|s)} \cdot A^{\pi_{\theta_{\rm old}}}(s,a) \right],\; \text{s.t.}\; D_{\rm KL}(\pi_\theta \,\|\, \pi_{\theta_{\rm old}}) \leq C.
\end{equation}
$$

Here:

$A^{\pi_{\theta_{\rm old}}}(s,a)$ is the advantage estimation for $s$, $a$, and $\pi_{\theta_{\rm old}}$;

$d^{\pi_{\theta_{\rm old}}}$ is the visiting frequency of states observed on $\pi_{\theta_{\rm old}}$;

the KL constraint enforces that the updated policy remains close to the old one.

Notice that, the expectation is taken with respect to $d^{\pi_{\theta_{\rm old}}}$ rather than $d^{\pi_\theta}$. If we could sample from the current policy $d^{\pi_\theta}$, or if $\pi_\theta = \pi_{\theta_{\text{old}}}$, then the objective $L(\theta)$ would coincide with $J(\theta)$ in the sense of gradient computation. However, in practice, we only have trajectories sampled from $\pi_{\theta_{old}}$ and the surrogate objective introduces a mismatch.

#### The Trade-off between Trust Region and REINFORCE

The key motivation of incorporating trust region is to achieve training stability. Directly optimizing $J(\theta)$ can cause overly aggressive updates or even collapse. Trust region algorithms stabilize training by enforcing conservative updates, e.g., $D_{\rm KL}(\pi_\theta \,\|\, \pi_{\theta_{\rm old}}) \leq C$. Also, $L(\theta)$ [provides a provable lower bound](https://arxiv.org/abs/1502.05477) on $J(\theta) - J(\theta_{\rm old})$.

Empirically, this trade-off has been found to yield more reliable improvements and TRPO/PPO have become standard practice. That said, recent work (e.g., [REINFORCE++](https://arxiv.org/abs/2501.03262)) shows that directly optimizing $J(\theta)$ can also be viable by introducing training stability control in reward definition instead of learning algorithm.

#### PPO-Clip

Optimizing [Equation 4](https://fengyao.notion.site/28b721e3f6c480b88b5be1d89512ac3a?pvs=25#28b721e3f6c4815c98f8f5d2bf76aa99) can be implemented by adding a KL penalty to the objective (as in PPO-KL). However, tuning the weight of this penalty can be tricky. PPO-Clip offers a simpler, more robust alternative. Instead of adding a soft penalty (the KL term) that discourages large updates, PPO-clip directly modifies the loss surface to remove the incentive for them.

![](https://fengyao.notion.site/image/attachment%3A5af608f9-2796-4f97-b8c8-696068277ed9%3Aclip.png?table=block&id=28b721e3-f6c4-8168-a9fa-ceece25e457e&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

Figure A. Visualization of the simplified loss clip mechanism; the color denotes the value of objective (weighted advantage/reward). Note PPO-Clip only clips towards better advantage/reward, while the visualization here clips at all directions.

The trust-region constraint ensures that each update remains close to the data distribution we actually have, effectively approximating a higher-order optimization step while avoiding instability.

💡

Key Takeaway: For trust region algorithms like PPO-Clip & PPO-KL, the [surrogate objective](https://fengyao.notion.site/28b721e3f6c480b88b5be1d89512ac3a?pvs=25#28b721e3f6c4815c98f8f5d2bf76aa99) $L(\theta)$ is not equivalent to the [original objective](https://fengyao.notion.site/28b721e3f6c480b88b5be1d89512ac3a?pvs=25#28b721e3f6c4813fb7b2eebe45a4117a) $J(\theta)$. Instead, $L(\theta)$ is a provable lower bound on $J(\theta) - J(\theta_{\rm old})$.

### Trust Region at Token/Sequence Level

At this point, the key question becomes: How exactly should we define “action” and “state” in language modeling? Depending on whether we treat each token, each turn, or the entire sequence as the action, the surrogate objective takes very different forms, while all of these choices are reasonable.

#### Token-Level Objective

Each prefix $y_{<t}$ is treated as the state $s_t$, and each token $y_t$ is the action. The state visitation distribution $d^{\pi_{\theta_{\rm old}}}(s_t)$ is implicitly captured by the probability of generating the prefix under $\pi_{\theta_{\rm old}}$.

The corresponding surrogate objective is defined as:

$$
\small \begin{equation}
\mathbb{E}_{y_{{<t}} \sim d^{\pi_{\theta_{\rm old}}},\;y_t \sim \pi_{\theta_{\rm old}}(y_t|y_{<t})} \left[ \frac{\pi_\theta(y_t|y_{<t})}{\pi_{\theta_{\rm old}}(y_t|y_{<t})} \cdot A^{\pi_{\theta_{\rm old}}}(y_{<t}, y_t) \right], \,\text{s.t.}\, D_{\rm KL}(\pi_\theta \,\|\, \pi_{\theta_{\rm old}}) \leq C
\end{equation}
$$

As previously mentioned, [Equation 5](https://fengyao.notion.site/28b721e3f6c480b88b5be1d89512ac3a?pvs=25#28b721e3f6c481b6909fecc6b6212d19) is not designed to be equivalent to $J(\theta)$, but rather to serve as a lower bound of $J(\theta) - J(\theta_{\rm old})$. This matches the [MDP](https://en.wikipedia.org/wiki/Markov_decision_process) interpretation, where each action has its own credit.

multi-turn sequence-level objective works similar to token-level objective.

#### Single-Turn Sequence-Level Objective

The entire generated output $y$ (all tokens) is viewed as a single action taken in the initial state (the prompt, marked as ${\rm p}$). The corresponding surrogate objective is defined as:

$$
\begin{equation}
\mathbb{E}_{{\rm p} \sim d,\;y \sim \pi_{\theta_{\rm old}}(y|{\rm p})} \left[ \frac{\pi_\theta(y|{\rm p})}{\pi_{\theta_{\rm old}}(y|{\rm p})} \cdot A^{\pi_{\theta_{\rm old}}}({\rm p}, y) \right],\; \text{s.t.}\; D_{\rm KL}(\pi_\theta \,\|\, \pi_{\theta_{\rm old}}) \leq C.
\end{equation}
$$

It is worth mentioning that, in this setting, since only the initial prompt is treated as state, this problem is simplified as the contextual bandit, i.e., $d$ is stationary regarding $\pi$ and $Q^{\pi_{\theta_{\rm old}}}({\rm p}, y) \propto R({\rm p}, y)$. Under this setting, its easy to see without KL constraint, $L(\theta)$ is equivalent to $J(\theta)$ in the sense of gradient computation, i.e.,

$$
\begin{equation}
\small \nabla \mathbb{E}_{{\rm p} \sim d,\;y \sim \pi_{\theta_{\rm old}}(y|{\rm p})} \left[ \frac{\pi_\theta(y|{\rm p})}{\pi_{\theta_{\rm old}}(y|{\rm p})} \cdot A^{\pi_{\theta_{\rm old}}}({\rm p}, y) \right] \propto \nabla \mathbb{E}_{{\rm p} \sim d,\;y \sim \pi_{\theta}(y|{\rm p})} \left[  R({\rm p}, y) \right]= \nabla J(\theta).
\end{equation}
$$

Accordingly, some have argued that only the sequence-level formulation is “correct” and all other formulations are “incorrect”—different from the sequence-level formulation, in token-level or multi-turn sequence-level formulations, $L(\theta)$ and $J(\theta)$ do not have the equivalence as in [Equation 7](https://fengyao.notion.site/28b721e3f6c480b88b5be1d89512ac3a?pvs=25#28b721e3f6c481949b79d94e1f157067).

This disparity is expected. Trust-region methods such as TRPO and PPO are not designed to make $L(\theta)$ identical to $J(\theta)$; instead, $L(\theta)$ serves as a theoretically provable lower bound on $J(\theta) - J(\theta_{\rm old})$. This conservative formulation ensures stable and monotonic policy improvement.

#### So... Token or Sequence?

So, which one is better? There's no absolute answer.

Single-turn sequence-level setting resembles the bandit setting, where an entire trajectory sequence is a single arm pull. The action is the whole sequence, and you get one reward. In this case, $L(\theta)$ and $J(\theta)$ aligns well, as the state distribution is stationary.

Multi-turn or token-level setting aligns better with the MDP setting. A new policy leads to a new state distribution. In this case, $L(\theta)$ and $J(\theta)$ diverge in a meaningful way, and the difference in the state distribution is reflected in the design of the trust region.

Empirically, some find token-level loss works better in their settings, while others find sequence-level work better. As described previously, both formulations are reasonable to a certain extent.

💡

Key Takeaway: The surrogate objective $L(\theta)$ is not equivalent to $J(\theta)$ in MDP settings like multi-turn/token-level loss. Instead, $L(\theta)$ is designed to be a provable lower bound of $J(\theta) - J(\theta_{\rm old})$,thus having different forms in different settings.

### The Further Complication: Learner-Sampler Mismatch

As mentioned in [our blog](https://fengyao.notion.site/off-policy-rl), we observe a notable disparity between the learner and the sampler, which is mainly caused by backend differences. Unlike conventional off-policy RL, such disparity cannot be attributed to parameter mismatch and persists even in the on-policy setting — that is, even when both sides use the same model parameters.

This is an open-ended question. In our original blog, we mentioned the TIS fix [is compilable with](https://fengyao.notion.site/237721e3f6c48094ad67dad3ac091c56?pvs=25#246721e3f6c4807389acd47b346a6aac) sequence-level loss (e.g., it operates at sequence-level for GSPO) and token-level loss (e.g., it operates at token-level for GRPO). Both approaches are reasonable and have pros and cons.

In [Part II](https://fengyao.notion.site/28b721e3f6c480f8a4b0e1f8301d90ac?pvs=25), we explored the impact of the mismatch, some simple adaptations, and some empirical verifications on this topic in more details.

### Acknowledgements

We thank Lihong Li, Linli Xu, Chujie Zheng, Jian Hu, and Chenyang Zhao for their insightful feedback on an earlier draft of this post.