---
updated: 2026-01-29T13:12
author:
  - Qiangwei Bai
url: https://bqw1013.github.io/posts/entropy-collapse-and-mitigation-strategies/
---
## 1. Policy Entropy and Entropy Collapse

### 1.1 Entropy Definition

Let $x$ denote the prompt and $y$ denote the response. The policy $\pi_\theta$ outputs a probability distribution for a token $t$ as follows:

$$
p_t = (p_{t,1}, \dots, p_{t,|V|}) = \pi_\theta(\cdot | x, y_{<t}) = \text{softmax}\left(\frac{z_t}{T}\right) \quad (1)
$$

Here, $|V|$ is the size of the vocabulary, $z_t \in \mathbb{R}^{|V|}$ are the `logits`, and $T \in \mathbb{R}$ is the decoding temperature.

The entropy for token $t$ is then given by:

$$
H_t = -\sum_{j=1}^{|V|} p_{t,j} \log p_{t,j} \quad (2)
$$

### 1.2 Entropy Collapse and Model Performance

In the early stages of RL training, the model’s entropy drops sharply. As entropy decreases, accuracy enters a period of rapid growth. However, the rapid depletion of entropy can lead to the model becoming overconfident, which in turn diminishes its exploration capabilities. Through empirical studies, [1] established a quantitative relationship between policy entropy $H$ and downstream task performance $R$:

$$
R = -a \cdot \exp(H) + b \quad (3)
$$

where $a$ and $b$ are fitting coefficients that reflect the intrinsic properties of the specific model and training data.

**Evidently, maintaining entropy within a reasonable range over the longer timeframe is crucial for the continuous improvement of the model’s capabilities.** By stabilizing entropy to enable longer RL training, [2] found that model can push past its original performance limits and achieve continuous improvement.

### 1.3 Token Entropy and Forking Tokens

Analysis by [3] reveals that while most tokens have very low entropy, a small number of tokens exhibit high entropy. Furthermore, the function of a token is highly correlated with its entropy:

- **High-entropy tokens** primarily serve as “logic connectors” and “hypothesis introducers”, such as `wait`, `however`, etc.
- **Low-entropy tokens**, on the other hand, act as “structure completers”, responsible for filling in details within well-established reasoning steps.

Consequently, [3] defines these high-entropy tokens as `forking tokens`.

---

## 2. GRPO

GRPO builds upon PPO by computing advantages through intra-group normalization. Specifically, given a prompt $x$, we sample $G$ responses $\{y_i\}_{i=1}^G$. The advantage is then calculated as:

$$
A_{i,t} = \frac{r_i - \text{mean}(\{r_i\}_{i=1}^G)}{\text{std}(\{r_i\}_{i=1}^G)} \quad (4)
$$

where $r_i$ is reward value for the response $y_i$.

The objective function of GRPO is:

$$
J_{GRPO} = \mathbb{E}_{x \sim p, \{y_i\}_{i=1}^G \sim \pi_{old}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min \left( r_{i,t}(\theta) A_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) A_{i,t} \right) \right] \quad (5)
$$

where $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta_{old}}(a_{i,t}|s_{i,t})}$ is the importance sampling ratio.

---

## 3. Direct Optimization for Forking Tokens

**Decoupling the $clip$ Upper and Lower Bounds**. [4] argues that the $clip$ operation is inequitable for high-probability (low-entropy) tokens and low-probability (high-entropy) tokens. For example, assume $\varepsilon=0.2$. For a token with a probability of $\pi_{\theta_{old}}=0.9$, the $clip$ operation limits its update, allowing for a maximum absolute probability increase of $1.08 - 0.9 = 0.18$. In contrast, for a token with a probability $\pi_{\theta_{old}}=0.01$, the maximum absolute increase is merely $0.01 \times 1.2 = 0.012$. Therefore, the $clip$ operation restricts the update magnitude for low-probability (high-entropy) tokens. To address this, [4] proposes decoupling the clipping bounds from a single $\varepsilon$ into $\varepsilon_{low}$ and $\varepsilon_{high}$, and appropriately increasing $\varepsilon_{high}$ to allow high-entropy tokens to receive larger updates.

$$
J_{DAPO} = \mathbb{E}_{x \sim p, \{y_i\}_{i=1}^G \sim \pi_{old}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min \left( r_{i,t}(\theta) A_{i,t}, \hat{r}_{i,t}(\theta) A_{i,t} \right) \right] \quad (6)
$$

where $\hat{r}_{i,t}(\theta) = \text{clip}(r_{i,t}(\theta), 1-\varepsilon_{low}, 1+\varepsilon_{high})$.

**Decoupling the Advantage from Clipping**. Building on [4], [5] further argues that the advantage term should be decoupled from the clipping mechanism. This is because when clipping is triggered, the gradient becomes zero. The importance sampling ratio should instead be viewed as a weight for the advantage. Therefore, the objective function is further modified to:

$$
J_{CISPO} = \mathbb{E}_{x \sim p, \{y_i\}_{i=1}^G \sim \pi_{old}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \text{sg}(\hat{r}_{i,t}(\theta)) A_{i,t} \right] \quad (7)
$$

where `sg` is the `stop_gradient` operation.

**Directly Increasing the Advantage of `forking tokens`.** [6] takes an even more direct approach: since high-entropy tokens are more important, their advantage should be directly increased.

$$
\psi(H_{i,t}) = \min \left( \alpha \cdot H_{i,t}^{\text{detach}}, \frac{|A_{i,t}|}{k} \right) \quad (8)
$$
$$
\hat{A}_{i,t} = A_{i,t} + \psi(H_{i,t}) \quad (9)
$$

Here, $\psi(H_{i,t})$ serves as an advantage bonus, which is capped to ensure it does not exceed $1/k$ of the original advantage’s magnitude.

---

## 4. The Impact of Positive and Negative Samples

[7] analyzes the impact of positive and negative samples on entropy collapse and finds in the experiments that surprisingly good results can be achieved using only negative samples. Consequently, for the case of binary rewards (+1 for positive samples, -1 for negative), [7] further analyzes the gradient of the REINFORCE loss with respect to the `logits`.

$$
-\frac{\partial L_t}{\partial z_{t,v}} = \begin{cases} r \cdot (1-\pi_v) & v = y_t \\ -r \cdot \pi_v & v \neq y_t \end{cases} \quad (10)
$$

where $z_{t,v}$ is the component for token $v$ in the `logits` vector $z_t$.

**Proof:**

The gradient at a single timestep $t$ is $\nabla L_t = -r \nabla \log \pi_\theta(y_t)$. Expanding $\log \pi_\theta(y_t)$, we get:
$$
\log \pi_\theta(y_t) = \log \left( \frac{\exp(z_{t,y_t})}{\sum_{v' \in V} \exp(z_{t,v'})} \right) = z_{t,y_t} - \log \left( \sum_{v' \in V} \exp(z_{t,v'}) \right)
$$
We discuss this in two cases:
Case 1: $v=y_t$, i.e., taking the derivative with respect to logit of the sampling token.
$$
\frac{\partial (\log \pi_\theta(y_t))}{\partial z_{t,y_t}} = \frac{\partial z_{t,y_t}}{\partial z_{t,y_t}} - \frac{\partial}{\partial z_{t,y_t}} \log \left( \sum_{v' \in V} \exp(z_{t,v'}) \right) = 1 - \frac{1}{\sum_{v'} \exp(z_{t,v'})} \cdot \exp(z_{t,y_t}) = 1 - \pi_{y_t}
$$
Case 2: $v \neq y_t$, i.e., taking the derivative with respect to logit of a non-sampled token.
$$
\frac{\partial (\log \pi_\theta(y_t))}{\partial z_{t,v}} = \frac{\partial z_{t,y_t}}{\partial z_{t,v}} - \frac{\partial}{\partial z_{t,v}} \log \left( \sum_{v' \in V} \exp(z_{t,v'}) \right) = 0 - \frac{1}{\sum_{v'} \exp(z_{t,v'})} \cdot \exp(z_{t,v}) = -\pi_v
$$
Combining these two cases, the gradient of the loss is:
$$
\frac{\partial L_t}{\partial z_{t,v}} = \begin{cases} -r \cdot (1-\pi_{y_t}) & v = y_t \\ r \cdot \pi_v & v \neq y_t \end{cases}
$$

**When $r=1$ (positive reward)**. For the sampled token $y_t$, the gradient increases its logits $z_{t,y_t}$ with a magnitude of $(1-\pi_{y_t})$. When the model is not confident about $y_t$ (i.e., $\pi_{y_t}$ is low), it is updated with larger magnitude. For non-sampled tokens, the gradient decreases its logits $z_{t,v}$ with a magnitude of $\pi_v$. From the perspective of forking tokens [3], if a positive sample contains a forking token, that token’s entropy will decrease more rapidly, meaning these become determined more quickly.

**When $r=-1$ (negative reward).** For the sampled token $y_t$, the gradient decreases its logits $z_{t,y_t}$ with a magnitude of $(1-\pi_{y_t})$. Simultaneously, the released probability mass is redistributed among the other tokens in proportion to their own probability.

In summary, tokens sampled in positive instances proportionally strip probability mass from the remaining tokens, whereas tokens sampled in negative instances cause the stripped probability mass to be proportionally redistributed among the remaining tokens. Therefore, negative samples are naturally conducive to increasing entropy.

Based on this analysis, [7] proposes reducing the signal strength of positive samples during training to maintain entropy.

> **My thought**: The effectiveness of training with purely negative samples might be because it continually reinforces the base model’s original distribution. Could this be analogous to the recent wave of methods based on internal feedback? [8]

---

## 5. The Change in Entropy

In contrast to the foregoing methods, [1] shifts the focus from entropy itself to the change in entropy. Viewing entropy as a function of the `logits`, i.e., $H_t(z_t)$, the change in entropy under a sufficiently small update step can be approximated as:

$$
H_t(z_t^{k+1}) - H_t(z_t^k) \approx -\text{Cov}_{y_t \sim \pi(\cdot|z_t^k)} \left( \log \pi(y_t|z_t^k), \Delta z_{t,y_t}^k \right) \quad (11)
$$

where $z_t^k$ and $z_t^{k+1}$ represent the `logits` vectors at two consecutive steps, and $\Delta z_{t,y_t}^k = z_{t,y_t}^{k+1} - z_{t,y_t}^k$ is the change in the `logits` of token $y_t$ between these steps.

Equation (11) reveals that the change in entropy is negatively correlated with the covariance between the log-probability and the change in `logits`. Specifically, entropy decreases when the `logits` of high-probability tokens increase or the `logits` of low-probability tokens decrease; otherwise, entropy increases. More intuitively, sharpening the current model’s distribution leads to a reduction in entropy.

**Proof:**

The definition of entropy can be expressed as:
$$
H_t(z_t) = -\mathbb{E}_{y_t \sim \pi(\cdot|z_t)} [\log \pi(y_t|z_t)]
$$
Let the `logits` at step $k$ be $z_t^k$, and after a training step, they become $z_t^{k+1}$ at step $k+1$. Since the learning rate is typically small in practice, we can estimate the change in entropy between $z_t^k$ and $z_t^{k+1}$ using Taylor expansion. Specifically, we perform a first-order Taylor expansion of the function $H_t(z_t)$ at point $z_t^k$:
$$
H_t(z_t) \approx H_t(z_t^k) + \langle \nabla_{z_t} H_t(z_t^k), z_t - z_t^k \rangle
$$
Substituting $z_t^{k+1}$ in the equation, we have:
$$
H_t(z_t^{k+1}) - H_t(z_t^k) \approx \langle \nabla_{z_t} H_t(z_t^k), z_t^{k+1} - z_t^k \rangle
$$
First, let's find $\nabla_{z_t^k} H_t(z_t^k)$:
$$
\begin{aligned}
\nabla_{z_t^k} H_t(z_t^k) &= \nabla_{z_t^k} \left( -\mathbb{E}_{y_t \sim \pi(\cdot|z_t^k)} [\log \pi(y_t|z_t^k)] \right) \\
&= -\sum_{y_t} \left[ \nabla_{z_t^k} \pi(y_t|z_t^k) \log \pi(y_t|z_t^k) + \pi(y_t|z_t^k) \nabla_{z_t^k} \log \pi(y_t|z_t^k) \right] \\
&= -\sum_{y_t} \left[ \pi(y_t|z_t^k) \nabla_{z_t^k} \log \pi(y_t|z_t^k) \cdot \log \pi(y_t|z_t^k) + \pi(y_t|z_t^k) \nabla_{z_t^k} \log \pi(y_t|z_t^k) \right] \\
&= -\mathbb{E}_{y_t \sim \pi(\cdot|z_t^k)} \left[ \nabla_{z_t^k} \log \pi(y_t|z_t^k) \cdot \log \pi(y_t|z_t^k) + \nabla_{z_t^k} \log \pi(y_t|z_t^k) \right] \\
&= -\mathbb{E}_{y_t \sim \pi(\cdot|z_t^k)} \left[ \nabla_{z_t^k} \log \pi(y_t|z_t^k) \cdot \log \pi(y_t|z_t^k) \right]
\end{aligned}
$$
(Note: $\mathbb{E}[\nabla \log \pi] = \sum \pi \frac{\nabla \pi}{\pi} = \nabla \sum \pi = \nabla 1 = 0$)

Now, we proceed to compute the inner product $\langle \nabla_{z_t^k} H_t(z_t^k), z_t^{k+1} - z_t^k \rangle$. For notational simplicity, let $\Delta z_t^k = z_t^{k+1} - z_t^k$.
$$
\begin{aligned}
\langle \nabla_{z_t^k} H_t(z_t^k), \Delta z_t^k \rangle &= -\left\langle \mathbb{E}_{y_t \sim \pi(\cdot|z_t^k)} [\nabla_{z_t^k} \log \pi(y_t|z_t^k) \cdot \log \pi(y_t|z_t^k)], \Delta z_t^k \right\rangle \\
&= -\mathbb{E}_{y_t \sim \pi(\cdot|z_t^k)} \left[ \log \pi(y_t|z_t^k) \langle \nabla_{z_t^k} \log \pi(y_t|z_t^k), \Delta z_t^k \rangle \right] \\
&= -\mathbb{E}_{y_t \sim \pi(\cdot|z_t^k)} \left[ \log \pi(y_t|z_t^k) \sum_{j=1}^{|V|} \frac{\partial \log \pi(y_t|z_t^k)}{\partial z_{t,j}^k} \Delta z_{t,j}^k \right]
\end{aligned}
$$
Based on the derivative of the log-softmax function, we have:
$$
\sum_{j=1}^{|V|} \frac{\partial \log \pi(y_t|z_t^k)}{\partial z_{t,j}^k} \Delta z_{t,j}^k = \sum_{j=1}^{|V|} (\mathbb{1}\{y_t=v_j\} - \pi(v_j|z_t^k)) \Delta z_{t,j}^k = \Delta z_{t,y_t}^k - \mathbb{E}_{v \sim \pi(\cdot|z_t^k)} [\Delta z_{t,v}^k]
$$
Therefore:
$$
\langle \nabla_{z_t^k} H_t(z_t^k), \Delta z_t^k \rangle = -\mathbb{E}_{y_t \sim \pi(\cdot|z_t^k)} \left[ \log \pi(y_t|z_t^k) \cdot (\Delta z_{t,y_t}^k - \mathbb{E}_{v \sim \pi(\cdot|z_t^k)} [\Delta z_{t,v}^k]) \right]
$$
Let the random variables be $X = \log \pi(y_t|z_t^k)$ and $Y = \Delta z_{t,y_t}^k$. The expression above is precisely the negative covariance of the two random variables, i.e.:
$$
H_t(z_t^{k+1}) - H_t(z_t^k) \approx -\text{Cov}_{y_t \sim \pi(\cdot|z_t^k)} (\log \pi(y_t|z_t^k), \Delta z_{t,y_t}^k)
$$

[1] further shows that under natural policy gradient, the change in entropy is related to advantage as follows:

$$
H_t(z_t^{k+1}) - H_t(z_t^k) \approx -\eta \cdot \text{Cov}_{y_t \sim \pi(\cdot|z_t^k)} (\log \pi(y_t|z_t^k), A^k(y_t, z_t^k)) \quad (12)
$$

where $\eta$ is the learning rate and $A^k(y_t, z_t^k)$ is the advantage of $y_t$ at the current state. In GRPO with binary rewards, positive samples have a positive advantage and negative samples have a negative one. This means that entropy decreases when updating on high-probability tokens from positive samples or low-probability tokens from negative samples.

Based on equation (12), [1] argues that update magnitude of high-covariance tokens should be limited, proposing `Clip-Cov` and `KL-Cov`. `Clip-Cov` primarily works by setting the gradients of identified high-covariance tokens to zero. `KL-Cov`, on the other hand, applies a KL-divergence constraint to identified tokens, preventing them from deviating too far from the original policy.

## 6\. A Unified Perspective

Fundamentally, the methods in \[1\], \[3\], \[4\], \[5\], and \[6\] all center on the entropy of high-entropy `forking tokens`. The approaches in \[4\], \[5\] and \[6\] mitigate the rapid decline in entropy by correcting the bias in GRPO against high-entropy(low-probability) tokens. In contrast, \[1\] alleviates rapid entropy decay by restricting the update magnitude of low-entropy(high-probability) tokens. Therefore, these approaches are theoretically compatible and could be combined, simultaneously increasing the update magnitude for high-entropy tokens while decreasing it for low-entropy ones.

\[7\] shifts the focus away from token-level entropy, proposing instead that the relative weight of negative samples should be increased. Interpreting this through the lens of \[1\]’s framework, since negative samples generated by model are unlikely to have extremely low probabilities, optimizing on these relatively high-probability negative samples also helps to counteract entropy collapse.

---

## References

\[1\]. [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/pdf/2505.22617)

\[2\]. [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/pdf/2505.24864)

\[3\]. [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/pdf/2506.01939)

\[4\]. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476)

\[5\]. [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/pdf/2506.13585)

\[6\]. [Reasoning with Exploration: An Entropy Perspective](https://arxiv.org/pdf/2506.14758)

\[7\]. [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](https://arxiv.org/pdf/2506.01347)

\[8\]. [No Free Lunch: Rethinking Internal Feedback for LLM Reasoning](https://arxiv.org/pdf/2506.17219)