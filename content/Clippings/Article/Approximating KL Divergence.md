---
author:
  - "[[John Schulman]]"
url: http://joschu.net/blog/kl-approx.html
updated: 2026-01-27T21:45
---

This post is about Monte-Carlo approximations of KL divergence.

$$
\mathrm{KL}[q, p] = \sum_x q(x) \log \frac{q(x)}{p(x)} = \mathbb{E}_{x \sim q} \left[ \log \frac{q(x)}{p(x)} \right]
$$

It explains a trick I've used in various code, where I approximate $\mathrm{KL}[q, p]$ as a sample average of $\frac{1}{2} (\log p(x) - \log q(x))^2$, for samples $x$ from $q$, rather than the more standard $\log \frac{q(x)}{p(x)}$. This post will explain why this expression is a good (though biased) estimator of KL, and how to make it unbiased while preserving its low variance.

Our options for computing KL depend on what kind of access we have to $p$ and $q$. Here, we'll be assuming that we can compute the probabilities (or probability densities) $p(x)$ and $q(x)$ for any $x$, but we can't calculate the sum over $x$ analytically.

Why wouldn't we be able to calculate it analytically?

* Computing it exactly requires too much computation or memory.
* There's no closed form expression.
* We can simplify code by just storing the log-prob, not the whole distribution. This is a reasonable choice if KL is just being used as a diagnostic, as is often the case in reinforcement learning.

The most common strategy for estimating sums or integrals is to use a Monte-Carlo estimate. Given samples $x_1, x_2, \dots \sim q$, how can we construct a good estimate?

A good estimator is unbiased (it has the right mean) and has low variance. We know that one unbiased estimator (under samples from $q$) is $\log \frac{q(x)}{p(x)}$. However, it has high-variance, as it's negative for half of the samples, whereas KL is always positive. Let's call this naive estimator $k_1$:

$$
k_1 = \log \frac{q(x)}{p(x)} = - \log r
$$

where we've defined the ratio $r = \frac{p(x)}{q(x)}$ that'll appear frequently in the subsequent calculations.

An alternative estimator, which has lower variance but is biased, is $\frac{1}{2} (\log \frac{p(x)}{q(x)})^2 = \frac{1}{2} (\log r)^2$. Let's call this estimator $k_2$:

$$
k_2 = \frac{1}{2} (\log r)^2
$$

Intuitively, $k_2$ seems to be better because each sample tells you how far apart $p$ and $q$ are, and it's always positive. Empirically, $k_2$ does indeed have much lower variance than $k_1$, and also has remarkably low bias. (We'll show this in an experiment below.)

There's a good reason why estimator $k_2$ has low bias: its expectation is an f-divergence.

An f-divergence is defined as $D_f(p, q) = \mathbb{E}_{x \sim q} [ f( \frac{p(x)}{q(x)} ) ]$ for a convex function $f$. KL divergence and various other well-known probability distances are f-divergences.

Now here's the key non-obvious fact: all f-divergences with differentiable $f$ look like KL divergence up to second order when $q$ is close to $p$. Namely, for a parametrized distribution $p_\theta$,

$$
D_f(p_0, p_\theta) = \frac{f''(1)}{2} \theta^T F \theta + O(\theta^3)
$$

where $F$ is the Fisher information matrix for $p_\theta$ evaluated at $p_\theta = p_0$.

$\mathbb{E}_q[k_2] = \mathbb{E}_q[\frac{1}{2}(\log r)^2]$ is the f-divergence where $f(x) = \frac{1}{2}(\log x)^2$, whereas $\mathrm{KL}[q, p]$ corresponds to $f(x) = -\log x$. It's easy to check that both have $f''(1) = 1$, so both look like the same quadratic distance function for $p \approx q$.

Is it possible to write down a KL divergence estimator that is unbiased but also low variance?

The general way to lower variance is with a control variate. I.e., take $k_1$ and add something that has expectation zero but is negatively correlated with $k_1$. The only interesting quantity that's guaranteed to have zero expectation is $\frac{p(x)}{q(x)} - 1 = r - 1$. So for any $\lambda$, the expression $-\log r + \lambda (r - 1)$ is an unbiased estimator of $\mathrm{KL}[q, p]$.

We can do a calculation to minimize the variance of this estimator and solve for $\lambda$. But unfortunately we get an expression that depends on $p$ and $q$ and is hard to calculate analytically.

However, we can choose a good $\lambda$ using a simpler strategy. Note that since log is concave, $\log(x) \le x - 1$. Therefore, if we let $\lambda = 1$, the expression above is guaranteed to be positive. It measures the vertical distance between $\log(x)$ and its tangent. This leaves us with the estimator $k_3$:

$$
k_3 = (r - 1) - \log r
$$

The idea of measuring distance by looking at the difference between a convex function and its tangent plane appears in many places. It's called a Bregman divergence and has many beautiful properties.

We can generalize the above idea to get a good, always-positive estimator for any f-divergence, most notably the other KL divergence $\mathrm{KL}[p, q]$ (note that $p$ and $q$ are switched here). Since $f$ is convex, and $\mathbb{E}_q[r] = 1$, the following is an estimator of the f-divergence: $f(r) - f'(1)(r - 1)$.

This is always positive because it's the distance between $f$ and its tangent at $r = 1$, and convex functions lie above their tangent lines. Now $\mathrm{KL}[p, q]$ corresponds to $f(x) = x \log x$, which has $f'(1) = 1$, leaving us with the estimator $r \log r - (r - 1)$.

In summary, we have the following estimators (for samples $x \sim q$, and $r = p(x)/q(x)$):

* $\mathrm{KL}[p, q]$: $r \log r - (r - 1)$
* $\mathrm{KL}[q, p]$: $(r - 1) - \log r$

Now let's compare the bias and variance of the three estimators for $\mathrm{KL}[q, p]$. Suppose $q = N(0, 1)$, $p = N(0.1, 1)$. Here, the true KL is 0.005.

| Estimator | Bias/True | Stdev/True |
| :--- | :--- | :--- |
| k1 | 0 | 20 |
| k2 | 0.002 | 1.42 |
| k3 | 0 | 1.42 |

Note that the bias of k2 is incredibly low here: it's 0.2%.

Now let's try for a larger true KL divergence. $p = N(1, 1)$ gives us a true KL divergence of 0.5.

| Estimator | Bias/True | Stdev/True |
| :--- | :--- | :--- |
| k1 | 0 | 2 |
| k2 | 0.25 | 1.73 |
| k3 | 0 | 1.7 |

Here, the bias of k2 is much larger. k3 has even lower standard deviation than k2 while being unbiased, so it appears to be a strictly better estimator.

Here's the code I used to get these results:

```python
import torch.distributions as dis

p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=0.1, scale=1)
x = q.sample(sample_shape=(10_000_000,))
truekl = dis.kl_divergence(p, q)

print("true", truekl)
logr = p.log_prob(x) - q.log_prob(x)
k1 = -logr
k2 = logr ** 2 / 2
k3 = (logr.exp() - 1) - logr

for k in (k1, k2, k3):
    print((k.mean() - truekl) / truekl, k.std() / truekl)