---
title: DPO 是如何简化 RLHF 的
author:
  - "[[朱小霖​]]"
url: https://zhuanlan.zhihu.com/p/671780768#/
created: 2026-01-25
tags:
  - LLM
  - RLHF
  - DPO
category: clippings/article
updated: 2026-01-25T22:36
---
> [!abstract] 摘要
> 最近通过 Mistral AI 7Bx8 模型的发布，我才后知后觉地了解到了 DPO（Direct Preference Optimization）这个算法，发现他用了一种很巧妙的思路，将 RLHF 的 2 阶段多个模型的训练简化为了 1 阶段的 SFT 训练。


## Notes

最近通过 Mistral AI 7Bx8 模型的发布，我才后知后觉地了解到了 DPO（Direct Preference Optimization）这个算法，发现他用了一种很巧妙的思路，将 RLHF 的 2 阶段多个模型的训练简化为了 1 阶段的 SFT 训练。在这里简单总结一下。

那么介绍 DPO 做了哪些简化之前，首先要提一下我们一般认为的 RLHF 是咋训练的。RLHF 一般会分 2 步:

- 第一步是训练 reward model。训练数据是同一个 prompt 的 2 个回答，让人或 GPT4 标注哪个回答更好，reward model 会去优化如下的 loss：

$$
\max_{r_{\phi}} \left\{ \mathbb{E}_{(x,y_{\text{win}},y_{\text{lose}}) \sim \mathcal{D}} \left[ \log \sigma \left( r_{\phi}(x,y_{\text{win}}) - r_{\phi}(x,y_{\text{lose}}) \right) \right] \right\}
$$

其中 $r_{\phi}$ 就是 reward model 用来给回答打分。$\mathcal{D}$ 是训练数据集，$x$ 是 prompt，$y_{\text{win}}$ 和 $y_{\text{lose}}$ 分别是好的回答和不好的回答。也就是说，要尽可能让好的回答的得分比不好的回答高，拉大他们之间的差别。

- 第二步是用 RL 算法来提升模型的得分。使用的 loss 是：

$$
\max_{\pi_{\theta}} \left\{ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ r_{\phi}(x,y) \right] - \beta D_{KL} \left[ \pi_{\theta}(y|x) \| \pi_{\text{ref}}(y|x) \right] \right\}
$$

其中 $\pi_{\theta}$ 是我们在训练的 LLM，$\pi_{\text{ref}}$ 是训练的初始值。这个 loss 意思是希望 LLM 输出的回答的评分能尽可能高，同时 $\pi_{\theta}$ 不要偏离 $\pi_{\text{ref}}$ 太多，保证它还能正常做回答，不要训成一个评分很高但是回答乱码的东西。

---

DPO 的作者们意识到，后面的这个式子是有显式解的。因为：

$$
\begin{aligned}
\max_{\pi_{\theta}} & \left\{ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ r_{\phi}(x,y) \right] - \beta D_{KL} \left[ \pi_{\theta}(y|x) \| \pi_{\text{ref}}(y|x) \right] \right\} \\
&= \max_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ r_{\phi}(x,y) - \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} \right] \\
&= \min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta} r_{\phi}(x,y) \right] \\
&= \min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x) e^{r_{\phi}(x,y)/\beta}} \right]
\end{aligned}
$$

如果我们归一化一下分母，即取 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) e^{r_{\phi}(x,y)/\beta}$，也就可以构造出一个新的概率分布：

$$
\pi^*(y|x) = \frac{\pi_{\text{ref}}(y|x) e^{r_{\phi}(x,y)/\beta}}{Z(x)}
$$

那么上式变成了：

$$
\begin{aligned}
\min_{\pi_{\theta}} & \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x) e^{r_{\phi}(x,y)/\beta}} \right] \\
&= \min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi^*(y|x)} - \log Z(x) \right] \\
&= \min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi^*(y|x)} \right] \\
&= \min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}} D_{KL}(\pi_{\theta}(y|x) \| \pi^*(y|x))
\end{aligned}
$$

由于 KL 散度在 2 个分布相等时取最小值，我们得到了这样的结论：RLHF 训练希望得到的最优的概率分布就是 $\pi^*$。

另一个角度来说，由 $\pi^*$ 的公式，我们相当于是得到了 $r_{\phi}$ 和 $\pi^*$ 的关系，那么是否我们可以把训练 $r_{\phi}$ 转化成直接去训练 $\pi^*$ 呢？

简单转换一下 $\pi^*$ 的定义式，可以得到：

$$
r_{\phi}(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

带入最上面优化 $r_{\phi}$ 的 loss，也就有了：

$$
\max_{\pi^*} \left\{ \mathbb{E}_{(x,y_{\text{win}},y_{\text{lose}}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi^*(y_{\text{win}}|x)}{\pi_{\text{ref}}(y_{\text{win}}|x)} - \beta \log \frac{\pi^*(y_{\text{lose}}|x)}{\pi_{\text{ref}}(y_{\text{lose}}|x)} \right) \right] \right\}
$$

或者说，我们可以直接用这个 loss 去求 $\pi_{\theta}$：

$$
\max_{\pi_{\theta}} \left\{ \mathbb{E}_{(x,y_{\text{win}},y_{\text{lose}}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{\text{win}}|x)}{\pi_{\text{ref}}(y_{\text{win}}|x)} - \beta \log \frac{\pi_{\theta}(y_{\text{lose}}|x)}{\pi_{\text{ref}}(y_{\text{lose}}|x)} \right) \right] \right\}
$$

这就是 DPO 的 loss。DPO 通过以上的公式转换把 RLHF 无损地转化为了 SFT，在训练的时候不再需要同时跑 4 个模型（reward model, ref model, critic, actor），而是只用跑 actor 和 ref 2 个模型，甚至由于不再在线采数据，ref model 的输出可以预先存下来，训练的时候重复使用。
