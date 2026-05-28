---
title: dpo 的局限性
author:
  - "[[ybq​]]"
url: https://zhuanlan.zhihu.com/p/1082394115#/
created: 2026-01-25
tags:
  - LLM
  - DPO
  - RLHF
category: clippings/article
updated: 2026-01-25T22:35
---
> [!abstract] 摘要
> 最近又认真读了一遍大佬的文章： 朱小霖：DPO 是如何简化 RLHF 的 。不由感叹，数学不愧是 AI 技术的第一生产力，但凡我当初对这个证明过程多深入思考一会儿，也不至于踩那么多坑。


## Notes

最近又认真读了一遍大佬的文章： [朱小霖：DPO 是如何简化 RLHF 的](https://zhuanlan.zhihu.com/p/671780768) 。不由感叹，数学不愧是 AI 技术的第一生产力，但凡我当初对这个证明过程多深入思考一会儿，也不至于踩那么多坑。

---

## 理论证明

我先把大佬的证明过程复述一遍，来一个保姆级翻译。首先请牢记下面这 3 个 loss 函数，$r_{\phi}$ 是 reward_model，$\pi_{\theta}$ 是我们要优化的模型，$\pi_{\text{ref}}$ 是 dpo 和 ppo 都用到的 reference_model。

**reward_model loss**

$$
\max_{r_{\phi}} \left\{ \mathbb{E}_{(x,y_{\text{win}},y_{\text{loss}}) \sim \mathcal{D}} \left[ \log \sigma \left( r_{\phi}(x, y_{\text{win}}) - r_{\phi}(x, y_{\text{loss}}) \right) \right] \right\}
$$

**ppo loss**

$$
\max_{\pi_{\theta}} \left\{ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ r_{\phi}(x,y) \right] - \beta D_{KL} \left[ \pi_{\theta}(y|x) \| \pi_{\text{ref}}(y|x) \right] \right\}
$$

**dpo loss**

$$
\max_{\pi_{\theta}} \left\{ \mathbb{E}_{(x,y_{\text{win}},y_{\text{loss}}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{\text{win}}|x)}{\pi_{\text{ref}}(y_{\text{win}}|x)} - \beta \log \frac{\pi_{\theta}(y_{\text{loss}}|x)}{\pi_{\text{ref}}(y_{\text{loss}}|x)} \right) \right] \right\}
$$

**KL 散度**

$$
D_{KL}(P \| Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)
$$

针对 ppo 的 loss 函数，我们做以下变换：

- 代入 KL 散度得（$y \sim \pi_{\theta}(y|x)$ 提取放到了左下角）：

$$
\max_{\pi_{\theta}} \left\{ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ r_{\phi}(x,y) - \beta \log\frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} \right] \right\}
$$

- 乘上 $-\frac{1}{\beta}$，把 max 变成 min 得：

$$
\min_{\pi_{\theta}} \left\{ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta} r_{\phi}(x,y) \right] \right\}
$$

- 等价变换得：

$$
\min_{\pi_{\theta}} \left\{ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} - \log e^{\frac{1}{\beta} r_{\phi}(x,y)} \right] \right\}
$$

- 根据 $\log(\frac{A}{B}) = \log A - \log B$ 得：

$$
\min_{\pi_{\theta}} \left\{ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x) e^{r_{\phi}(x,y) / \beta}} \right] \right\}
$$

这里我们可以构造出一个新的概率分布：

$$
\pi^{*}(y|x) = \frac{\pi_{\text{ref}}(y|x) e^{r_{\phi}(x,y) / \beta}}{Z(x)}
$$

进行分母归一化，为的是保证分布的概率和等于 1，分母：

$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y|x) e^{r_{\phi}(x,y) / \beta}
$$

- 继续对 ppo 的 loss 函数等价变换得：

$$
\min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\frac{\pi_{\text{ref}}(y|x) e^{r_{\phi}(x,y) / \beta}}{Z(x)} Z(x)} \right]
$$

- 化简得：

$$
\min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi^{*}(y|x)} - \log Z(x) \right]
$$

- 由于 $\pi_{\theta}$ 和 $Z(x)$ 完全没有关系，可以省略，得：

$$
\min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi^{*}(y|x)} \right]
$$

- 代入 KL 散度得（左下角的 $y \sim \pi_{\theta}(y|x)$ 没有了）：

$$
\min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}} D_{KL}(\pi_{\theta}(y|x) \| \pi^{*}(y|x))
$$

由于 KL 散度的非负性质，$\pi_{\theta}(y|x)$ 的最优解就是 $\pi^{*}(y|x)$，PPO 的最优概率分布就是 $\pi^{*}$。

**也就是说，在已知 Reward_model 的参数 $r_{\phi}$ 的情况下，我们可以求得 PPO 的最优解 $\pi^{*}$；那如果我们已知 PPO 的最优解 $\pi^{*}$，是不是也可以反向求解 $r_{\phi}$ 呢？**

转换一下 $\pi^{*}$ 的定义式可以得到：

$$
\frac{\pi^{*}(y|x) Z(x)}{\pi_{\text{ref}}(y|x)} = e^{r_{\phi}(x,y) / \beta}
$$

等价变换得：

$$
r_{\phi}(x,y) = \beta \log\frac{\pi^{*}(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

把 $r_{\phi}(x,y)$ 的等价表达代入到 Reward_model 的 loss 函数：

$$
\begin{aligned}
\text{reward\_model\_loss} &= \max_{\pi^{*}} \left\{ \mathbb{E}_{(x,y_{\text{win}},y_{\text{loss}}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi^{*}(y_{\text{win}}|x)}{\pi_{\text{ref}}(y_{\text{win}}|x)} + \beta \log Z(x) - \beta \log \frac{\pi^{*}(y_{\text{loss}}|x)}{\pi_{\text{ref}}(y_{\text{loss}}|x)} - \beta \log Z(x) \right) \right] \right\} \\
&= \max_{\pi_{\theta}} \left\{ \mathbb{E}_{(x,y_{\text{win}},y_{\text{loss}}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{\text{win}}|x)}{\pi_{\text{ref}}(y_{\text{win}}|x)} - \beta \log \frac{\pi_{\theta}(y_{\text{loss}}|x)}{\pi_{\text{ref}}(y_{\text{loss}}|x)} \right) \right] \right\} \\
&= \text{dpo\_loss}
\end{aligned}
$$

至此，艺术已成！致敬大佬朱小霖。

---

## dpo 的局限性

经过上面的证明，一切都清晰了，**dpo 对标的从来都是不是 ppo，而是 reward_model。** 二者不仅训练数据一样，loss 函数本质上也一致。那么即使不去做实验验证，dpo 的局限性也已经跃然纸上了。

### dpo 的大前提未被验证

dpo 从头到尾都在以 reward_model 的方式让模型学习 evaluate 能力，但是却并没有证明一个重要假设："**模型的 evaluate 能力和 generate 能力到底是不是相互促进的？**" dpo 后的模型具有了更强的 evaluate 能力，但我们的目标是提升模型的 generate 能力啊。如果这个基本假设不成立，那 dpo 的学习过程就没有什么价值。

不要认为这个假设是显而易见的，美食家并不一定做得一手好饭。这个大前提的成立与否，还关系到 SPIN、self-reward 等左脚踩右脚的训练方式是否有存在的意义。

也正是因为 dpo 是在让模型具有 reward_model 的能力，所以它并不在乎模型能不能说出一个好的句子，只在乎 loss margin 是否在变大。大家训练 dpo 的时候，基本都遇到过 good_sentence 和 bad_sentence 的 loss 都上升的尴尬现象，往往需要我们加系数和调参数才能解决。

reward_model 的训练方式根本不在乎模型的 generate 能力，因此稳定训练的 dpo 需要魔改 loss 函数。

### dpo 缺乏在线采样数据

我们再从另外一个角度来思考 dpo 的不足，已知：

- dpo 和 reward_model 完全等价；
- rlhf = reward_model + ppo。

可得：**ppo 所做的一切操作，便是 dpo 效果不如 rlhf 的原因**。

那我们思考一下，ppo 到底做了什么？ppo 没有使用任何训练数据，它唯一做的事情便是 generate，通过对模型 generate 的文本进行打分，把 reward_model 的 evaluate 能力转换成 generate 能力喂给模型。因此，缺乏的这个 generate 的过程就是 dpo 效果不行的原因。那这个 generate 的过程又带来了什么呢？online 和 explore。

老生常谈的一个知识点了：rlhf 是 online 学习方式，dpo 是 offline 学习方式。rlhf 是修复模型当前已有的知识，属于因材施教，并且由于 do_sample 的随机性，它可以进行 explore；但 dpo 则是强制模型学习训练者认为正确的知识（偏好数据对中的 good sentence），沿着一条被设定的正确的路线使劲走，这种 offline 的学习方式也导致它不具备 explore 的空间。

我们在 dpo 训练中常用的一个技巧：让模型先对"偏好数据对的 good sentence"做一次 sft，再进行 dpo 训练。不就是强行让 dpo 变成 online 的学习方式吗？毕竟训过的知识大概率也是可以 generate 出来的。

我们在 dpo 数据中常用的一个技巧：把模型自己生成的 pass@N 结果，拿来构造成偏好 pair 对，不就是同时在模拟 ppo 的 online 和 explore 能力吗？

因为欠缺，所以需要弥补。所有的 dpo 数据处理技巧，大多都能从 ppo 身上找到一些痕迹。
