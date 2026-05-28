---
title: RLHF 常见的思维误区
author:
  - "[[ybq​]]"
url: https://zhuanlan.zhihu.com/p/17657567877#/
created: 2026-01-25
tags:
  - LLM
  - RLHF
  - openai-o1
category: clippings/article
updated: 2026-01-27T12:23
---
---

本文分享下我在学习和实践 [RLHF](https://zhida.zhihu.com/search?content_id=252553554&content_type=Article&match_order=1&q=RLHF&zhida_source=entity) 时，曾经陷入过的一些思维误区。这些误区的产生大多和我的强化基础知识理解不到位有关，我建议非强化出身的同学都耐心读读下面这些文章。

-   RLHF 圣经：[猛猿：人人都能看懂的RL-PPO理论知识](https://zhuanlan.zhihu.com/p/7461863937) - [[人人都能看懂的RL-PPO理论知识]]
-   RLHF 必读论文：[GAE](https://zhida.zhihu.com/search?content_id=252553554&content_type=Article&match_order=1&q=GAE&zhida_source=entity) [https://arxiv.org/pdf/1506.02438](https://arxiv.org/pdf/1506.02438) 和 PPO [arXiv reCAPTCHA](https://arxiv.org/pdf/1707.06347)

---

### RLHF 代码流程（背景知识）

***Input：***prompt，sft\_model，reward\_model

***Initialize：*** actor\_model / ref\_model = sft\_model，critic\_model / reward\_model = reward\_model

***Training：***

step1：生产计算 loss 的中间数据（每次 rollout\_batch\_size 条数据）：

1.  actor\_model generate，得到 prompt + response
2.  reward\_model predict， 得到 reward，需要 clip 到一个区间内；
3.  reference\_model / actor\_model forward，得到两个 log\_probs，分别是 $\pi_{ref}$ 和 $\pi_{old}$ ，计算 KL\_penalty；
4.  critic\_model forward，得到 values ，此处记作 $v_{old}$ ；
5.  reward - KL\_penalty：得到修正后的 reward（此处是 **reference\_model** 的生效位置）；
6.  reward 和 values 反传，利用 PPO 论文中的计算公式，得到 advantages 和 returns（此处是 **GAE** 的生效位置）。

step2：更新 loss（每次 train\_batch\_size 条数据，反复调用，直到 step1 的数据用尽）：

1.  actor\_model forward，又得到一个 log\_probs，这里的 actor\_model 是 $\pi_{\theta}$ ，与 step1 中得到的 $\pi_{old}$ 和 advantages 一起计算 loss（此处是 **Importance Sampling** 和 **CLIP** 的生效位置）；
2.  actor\_model backward；
3.  critic\_model forward，又得到一个 values，此处记作 v，与 step1 中得到的的 $v_{old}$ 和 returns 一起计算 loss，引入 $v_{old}$ 进行 clip，防止 critic\_model 的更新幅度太大；
4.  critic\_model backward。

  

RLHF 训练流程的难点主要集中在以下几个问题：

-   PPO 的处理技巧：CLIP，GAE，Importance Sampling
-   advantages 和 returns 是怎么计算得到的？
-   凭什么 advantages 和 returns 可以作为 actor\_model 和 critic\_model 的优化目标？
-   整个训练过程为什么存在三个 policy 模型： $\pi_{ref}  \quad \pi_{old} \quad \pi_{\theta}$

回答不上来的同学建议好好读下前面推荐的文章，然后死磕下 [OpenRLHF](https://zhida.zhihu.com/search?content_id=252553554&content_type=Article&match_order=1&q=OpenRLHF&zhida_source=entity) 的源码。

### RLHF 不等于 PPO

RLHF 的含义是“通过强化学习的训练方式，利用人类反馈来优化语言模型”，其和 PPO 是不能完全划等号的，二者的区别主要在于：

-   PPO 仅仅是 OpenAI 最喜欢的强化学习训练方法，其他强化学习训练方法也可以；
-   Critic\_model 是 PPO 需要的模块，并非 RLHF 必备模块，如果 RL 算法是 actor-critic 系列，那么就会引入 critic\_model，如果是 [REINFORCE](https://zhida.zhihu.com/search?content_id=252553554&content_type=Article&match_order=1&q=REINFORCE&zhida_source=entity) 系列，那么就可以省掉 critic\_model；
-   Reference\_model 是 RLHF 提出的概念，目标是防止语言模型在训练中崩溃，和 PPO 没有任何关系；
-   Reward\_model 也是 RLHF 提出的概念，目标是自动生产训练数据，同样和 PPO 没有关系，其他生产数据的方法（比如 [verifier](https://zhida.zhihu.com/search?content_id=252553554&content_type=Article&match_order=1&q=verifier&zhida_source=entity) ）也可以。

这也就是说，虽然我们熟知的 RLHF 是 4 个模型组成的，但实际上只需要准备好一个微调后的 sft\_model 即可启动，毕竟 reference\_model 与 sft\_model 是同一个模型，critic\_model 和 reward\_model 并非必备模块。

一言以概之**，RLHF = LLM + 任意 RL 算法 + 数据打分工具**。

### RL 的常用技巧并非 RLHF 必备

强化学习和传统监督学习一个很大的区别就是“训练数据是当场采集出来的”，**一边训模型，一边造数据**。在传统的强化学习任务中，训练数据的生产是很困难的，比如下完一盘围棋、打完一盘马里奥游戏 …… 往往是十几分钟产生一条训练数据（trajectory），但不到一秒就训完了。然而，在 RLHF 的场景下，训练数据还真就不难生产，生产 1 条 repsonse 和训练 1 条 response 还真不一定谁更快（不过 1 次生产 N 条 response 确实快于 N 次生产 1 条 response）。

下面我就围绕重要性采样这一技巧来展开讲讲。RL 算法引入重要性采样这一概念，其目的是：可以一次性生产多条数据，或者说让生产的数据可以反复使用。

这里，有同学会疑惑为什么“生产的数据能反复使用”还需要特殊处理，这不是天经地义的吗？问题出就出在我们想进行的是 on-policy 的训练方法。假设我用 llm 生产了两条 response，训练完第 1 条 response 后，我便会得到一个 llm\_1，此时这个 llm\_1 根本就说不出来第 2 条 response，那么第 2 条数据原则上就不能再使用了，硬要使用就需要引入重要性采样。

此外，重要性采样还有稳定训练的作用，引用一下[[猛猿]]的科普：由于采样具有随机性，on-policy 算法采样出的 bacth 可能会把 actor 往错误的方向更新。比较谨慎的想法是，每次更新不要离 actor\_old 太远（在信任域内）。那实现这个想法的方式之一，就是我拿 old 产出一波数据，分成 K 份，吃每份做更新时，我去参考 old 的结果调整更新方向，相当于 new 在 old 上做了 K 次验证，使得整个过程有“探索-利用”的意味，这也就是 PPO 前身 [TRPO](https://zhida.zhihu.com/search?content_id=252553554&content_type=Article&match_order=1&q=TRPO&zhida_source=entity) 的初衷。

$\mathbb{E}_{x \sim p(x)}[f(x)] = \int p(x) f(x) \, dx \\ = \int \frac{p(x)}{q(x)} q(x) f(x) \, dx \\ = \mathbb{E}_{x \sim q(x)}\left[\frac{p(x)}{q(x)} f(x)\right]$

重要性采样有一个重大缺陷：当 p(x) 和 q(x) 差异较大时，需要通过足够多的采样来抵消这种差异对期望的最终影响，可以执行这段代码自行体验，采样少了就是不准。

```python3
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, exp
np.random.seed(1234)

mu_p, sigma_p = -0.5, 0.5
def p(x):
    return 1/(sqrt(2*pi)*sigma_p) * exp(-((x - mu_p)**2)/(2*sigma_p**2))

mu_q, sigma_q = 1.5, 0.8
def q(x):
    return 1/(sqrt(2*pi)*sigma_q) * exp(-((x - mu_q)**2)/(2*sigma_q**2))

def f(x):
    return 1 / (1 + exp(-x)) - 0.5

xs = np.linspace(-3, 3, 200)
pxs = [p(x) for x in xs]
qxs = [q(x) for x in xs]
fxs = [f(x) for x in xs]

plt.figure(figsize=(8,4))
plt.plot(xs, pxs, label='$p(x)$', color='blue')
plt.plot(xs, qxs, label='$q(x)$', color='green')
plt.plot(xs, fxs, label='$f(x)$', color='red')
plt.ylim(-0.5, 1)
plt.legend()
plt.title('Importance Sampling')
plt.show()

samples = np.random.normal(loc=mu_p, scale=sigma_p, size=1000000)
mean_fp = np.mean([f(x) for x in samples])
print(f'在p(x)上采样，期望为：{mean_fp}')

samples = np.random.normal(loc=mu_q, scale=sigma_q, size=1000000)
mean_fq = np.mean([f(x) for x in samples])
print(f'在q(x)上采样，期望为：{mean_fq}')

for size in [10 ** i for i in range(0, 8)]:
    samples = np.random.normal(loc=mu_q, scale=sigma_q, size=size)
    mean_is = np.mean([p(x) / q(x) * f(x) for x in samples])
    print(f"采集{str(size).zfill(8)}个样本点之后，重要性采样的期望为：{mean_is}")
```

![图片](https://pic1.zhimg.com/v2-2b06f0dd585be9be8a2b8bb5133c87c0.jpg)

重要性采样

言归正传，RLHF 虽然也是 on-ploicy 训练，但我们却不需要太过于依赖重要性采样。

先普及两个 RLHF 算法中的重要参数：**rollout\_batch\_size** 和 **train\_batch\_size**，前者代表一次性生成多少条训练数据（response 和 reward），后者代表每次用多少条数据来更新模型，前者是后者的 N 倍。

随着训练框架的不断优化， RLHF 的训练数据并没有那么难生产了，尤其是像 OpenRLHF 这种框架，引入了 [vllm](https://zhida.zhihu.com/search?content_id=252553554&content_type=Article&match_order=1&q=vllm&zhida_source=entity) 来生产 response，效率极高。我们完全可以令 N = 1 / 2 / 4 这种很小的值，且每条训练数据仅使用一次。事实上，由于重要性采样需要大量样本才能做到无偏替代，这个 N 值还真不能很大，越大就越容易训崩。

> 一个比较坑的点是，OpenRLHF 在 forward 计算 log\_probs 时，micro\_batch\_size 取不同的值会导致计算结果也不同。这并不怎么影响模型训练，但是会在 debug 的时候让人疑惑，为什么重要性采样的 ratio 最初不是 1 呢？下图里的 old\_action\_log\_probs 是 batch=1 计算所得，action\_log\_probs 是 batch=2 计算所得

![图片](https://pic2.zhimg.com/v2-d4e869ad214ae0dc00145bb54842d2e9.jpg)

batch\_size 不同，结果不同

综上，**RL 的很多技巧，是围绕着“训练数据不好生产”，“训练数据量太少”，“训练数据波动太大”，“训练数据分布不均”等出发点提出的**。在 RLHF 中使用这些技巧的时候，不妨多思考一下：语言模型场景下是否也有这些痛点？例如，PPO 的 clip 操作，会导致这条数据没有梯度，这条训练数据就起不到贡献了。真的必须这样吗？我能不能换一种 clip 操作，既 clip 了又能保留梯度，说不定就更适合 LLM 的训练呢？

$L^{CLIP}(\theta) = \widehat{\mathbb{E}}_t \left[  \min \left(  \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \widehat{A}_t,  \text{clip}\left(\frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}, 1 - \epsilon, 1 + \epsilon\right) \widehat{A}_t  \right)  \right]$

### 省去 Critic model 未必更好

作为一个 NLPer，我在学习 RLHF 的时候，是先学 PPO，再接触 GRPO、RLOO 等优化算法。这就导致我产生了一种思维误区：“PPO 的 critic model 让训练显得臃肿，省掉 critic model 的算法更好更先进”。

实际上，在 RL 算法的发展史上，是先有 REINFORCE 这种没 critic model 的算法，因为效果不佳或训练不稳定，才出现了 PPO 这种有 critic\_model 的算法。因此，GRPO、RLOO 这种省略 critic\_model 的 RLHF 算法，它们本质上都是 REINFORCE 算法的变种，而 PPO 则属于 Actor-Critic 算法的范畴。

在分析这些算法之前，我们先复习一下 GAE 这篇论文讲述的内容，熟悉的同学则可以跳过。

对我而言，GAE 这篇论文最大的贡献不在于 GAE 算法本身，而是下面这段内容背后的道理：**策略梯度可以是任意形式的函数**。

![图片](https://pic2.zhimg.com/v2-a6f5918eb02b39103be705d681ae552b.jpg)

Policy Gradient

说的更直白点：强化学习的 policy\_model 在优化的时候，其优化目标没有固定形式，make sense 即可：

-   reward，可以
-   reward - baseline，可以
-   Q 函数，可以
-   V 函数，可以
-   Advantage 函数，可以
-   TD-error，可以
-   ……

这个优化目标的设计，就像是神经网络激活函数的选取一样，Relu 可以，0.5 \* Relu 也可以，Relu + 0.5 还可以…… 只要保证你新设计的函数，其单调性和原来的函数是一样的，那就可以代替原函数。所以，不要去纠结为什么这么设计优化目标，只要能让模型收敛就是一个好的优化目标（大概率是为了稳定方差，让分布更加均匀）。

有了这个认知，我们就能把 policy-based rl 算法分为三类：

-   REINFORCE：让采集到的 Reward 最大化来优化策略；
-   REINFORCE with (constant) baseline：使用一个常数作为基线，将 Reward 减去该基线来优化策略；
-   REINFORCE with learned baseline / Actor-Critic：使用一个学习得到的基线（例如值函数），将 Reward 减去该基线来优化策略。

PPO 就是第三种，用 Advantage 来优化策略， $A_{\pi}(s_t, a_t) = Q_{\pi}(s_t, a_t) - V_{\pi}(s_t)$ ，其中， $Q_{\pi}(s_t, a_t)$ 是通过样本的 Reward 和 $V_{\pi}(s_t)$ 共同估算出来的， $V_{\pi}(s_t)$ 是 critic\_model 估算出来的。 Advantage 的实际计算过程，是通过 GAE 算法完成的，GAE 的动机是： **Critic\_model 可能不准（偏差大），采样的结果随机性又很大（方差大），因此在计算优势函数时引入一个参数** $\lambda$ **，来控制是更相信实际采样得到的 Reward，还是更相信 Critic\_model 的结果**。具体细节不再赘述了，大家详看猛猿小姐姐的圣经。

至于其他 RLHF 算法，无一例外都是第二种算法，寻找一个不用学习的 baseline，也就是：

$\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ \left( R(y, x) - b \right) \nabla_\theta \log \pi_\theta(y \mid x) \right]$

下面简单展开说下每个优化算法的原理，公式取自每个算法的原论文，所以符号不是很一致。

**REINFORECE++**：没有 baseline，引入所有 PPO 的 trick（初七大佬说 KL\_penalty 是 baseline，但我更倾向于把 KL\_penalty 看作是 Reward 的一种修正，毕竟其他 RLHF 算法都是既有 KL\_penalty 又有 baseline）：

$A_t(s_t, a_t) = r(x, y) - \beta \cdot \sum_{i=t}^T \text{KL}(i)$

**RLOO（REINFORCE leave-one-out）**：一次生成多条 response，每条 response 用其他 response 的 Reward 均值作为 baseline：

$\frac{1}{k} \sum_{i=1}^k \left[ R(y_{(i)}, x) - \frac{1}{k-1} \sum_{j \neq i} R(y_{(j)}, x) \right] \nabla \log \pi(y_{(i)} \mid x) \quad \\ \text{for} \quad y_{(1)}, \dots, y_{(k)} \overset{\text{i.i.d.}}{\sim} \pi_\theta(\cdot \mid x)$

**GRPO（Group related policy Optimization）**：一次生成多条 response，每条 response 用所有 response 的 Reward 均值作为 baseline：

$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}$

**ReMAX（REINFORCE argmax）**：用 greedy decode 的结果作为 baseline：

$$\tilde{g}(\theta) = \sum_{i=1}^N \sum_{t=1}^T \left[ \nabla_\theta \log \pi_\theta(a_t^i \mid x^i, a_{1:t-1}^i) \cdot \left( r(x, a_{1:T}^i) - r(x, \bar{a}_{1:T}^i) \right) \right] \\ a_t^i \sim \pi_\theta(\cdot \mid x^i, a_{1:t-1}^i) \quad \bar{a}_t^i \in \arg\max \pi_\theta(\cdot \mid x^i, \bar{a}_{1:t-1}^i)  $$

  

关于 critic\_model 的大背景终于说完了，可以开始讨论下到底能不能省掉 critic\_model 了？

其实不太好说，critic\_model 毕竟是一个 token 粒度的评估模型，能对 response 的中间过程起到一个很好的监控作用。但是大部分的 math 数据集，好像也不太在乎中间过程，结果对就是高 reward，结果错就是低 reward，这时候 critic\_model 可能就没太大意义。

然而，就算 critic\_model 本身没意义，也不代表说 RLOO / GRPO 算法可以更好的取代 PPO，这些算法强调的省去 critic\_model 的显存只对算力稀缺的同学有意义。以 OpenRLHF 为例，ray 会把 critic\_model 放在一台单独的机器上，只需要多加一台机器就不存在 OOM 的问题，而工业界做 RLHF 往往都是 4 / 8 / 10 / 20 机这样的资源数，不是很在乎 critic\_model 占据的资源。反倒是生成多条 response，可能会严重降低训练速度。

所以，是否使用 GRPO / RLOO，出发点应该是目标任务是否需要一次性生成多个 response ，而不是要省资源，例如训 o1 可能就需要模型一次性生成多条 cot 路径。真正做到省资源且训得快的应该是 REINFORCE ++。

> 个人观点：各 RLHF 变种算法的论文，都论证了其算法相较于 PPO 的优异性，但其实未必公平。on-policy 的算法是很容易训崩溃的：采样数量不够，explore 到 reward model OOD 的样本，critic 和 actor 的收敛时机不 match 等很多细节都会让模型表现不佳。我总感觉 ablation study 的时候，一些工作只对自己提出的算法做了调参优化，而没有对 baseline 算法进行同等精力的调参。不过考虑另外一个角度，对参数的容错率高，可能也是某个算法的优势所在。只要能让自己的模型训练不崩溃，就是一个好的 RLHF 算法。

### Reward\_model 非必须

Reward\_model 是很难训练的，如果 prompt 没有出现在 reward\_model 的训练集合中，那它就是打不准，也就是 out of distribution。基本上可以认为，reward\_model 只能给它见过的 prompt 进行较准确的打分。同理，如果见过一个 prompt，但是没见过相似的 response 且找不到判别的技巧，也会 OOD。就比如，小说生成创作任务，reward\_model 不好训；作诗任务，可以通过判断韵脚来打分，reward\_model 好训。

这也是为什么国内的 RLHF 普遍都是在 math / code 进行大量使用的原因之一，训练数据易于大量生产（结果是否正确，代码是否能执行），也就不太容易让 reward\_model OOD。

因此用 verifier 取代 reward\_model 就是一个训 RLHF 很好用的技巧：首先保证每一条 prompt 都有一个 gold\_answer，然后 verifier 的职责就是信息抽取，**判断 response 是否包含 gold\_answer**。这个任务是很简单的，一个 7B 左右的模型就能很好胜任，也基本不会 OOD （外接 GPT4o 作为 verifier 是一个很流行的做法）。generate\_reward\_model （通过 cot 来判断 response 的质量）则可以视为是一种没有 gold\_answer 的 verifier，除了训练有点慢之外它没有缺点。

不过 verifier 也不全都是优点，它的输出结果毕竟是离散的，引用一下

[@真中合欢](//www.zhihu.com/people/bf1764dccc55b8f831b89c9103f41564)

说过的话：“RLHF 用 llm 做 verifier 直接产生奖励信号，或者直接使用基于规则的奖励信号也可以，这种叫做稀疏奖励，用 reward 的叫稠密奖励。深度学习一路走来的经验就是，稠密通常优于稀疏，soft 通常优于 hard。一般认为稠密，soft 的信号包含更多的信息。”

由于 Critic\_model 是需要用 reward\_model 进行初始化的，那我们没有 reward\_model 怎么办，两个方案：

1.  用上文介绍过的 GRPO / RLOO / ReMAX / REINFORCE++ 算法；
2.  OpenRLHF 代码中给的解决方案是：冻结 actor\_model 一段时间，先让 critic\_model 学习一段时间（[actor-critic 算法](https://zhida.zhihu.com/search?content_id=252553554&content_type=Article&match_order=1&q=actor-critic+%E7%AE%97%E6%B3%95&zhida_source=entity)中，两个模型是有对抗关系的，先保护一下不太准的 critic\_model）。

前段时间炒的比较热的一个点是强化微调（RFT），其核心点总共两个，一个是 base 模型需要是 o1 这种有较强推理能力的模型，另一个就是用 verifier 来进行 RL 训练，来定向增强模型的领域能力。

---

暂时就写到这里吧，后面意识到更多的思维误区我再更新。之前我以为我只要会用 trl / deepspeed-chat / OpenRLHF 这种开源 RLHF 框架进行训练就够了，基本只看了 PPO 相关的知识。事实证明这样是不对的，老老实实按照 RL 发展史的顺序去学习，搞懂 TD-error，GAE 这种基础概念，很多认知都会焕然一新的。

我和

[@真中合欢](//www.zhihu.com/people/bf1764dccc55b8f831b89c9103f41564)

最近在尝试沿着 RLHF 的技术路线来复现 o1，如果有对 o1 感兴趣且愿意卷的从业人士，可以私信我们。利用周末 / 过年的时间，大家用开源模型、开源数据做点实验，在不涉及各自公司机密的情况下交流认知，一起茁壮成长。