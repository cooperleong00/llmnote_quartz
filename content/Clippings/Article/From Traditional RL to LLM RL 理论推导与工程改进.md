---
title: From Traditional RL to LLM RL 理论推导与工程改进
author:
  - "[[extreme1228AI]]"
url: https://zhuanlan.zhihu.com/p/1997363850849300572
created: 2026-02-26
tags:
category: clippings/article
updated: 2026-02-26T22:20
---


过去半年到一年时间，自己也算是在[LLM RL](https://zhida.zhihu.com/search?content_id=269378271&content_type=Article&match_order=1&q=LLM+RL&zhida_source=entity)领域的一个科研工作者。自从25年年初[DeepSeek-R1](https://zhida.zhihu.com/search?content_id=269378271&content_type=Article&match_order=1&q=DeepSeek-R1&zhida_source=entity)横空出世后，LLM RL就变成了一个非常火爆的方向，与此同时基于GRPO改进的各种XXPO方法如“雨后春笋”般涌出，大家都在不同的方向尝试应用RL来取得更好的效果。我是25年年初刚开始接触LLM，之前对于RL的理解也不是很深刻，中间断断续续学习了好几次这其中的理论推导，同时也实际上手尝试了LLM RL理解了一些工程上的问题，特此总结，希望能够帮助初学者从零理解这其中的理论推导和工程改进。

## 从零理解传统RL的理论推导

强化学习是一种无监督机器学习方法，它的核心思想就是不断与环境交互并利用环境的反馈来调整模型的行为，进而能够更好地解决对应的问题。这种思想和人类学习新东西的过程类似，因此理论上具备比较强的泛化性，但通常相比传统的监督学习方法也更难以收敛，奖励信号是否准确，参数更新是否平稳，交互环境是否稳定等都影响着RL训练的收敛程度。

RL按照模型建模的对象分为Value-based和Policy-Based两个大方向，因为LLM领域中action space及其庞大，所以目前value-based方法不属于主流，这里主要以介绍Policy-Based方法为主（想从头开始了解RL的推荐参考西湖大学赵世钰老师的这门课程[强化学习的数学原理](https://www.bilibili.com/video/BV1sd4y167NS/?spm_id_from=333.337.search-card.all.click&vd_source=ba78eb27a55cbb3bb35bff17ca84ef3f)）。

为方便理解，我们以Policy-Based中最典型的[Reinforce算法](https://zhida.zhihu.com/search?content_id=269378271&content_type=Article&match_order=1&q=Reinforce%E7%AE%97%E6%B3%95&zhida_source=entity)为例进行推导，首先Reinforce算法的目标函数是最大化当前策略函数下所有轨迹的奖励期望： $$$\max_{\theta} L(\theta) = \mathbb{E}_{\tau \sim \pi(\cdot|\theta)} [R(\tau)]$$$ ，

这也很符合我们对强化学习算法的直观认知：一个策略产生轨迹的奖励期望越高，这个策略也自然更好。因此我们用梯度上升方法更新参数时即是： $$$\theta' = \theta + \alpha \nabla_{\theta} L(\theta)$$$ 。

接下来我们对梯度进行求解：

$\begin{aligned} \nabla_{\theta} L(\theta) &= \nabla_{\theta} \mathbb{E}_{\tau \sim \pi(\cdot|\theta)} [R(\tau)] \quad \text{--- (1)} \\ &= \nabla_{\theta} \sum_{\tau} [\pi(\tau|\theta) \cdot R(\tau)] \quad \text{--- (2)} \\ &= \sum_{\tau} [\nabla_{\theta} \pi(\tau|\theta) \cdot R(\tau)] \quad \text{--- (3)} \\ &= \sum_{\tau} \left[ \pi(\tau|\theta) \frac{\nabla_{\theta} \pi(\tau|\theta)}{\pi(\tau|\theta)} \cdot R(\tau) \right] \quad \text{--- (4)} \\ &= \sum_{\tau} [\pi(\tau|\theta) R(\tau) \cdot \nabla_{\theta} \log \pi(\tau|\theta)] \quad \text{--- (5)} \\ &= \mathbb{E}_{\tau \sim \pi(\cdot|\theta)} [R(\tau) \nabla_{\theta} \log \pi(\tau|\theta)] \quad \text{--- (6)} \\ &= \mathbb{E}_{\tau \sim \pi(\cdot|\theta)} \left[ R(\tau) \cdot \sum_{t} \nabla_{\theta} \log \pi(a_t|s_t, \theta) \right] \quad \text{--- (7)} \end{aligned}$

这其中(1) -> (2)是按照期望的定义展开，(3) -> (4) 是利用了常见的对数变化公式来凑出log的形式，(5) -> (6) 是写为期望的定义。

而随着Reinforce算法继续发展会得到[Actor-Critic算法](https://zhida.zhihu.com/search?content_id=269378271&content_type=Article&match_order=1&q=Actor-Critic%E7%AE%97%E6%B3%95&zhida_source=entity)，[TRPO算法](https://zhida.zhihu.com/search?content_id=269378271&content_type=Article&match_order=1&q=TRPO%E7%AE%97%E6%B3%95&zhida_source=entity)以及PPO算法，这里具体的推导过程不在赘述，如果有感兴趣的可以参考这篇[blog](https://medium.com/data-science/understand-reinforce-actor-critic-and-ppo-in-one-go-2569f520c066)。这里直接给出PPO的公式(篇幅起见省略了clip的部分，详情参考[PPO](https://arxiv.org/abs/1707.06347))：

$\begin{aligned} L(\theta) &= \max_{\theta} \mathbb{E}_{\tau \sim \pi(\cdot|\theta)} \left[ \frac{\pi_{\theta}(\tau)}{\pi_{\theta_{\text{old}}}(\tau)} \cdot R(\tau) \right] \\ \nabla_{\theta} L(\theta) &=  \mathbb{E}_{x \sim D, y \sim \pi_{\theta}(\cdot|x)} \left[ \frac{\pi_{\theta}(y|x)}{\pi_{\theta_{\text{old}}}(y|x)} \cdot A(x, y) \cdot \sum_{t} \nabla_{\theta} \log \pi_{\theta}(y_t | x, y_{<t}) \right]  \end{aligned}$

## 传统RL到LLM RL的过渡

从传统RL到LLM RL实际上没有很多新定义的引入，主要需要熟悉并了解下LLM背景下一些常见术语的定义：

$\begin{aligned} & \textbf{1. State (状态):} \\ & s_t = [x; y_{<t}] \\ & \text{定义：在 LLM 中，状态是初始 Prompt } x \text{ 与当前已生成的 Token 序列 } y_{<t} \text{ 的拼接。} \\ \\ & \textbf{2. Action Space (动作空间):} \\ & \mathcal{A} = \mathcal{V} \\ & \text{定义：动作空间即词表（Vocabulary），智能体每一步动作是从数万个候选 Token 中选择一个。} \\ \\ & \textbf{3. State Transition Function (状态转移方程):} \\ & s_{t+1} = [s_t; a_t] \\ & \text{定义：LLM 的转移具有确定性，新的状态只是简单地将当前动作（Token）附加到现有序列末尾。} \\ \\ & \textbf{4. Reward Function (奖励函数):} \\ & \text{定义：用于评价生成序列质量的标量数值，通常在序列生成完成后由 Reward Model 给出。} \end{aligned}$

25年年初DeepSeek-R1的爆火使得GRPO这个算法进入大家的视线，GRPO通过利用组间奖励值来计算每条轨迹的优势值，简单有效地去除了PPO算法中Critic model和Reward Model两部分组件，极大地减少了LLM RL中所需要的显存，RLVR（Reinforcement Leaning with Verifable Reward）也因此成为25年的一个热点主题。GRPO算法公式：

$\begin{aligned} \\ & L_{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_{\theta}(y_i|q)}{\pi_{\theta_{\text{old}}}(y_i|q)} \hat{A}_i, \text{clip} \left( \frac{\pi_{\theta}(y_i|q)}{\pi_{\theta_{\text{old}}}(y_i|q)}, 1-\epsilon, 1+\epsilon \right) \hat{A}_i \right) - \beta D_{KL} \right) \\ \\ & \textbf{优势函数 (组内归一化):} \\ & \hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)} \\ \\ & \textbf{KL 惩罚项 (直接计算):} \\ & D_{KL} = \frac{\pi_{\text{ref}}(y_i|q)}{\pi_{\theta}(y_i|q)} - \log \frac{\pi_{\text{ref}}(y_i|q)}{\pi_{\theta}(y_i|q)} - 1 \end{aligned}$

## 工业级RL所面临的问题与解决方案

如果不考虑显存占用和运行时间，LLM RL可以简单地写为 模型采样得到样本和Reward、前向传播计算logprobs、反向传播更新模型这三个步骤。但LLM RL相比于传统RL，采样是一个非常耗时、耗资源的时间。因此现有的工业级RL框架大部分使用训练框架和推理框架（采样框架）相分离的架构，训练框架通常采用 FSDP / Megatron，而推理框架通常采用[VLLM](https://zhida.zhihu.com/search?content_id=269378271&content_type=Article&match_order=1&q=VLLM&zhida_source=entity) / SGLang等来加快推理速度。

### RL框架介绍

目前开源社区已经有很多非常出色的RL框架，包括但不限于Verl、Areal、Slime、AgentRL等等。这些开源框架如上所述基本都采用了训推分离的架构，但具体的设计细节可能不尽相同，具体的分析与讨论已经有很多出色的帖子做出了分析（[框架对比分析](https://zhuanlan.zhihu.com/p/1979237927641949997)），这里不再赘述。下面主要讨论下我认为几个RL工程上改进比较大的部分:

### 异步加速的引入

LLM RL中rollout属于比较耗时的过程，如果采用完全on-policy的实现，rollout和forward\_backward顺序执行，会带来GPU利用率的降低和整体RL速度的减慢。为此多数RL框架中先后引入了异步rollout的逻辑，实现方式上大体都是利用python的asyncio来管理，维护一个buffer队列，rollout engine时刻在进行采样并将完成的轨迹存入buffer，actor、ref需要前向传播时从buffer队列中取出一定数量的样本进行前向传播和反向更新。本质上类似于流水线的思想，通过合理的资源调配，实现GPU的满载利用，这部分的详细介绍在[AgentRL](https://arxiv.org/abs/2510.04206)、[AReal](https://arxiv.org/abs/2505.24298) 中均有介绍。但需要注意的一点是，采用异步加速后的RL算法并不能算是严格的on-policy的算法，引入的一定程度的off-policy可能会导致算法收敛收到影响，对于这个问题，一方面在异步框架中都有一个类似staleness的参数控制或通过限制buffer size的大小以保证当前更新的数据和当前的policy不会超过很多的版本；另一方面则是通过重要性采样的修正。

### 重要性采样修正

现有的RL框架使用推理引擎VLLM / SGLang作为 Engine进行采样就是为了最大化加速推理（采样）的速度，但这些推理引擎中会有精度损失等问题，进而导致对于同一条轨迹，训练框架和推理框架产生的log\_probs存在误差，而这个误差会影响算法的收敛速度和训练的稳定运行。为此，社区提出重要性采样的概念，即在短时间内无法做到训推一致的情况下，从算法层面修正这个问题：通过在目标函数中乘以对应的重要性采样修正比例来修复。

$\mathbb{E}_{x \sim \mathcal{D}, y \sim \mu_{\theta_{\text{old}}}(\cdot|x)} \left[ \underbrace{\frac{\pi_{\theta}(y|x)}{\mu_{\theta_{\text{old}}}(y|x)}}_{\text{IS weight}} R(x,y) \right] $

详细地推导和理解过程可以参考[TIS blog](https://fengyao.notion.site/off-policy-rl)。

在此基础上大家又继续探索了[MoE模型](https://zhida.zhihu.com/search?content_id=269378271&content_type=Article&match_order=1&q=MoE%E6%A8%A1%E5%9E%8B&zhida_source=entity)上的重要性采样，MoE模型相比Dense 模型多出了expert router这个变量，如果训练和推理时激活的expert router不一致，也会导致训推不一致的问题，为稳定MoE 模型的RL训练，在TIS修正的基础上进行改进，引入per-token的mask来对TIS进行clip，公式如下：

$$$\mathcal{J}_{\text{IcePop}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\} \sim \pi_{\text{infer}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left\{ \mathcal{M}(k_{i,t}, \alpha, \beta) \cdot f_{\text{clip}}(r_{i,t} \hat{A}_{i,t}) \right\} \right]$$$

$$$\mathcal{M}(k) = \begin{cases} k, & \text{if } k \in [\alpha, \beta] \\ 0, & \text{otherwise} \end{cases}$$$

详细地推导和细节可以参考[IcePop](https://www.emergentmind.com/topics/icepop)、[MiniRL](https://arxiv.org/pdf/2512.01374v1)

### RL loop token 流的维护

在Agentic RL中我们一般会涉及到Multi-turn的任务处理，一般会有一个loop函数来调用llm生成assitant回复，同时会调用类似env.step()函数得到环境反馈，在这个过程中如果我们选择以text作为媒介进行交流（即每次将新的history tokenize为新的一个token ids list输入给模型）很容易导致tokenizer在进行merge的时候本轮和上一轮merge出来的token不一致，进而导致出现少数token概率异常的情况。随着轮数的累积，这种不稳定性很容易导致整个RL训练的崩溃。因此我们需要为训练端维护一个独立的tokens id流，每次得到环境反馈后增量式的更新这个ids列表而非整个替换，进而能够保证RL训练的稳定。

## 参考文献

1.  [Understand REINFORCE, Actor-Critic, and PPO in One Go](https://medium.com/data-science/understand-reinforce-actor-critic-and-ppo-in-one-go-2569f520c066)
2.  [关于Agentic RL训推框架的一点看法和思考](https://zhuanlan.zhihu.com/p/1979237927641949997)
3.  [MiniRL](https://arxiv.org/pdf/2512.01374v1)
4.  [TIS](https://fengyao.notion.site/off-policy-rl)
5.  [IcePop](https://www.emergentmind.com/topics/icepop)