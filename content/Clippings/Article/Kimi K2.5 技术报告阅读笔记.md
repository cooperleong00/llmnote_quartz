---
title: Kimi K2.5 技术报告阅读笔记
author:
  - "[[YiFan-Zhang]]"
url: https://zhuanlan.zhihu.com/p/2000719027690030326
created: 2026-02-09
tags:
category: clippings/article
updated: 2026-02-09T14:45
---

先说架构，预训练，后训练，reward system设计，PPO算法改进，后面再说一些有意思的细节设计和finding。

## 模型结构：[MoonViT-3D](https://zhida.zhihu.com/search?content_id=269796559&content_type=Article&match_order=1&q=MoonViT-3D&zhida_source=entity) + MLP + K2 MoE [LLM](https://zhida.zhihu.com/search?content_id=269796559&content_type=Article&match_order=1&q=LLM&zhida_source=entity)

1.  MoonViT-3D：从 SigLIP-SO-400M初始化，对于video，把连续 4 帧视频看作一个“时空块”，将这 4 帧的 2D patches 一起展平、打包成同一条 1D 序列，让 **同一个注意力机制**同时处理空间与时间，在进入Projector之前，对时间维度进行 4 倍压缩（Pooling）。这使得 K2.5 可以在相同的上下文窗口下，处理比原本长 4 倍的视频内容。
2.  LLM 1.04 T A 32 ( 384 expert A 8)
3.  使用了 **MuonClip + [QK-Clip](https://zhida.zhihu.com/search?content_id=269796559&content_type=Article&match_order=1&q=QK-Clip&zhida_source=entity)**

## **预训练**

![图片](https://pic4.zhimg.com/v2-1abc05eb6102205bcd02210ede071df9.jpg)

Kimi K2.5 的预训练消耗了约 15T 的 Token，分为三个精心设计的阶段：

-   **ViT 训练阶段（1T Token）**：

-   第一段：只优化 **caption 生成的交叉熵损失**，把 MoonViT-3D 对齐到一个 16B 级别的语言模型（文中是 Moonlight-16B-A3B），消耗约 **1T token**，此时主要更新 ViT 权重
-   第二段：很短，只更新 **MLP projector**，目的是让视觉输出更平滑地接入万亿级 LLM，方便后续联合预训练

-   **联合预训练阶段（15T Token）**：

-   从接近训练末期的 K2 checkpoint 接着训
-   混合视觉和文本数据继续训练。特别增加了代码相关数据的权重。对每个数据源设置最大 epoch，避免某些源被反复过拟合

-   **mid-training**：

-   训练数据更偏高质量与长样本：长文本、长视频, 推理数据、长链路 CoT
-   用**YaRN 插值**逐步把长度从**32768 拉到 262144**

## **后训练-[SFT](https://zhida.zhihu.com/search?content_id=269796559&content_type=Article&match_order=1&q=SFT&zhida_source=entity)**

### 监督微调（SFT）：多模型合成 + 人工与验证管线

-   候选答案来源：K2、K2 Thinking，以及多种内部专家模型
-   diverse prompt，intricate reasoning trajectories，tool-call三个核心特性

### 强化学习（[RL](https://zhida.zhihu.com/search?content_id=269796559&content_type=Article&match_order=1&q=RL&zhida_source=entity)）：统一环境 + 稳定的 token 级裁剪策略

文本多模态不拆分，训练函数如下

![图片](https://pica.zhimg.com/v2-411448b1e13c5f3edb9028286b01792c.jpg)

两个核心的点

1.  token-level的clip，threshold在 \[alpha, beta\]之间
2.  不考虑advantage的符号（在需要长视野、多步骤工具调用推理的场景中对稳定性尤其重要）

### 更好的token效率

施加问题相关的 token 预算可以有效约束推理阶段的计算开销，但是在严格预算约束下训练的模型，无法有效利用额外的推理 token 来解决复杂问题，反而倾向于生成截断的推理模式。因此作者的reward相对这两个目标同时优化，做了个启发式的算法

其中，λ 和 m 为超参数，K 为每个问题生成的 rollout 数量。算法每 *m* 次迭代在两个阶段间切换：

-   **Phase 0（预算受限阶段）**：模型需在任务相关的 token 预算内解决问题。为避免过早牺牲质量换取效率，该约束仅在模型对某问题的平均准确率超过阈值 *λ* 时才被激活（大概率能做对才约束预算）。
-   **Phase 1（标准扩展阶段）**：模型可生成至最大 token 长度，鼓励其利用更多计算资源实现更好的测试时扩展能力。  
    $\tilde{r}(x, y) =  \begin{cases} r(x, y) \cdot \mathbb{I}\left\{ \frac{1}{K} \sum_{i=1}^{K} r(x, y_i) < \lambda \ \text{或} \ |y_i| \leq \text{budget}(x) \right\}, & \text{若 } \lfloor t/m \rfloor \bmod 2 = 0 \quad \text{(Phase 0)} \\ r(x, y), & \text{若 } \lfloor t/m \rfloor \bmod 2 = 1 \quad \text{(Phase 1)} \end{cases}$

$\text{budget}(x) = \text{Percentile}\left( \{ |y_j| \mid r(x, y_j) = 1,\ j = 1, \dots, K \},\ \rho \right). \tag{2} $  
任务预算是通过做对的题的分位数进行估算该预算在训练开始时估算一次，之后保持固定。几乎所有基准测试中的输出长度均显著下降。平均而言，Toggle 在性能几乎无损的情况下，将输出 token 减少了 25%∼30%。同时，思维链中的冗余模式（如重复验证、机械计算等）大幅减少。

![图片](https://pic3.zhimg.com/v2-76ebe2023e3df8c109b5fa187d7baf8c.jpg)

### Reward System

-   可验证任务：规则型 outcome reward 用于推理题、工具任务这类“对错明确”的场景比如定位用IOU，分割用分割的IOU，OCR用编辑距离，计数用真实误差。
-   通用开放任务：rubric GRM 而是按价值观与体验指标做细粒度评分，有效缓解reward hacking：

-   有用性，响应及时性，上下文相关性，信息详略得当，生成内容的美学质量，严格遵循指令等

-   使用场景不仅是聊天，还覆盖：coding agent；search agent；产物生成类 agent

## Infra：Decoupled Encoder Process ([DEP](https://zhida.zhihu.com/search?content_id=269796559&content_type=Article&match_order=1&q=DEP&zhida_source=entity))

使用了类似 LongCat-Flash-Omni 的思路，传统的PP，视觉编码器与文本嵌入通常被部署在同一阶段（即流水线的第一阶段，Stage-0），但是因为图像/视频 tokens数差别非常大，stage 0不管是内存占用还是计算burden波动都很剧烈。因此

1.  **先统一处理vision data**：把所有视觉数据先在vision tower这forward下，而且是均匀分配至所有 GPU，只保留输出激活并将结果收集回 PP 的 Stage-0。
2.  训练backbone：按照text的方式即可，梯度反传在vision tower这里停止
3.  重新执行视觉编码器的前向传播（用于恢复中间状态），随后进行反向传播，以计算视觉编码器参数的梯度。

**解耦了视觉编码器与主干网络的优化策略**。在多模态训练中达到相当于纯文本训练 **90% 的训练效率**。

## 一些有意思的点

### 原生多模预训练的必要性

引了Qwen 3 VL， Seed 1.5 VL，他们的point在于应在LLM训练的后期阶段以高比例（例如 50% 或更高）引入视觉 token，将多模态能力视为在语言能力基础上的post-hoc add-on，从而加速多模态能力的获得。

![图片](https://pica.zhimg.com/v2-b22a5af09920ff14ab1a1648cb2aed72.jpg)

核心就是说，我们先把LLM train好，再train mlp，vit，再合起来train数据效率太低，

-   控制变量：总的视觉token数与文本token数保持不变
-   改动变量：

-   视觉注入时机（早/中/晚）
-   训练过程中的视觉占比（例如 10%:90%、20%:80%、50%:50%）

-   为了严格满足不同配比，模型在引入视觉数据之前，会先跑一段“纯文本预训练”，长度按配比反推计算，确保最终视觉/文本预算一致。

**实验结论（很反直觉但很关键）**

-   视觉占比本身对最终多模态能力影响没想象中大，真正影响更大的是“什么时候开始混”
-   在固定预算下，“更早融合 + 更低视觉占比”反而整体最好

-   表现上不仅视觉相关能力（视觉知识、视觉推理、OCR）更好
-   文本侧能力（文本知识、文本推理、代码）也没有被牺牲，甚至更稳

### [Zero-Vision SFT](https://zhida.zhihu.com/search?content_id=269796559&content_type=Article&match_order=1&q=Zero-Vision+SFT&zhida_source=entity)

这块也有点意思，当时做Thyme的时候就发现，大多数VLM其实并没见过“看图→调用工具→再推理”的链路，包括Deepeys他们在qwen上直接zero rl，train出来的模型也一堆问题。thyme自己设计了很多类型的数据但是确实

-   覆盖面窄、花样少
-   视觉操作通常停留在裁剪、旋转这类低级工具调用
-   难支撑复杂的视觉推理与泛化

Kimi的做法是Zero-Vision SFT。SFT阶段完全不使用图像数据，模型学习通过**IPython**环境（代码解释器）来执行各种程序化操作。把“处理图像”看作是“处理数据”的一种特例。如果模型学会了用代码去处理复杂的数据结构，这种能力就能自然迁移到图像处理上。

这个motivation我们在thyme中就提到了，code的灵活性是定义各种image tool无法媲美的，但是这个能力需要模型具有较强的基础能力，一个小的code写不好的模型，花很大功夫可能带来的提升也有限，但是显然，这么大体量的模型，以及最近的Gemini 3.0 flash，都验证了coding 在think with image这条路上的有效性。

作者把成功的原因归根于 **联合预训练**。因为模型底层的视觉和文本已经对齐得非常好，只要通过文本 SFT 教会它“思考和写代码的逻辑”，它就能自动把这套逻辑套用到视觉信息上。

同时有个有趣的对比实验，尝试在 SFT 阶段加入视觉数据（Text-Vision SFT）。结果发现，效果反而变差了。这很可能是因为目前的视觉 SFT 数据质量远不如纯文本数据高，混进去反而拉低了模型的智商。而且text only sft之后，vision rl的曲线也还是比较不错的

![图片](https://pica.zhimg.com/v2-82fedeeaf2427051f75b89f384fc3c70.jpg)

### Joint Multimodal Reinforcement Learning (RL)

纯文本sft造成的后果就是，多模态模型不看图了，或者看的不准，因此需要一轮RL来矫正

1.  Outcome-Based Visual RL ：定位计数+图标文档理解+vision centric的STEM 问题。
2.  视觉 RL 过程中产生的高质量轨迹会被抽取出来，用于RFT。

几个finding

1.  从极少量零视觉 SFT 起步，只要把视觉 RL 的FLOPs持续加上去，视觉基准表现还会继续稳步提升。
2.  视觉 RL 反而提升纯文本能力，文中给出的解释倾向于：视觉 RL 强化了“结构化信息抽取 + 不确定性校准”这类通用能力，进而让模型在类似计数、抽取、对齐的文本问题上更稳，减少乱猜。

也是因为发现视觉RL能boost 纯文本，这里又整了个活，即上面提到过的任务不按照模态拆分，而是按照domain 拆分，比如知识，推理，编程，agentic等。domain expert同时学多模态和text query。

### Agent Swarm的定义

一个很有意思的设计，现有 agent 系统通常按“推理一步→调用一次工具→再推理一步”的方式串行推进。当任务变复杂（信息搜集更广、推理分支更多、上下文更长）时，串行会出现两个硬限制：

-   **推理步数/工具调用预算耗尽**：没做完就到上限。
-   **延迟线性累积**：步骤越多越慢，复杂任务几乎不可用。

![图片](https://pic4.zhimg.com/v2-bc8ffa29fa83dd6ef9dcdde2fa817001.jpg)

本文的做法是，把复杂任务拆分成子问题，1）动态的任务分解 2）动态的实例化子智能体 3）并行的子任务调度；主智能体在合适时机汇总各路结果继续推进。

### 如何强化学习训练Agent Swarm - [PARL](https://zhida.zhihu.com/search?content_id=269796559&content_type=Article&match_order=1&q=PARL&zhida_source=entity)

PARL 采用解耦结构：

-   **planner是可训练的**：负责高层决策（拆什么、开几个子智能体、如何汇总）
-   **子智能体是冻结的**：从“固定的中间策略检查点”实例化而来（相当于能力固定的执行器）

这样做刻意避免端到端联训，主要是为了解决多智能体强化学习里两个老大难：

-   **credit assignment ambiguity**：最终答案对了，不代表每个子任务都做得对；最终失败，也不一定是所有子智能体都错。奖励信号对单个子决策非常含糊。
-   **训练不稳定**：多智能体同时学习会造成环境非平稳、梯度互相干扰，更容易崩。

冻结子智能体后，子智能体输出被当作“环境观察”而不是可微决策点，训练就聚焦在“编排与调度策略”，更容易稳定收敛。

此外还有两点工程策略提升效率：

-   **先用小子智能体训练编排器**，再过渡到更大模型子智能体，降低早期探索成本
-   **支持动态调整编排器与子智能体的推理实例比例**，让集群资源利用更充分（哪个环节更卡就给哪个更多算力）

![图片](https://pic1.zhimg.com/v2-3fc5ad499ba2910b1df7f754ff3fb814.jpg)

为了在稀疏和嘈杂的环境反馈中训练出一个靠谱的指挥官，设计了包含三部分的复合奖励函数​：

-   **性能奖励 (*rperf*​)**：评估任务最终解决的质量和成功率。
-   **实例化奖励 (*rparallel*​)**：这是一个辅助奖励，用于防止模型陷入局部最优（即退化回单线程模式），鼓励模型大胆尝试并发调度。
-   **完成率奖励 (*rfinish*​)**：另一个辅助奖励，用于防止模型“刷分”（即生成大量子智能体却不进行有效的任务拆解）。只奖励“真正完成了分配子任务”的行为，逼迫并行必须可行且有效。
-   **动态调整**：上述两个辅助奖励的权重（*λ*1​ 和 *λ*2​）会在训练过程中逐渐衰减至零，确保模型最终专注于解决任务本身。

还有一个小trick，因为在并行系统里，用“总步数”衡量成本不合理，该方法引入类似计算图“关键路径”的概念：**关键步数（Critical Steps）**。

![图片](https://pic2.zhimg.com/v2-7dc7f23541a1fbff0a701e08fb917ac7.jpg)

数据方面，整了一套合成任务，主要覆盖：

-   **宽搜索（wide search）**：需要同时查很多独立信息源（天然可并行）
-   **深搜索（deep search）**：需要多条推理分支各自深入，最后再汇总（串行会超预算）
-   **贴近真实负载的任务**：如长上下文文档分析、大规模文件下载等

prompt里不会明确说要parallel去做，而是通过任务去要求。

## 总结

总结下，很多有意思的点，虽然很多技术细节很多没有真正开源，但是rubric reward system的探索，多模态数据联合训练的ablation，zero-vision激活think with image的能力，以及agent swarm系统其实都挺有趣的。

最后，欢迎大家关注github，聚合了Multimodal Large Language Models, Large Language Models, and Diffusion Models以及一些前沿研究方向的一些阅读笔记，非常欢迎大家补充完善

[链接](https://github.com/yfzhang114/Awesome-Multimodal-Large-Language-Models)