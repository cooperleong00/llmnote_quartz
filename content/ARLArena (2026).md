---
type: paper
description: 系统分析 Agentic RL 训练不稳定性的统一框架，通过四维 policy gradient 分解提出 SAMPO 方法，相比 GRPO 平均提升 25.2%
aliases:
  - ARLArena
  - Stable Agentic RL
prerequisites:
  - "[[Agentic RL]]"
  - "[[GRPO]]"
  - "[[PPO]]"
  - "[[Importance Sampling]]"
tags:
  - post-training
  - reinforcement-learning
  - agent
created: 2026-03-05
updated: 2026-03-31T10:34
---

# ARLArena (2026)

ARLArena 是 UCLA 提出的一个分析 [[Agentic RL]] 训练稳定性的框架。做法是把 policy gradient 拆成四个正交维度，逐个分析哪些设计选择会导致训练崩溃，最终整合出 [[SAMPO]] 方法。

> [!paper] 论文信息
> **标题**: ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning
> **作者**: Xiaoxuan Wang, Han Zhang, Haixin Wang et al. (UCLA)
> **时间**: 2026.02
> **链接**: https://github.com/WillDreamer/ARL-Arena

## 问题

[[Agentic RL]] 训练很不稳定，经常崩溃（training collapse）。原因有几个：

1. 多轮交互的复合效应——早期决策的小偏差在后续轮次中放大
2. 稀疏奖励——只有完成任务才有奖励，中间步骤没有反馈
3. 长时域信用分配——难以判断哪个动作导致了最终结果
4. 非平稳动态——agent-environment 交互导致数据分布不断变化

结果就是训练难以复现，也很难扩展到更长的交互时域。

## 方法论：四维分解框架

ARLArena 把 policy gradient 拆成四个正交的设计维度，分别分析：

### Policy Gradient 公式

多轮 agentic 交互的 policy gradient：

$$
\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}} \left[ \sum_{k=1}^K \sum_{t=0}^{T_k} \underbrace{w_t(y^{(k)})}_{\text{IS}} \underbrace{\nabla_\theta \log \pi_\theta(y_t^{(k)} | x^{(k)}, y_{<t}^{(k)})}_{\text{Log prob}} \underbrace{A(x^{(k)}, y^{(k)})}_{\text{Advantage}} \right]
$$

其中：
- $K$：交互轮数
- $T_k$：第 $k$ 轮生成的 token 数
- $w_t$：[[Importance Sampling|重要性采样]]权重
- $A$：优势函数

### 四个设计维度

| 维度                | 问题             | 代表方法                                    |
| ----------------- | -------------- | --------------------------------------- |
| Loss Aggregation  | 如何聚合不同长度序列的损失？ | Token-mean vs Seq-mean-token-mean       |
| IS Clipping       | 如何限制策略更新幅度？    | [[GRPO]], [[CISPO]], [[SAPO]], [[GSPO]] |
| Advantage Design  | 如何设计优势函数？      | [[GiGPO]], [[EMPG]]                     |
| Dynamic Filtering | 是否动态过滤无信息样本？   | [[DAPO]]                                |

> [!intuition] 为什么需要分解？
> 现有方法往往同时修改多个维度，很难判断哪个设计选择真正起了作用。拆开来看，才能隔离变量。

## 标准化测试平台

为了公平比较，ARLArena 构建了标准化测试平台（Standardized Testbed）：

### 1. Behavior Cloning 初始化

用 Qwen3 在目标环境中自举生成高质量轨迹，做 SFT 初始化。这样策略从一个合理的行为流形（behavioral manifold）出发，而不是从随机策略开始。

### 2. Format Penalty

强制输出结构化格式：`<think>...</think>` 和 `<action>...</action>`。这提供了密集的 shaping signal，减少无效 rollout。

### 3. KL Regularization

用 $k_3$ 估计器约束策略与参考模型的 KL 散度：

$$
k_3(x) = \delta(x) - 1 - \log \delta(x), \quad \delta(x) = \frac{p(x)}{q(x)}
$$

防止策略过度偏离参考模型，保留预训练知识。

### 4. 方法特定的超参数搜索

对每个方法进行网格搜索，直到训练曲线稳定（最后 20% 步骤的成功率方差低于阈值）。

> [!warning] 注意
> 这些"稳定化策略"不是 trick，是必要条件。没有它们，大多数方法直接崩溃。

## 三个主要发现

### Finding 1: IS Clipping 的敏感性

Tolerant clipping 容易导致训练崩溃，sequence-level clipping 则稳定得多。

实验中，[[CISPO]] 和 [[SAPO]]（tolerant clipping）早期快速提升，但在 ~130 步崩溃；[[GSPO]]（sequence-level clipping）则稳定单调改进，最终性能最好。

崩溃时发生了什么？Token-level 和 sequence-level 分析显示：
- 崩溃时，负优势 + 低 IS ratio 的样本急剧增加
- 这些样本的 KL 散度占比突然上升，主导了策略更新
- Tolerant clipping 保留了这些样本的梯度，导致过度探索

解决方案是 sequence masking——屏蔽负优势且低 IS ratio 的序列：

$$
M_i = \mathbb{1}\left[ A_i \geq 0 \text{ or } \frac{1}{|T_i|} \sum_{t=0}^{T_i-1} \log \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_\text{old}}(y_t|x,y_{<t})} \leq \delta \right]
$$

应用后，SAPO$_{\text{SM}}$ 和 CISPO$_{\text{SM}}$ 的成功率从 25.16%/54.42% 提升到 76.92%/78.88%。

> [!comparison] GSPO vs Tolerant Clipping
> - [[GSPO]]：用 sequence-level IS ratio $s_i = \exp\left(\frac{1}{|T_i|}\sum_t \log w_t\right)$ 裁剪
> - Tolerant clipping：用 stop-gradient 或 soft clipping 保留 out-of-bounds token 的梯度
>
> GSPO 在 sequence level 做约束，天然抑制有害轨迹；tolerant clipping 在 token 级别太宽容了。

### Finding 2: 环境级优势设计的收益

把环境级信息融入优势设计，能同时提升性能和稳定性。

[[GiGPO]]（Group-level Importance-weighted Generalized Policy Optimization）的做法是把同一环境状态下的动作分组，赋予相对优势：

$$
A_i' = A_i + \omega \cdot A_{\text{step}}(\hat{y}_{i,k})
$$

其中 $A_{\text{step}}$ 是 state-level 的相对优势。

在 ALFWorld 上提升 34.4%，平均提升 3.4%。有效的原因是：同一状态下的不同动作有了相对比较，缓解了奖励稀疏性，也让信用分配更细粒度。

### Finding 3: Dynamic Filtering 的条件收益

Dynamic filtering 是过滤掉所有样本都成功或都失败的 group，重新采样。

有意思的是，它和 [[GiGPO]] 结合有效（+11.0%），但和 GRPO 结合反而有害（-7.6%）。

原因在于早期训练中 format 错误导致整组失败，format penalty 提供了强信号。GRPO 的优势信号单一，过滤后丢失了 format 学习信号；[[GiGPO]] 的优势信号更多样，过滤后仍能稳定学习 format。

> [!warning] 设计选择的耦合
> Dynamic filtering 的效果取决于优势设计的多样性。ARL 中各维度的设计选择是强耦合的。

## SAMPO

基于上面三个发现，论文整合出 SAMPO（Stable Agentic Multi-turn Policy Optimization）：

$$
\mathcal{L}(\theta) = \frac{1}{\sum_{i=1}^N T_i} \sum_{i=1}^N \sum_{t=0}^{T_i-1} \min\Big( s_i(\theta) A_i', \text{clip}(s_i(\theta), 1 \pm \varepsilon) A_i' \Big)
$$

$$
\text{s.t.} \quad 0 < |\{y \mid \text{is\_equivalent}(a, y)\}| < G
$$

其中：
- $s_i(\theta)$：sequence-level IS ratio（来自 Finding 1）
- $A_i' = A_i + \omega \cdot A_{\text{step}}(\hat{y}_{i,k})$：环境级优势（来自 Finding 2）
- 约束条件：dynamic filtering（来自 Finding 3）

性能：
- ALFWorld：92.72%（GRPO 62.36%）
- WebShop：77.73%（GRPO 57.71%）
- Sokoban：88.86%（GRPO 83.90%）
- 平均提升 25.2%

> [!intuition] SAMPO 的思路
> 不是发明新算法，而是把已验证的设计选择组合起来。每个组件有明确的稳定性保证，组合后整体鲁棒。

## 额外发现

### Off-Policy Staleness

批量 rollout 导致后续更新用的是旧策略的数据（off-policy staleness）。控制 rollout batch size（128/512/1024）的实验显示：
- TIR Math：低 staleness 下 AIME avg@32 = 87.34%，高 staleness 下 = 74.99%
- ALFWorld：低 staleness 下成功率 60.80%，高 staleness 下 = 52.71%

Agentic RL 对 off-policy ratio 比较敏感，应尽量减少 staleness。

### 与闭源模型的对比

Qwen3-4B + SAMPO 在 ALFWorld 上达到 92.72%，超过：
- GPT-5.2 (SLA)：51.56%
- o3 (Multi-Agent Debate)：56.25%

> [!intuition] 一个值得注意的结果
> 环境对齐的 RL 训练比单纯堆规模和推理更有效。4B 的小模型通过稳定 RL 训练超过了大模型的 inference-time 工程。

## 启示

### 1. Clean Training Recipe 是基础

ARL 对初始化和早期动态很敏感。Training recipe 不是 trick，是算法的一部分。

### 2. IS Clipping 高风险高回报，Advantage Design 稳定但收益有限

- IS clipping：微小变化导致巨大差异
- Advantage design：改进稳定但幅度有限

### 3. 稳定 ARL 解锁长时域扩展

一旦训练崩溃得到缓解，agentic 策略能在更多优化步骤上持续改进，为扩展交互时域和多任务课程打开了空间。

## 相关概念

前置知识：
- [[Agentic RL]] — 本文研究的范式
- [[GRPO]] — 基线方法
- [[PPO]] — Policy gradient 的经典实现
- [[Importance Sampling]] — IS clipping 的理论基础

相关方法：
- [[CISPO]] — Tolerant clipping 的代表
- [[SAPO]] — Soft clipping 方法
- [[SAMPO]] — 本文提出的方法

应用场景：
- [[SWE-Bench]] — 软件工程 agent 基准
- [[ASTRA]] — 自动化合成 agentic 轨迹

相关论文：
- [[GLM-5 (2026)]] — 使用异步 Agentic RL
- [[Kimi K2 (2025)]] — 大规模 agent 数据合成

## 延伸阅读

原始论文：
- [[Clippings/Paper/2602.21534/2602.21534|ARLArena 论文全文]]

实现资源：
- GitHub: https://github.com/WillDreamer/ARL-Arena
- HuggingFace: https://huggingface.co/UCLA-SCAI/models
