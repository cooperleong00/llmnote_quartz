---
type: paper
description: Cursor 的 agentic coding 专用模型，展示了 continued pretraining + 大规模 RL 两阶段范式如何将开源 MoE 基座提升到 frontier 水平
aliases:
  - Composer 2
prerequisites:
  - "[[GRPO]]"
  - "[[Mixture of Experts]]"
  - "[[KL Divergence]]"
  - "[[Policy Gradient]]"
tags:
  - agentic
  - reinforcement-learning
  - software-engineering
  - infrastructure
created: 2025-03-25
updated: 2025-03-25
---

# Composer 2

Composer 2 是 Cursor（Anysphere）为 agentic software engineering 打造的专用模型。它的核心叙事很清晰：拿一个开源 [[Mixture of Experts|MoE]] 基座（Kimi K2.5，1.04T 参数 / 32B active），通过 continued pretraining 注入代码能力，再用大规模 [[RLHF|RL]] 在真实编程环境中训练 agent 行为，最终达到与 frontier 模型（Claude Opus 4.6、GPT-5.4）接近的性能，但推理成本显著更低。

这篇 technical report 的价值不仅在于模型本身，更在于它系统性地记录了训练过程中的工程决策和 ablation 发现——哪些标准做法被保留，哪些被刻意移除，以及为什么。

## 动机

> [!intuition] 为什么不直接用 frontier 模型？
> Frontier 模型（如 Claude、GPT）是通用的，但 agentic coding 有独特需求：长 horizon 多步操作、工具调用、代码风格一致性、成本敏感。Cursor 需要一个在这些维度上专门优化的模型，同时控制推理成本。

核心问题是：**如何用有限资源把一个开源基座变成 domain-specific 的 frontier 模型？**

Composer 2 的答案是两阶段范式：先通过 continued pretraining 建立强大的代码基础能力，再通过 RL 在模拟真实编程环境中学习 agent 行为。

## 训练流程

### Phase 1: Continued Pretraining

在 Kimi K2.5 基础上做 code-dominated 数据的 continued pretraining，分三个子阶段：

1. **主训练**（32k 序列长度）：大量代码数据，建立核心编码能力
2. **长上下文扩展**（256k）：扩展上下文窗口，适应大型代码库的场景
3. **短 [[SFT]]**：少量高质量指令数据，激活对话和指令遵循能力

> [!intuition] 为什么 continued pretraining 重要？
> 一个关键发现：continued pretraining 的 loss 与下游 RL 性能有预测性关系。也就是说，基座的代码能力越强，后续 RL 的天花板越高。这在 Qwen3-Coder 上也得到了独立验证。
>
> 这意味着 Phase 1 不是可选的"预热"，而是决定最终性能上限的关键阶段。

这个阶段与 [[Mid-Training]] 的理念一致——在通用预训练和任务特定训练之间插入一个领域适配阶段。

同时训练了 [[Multi-Token Prediction|MTP]] 层，用于 [[Speculative Decoding]] 加速推理。这是一个务实的工程选择：MTP 在训练时几乎不增加成本，但在推理时能显著降低延迟。

### Phase 2: Reinforcement Learning

大规模异步 RL，在模拟真实 Cursor 会话的环境中训练。这是 Composer 2 最有技术深度的部分。

**基本框架**：使用 [[Policy Gradient|policy gradient]] + 多 sample per prompt，单 epoch 训练（同一 prompt 不重复使用）。

#### 对 GRPO 的关键修改

Composer 2 的 RL 算法基于 [[GRPO]] 框架，但做了几个刻意的偏离，每个都有明确的 rationale：

> [!comparison] 与标准 GRPO 的区别
>
> **1. 移除 length standardization**
>
> 标准 GRPO 对每个 response 的 loss 按长度归一化（目标函数中的 $\frac{1}{|o_i|}$），这意味着短回答和长回答对梯度的贡献相同。Composer 2 移除了这个归一化，**刻意引入 length bias**——让长回答贡献更多梯度。
>
> 为什么？因为在 agentic coding 中，长回答通常意味着更复杂的操作序列，这些样本包含更丰富的学习信号。
>
> **2. 不做 group advantage 的标准差归一化**
>
> 标准 GRPO 将 advantage 归一化为 $\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$。当一个 group 内所有 sample 的 reward 相同时（全对或全错），$\text{std} = 0$，导致 degenerate case。Composer 2 直接跳过这个归一化。
>
> **3. 不做 overlong masking**
>
> 一些 RL 实现会 mask 掉超长 response 的 loss。Composer 2 认为这会丢失有价值的训练信号。

#### KL 正则化：k1 vs k3

这是一个值得深入理解的技术选择。[[KL Divergence]] 在 RL 中用于约束新策略不要偏离参考策略太远，但真实 KL 散度无法精确计算，需要用估计器。

> [!math] 两种 KL 估计器
>
> **k1 估计器**（Composer 2 使用）：
> $$\hat{D}_{k1} = -\log r_{i,t}, \quad r_{i,t} = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{ref}(o_{i,t}|q, o_{i,<t})}$$
>
> **k3 估计器**（标准 GRPO 使用）：
> $$\hat{D}_{k3} = \frac{1}{r_{i,t}} - \log \frac{1}{r_{i,t}} - 1$$
>
> k3 保证非负且无偏，理论上更"正确"。但 Composer 2 选择 k1，原因是：当 $\pi_\theta$ 和 $\pi_{ref}$ 偏离较大时（RL 训练后期常见），k3 的方差会爆炸（因为包含 $1/r$ 项），导致训练不稳定。k1 虽然可以为负，但方差更可控。

#### RL 的效果：不只是 probability mass 集中

> [!intuition] RL 到底在做什么？
> 一个常见的误解是：RL 只是让模型把概率集中到已经能生成的好回答上（即 best-of-K 不变，但 average 提升）。
>
> Composer 2 的实验否定了这个假设——RL 训练同时提升了 average 和 best-of-K 性能。这意味着 RL 确实在教模型新的能力，而不仅仅是重新分配概率。

### Self-Summarization

继承自 Composer 1.5 的技术。在长 horizon agentic 任务中，模型需要处理的上下文会不断增长。Self-summarization 允许 agent 在执行过程中多次自我总结，形成链式生成：

$$\text{context}_1 \xrightarrow{\text{summary}} \text{context}_2 \xrightarrow{\text{summary}} \text{context}_3 \rightarrow \cdots$$

关键设计：用最终 reward 训练整个链（包括 summary 部分）。好的 summary 被 upweight，丢失关键信息的 summary 被 downweight。这让模型学会"什么信息值得保留"。

### Agent Behavior 优化

除了任务完成的主 reward，Composer 2 还使用辅助 reward 信号：coding style、communication 质量、tool call 准确性。

> [!math] 非线性长度惩罚
> $$C_{\text{length}}^{k,q}(x) = \frac{(1+kx)^{1-q} - 1}{k(1-q)}$$
>
> 这个函数是 concave down + increasing 的，直觉是：简单任务应该快速完成（短回答的边际惩罚大），复杂任务允许更多思考步骤（长回答的边际惩罚递减）。

## CursorBench

Composer 2 自建的 benchmark，基于真实软件工程问题。

> [!comparison] 与 SWE-bench 的区别
>
> | 维度 | [[SWE-Bench]] | CursorBench |
> |------|---------------|-------------|
> | 中位修改行数 | 7-10 行 | 181 行 |
> | Prompt 长度 | 1185-3055 字符 | 390 字符 |
> | 任务类型 | 主要是 bug fixing | 多样化（feature、refactor 等） |
> | 更新策略 | 固定 | 持续迭代（v3 比 v1 复杂度翻倍） |
>
> CursorBench 的 prompt 更短更模糊，更接近真实用户的表达方式——用户不会写一个详细的 bug report，而是说"这个功能不对"。

### Benchmark 结果

| Model | CursorBench | SWE-bench Multi. | Terminal-Bench |
|-------|-------------|-------------------|----------------|
| Composer 2 | 61.3 | 73.7 | 61.7 |
| Composer 1.5 | 44.2 | 65.9 | 47.9 |
| Claude Opus 4.6 (High) | 58.2 | 75.8 | 58.0 |
| GPT-5.4 | 63.9 | 76.8 | 66.5 |
| Kimi K2.5 (base) | 36.0 | 65.1 | 47.3 |

从 Kimi K2.5 的 36.0 到 Composer 2 的 61.3（+70%），展示了 continued pretraining + RL 两阶段范式的威力。

## 基础设施

### 训练并行策略

使用 [[FSDP]] + Expert Parallelism (EP) + Context Parallelism (CP) 的组合，有几个值得注意的设计选择：

- **解耦 EP 和 TP**：传统 [[Mixture of Experts|MoE]] 训练常将 Expert Parallelism 和 Tensor Parallelism 绑定。Composer 2 将它们解耦，获得更灵活的并行配置
- **CP 作为长上下文主要扩展轴**：Context Parallelism 比 Tensor Parallelism 通信量更少，更适合扩展长上下文训练

### 混合精度训练

在 NVIDIA B300 上使用 MXFP8 训练，[[Mixture of Experts|MoE]] 层进一步使用 NVFP4（forward）+ MXFP8（backward）的混合精度。

> [!warning] NVFP4 的 scaling 陷阱
> NVFP4 必须使用 per-token scaling。如果使用 per-tensor scaling，RL 训练会发散。这是一个容易被忽略但后果严重的工程细节。

### RL 推理基础设施

与 Fireworks AI 合作，跨地理分布式集群做 RL rollout。两个关键工程问题：

- **Router replay**：MoE 的 expert routing 在训练和推理时可能不一致（不同的并行策略导致不同的 dispatch 顺序）。Router replay 通过在训练时重放推理时的 routing 决策来解决这个问题
- **Delta compression**：1T 参数的模型做权重同步代价巨大。通过只传输权重差异（delta），将同步数据量压缩到几 GB

## 局限性

> [!warning] 边界条件
> - 报告未公开完整的训练数据构成和 reward 函数细节
> - CursorBench 是自建 benchmark，存在 "teaching to the test" 的风险（尽管报告声称持续迭代更新）
> - Continued pretraining loss 与 RL 性能的预测关系是否在所有基座上成立，仍需更多验证
> - Self-summarization 在信息密度极高的场景下（如大量数值数据）可能丢失关键细节

> [!interview] 面试视角
> **Q: Composer 2 为什么选择在 GRPO 基础上移除 length standardization？**
> A: 标准 GRPO 的 length normalization 让短回答和长回答对梯度贡献相同。但在 agentic coding 中，长回答通常包含更复杂的操作序列和更丰富的学习信号，移除归一化让这些高信息量样本贡献更多梯度。
>
> **Q: 为什么用 k1 而不是 k3 KL 估计器？**
> A: k3 虽然理论上无偏且非负，但当策略偏离参考策略较大时（RL 后期常见），k3 中的 $1/r$ 项导致方差爆炸。k1 = $-\log r$ 方差更可控，训练更稳定。这是一个理论正确性 vs 实践稳定性的权衡。

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/Composer2/Composer2|Composer 2 Technical Report]] -- Cursor Research Team, 2025

**后续发展**：
- [[Kimi K2.5]] -- Composer 2 的基座模型
- [[CursorBench]] -- Cursor 自建的 agentic coding benchmark

## 参考资料

> [!paper] 论文出处
> Cursor Research Team (Anysphere). *Composer 2 Technical Report*. 2025.
