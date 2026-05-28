---
type: method
description: 训练 LLM 将推理链分段并生成压缩摘要（memento），通过原位 KV cache 遮蔽实现 2-2.5× 峰值内存减少，同时保持推理准确率
aliases:
  - Memento
  - 推理上下文自管理
prerequisites:
  - "[[KV Cache]]"
  - "[[Chain-of-Thought]]"
  - "[[Attention]]"
  - "[[SFT]]"
tags:
  - inference
  - efficiency
  - reasoning
  - post-training
  - serving
created: 2026-04-09
updated: 2026-04-09
---

# MEMENTO

MEMENTO（Microsoft Research, 2026）解决了 reasoning model 的一个核心基础设施问题：长链推理（[[Chain-of-Thought]]）过程中 [[KV Cache]] 线性增长导致的显存瓶颈。核心思想是训练模型自己管理推理上下文——将推理链分段，每段结束后生成一个压缩摘要（memento），然后遮蔽原始段落的 KV cache，只保留 memento。这样推理继续进行，但显存占用呈锯齿形而非单调上升，实现 $\sim 2\text{-}2.5\times$ 峰值 KV cache 减少，同时保持推理准确率。

## 动机

> [!intuition] 为什么需要这个？
> Reasoning model（如 QwQ, DeepSeek-R1）在推理时会生成数千甚至数万 token 的思维链。这个过程中的每个 token 都平等地占据 [[Attention]] 窗口中的位置——模型没有任何内建机制来标记"这个中间结果值得保留"或"这段推导可以压缩成一个简短结论"。一个 32K token 的 CoT 就是一个**扁平的、无结构的流**。
>
> 这直接导致两个问题：
> 1. **显存**：KV cache 随 token 数线性增长，限制了 batch size 和并发能力
> 2. **计算**：每个新 token 需要 attend to 所有历史 token，注意力计算量呈二次增长
>
> 已有方案要么丢弃信息（如 sliding window attention），要么需要重新开始生成（iterative summarization）。MEMENTO 的关键洞察是：**让模型自己学会何时压缩、如何压缩**，并且通过原位 KV cache 遮蔽在单次生成中完成。

## 核心机制

### 推理时的工作流程

MEMENTO 在推理时的行为可以概括为一个**生成-压缩-遮蔽**的循环：

1. **生成 thinking block $T_i$**：模型正常推理，生成一段语义连贯的推理
2. **生成 memento $M_i$**：模型将 $T_i$ 压缩成一个简短的状态摘要（通常是原始长度的 $5\text{-}20\times$ 压缩），保留关键结论、中间值、策略决策
3. **遮蔽 $T_i$ 的 KV cache**：$T_i$ 的 KV entries 被物理移除，但 $M_i$ 的 KV entries 保留
4. **继续推理**：后续 token 只能 attend to prompt + 所有 mementos ($M_1, M_2, \ldots$) + 当前 block

这产生了一个**锯齿形**的 KV cache 占用模式：每段推理结束时 cache 骤降，然后随新 block 的生成再次增长。

> [!intuition] 为什么是原位遮蔽而不是重新生成？
> 这是 MEMENTO 区别于传统 iterative summarization 方法的关键设计。当模型生成 $M_i$ 时，它可以 attend to $T_i$ 的全部内容——$M_i$ 的 KV states 是在**完整上下文**中计算的。遮蔽后，虽然 $T_i$ 的原始 token 不可见了，但 $M_i$ 的 KV states 隐式地编码了 $T_i$ 的信息。
>
> 这构成了一个**双信息流（dual information stream）**：
> - **显式通道**：memento 文本本身
> - **隐式通道**：memento KV states 中编码的原始 block 信息
>
> 实验表明，如果重新计算 $M_i$ 的 KV（让 $M_i$ 只 attend to 前面的 mementos，不看 $T_i$），AIME'24 准确率从 66.1% 暴跌到 50.8%（-15 pp）。这证明隐式 KV 通道承载了大量信息。

### 四个特殊 token

模型通过四个特殊 token 来结构化推理流程：

- `<|block_start|>` / `<|block_end|>`：标记 thinking block 边界
- `<|summary_start|>` / `<|summary_end|>`：标记 memento 边界

当 `<|summary_end|>` 生成时，推理引擎自动触发前一个 block 的 KV cache 遮蔽。

## 训练方法

### OpenMementos 数据集

训练数据从 OpenThoughts-v3（228K 推理 traces，来自 QwQ-32B）出发，通过一个四阶段 pipeline 构建：

1. **句子切分**：将连续推理流拆成原子"句子"（保护代码块和多行公式的完整性），将候选边界从 naive 切分的 ~397 减少到 ~187
2. **边界评分**：用 LLM judge 对每个句间边界打分 0-3（0 = 正在推导中间，3 = 主题转换的自然断点）
3. **分段优化**：算法化地选择最优分段——最大化边界得分的同时惩罚 block 长度不均匀（使用变异系数 $\sigma / \mu$ 作为正则项），每个 block 至少 200 tokens
4. **迭代 memento 生成**：LLM 压缩每个 block 为 memento，然后由 judge 按 6 维评分（公式、数值、方法命名、验证、无幻觉、结果优先），不通过（<8/10）的迭代改进。单次通过率仅 28%，两轮迭代提升到 92%

> [!warning] 为什么不直接让 LLM 做分段？
> 团队最初尝试过让 frontier LLM 直接分段，但失败了。分段本质上是一个**组合优化问题**，需要同时考虑 block 语义连贯性、大小均衡、边界自然性。LLM 擅长的是**局部判断**（"这里是不是好的切分点？"），而全局优化更适合算法化解决。

### 两阶段 SFT

| 阶段 | 注意力模式 | 目标 |
|------|-----------|------|
| Stage 1: Full Attention | 标准因果注意力，所有 token 可见 | 学习 block-memento 格式 |
| Stage 2: Memento Attention | 每个 memento 完成后遮蔽前一个 block | 学习在压缩上下文下继续推理 |

这遵循 **curriculum learning** 的思路：先在简单条件下学格式，再在困难条件下学内容管理。

关键超参数：学习率 $8 \times 10^{-5}$，cosine schedule，5% warmup，5 epochs/stage，batch size 512。

### RL 微调（可选）

> [!intuition] 为什么 RL 有效？
> SFT 后的准确率下降是**一致性问题**而非能力问题。Pass@64 分析表明 MEMENTO 和 base model 能解决几乎相同的问题集（Jaccard 相似度 96.4%），只是 MEMENTO 的正确答案不总是 mode。Majority voting 只需 $k=2\text{-}3$ 就能恢复 base 准确率。因此 RL 的作用是**重新分配概率质量**到正确的压缩 trace 上。

使用 CISPO（Clipped Importance-Sampled Policy Optimization，GRPO 变体）：

$$L = -\text{sg}\big(\text{clip}(r_t(\theta), 1-\epsilon_\text{low}, 1+\epsilon_\text{high})\big) \cdot A_t \cdot \log \pi_\theta(a_t \mid s_t)$$

配合 KL penalty（$\beta = 0.001$）防止 response-length collapse，以及 block 长度上限（7K tokens）防止模型学会生成更少更长的 block 来"绕过"压缩。

RL 后 Qwen3-8B 在 AIME'26 从 57.3% 提升到 64.9%，基本恢复到 control（64.7%）水平。

## 实验结果

> [!example] 关键数据（Qwen3-8B 为例）
>
> | 配置 | AIME'26 | Peak KV (GB) | KV 减少 |
> |------|---------|-------------|---------|
> | Base | 66.8% | 2.41 | — |
> | MEMENTO SFT | 57.3% | 1.02 | 0.39× |
> | MEMENTO + RL | 64.9% | 1.45 | 0.56× |
>
> Qwen3-32B 更大规模下差距更小：AIME'26 仅 -2.6 pp（72.6% vs 75.2%），peak KV 0.44×。
>
> **规模效应**：准确率差距随模型规模缩小（8B: -6.3 pp → 32B: -3.5 pp），表明更大模型更善于管理压缩上下文。

**跨模型族泛化**：在 Qwen3（8B/32B）、Phi-4-reasoning（14B）、Olmo-3-7B-Think 上均有效，使用同一份 OpenMementos 数据（来自 QwQ-32B 的 trace）训练。

**推理吞吐提升**：通过定制 vLLM 引擎实现原位 block masking，单 B200 GPU 上实现 $1.75\times$ token 吞吐提升、$1.58\times$ batch 完成加速。

> [!warning] 局限性：Hybrid attention 架构
> Olmo-3-7B-Think 使用 sliding-window + full attention 混合架构（32 层中 24 层是 sliding window），block masking 只对 8 个 full attention 层有效，KV 减少仅 $0.85\text{-}0.93\times$，远低于纯 full attention 模型的 $2\text{-}3\times$。

## 局限性

> [!warning] 边界条件
> - **Hybrid attention 架构受限**：sliding-window 层已有固定 KV cache 上限，block masking 带来的额外收益有限
> - **Failure mode：过度生成**：某些问题上 MEMENTO 会生成 $3\times$ 更多 token（可能因为频繁压缩导致信息丢失后重新推导），虽然峰值 KV 更低，但总 KV AUC 反而更高
> - **SFT 带来的固有损失**：在 strong reasoning model 上做 SFT 本身会导致性能下降（control run 也有类似下降），MEMENTO 的损失部分来自 SFT 本身
> - **依赖 vLLM 定制**：需要对推理引擎做 non-trivial 修改，现有框架均无原生支持动态 sparse attention mask
> - **数据生成成本**：OpenMementos pipeline 依赖 frontier LLM（GPT-5.x）做边界评分和 memento 生成/评判

> [!interview] 面试视角
> **Q: MEMENTO 和 sliding window attention 有什么区别？**
> A: Sliding window 是固定模式的注意力稀疏化，对所有 token 统一截断历史窗口。MEMENTO 是**语义感知**的——模型自己决定何时压缩，压缩的是语义完整的推理段落，并保留关键信息在 memento 中。更重要的是，MEMENTO 的 memento KV states 隐式编码了被遮蔽 block 的信息（双信息流），而 sliding window 丢弃的信息就完全消失了。
>
> **Q: 为什么 memento 的 KV states 如此重要？**
> A: 因为 memento 的 KV 是在完整 block 上下文中计算的。上层 attention 层的 KV states 编码了原始 block 的任务相关信息，即使这些信息不在 memento 文本中。移除这个隐式通道（recompute KV without block context）导致 15 pp 的准确率下降。这也是 MEMENTO 优于 restart-based summarization 方法的关键原因。

> [!paper] 论文出处
> Kontonis, Zeng, Garg, Chen, Tang, Wang, Awadallah, Horvitz, Langford, Papailiopoulos. *MEMENTO: Teaching LLMs to Manage Their Own Context.* Microsoft Research, 2026.
> 数据集：OpenMementos（228K traces，公开发布）

## 延伸阅读

**原始材料**：
- [[Clippings/Paper/memento/memento|MEMENTO 论文 Clipping]] — 完整论文提取

**相关方法**：
- [[KV Cache]] — MEMENTO 优化的核心对象
- [[Chain-of-Thought]] — MEMENTO 压缩的对象：长链推理 trace
