---
type: paper
description: OpenAI 提出的 RLHF 经典论文，首次将三阶段对齐流程（SFT → RM → PPO）应用于大规模语言模型
aliases:
  - InstructGPT (2022)
prerequisites:
  - "[[SFT]]"
  - "[[Reward Model]]"
  - "[[PPO]]"
  - "[[RLHF]]"
tags:
  - post-training
  - alignment
  - paper
created: 2026-01-28
updated: 2026-02-01T01:03
---

# InstructGPT (2022)

**Training language models to follow instructions with human feedback**

Ouyang et al., OpenAI, NeurIPS 2022

> [!paper] 论文信息
> - 作者：Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida 等
> - 机构：OpenAI
> - 发表：NeurIPS 2022
> - 链接：[arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

---

## 核心贡献

> [!intuition] 一句话总结
> InstructGPT 首次在大规模语言模型上验证了 [[RLHF]] 的有效性：**1.3B 参数的对齐模型优于 175B 的未对齐模型**。

这篇论文的历史意义在于：
1. **确立了三阶段对齐范式**：SFT → Reward Model → PPO，成为后续 ChatGPT、Claude 等模型的基础
2. **证明对齐比规模更重要**：小模型 + 对齐 > 大模型 + 无对齐
3. **提出 HHH 框架**：Helpful（有帮助）、Honest（诚实）、Harmless（无害）

---

## 方法：三阶段 RLHF

![[Clippings/Paper/220302155v1/_page_2_Figure_0.jpeg]]

### Step 1: 监督微调 (SFT)

收集人工编写的高质量示范数据，对 GPT-3 进行监督微调。

**数据来源**：
- Labeler 编写的 prompt（用于 bootstrap）
- OpenAI API Playground 用户提交的 prompt

**数据规模**：约 13k 训练 prompt

**训练细节**：
- 16 epochs，cosine learning rate decay
- Residual dropout 0.2
- 虽然 1 epoch 后 validation loss 开始过拟合，但继续训练能提升 RM score 和人类偏好

### Step 2: 奖励模型训练 (RM)

从 [[SFT]] 模型移除最后的 unembedding layer，训练输出标量 reward。

**数据收集**：
- 每个 prompt 展示 K=4~9 个回答让 labeler 排序
- 产生 $\binom{K}{2}$ 个 pairwise 比较
- 数据规模：约 33k 训练 prompt

**损失函数**（基于 [[Bradley-Terry Model]]）：

$$\mathcal{L}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x,y_w,y_l)\sim D} \left[\log\sigma\left(r_\theta(x,y_w) - r_\theta(x,y_l)\right)\right]$$

> [!warning] 训练技巧
> 将同一 prompt 的所有 $\binom{K}{2}$ 比较作为单个 batch element 训练，避免过拟合且更高效（只需 K 次前向传播而非 $\binom{K}{2}$ 次）。

**模型选择**：只使用 6B RM，因为 175B RM 训练不稳定，不适合作为 RL 的 value function。

### Step 3: 强化学习优化 (PPO)

使用 [[PPO]] 优化 SFT 模型，最大化 RM 给出的 reward。

**目标函数**：

$$\text{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi_\phi^{RL}}} \left[r_\theta(x,y) - \beta \log\frac{\pi_\phi^{RL}(y|x)}{\pi^{SFT}(y|x)}\right] + \gamma \mathbb{E}_{x\sim D_{pretrain}} \left[\log\pi_\phi^{RL}(x)\right]$$

其中：
- 第一项：最大化 reward
- 第二项：[[KL Divergence]] 惩罚，防止偏离 SFT 模型太远（防止 [[Reward Hacking]]）
- 第三项（PPO-ptx）：混合预训练梯度，减少 alignment tax

**两种变体**：
- **PPO**：$\gamma = 0$，纯 RLHF
- **PPO-ptx**：$\gamma > 0$，混合预训练数据，减少公共 NLP 数据集上的性能退化

---

## 关键实验结果

### 1. 对齐效果显著

![[Clippings/Paper/220302155v1/_page_1_Figure_0.jpeg]]

- **1.3B InstructGPT > 175B GPT-3**：小模型对齐后优于 100 倍大的未对齐模型
- **175B InstructGPT vs 175B GPT-3**：85% 的情况下人类更偏好 InstructGPT
- **vs few-shot GPT-3**：71% 偏好 InstructGPT

### 2. 真实性提升

- TruthfulQA 上，InstructGPT 生成真实且有信息量的回答的频率是 GPT-3 的两倍
- 闭域任务（如摘要）的幻觉率：InstructGPT 21% vs GPT-3 41%

### 3. 毒性降低

- 在 RealToxicityPrompts 上，使用 "respectful" 指令时，InstructGPT 毒性输出减少约 25%
- 但在 bias 指标上没有显著改善

### 4. [[Alignment Tax]] 可控

> [!definition] Alignment Tax
> 对齐过程导致的能力退化。InstructGPT 在某些公共 NLP 数据集（SQuAD, DROP, HellaSwag）上性能下降。

**解决方案**：PPO-ptx 通过混合预训练梯度，大幅减少性能退化，同时不影响人类偏好评分。

### 5. 泛化能力

- 对 held-out labeler 的偏好泛化良好（与训练 labeler 一致）
- 能泛化到训练分布外的任务（如非英语指令、代码任务）

---

## 数据集构成

| 数据集 | 规模 | 用途 |
|--------|------|------|
| SFT 数据 | ~13k prompts | 监督微调 |
| RM 数据 | ~33k prompts | 奖励模型训练 |
| PPO 数据 | ~31k prompts | RL 优化（无人工标签） |

**任务分布**（RM 数据集）：
- 生成类任务（开放式生成、头脑风暴）：~57%
- 分类和 QA：~18%
- 其他（摘要、对话、提取等）

---

## 局限性

> [!warning] 论文承认的局限性

1. **对齐目标的局限**：
   - 只对齐到约 40 名标注员的偏好
   - 标注员主要是英语使用者，不代表全球用户
   - 标注员之间的一致率约 73%

2. **模型行为的局限**：
   - 仍会生成有毒、有偏见的内容
   - 仍会编造事实
   - **最大问题**：即使指令有害，模型也会遵循

3. **方法论的局限**：
   - 大多数比较只有 1 个标注员标注
   - 训练时优先 helpfulness，可能与 harmlessness 冲突

---

## 历史意义与后续影响

> [!intuition] 为什么这篇论文重要？

InstructGPT 是 **RLHF 从学术研究走向工业应用的里程碑**：

1. **ChatGPT 的前身**：InstructGPT 的方法直接演化为 ChatGPT
2. **确立行业标准**：三阶段流程成为对齐的标准范式
3. **启发后续研究**：
   - [[DPO]]：简化 RLHF，去掉 RM 和 RL
   - [[Constitutional AI]]：用 AI 反馈替代人类反馈
   - [[RLAIF]]：AI 生成偏好数据
   - [[KTO]]：不需要 pairwise 数据

---

## 面试要点

> [!interview] 常见问题

**Q: InstructGPT 的核心创新是什么？**
A: 首次在大规模 LLM 上验证 RLHF 有效性，证明 1.3B 对齐模型 > 175B 未对齐模型。

**Q: 三阶段流程是什么？**
A: SFT（监督微调）→ RM（奖励模型训练）→ PPO（强化学习优化）。

**Q: 什么是 Alignment Tax？如何解决？**
A: 对齐导致的能力退化。InstructGPT 用 PPO-ptx（混合预训练梯度）来缓解。

**Q: 为什么用 6B 而不是 175B 的 RM？**
A: 175B RM 训练不稳定，不适合作为 RL 的 value function。

**Q: InstructGPT 的主要局限是什么？**
A: 即使指令有害也会遵循；只对齐到少数标注员的偏好；仍会幻觉和生成有毒内容。

---

## 相关概念

**核心方法**：
- [[RLHF]] — 本文的核心方法
- [[SFT]] — 第一阶段
- [[Reward Model]] — 第二阶段
- [[PPO]] — 第三阶段的优化算法

**数学基础**：
- [[Bradley-Terry Model]] — RM 损失函数的理论基础
- [[KL Divergence]] — PPO 中的约束项

**后续发展**：
- [[DPO]] — 简化 RLHF，直接优化偏好
- [[Constitutional AI]] — Anthropic 的自我改进方法
- [[RLAIF]] — 用 AI 反馈替代人类反馈
- [[KTO]] — 不需要 pairwise 数据的对齐方法

**相关问题**：
- [[Reward Hacking]] — RLHF 的核心挑战
