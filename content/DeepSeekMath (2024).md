---
type: paper
description: DeepSeek 数学推理模型 technical report，首次提出 GRPO 算法，通过 120B 数学 token 预训练和 RL 微调在 MATH 达到 51.7%
aliases:
  - DeepSeekMath
prerequisites:
  - "[[RLHF]]"
  - "[[PPO]]"
  - "[[SFT]]"
tags:
  - post-training
  - math-reasoning
  - deepseek
created: 2026-02-04
updated: 2026-02-04T01:12
---

# DeepSeekMath (2024)

DeepSeekMath 是 DeepSeek 发布的数学推理专用模型，在 MATH benchmark 上达到 51.7%（接近 GPT-4 的 52.9%），是当时最强的开源数学模型。这篇 technical report 的核心贡献有两个：一是构建了 120B token 的高质量数学预训练语料 DeepSeekMath Corpus；二是首次提出 [[GRPO]]（Group Relative Policy Optimization），一种比 [[PPO]] 更高效的 RL 算法。

> [!paper] 论文信息
> - **标题**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
> - **机构**: DeepSeek-AI
> - **时间**: 2024.02
> - **链接**: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

## 核心贡献

1. **DeepSeekMath Corpus**: 从 Common Crawl 构建的 120B token 数学语料，是 OpenWebMath 的 9 倍
2. **GRPO 算法**: 用 group relative advantage 替代 value function，显著降低 RL 训练的内存开销
3. **训练洞察**: 代码预训练有助于数学推理，arXiv 论文对数学能力提升有限

## 模型架构与规格

DeepSeekMath 基于 DeepSeek-Coder-Base-v1.5 7B 初始化，而非通用 LLM。这一选择基于一个重要发现：**代码预训练有助于数学推理能力**。

| 模型 | 参数量 | 基座 | 训练数据 |
|------|--------|------|----------|
| DeepSeekMath-Base | 7B | DeepSeek-Coder-v1.5 | 500B tokens (含 120B 数学) |
| DeepSeekMath-Instruct | 7B | DeepSeekMath-Base | 数学指令数据 |
| DeepSeekMath-RL | 7B | DeepSeekMath-Instruct | GRPO + 144K 问题 |

## 数学预训练数据

### 数据构建流程

DeepSeekMath Corpus 通过迭代式数据挖掘从 Common Crawl 构建：

1. **种子语料**: 以 OpenWebMath 作为初始正样本
2. **fastText 分类器**: 训练二分类器识别数学相关网页
3. **迭代扩展**: 识别高数学密度的域名，人工标注 URL 模式，扩充正样本
4. **去污染**: 过滤包含 benchmark 题目的网页（10-gram 精确匹配）

经过 4 轮迭代，最终获得 35.5M 数学网页，共 120B tokens。

### 与其他数学语料对比

| 语料 | 规模 | GSM8K | MATH |
|------|------|-------|------|
| MathPile | 8.9B | 2.7% | 3.3% |
| OpenWebMath | 13.6B | 11.5% | 8.9% |
| Proof-Pile-2 | 51.9B | 14.3% | 11.2% |
| **DeepSeekMath Corpus** | **120B** | **23.8%** | **13.6%** |

> [!intuition] 为什么 DeepSeekMath Corpus 更好？
> 1. **规模更大**: 是 OpenWebMath 的 9 倍，避免过早过拟合
> 2. **多语言**: 包含中英文数学内容
> 3. **迭代优化**: 通过人工标注不断提升分类器质量

## GRPO: 首次提出

这篇论文首次提出了 [[GRPO]]，详细内容见独立笔记。这里概述其核心思想：

### 动机

[[PPO]] 需要训练一个与 policy model 同等规模的 value function（critic），带来显著的内存和计算开销。在 LLM 场景下，reward 通常只在序列末尾给出，这使得训练一个 token-level 准确的 value function 变得困难。

### 核心改进

GRPO 用 **group relative advantage** 替代 value function：
- 对每个问题采样 G 个输出
- 用组内 reward 的均值和标准差归一化，得到相对优势
- 无需额外的 critic 网络

$$\hat{A}_{i} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

### 两种监督模式

1. **Outcome Supervision**: 只在序列末尾给 reward，所有 token 共享同一 advantage
2. **Process Supervision**: 在每个推理步骤给 reward，advantage 为后续步骤 reward 之和

## 训练流程

### 三阶段训练

```
DeepSeek-Coder-v1.5 7B
        ↓ 继续预训练 (500B tokens, 含 120B 数学)
DeepSeekMath-Base 7B
        ↓ SFT (CoT + PoT + Tool-use 数据)
DeepSeekMath-Instruct 7B
        ↓ GRPO (144K 问题, 64 samples/question)
DeepSeekMath-RL 7B
```

### SFT 数据

指令微调数据包含三种推理格式：
- **Chain-of-Thought (CoT)**: 自然语言推理
- **Program-of-Thought (PoT)**: 生成代码求解
- **Tool-Integrated Reasoning**: 结合代码执行

### GRPO 训练细节

| 参数 | 值 |
|------|-----|
| 学习率 | 1e-6 |
| KL 系数 | 0.04 |
| 每问题采样数 | 64 |
| 最大长度 | 1024 |
| Batch size | 1024 |
| 训练数据 | 144K 问题 (GSM8K + MATH 格式) |

## 性能表现

### 主要结果

| 模型 | 参数量 | GSM8K | MATH |
|------|--------|-------|------|
| GPT-4 | - | 92.0% | 52.9% |
| Gemini Ultra | - | 94.4% | 53.2% |
| Qwen | 72B | 78.9% | 35.2% |
| WizardMath-v1.1 | 7B | 83.2% | 33.0% |
| **DeepSeekMath-Instruct** | **7B** | 82.9% | 46.8% |
| **DeepSeekMath-RL** | **7B** | **88.2%** | **51.7%** |

> [!intuition] 关键观察
> - 7B 模型超越所有 7B-70B 开源模型
> - RL 阶段仅用 GSM8K + MATH 数据，但在 out-of-domain 任务（如 CMATH）也有提升
> - Self-consistency (64 samples) 可达 60.9% on MATH

### RL 为什么有效？

论文通过 Pass@K 和 Maj@K 分析发现：
- RL 提升了 Maj@K 但没有提升 Pass@K
- 这说明 RL 的作用是**让输出分布更 robust**，而非增强基础能力
- 本质上是将正确答案从 TopK 提升到 Top1

> [!warning] 这意味着什么？
> RL 更像是"对齐"而非"能力增强"。模型本来就"会"解这些题，RL 只是让它更稳定地输出正确答案。

## 统一范式

论文提出了一个统一框架来理解 [[SFT]]、RFT、[[DPO]]、[[PPO]]、[[GRPO]] 等方法：

$$\nabla_{\theta} \mathcal{J}(\\theta) = \mathbb{E}_{(q, o) \sim \mathcal{D}} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} GC(q, o, t) \nabla_{\theta} \log \pi_{\theta}(o_t|q, o_{< t}) \right]$$

三个关键维度：
1. **Data Source**: 离线（SFT model 采样）vs 在线（当前 policy 采样）
2. **Reward Function**: 规则（答案正确性）vs 模型（[[Reward Model]]）
3. **Gradient Coefficient**: 如何根据 reward 调整梯度强度

| 方法 | Data Source | Reward | 特点 |
|------|-------------|--------|------|
| SFT | 人工标注 | - | GC = 1 |
| RFT | 离线采样 | 规则 | 只保留正确答案 |
| DPO | 离线采样 | 规则 | Pairwise 对比 |
| Online RFT | 在线采样 | 规则 | 实时探索 |
| PPO/GRPO | 在线采样 | 模型 | 差异化强化 |

> [!intuition] 核心洞察
> - **在线 > 离线**: Online RFT 后期显著优于 RFT
> - **差异化强化 > 二元强化**: GRPO 根据 reward 大小调整梯度，优于只区分对错的 RFT
> - **Process > Outcome**: 细粒度的步骤级监督优于序列级监督

## 训练洞察

### 代码训练有助于数学推理

从 DeepSeek-Coder 初始化比从通用 LLM 初始化效果更好。这部分回答了一个长期问题：**代码训练是否提升推理能力？** 至少对数学推理，答案是肯定的。

### arXiv 论文效果有限

尽管 arXiv 论文在数学预训练中被广泛使用，但实验表明它对数学 benchmark 的提升有限。可能原因：
- arXiv 论文的数学内容过于专业
- 与 benchmark 的问题类型不匹配

## 局限性

1. **几何和定理证明较弱**: 模型在三角形、椭圆等几何问题上表现不佳
2. **Few-shot 能力有限**: 受模型规模限制，few-shot 提升不如 GPT-4 明显
3. **数据选择偏差**: 预训练和微调数据可能存在领域偏差

## 后续影响

- [[GRPO]] 被 [[DeepSeek-V3 (2024)]] 采用作为 RL 训练算法
- 数学预训练数据构建方法被后续工作借鉴
- 统一范式为理解各种对齐方法提供了理论框架

## 延伸阅读

**原始论文**:
- [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

**相关方法**:
- [[GRPO]] — 本文首次提出的 RL 算法
- [[PPO]] — GRPO 的前身
- [[DPO]] — 另一种简化 RLHF 的方法
