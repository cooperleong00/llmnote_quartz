---
type: method
status: complete
description: Post-training 第一阶段，通过高质量指令数据将预训练模型转化为能遵循指令的助手
aliases:
  - 监督微调
  - Supervised Fine-Tuning
  - 指令微调
prerequisites:
  - "[[Transformer]]"
  - "[[Cross-Entropy Loss]]"
tags:
  - post-training
  - alignment
  - fine-tuning
created: 2026-01-27
updated: 2026-02-01T01:12
---

# SFT (Supervised Fine-Tuning)

监督微调（SFT, Supervised Fine-Tuning）是 Post-training 的第一阶段，通过在高质量的指令-回答数据上进行监督学习，将预训练模型转化为能够遵循指令、进行对话的助手。

> [!paper] 论文出处
> Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), NeurIPS 2022
> Wei et al., "Finetuned Language Models Are Zero-Shot Learners" (FLAN), ICLR 2022

---

## 动机

> [!intuition] 为什么需要 SFT？
> 预训练模型学会了"语言"，但不知道"如何对话"。
>
> - **预训练目标**：预测下一个 token（next token prediction）
> - **用户期望**：回答问题、遵循指令、进行有帮助的对话
>
> 这两者之间存在 **gap**：预训练模型可能会续写你的问题，而不是回答它。SFT 的目标就是弥合这个 gap。

**示例对比：**

```
用户输入: "什么是机器学习？"

预训练模型可能输出: "什么是深度学习？什么是神经网络？..."  (续写问题)
SFT 模型输出: "机器学习是一种人工智能方法，它使计算机能够..."  (回答问题)
```

---

## 核心机制

### 数据格式

SFT 使用 **instruction-response pairs**（指令-回答对）进行训练：

```json
{
  "instruction": "解释什么是梯度下降",
  "input": "",  // 可选的额外上下文
  "output": "梯度下降是一种优化算法，用于最小化损失函数..."
}
```

对于多轮对话，数据格式通常包含完整的对话历史：

```json
{
  "messages": [
    {"role": "user", "content": "什么是 Transformer？"},
    {"role": "assistant", "content": "Transformer 是一种基于注意力机制的神经网络架构..."},
    {"role": "user", "content": "它和 RNN 有什么区别？"},
    {"role": "assistant", "content": "主要区别在于..."}
  ]
}
```

### 训练目标

> [!math] 数学形式
> SFT 的训练目标是标准的 **交叉熵损失**（Cross-Entropy Loss），只在 response 部分计算 loss：
>
> $$
> \mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \mathbb{1}[t \in \text{response}] \cdot \log p_\theta(y_t | y_{<t}, x)
> $$
>
> 其中：
> - $x$ 是 instruction（输入）
> - $y$ 是 response（输出）
> - $\mathbb{1}[t \in \text{response}]$ 表示只在 response token 上计算 loss

> [!warning] 常见误区
> **误区**：SFT 在整个序列上计算 loss
>
> **事实**：通常只在 response 部分计算 loss（称为 "causal LM with response-only loss"）。在 instruction 部分计算 loss 会让模型学习"如何提问"，这不是我们想要的。

---

## 与 Pretraining 的区别

> [!comparison] Pretraining vs SFT

| 方面 | Pretraining | SFT |
|------|-------------|-----|
| **数据规模** | 万亿 tokens | 数万到数十万样本 |
| **数据来源** | 互联网爬取 | 人工编写/筛选 |
| **数据质量** | 参差不齐 | 高质量、精心设计 |
| **训练目标** | 全序列 next token prediction | Response-only loss |
| **学习内容** | 语言知识、世界知识 | 指令遵循、[[Chat Template|对话格式]] |
| **训练时长** | 数周到数月 | 数小时到数天 |
| **Learning Rate** | 较大（1e-4 ~ 6e-4） | 较小（1e-5 ~ 5e-5） |

> [!intuition] 本质区别
> - **Pretraining** 学习"语言是什么"（知识获取）
> - **SFT** 学习"如何使用语言"（行为塑造）
>
> SFT 不是让模型学习新知识，而是**激活**预训练中已有的知识，并以用户期望的方式呈现出来。

---

## 训练细节

### 超参数设置

> [!example] 典型超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| **Learning Rate** | 1e-5 ~ 5e-5 | 比预训练小 10-100 倍，避免灾难性遗忘 |
| **Epochs** | 1-3 | 过多 epoch 容易过拟合 |
| **Batch Size** | 32-128 | 根据 GPU 内存调整 |
| **Warmup Ratio** | 0.03-0.1 | 学习率预热 |
| **LR Scheduler** | Cosine | 余弦退火 |
| **Weight Decay** | 0.01-0.1 | 正则化 |

### 关键技巧

**1. Learning Rate 选择**

> [!warning] 学习率过大的风险
> 学习率过大会导致**[[Catastrophic Forgetting|灾难性遗忘]]**（Catastrophic Forgetting）：模型在学习新任务时忘记预训练知识。
>
> 经验法则：SFT 的学习率通常是预训练的 1/10 到 1/100。

**2. 数据混合**

为了保持模型的通用能力，常见做法是在 SFT 数据中混入部分预训练数据：

```
SFT 数据 : 预训练数据 = 9 : 1  (典型比例)
```

**3. Epoch 数量**

> [!intuition] 为什么 SFT 通常只训练 1-3 个 epoch？
> - SFT 数据量小，多次遍历容易过拟合
> - 目标是"激活"能力，不是"记忆"数据
> - 过拟合会导致模型只会回答训练集中的问题

**4. Packing vs Padding**

- **Padding**：每个样本独立，短样本用 padding token 填充
- **Packing**：多个样本拼接成一个长序列，提高 GPU 利用率

```
Padding: [样本1][PAD][PAD] | [样本2][PAD][PAD][PAD]
Packing: [样本1][样本2][样本3]  (用 attention mask 隔离)
```

---

## 在 RLHF Pipeline 中的位置

```
Pretrained Model
       ↓
   ┌───────────────────────────────────────┐
   │  SFT (Supervised Fine-Tuning)         │  ← 你在这里
   │  - 学习指令遵循                        │
   │  - 学习对话格式                        │
   │  - 输出：SFT Model (π_ref)            │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │  Reward Model Training                │
   │  - 基于 SFT Model 生成回答            │
   │  - 人类标注偏好                        │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │  RL Optimization (PPO)                │
   │  - SFT Model 作为 Reference (π_ref)   │
   │  - KL 惩罚防止偏离太远                 │
   └───────────────────────────────────────┘
       ↓
   Aligned Model
```

> [!intuition] SFT 的双重角色
> 在 [[RLHF]] 中，SFT 模型扮演两个角色：
> 1. **初始化**：RL 优化的起点
> 2. **参考策略**（$\pi_{ref}$）：KL 惩罚的锚点，防止模型偏离太远

---

## 与 RLHF、DPO 的关系

> [!comparison] SFT vs RLHF vs DPO

| 方面 | SFT | RLHF | DPO |
|------|-----|------|-----|
| **学习信号** | 正确答案 | 偏好排序 | 偏好排序 |
| **数据格式** | (instruction, response) | (prompt, chosen, rejected) | (prompt, chosen, rejected) |
| **优化方式** | 监督学习 | 强化学习 | 监督学习 |
| **学习目标** | 模仿示范 | 最大化奖励 | 增大偏好 margin |
| **能力获取** | 指令遵循 | 偏好对齐 | 偏好对齐 |

> [!intuition] 三者的关系
> - **SFT** 让模型"会回答"（基础能力）
> - **RLHF/DPO** 让模型"回答得好"（质量提升）
>
> SFT 是必要的前置步骤：没有 SFT，模型连基本的对话能力都没有，更谈不上优化回答质量。

### SFT 的局限性

> [!warning] 为什么 SFT 不够？
> 1. **数据瓶颈**：高质量示范数据难以覆盖所有场景
> 2. **模仿上限**：模型最多只能达到示范数据的水平
> 3. **无法表达偏好**：SFT 只能说"这是对的"，无法说"这个比那个好"
>
> 这就是为什么需要 [[RLHF]] 或 [[DPO]] 进一步优化。

---

## 数据质量的重要性

> [!intuition] 核心洞察
> **SFT 的效果 80% 取决于数据质量，20% 取决于训练技巧。**

### 高质量 SFT 数据的特征

1. **多样性**：覆盖各种任务类型（问答、写作、代码、推理等）
2. **准确性**：回答内容正确、无事实错误
3. **格式规范**：遵循一致的对话格式和风格
4. **难度适中**：太简单学不到东西，太难模型学不会

### 数据来源

| 来源 | 优点 | 缺点 |
|------|------|------|
| **人工编写** | 质量高、可控 | 成本高、规模有限 |
| **众包标注** | 规模大 | 质量参差不齐 |
| **模型生成** | 成本低、规模大 | 可能有幻觉、需要筛选 |
| **开源数据集** | 免费、即用 | 可能不适合特定场景 |

### 常用开源数据集

- **Alpaca**：52K 指令数据，GPT-3.5 生成
- **Dolly**：15K 人工编写数据
- **OpenAssistant**：多轮对话数据
- **ShareGPT**：用户分享的 ChatGPT 对话

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: SFT 和 Pretraining 的主要区别是什么？**
> A: 三个关键区别：(1) 数据规模——预训练用万亿 tokens，SFT 用数万样本；(2) 数据质量——SFT 需要高质量指令数据；(3) 训练目标——SFT 通常只在 response 部分计算 loss。
>
> **Q2: 为什么 SFT 的学习率要比预训练小很多？**
> A: 防止灾难性遗忘。学习率过大会破坏预训练学到的知识。SFT 的目标是"激活"能力，不是"重新学习"。
>
> **Q3: SFT 在 RLHF 中扮演什么角色？**
> A: 双重角色：(1) 作为 RL 优化的初始化；(2) 作为参考策略 $\pi_{ref}$，用于计算 KL 惩罚，防止模型偏离太远。
>
> **Q4: 为什么 SFT 之后还需要 RLHF/DPO？**
> A: SFT 只能让模型"模仿"示范数据，存在上限。RLHF/DPO 通过偏好信号让模型学习"什么更好"，可以超越示范数据的水平。
>
> **Q5: SFT 数据中，为什么通常只在 response 部分计算 loss？**
> A: 因为我们希望模型学习"如何回答"，而不是"如何提问"。在 instruction 部分计算 loss 会浪费模型容量。

---

## 延伸阅读

- [[InstructGPT]] — RLHF 的经典工作，详细描述了 SFT 阶段
- [[FLAN (2022)]] — 指令微调的早期工作
- [[Alpaca]] — 开源 SFT 数据集的代表
- [[LoRA]] — 参数高效的微调方法

