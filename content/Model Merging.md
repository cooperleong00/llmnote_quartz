---
type: method
description: 将多个任务特定的微调模型合并为单一多能力模型，无需原始训练数据，只需模型参数
aliases:
  - 模型合并
  - Model Fusion
  - Weight Merging
prerequisites:
  - "[[SFT]]"
  - "[[LoRA]]"
tags:
  - fine-tuning
  - efficiency
  - multi-task
created: 2026-01-28
updated: 2026-02-11T23:26
---

# Model Merging

模型合并（Model Merging）是一种将多个任务特定的微调模型合并为单一多能力模型的技术。核心优势：**不需要原始训练数据，只需要模型参数**，就能获得多任务能力。

> [!paper] 论文出处
> - Model Soups: Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time", ICML 2022
> - Task Arithmetic: Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023
> - TIES-Merging: Yadav et al., "Resolving Interference When Merging Models", NeurIPS 2023
> - DARE: Yu et al., "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch", ICML 2024

---

## 动机

> [!intuition] 为什么需要 Model Merging？
> **[[Multi-task Learning]] 的困境**：
> - 需要收集所有任务的训练数据
> - 需要同时访问所有数据进行联合训练
> - 任务间可能存在负迁移（negative transfer）
> - 添加新任务需要重新训练
>
> **Model Merging 的解决方案**：
> - 各任务独立微调，互不干扰
> - 合并时只需要模型权重，不需要数据
> - 可以灵活组合不同来源的模型
> - 推理时只需要一个模型，无额外开销

**核心前提**：所有待合并模型必须从**同一个预训练模型**微调而来（homologous models）。

> [!comparison] 与其他多能力整合方式的对比
> - **[[Mixture of Experts]]**：通过路由机制动态选择专家，推理时有额外开销；Model Merging 合并后是单一模型
> - **[[LoRA]]**：参数高效微调方法，LoRA adapters 也可以用类似方式合并（LoRA Merging）

---

## 核心概念：Delta Parameters

理解 Model Merging 的关键是 **delta parameters**（任务向量）：

$$
\delta_t = \theta_t - \theta_{\text{base}}
$$

其中：
- $\theta_{\text{base}}$：预训练基座模型参数
- $\theta_t$：在任务 $t$ 上微调后的模型参数
- $\delta_t$：任务 $t$ 的 delta parameters（也称 task vector）

> [!intuition] 关键洞察
> [[SFT]] 产生的 delta parameters 通常**非常小**（约 0.002 范围内），且**极度冗余**。这说明 SFT 主要是"解锁"预训练模型已有的能力，而非引入全新的能力。

---

## 核心方法

### 1. Average Merging (Model Soups)

最简单的方法：直接平均多个模型的权重。

$$
\theta_{\text{merged}} = \frac{1}{n} \sum_{i=1}^{n} \theta_i
$$

**优点**：简单、无超参数
**缺点**：对所有参数同等对待，忽略了参数重要性差异

### 2. Task Arithmetic

在 delta parameters 空间进行操作，引入 scaling term $\lambda$：

$$
\theta_{\text{merged}} = \theta_{\text{base}} + \lambda \sum_{t=1}^{n} \delta_t
$$

**核心思想**：任务向量可以像向量一样进行算术运算
- **加法**：合并多个任务能力
- **减法**：移除某种能力（如去除有害行为）
- **缩放**：控制任务能力的强度

**超参数**：$\lambda$ 通常在 0.3-1.0 之间，需要调优

### 3. Fisher Merging

使用 Fisher Information Matrix 作为参数重要性的权重：

$$
\theta_{\text{merged}} = \sum_{t=1}^{n} \frac{F_t}{\sum_{j} F_j} \odot \theta_t
$$

其中 $F_t$ 是任务 $t$ 的 Fisher Information，衡量参数对该任务的重要性。

**优点**：考虑参数重要性
**缺点**：需要计算 Fisher Information，计算开销大

### 4. RegMean

将合并问题建模为线性回归优化：

$$
\theta_{\text{merged}} = \arg\min_\theta \sum_{t=1}^{n} \| X_t \theta - X_t \theta_t \|^2
$$

通过最小化合并模型与各任务模型在特征空间的差异来求解。

### 5. [[TIES-Merging]]

专门解决参数干扰（interference）问题，识别了两种干扰来源：冗余参数的干扰和符号方向的冲突。

**三步流程**：
1. **Trim**：移除小幅度的 delta（保留 top-k%），消除冗余参数干扰
2. **Elect Sign**：对每个参数，按总变化量选择多数方向，解决符号冲突
3. **Disjoint Merge**：只合并与选定符号一致的参数值，取均值

> [!intuition] 为什么需要 TIES？
> 不同任务的 delta 可能在同一参数上有**相反方向**的更新（sign disagreement），直接相加会相互抵消。同时大量冗余参数会稀释真正重要的更新。TIES 通过修剪 + 符号选举同时解决这两个问题。
>
> 详见 [[TIES-Merging]] 笔记。

---

## DARE：Drop And REscale

[[231103099v3|DARE]] 是一种**预处理方法**，可以作为 plug-in 与其他合并方法结合使用。

### 核心思想

随机丢弃大部分 delta parameters，然后重新缩放保留的参数以保持期望不变。

### 算法流程

**Step 1: Drop**
$$
m \sim \text{Bernoulli}(p)
$$
$$
\tilde{\delta} = (1 - m) \odot \delta
$$

**Step 2: Rescale**
$$
\hat{\delta} = \frac{\tilde{\delta}}{1 - p}
$$

其中：
- $p$：drop rate（丢弃概率），通常 0.9-0.99
- $m$：二值 mask，1 表示丢弃
- $\odot$：逐元素乘法

### 理论分析

> [!math] 期望保持不变
> $$
> \mathbb{E}[\hat{\delta}] = \mathbb{E}\left[\frac{(1-m) \odot \delta}{1-p}\right] = \frac{(1-p) \cdot \delta}{1-p} = \delta
> $$
>
> 通过 rescale，保证了处理后的 delta 在期望意义上与原始 delta 相同。

### 关键发现

> [!intuition] SFT Delta 的冗余性
> DARE 论文的核心发现：SFT 产生的 delta parameters **极度冗余**，可以丢弃 90%-99% 而几乎不影响性能。
>
> 这意味着：
> 1. SFT 主要是"解锁"预训练能力，而非引入新能力
> 2. 大部分 delta 是噪声或冗余信息
> 3. 模型越大，能容忍的 drop rate 越高

> [!warning] 重要区别
> **丢弃 delta parameters** vs **丢弃 fine-tuned parameters**：
> - 丢弃 delta（$\delta = \theta_{\text{ft}} - \theta_{\text{base}}$）：性能几乎不变
> - 直接丢弃 fine-tuned 参数（$\theta_{\text{ft}}$）：**灾难性性能下降**
>
> 这说明预训练权重是基础，delta 只是微调。

### DARE 的优势

1. **减少参数干扰**：稀疏化后的 delta 更不容易相互冲突
2. **即插即用**：可与 Task Arithmetic、TIES 等方法结合
3. **无需额外数据**：纯参数空间操作

---

## 方法对比

> [!comparison] 各方法对比

| 方法 | 核心思想 | 优点 | 缺点 |
|------|----------|------|------|
| Average | 简单平均 | 无超参数 | 忽略重要性 |
| Task Arithmetic | 缩放求和 | 支持加减操作 | 需调 $\lambda$ |
| Fisher | 重要性加权 | 理论优雅 | 计算开销大 |
| [[TIES-Merging\|TIES]] | 符号选举 | 解决冲突 | 多步骤 |
| DARE | 稀疏化 | 即插即用 | 引入随机性 |

---

## 局限性

> [!warning] Model Merging 的边界

1. **同源要求**：所有模型必须从同一个 base model 微调，不能合并不同架构或不同预训练的模型

2. **任务冲突**：当任务差异过大时，合并效果会显著下降
   - 例如：代码生成 + 创意写作可能冲突

3. **性能上限**：合并后的模型通常不如针对单任务专门训练的模型
   - 是"多面手"而非"专家"

4. **超参数敏感**：$\lambda$、drop rate 等需要调优，不同任务组合可能需要不同设置

5. **理论理解有限**：为什么 weight averaging 能 work？目前缺乏完整的理论解释
   - 与 loss landscape 的平坦性有关（[[Mode Connectivity]]）

---

## 面试视角

> [!interview] 常见问题

**Q: Model Merging 和 Multi-task Learning 有什么区别？**
A: Multi-task Learning 需要同时访问所有任务的数据进行联合训练；Model Merging 只需要各任务独立微调后的模型参数，不需要原始数据。这使得 Model Merging 更灵活，可以合并来自不同来源的模型。

**Q: DARE 为什么能丢弃 90% 的 delta 而不影响性能？**
A: 这揭示了 SFT delta 的冗余性。SFT 主要是"解锁"预训练模型已有的能力，而非引入全新能力。大部分 delta 是噪声或冗余信息，真正重要的信息只占很小比例。

**Q: 什么情况下 Model Merging 效果不好？**
A: 1) 任务差异过大导致严重冲突；2) 模型不是从同一个 base model 微调；3) 某些任务需要的能力在预训练中完全没有。

**Q: TIES-Merging 解决了什么问题？**
A: 解决了参数干扰——包括冗余参数的干扰和符号方向的冲突。不同任务可能在同一参数上有相反方向的更新，直接相加会相互抵消；同时大量冗余参数会稀释重要更新。TIES 通过修剪冗余 + 符号选举 + 不相交合并三步解决。详见 [[TIES-Merging]]。

