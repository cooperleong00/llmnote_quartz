---
type: paper
description: 通过 segment-based memory overwrite 和 Multi-Conv DAPO 实现线性复杂度的无限长文本处理，从 8K 训练外推到 3.5M 任务性能损失 <5%
aliases:
  - MemAgent
tags:
  - long-context
  - rl
  - post-training
  - memory
created: 2026-03-06
updated: 2026-03-06
---

# MemAgent (2025)

MemAgent 提出了一种基于强化学习的长文本处理新范式：将文档分段处理，用固定长度的 memory 通过 **overwrite 策略** 迭代更新关键信息，最终基于 memory 生成答案。通过扩展 [[DAPO]] 算法到 Multi-Conv DAPO，实现了对这种 agent workflow 的端到端训练。

**核心贡献**：
- 线性复杂度 $O(N)$ 处理任意长度文本
- 从 8K 训练外推到 3.5M QA 任务，性能损失 <5%
- 在 512K RULER 测试中达到 95%+ 准确率

> [!paper] 论文信息
> **标题**: MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent
> **作者**: Hongli Yu et al. (ByteDance Seed & Tsinghua AIR)
> **发布**: 2025-07-04
> **arXiv**: 2507.02259
> **项目页**: https://memagent-sialab.github.io/

---

## 动机：长文本处理的三难困境

现有长文本处理方法面临三难选择：

| 方法 | 复杂度 | 外推能力 | 局限 |
|------|--------|----------|------|
| **Length Extrapolation** | $O(n^2)$ | 中等 | 性能下降、速度慢 |
| **Sparse/Linear Attention** | $O(n)$ | 差 | 需从头训练、并行化困难 |
| **Context Compression** | 可变 | 差 | 外推困难、破坏生成流程 |

> [!intuition] 人类如何处理长文本？
> 人类阅读长文档时，不会记住每个细节，而是：
> - **选择性记录**：提取关键信息做笔记
> - **主动遗忘**：丢弃冗余和无关信息
> - **固定容量**：笔记长度有限，需要不断更新覆盖
>
> MemAgent 将这一过程形式化为 RL 问题。

---

## 核心机制

### 1. Segment-based Memory Workflow

```
┌─────────────────────────────────────────────────────────┐
│  Long Document (N tokens)                               │
│  ┌────────┬────────┬────────┬─────┬────────┐           │
│  │ Chunk1 │ Chunk2 │ Chunk3 │ ... │ ChunkK │           │
│  └────────┴────────┴────────┴─────┴────────┘           │
└─────────────────────────────────────────────────────────┘
         │         │         │              │
         ▼         ▼         ▼              ▼
    ┌────────┐ ┌────────┐ ┌────────┐   ┌────────┐
    │ Memory │→│ Memory │→│ Memory │...│ Memory │
    │   m¹   │ │   m²   │ │   m³   │   │   mᴷ   │
    └────────┘ └────────┘ └────────┘   └────────┘
                                             │
                                             ▼
                                        ┌─────────┐
                                        │ Answer  │
                                        └─────────┘
```

**关键设计**：
- **固定长度 memory**：$|\mathbf{m}^k| = M$ 始终不变（如 1024 tokens）
- **Overwrite 策略**：每次读取新 chunk 后，完全覆盖旧 memory
- **线性复杂度**：每个 chunk 的计算量 $O(C + M)$，总复杂度 $O(N)$

### 2. 两阶段推理流程

**Context-Processing Module**（迭代更新 memory）：
```
Input: <problem> {question} </problem>
       <memory> {previous_memory} </memory>
       <section> {current_chunk} </section>

Output: Updated memory
```

**Answer-Generation Module**（基于 memory 生成答案）：
```
Input: <problem> {question} </problem>
       <memory> {final_memory} </memory>

Output: Answer in \boxed{}
```

---

## Multi-Conv DAPO 训练算法

### 问题：多轮独立上下文的优化

MemAgent 的一个样本会生成 **多个独立上下文的对话**（每次 memory 更新是一轮对话），这与传统 multi-turn 对话不同：

| 传统 Multi-turn | MemAgent Multi-Conv |
|----------------|---------------------|
| 对话轮次共享上下文 | 每轮对话上下文独立 |
| 可用 attention mask 处理 | 需要新的优化方法 |

### 解决方案：Multi-Conv DAPO

扩展 [[DAPO]] 算法，将每个对话视为独立优化目标：

> [!math] Multi-Conv DAPO 目标函数
>
> **Advantage 计算**（从最终答案的 reward 分配到所有对话）：
> $$
> \hat{A}_{i,j,t} = r_i - \text{mean}(\{R_i\}_{i=1}^G)
> $$
>
> **损失函数**（扩展到 (group, conversation, token) 三维）：
> $$
> \mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E} \left[ \frac{1}{\sum_{i=1}^G \sum_{j=1}^{n_i} |o_{i,j}|} \sum_{i=1}^G \sum_{j=1}^{n_i} \sum_{t=1}^{|o_{i,j}|} \left( \mathcal{C}_{i,j,t} - \beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) \right) \right]
> $$
>
> 其中：
> - $n_i$：样本 $i$ 生成的对话数（= chunk 数 + 1）
> - $o_{i,j}$：第 $i$ 个样本的第 $j$ 个对话
> - $\mathcal{C}_{i,j,t}$：clipped objective（继承自 [[GRPO]]）

**关键创新**：
- 只有最后一轮对话（包含答案）用于计算 reward
- Advantage 均匀分配到所有前序对话（memory 更新轮次）
- 遵循 [[DAPO#Dr. GRPO|Dr. GRPO]] 不对 advantage 做标准差归一化

---

## 从自回归建模视角理解 MemAgent

### 传统 LLM 的瓶颈

标准自回归 LLM 的联合概率：
$$
p(\mathbf{x}_{1:N}) = \prod_{n=1}^N p(x_n \mid \mathbf{x}_{1:n-1})
$$

**问题**：每个 token 都依赖完整历史，导致 $O(N^2)$ 复杂度。

### MemAgent 的分解

引入固定长度的 latent memory $\mathbf{m} \in \mathbb{V}^M$，将文档分为 $K$ 个 chunks：

$$
p(\mathbf{x}_{1:N}) = \sum_{\mathbf{m}^{1:K-1}} \prod_{k=1}^K \underbrace{p(\mathbf{c}^k \mid \mathbf{m}^{k-1})}_{\text{read}} \underbrace{p(\mathbf{m}^k \mid \mathbf{c}^k, \mathbf{m}^{k-1})}_{\text{write}}
$$

**解释**：
- **Read path**：基于当前 memory 读取 chunk（token-by-token 自回归）
- **Write path**：基于 chunk 和旧 memory 生成新 memory（自回归生成）
- **复杂度**：每步 $O(C + M)$，总计 $O(N)$

> [!intuition] 为什么需要 RL？
> Memory tokens 是 **latent variables**（没有监督信号），且通过 discrete overwrite 更新。
>
> 反向传播无法直接教会模型"什么该保留、什么该丢弃"。RL 通过最终答案的 reward 反向指导 memory 更新策略。

---

## 实验结果

### 主要发现

**训练设置**：
- 基座模型：Qwen2.5-7B/14B-Instruct
- 训练窗口：8K（1024 query + 5000 chunk + 1024 memory + 1024 output）
- 训练数据：32K 长度的 HotpotQA 合成数据

**外推能力**（RULER-HotpotQA 准确率）：

| 模型 | 7K | 28K | 112K | 448K | 896K | 1.75M | 3.5M |
|------|-----|-----|------|------|------|-------|------|
| Qwen2.5-7B-1M | 61.7 | 53.9 | 51.6 | 12.5 | 0.0 | - | - |
| QwenLong-L1-32B | 72.7 | 72.7 | 31.2 | 13.3 | 11.7 | - | - |
| **MemAgent-7B** | **82.0** | **78.9** | **79.7** | **74.2** | **76.6** | **75.8** | **71.1** |
| **MemAgent-14B** | **83.6** | **84.4** | **76.6** | **75.0** | **77.3** | **76.6** | **78.1** |

**关键观察**：
- 基线模型在 896K 时性能崩溃（远低于其 1M 理论窗口）
- MemAgent 从 8K 外推到 3.5M，性能仅下降 <5%
- 在 512K RULER 综合测试中达到 95%+ 准确率

### Ablation Study

**RL 训练的必要性**：

| 模型 | 28K | 112K | 448K | 896K |
|------|-----|------|------|------|
| Qwen2.5-7B-Instruct | 53.9 | 51.6 | 12.5 | 0.0 |
| + Memory（无 RL） | 65.2 | 58.3 | 42.7 | 35.9 |
| + Memory + RL | **78.9** | **79.7** | **74.2** | **76.6** |

**结论**：Memory 机制提供结构支持，但 RL 训练是学会正确使用 memory 的关键。

---

## 与相关工作的对比

### vs Length Extrapolation

| 方法 | 代表 | 复杂度 | 外推能力 |
|------|------|--------|----------|
| [[Length Extrapolation\|RoPE 外推]] | [[ALiBi]], NTK, YaRN | $O(n^2)$ | 中等 |
| **MemAgent** | - | $O(n)$ | 强（3.5M 无性能崩溃） |

MemAgent 不修改 positional encoding，而是通过 segment-based 处理绕过长度限制。

### vs Memory Mechanisms

| 方法 | Memory 类型 | 训练方式 | 可解释性 |
|------|------------|----------|----------|
| External Memory Modules | Feature space | 监督学习 | 低（隐式表示） |
| **MemAgent** | Token space | RL | 高（可读可编辑） |

MemAgent 的 memory 是 **token-level**，每个中间 memory 都是人类可读的文本。

### vs Context Compression

| 方法 | 压缩方式 | 生成流程 | 外推能力 |
|------|----------|----------|----------|
| Context Compression | Token/feature 压缩 | 需要额外模块 | 差 |
| **MemAgent** | Overwrite 策略 | 标准自回归 | 强 |

MemAgent 不破坏标准生成流程，兼容性和并行化更好。

---

## 局限性与未来方向

> [!warning] 当前局限
>
> **1. 任务范围**：
> - 主要在 QA 任务上验证，其他长文本任务（摘要、编辑）待探索
>
> **2. Memory 容量**：
> - 固定 1024 tokens，对于极复杂任务可能不足
> - 需要研究动态 memory 大小调整
>
> **3. 训练成本**：
> - Multi-Conv DAPO 需要多轮 rollout，训练成本高于单轮 RL
>
> **4. Chunk 划分**：
> - 当前使用固定长度 chunk，语义边界划分可能更优

**未来方向**：
- 扩展到更多长文本任务类型
- 探索 hierarchical memory（多层次 memory）
- 结合 [[Process Reward Model|PRM]] 提供 step-level 反馈
- 研究 memory 的可编辑性（人工干预 memory 内容）

---

## 延伸阅读

**核心依赖**：
- [[DAPO]] — Multi-Conv DAPO 的基础算法
- [[GRPO]] — DAPO 的前身，group-based RL
- [[Length Extrapolation]] — 传统长文本外推方法

**相关工作**：
- [[Context-Folding (2025)]] — 另一种基于 RL 的长文本处理方法，通过 branch-fold 管理上下文
- [[Agentic RL]] — RL 训练 agent workflow 的通用框架

**原始论文**：
- [[Clippings/Paper/2507.02259/2507.02259]] — 完整论文内容
