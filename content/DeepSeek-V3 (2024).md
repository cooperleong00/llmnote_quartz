---
type: paper
title: DeepSeek-V3 Technical Report
authors:
  - DeepSeek-AI
year: 2024
arxiv: "2412.19437"
description: 671B MoE 模型，采用 MLA 和 DeepSeekMoE 架构，首创无辅助损失的负载均衡策略和多 token 预测目标，仅需 2.788M H800 GPU 小时完成训练
aliases:
  - DeepSeek-V3
  - DeepSeek V3
prerequisites:
  - "[[MoE]]"
  - "[[GRPO]]"
  - "[[RoPE]]"
tags:
  - architecture
  - pretraining
  - post-training
  - moe
  - efficiency
  - rlhf
created: 2026-02-04
updated: 2026-02-04T00:32
---

# DeepSeek-V3

DeepSeek-V3 是 DeepSeek-AI 发布的 671B 参数 MoE 大语言模型，每个 token 激活 37B 参数。其核心贡献在于**极致的训练效率**：仅用 2.788M H800 GPU 小时（约 $5.576M）完成全部训练，且训练过程极其稳定，未出现任何不可恢复的 loss spike 或回滚。

## 核心贡献

### 1. 架构：MLA + DeepSeekMoE

DeepSeek-V3 继承了 DeepSeek-V2 验证过的两大架构创新：

#### Multi-head Latent Attention (MLA)

MLA 通过**低秩联合压缩** Key 和 Value 来减少 KV cache：

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t$$

其中 $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ 是压缩后的 latent vector，$d_c \ll d_h n_h$。推理时只需缓存压缩后的 $\mathbf{c}_t^{KV}$ 和解耦的 RoPE key $\mathbf{k}_t^R$，显著减少 KV cache 同时保持与标准 MHA 相当的性能。

#### DeepSeekMoE

相比传统 MoE（如 GShard），DeepSeekMoE 使用**更细粒度的专家**并**隔离部分共享专家**：

$$\mathbf{h}'_t = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}_i^{(r)}(\mathbf{u}_t)$$

- $N_s$ 个共享专家：所有 token 都经过
- $N_r$ 个路由专家：每个 token 选择 Top-K 个
- 使用 **Sigmoid** 计算 affinity score（而非 softmax）

### 2. Auxiliary-Loss-Free Load Balancing

传统 MoE 使用辅助损失（auxiliary loss）来平衡专家负载，但过大的辅助损失会损害模型性能。DeepSeek-V3 首创**无辅助损失的负载均衡策略**：

为每个专家引入 bias term $b_i$，仅用于路由决策（不影响 gating value）：

$$g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \text{Topk}(\{s_{j,t} + b_j\}, K_r) \\ 0, & \text{otherwise} \end{cases}$$

训练过程中动态调整 bias：
- 专家过载 → 降低 bias
- 专家欠载 → 提高 bias

这种方法比纯辅助损失方法取得更好的性能，同时保持负载均衡。

> [!intuition] 为什么这样设计？
> Bias 只影响"谁被选中"，不影响"选中后的权重"。这样既能控制负载分布，又不会扭曲专家的实际贡献。

### 3. Multi-Token Prediction (MTP)

DeepSeek-V3 采用**多 token 预测**训练目标，在每个位置预测多个未来 token：

- 使用 D 个顺序模块预测 D 个额外 token
- 保持每个预测深度的完整因果链
- 可用于 speculative decoding 加速推理

与 Gloeckle et al. (2024) 的并行预测不同，DeepSeek-V3 采用**顺序预测**，每个深度的输入结合了：
- 上一深度的表示 $\mathbf{h}_i^{k-1}$
- 下一个 token 的 embedding $\text{Emb}(t_{i+k})$

### 4. 训练效率优化

#### FP8 混合精度训练

首次在超大规模模型上验证 FP8 训练的可行性和有效性，实现训练加速和显存节省。

#### DualPipe 算法

高效的流水线并行算法：
- 更少的 pipeline bubbles
- 通过计算-通信重叠隐藏大部分通信开销
- 支持跨节点的细粒度专家，几乎零 all-to-all 通信开销

#### 训练成本

| 阶段 | GPU 小时 | 成本 |
|------|----------|------|
| Pre-training | 2,664K | $5.328M |
| Context Extension | 119K | $0.238M |
| Post-training | 5K | $0.01M |
| **总计** | **2,788K** | **$5.576M** |

每 1T tokens 仅需 180K H800 GPU 小时，在 2048 卡集群上约 3.7 天。

## Post-Training

### Supervised Fine-Tuning

1.5M 样本，覆盖多个领域：

**Reasoning Data**：从 DeepSeek-R1 蒸馏
- R1 生成的数据准确但存在 overthinking、格式差、过长等问题
- 通过 SFT + RL pipeline 训练 expert model
- 使用 rejection sampling 筛选高质量数据
- 目标：保留 R1 的准确性，同时保持简洁清晰

**Non-Reasoning Data**：DeepSeek-V2.5 生成 + 人工验证

### Reinforcement Learning

使用 [[GRPO]]（Group Relative Policy Optimization）：

$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

**Reward Model**：
- **Rule-based RM**：数学、代码等可验证任务
- **Model-based RM**：自由形式回答，训练时包含 CoT 以减少 reward hacking

## 性能表现

- **MMLU**: 88.5（超越所有开源模型）
- **MMLU-Pro**: 75.9
- **GPQA**: 59.1
- **MATH-500**: 超越 o1-preview
- **LiveCodeBench**: 开源模型 SOTA
- **Chinese SimpleQA**: 超越 GPT-4o 和 Claude-3.5-Sonnet

## 模型规格

| 参数 | 值 |
|------|-----|
| 总参数 | 671B |
| 激活参数 | 37B |
| 路由专家数 | 256 |
| 激活专家数 | 8 |
| 共享专家数 | 1 |
| 预训练数据 | 14.8T tokens |
| 最大上下文 | 128K |

## 相关资源

- 模型权重：[GitHub](https://github.com/deepseek-ai/DeepSeek-V3)
- 原始论文：[[Clippings/Paper/241219437v2/241219437v2|Paper Clipping]]
