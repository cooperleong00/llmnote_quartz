---
type: paper
description: 美团 560B MoE 模型，通过 Zero-Computation Experts 动态分配计算预算（18.6B-31.3B/token）和 Shortcut-Connected MoE 扩大计算-通信重叠窗口，30 天完成 20T token 训练
aliases:
  - LongCat-Flash
  - 美团 LongCat
prerequisites:
  - "[[Mixture of Experts]]"
  - "[[Multi-head Latent Attention]]"
  - "[[Multi-Token Prediction]]"
tags:
  - foundation-model
  - moe
  - architecture
  - efficiency
created: 2025-02-04
updated: 2026-02-04T18:38
---

# LongCat-Flash (2025)

LongCat-Flash 是美团 LongCat 团队发布的 560B 参数 [[Mixture of Experts|MoE]] 基础模型，核心创新在于**动态计算预算分配**和**高效训练推理架构**。通过 Zero-Computation Experts 机制，模型根据 token 的上下文重要性动态激活 18.6B-31.3B 参数（平均 27B），实现"不是所有 token 都需要同等计算量"的直觉。结合 Shortcut-Connected MoE 架构扩大计算-通信重叠窗口，在数万张加速卡上仅用 30 天完成 20T token 训练，推理速度超过 100 TPS，成本仅 $0.70/M output tokens。

> [!paper] 论文信息
> - **标题**: LongCat-Flash Technical Report
> - **机构**: Meituan LongCat Team
> - **时间**: 2025
> - **链接**: [HuggingFace](https://huggingface.co/meituan-longcat) | [GitHub](https://github.com/meituan-longcat)

## 动机

> [!intuition] 为什么需要动态计算分配？
> 传统 MoE 模型对每个 token 激活固定数量的专家（如 top-8），但实际上：
> - **简单 token**（如常见词、标点）只需少量计算即可准确预测
> - **困难 token**（如专业术语、复杂推理步骤）需要更多计算资源
>
> 这一现象被 [[Speculative Decoding]] 经验性地验证：小模型能可靠预测大模型在大多数"简单"token 上的输出。LongCat-Flash 将这一洞察内化到架构中，让模型自主学习计算资源的分配。

## 架构创新

### Zero-Computation Experts

Zero-Computation Experts 是 LongCat-Flash 的核心创新，通过在专家池中引入"零计算专家"实现动态计算预算。

> [!definition] 机制定义
> 在 $N$ 个标准 FFN 专家之外，增加 $Z$ 个零计算专家。零计算专家直接返回输入 $x_t$，不引入额外计算：
>
> $$\text{MoE}(x_t) = \sum_{i=1}^{N+Z} g_i \cdot E_i(x_t)$$
>
> 其中：
> $$E_i(x_t) = \begin{cases} \text{FFN}_i(x_t), & \text{if } 1 \leq i \leq N \\ x_t, & \text{if } N < i \leq N+Z \end{cases}$$
>
> 路由器选择 top-K 专家时，选中零计算专家意味着该 token 不需要那么多计算。

**计算预算控制**：为防止模型过度使用或不足使用零计算专家，LongCat-Flash 采用 PID 控制器动态调整专家偏置 $b_i$：

$$\Delta b_i = \mu \left( \frac{K_e}{K} \cdot \frac{1}{N} - \frac{T_i}{KT_{\text{all}}} \right)$$

其中 $K_e$ 是期望激活的 FFN 专家数量，$T_i$ 是路由到第 $i$ 个专家的 token 数。这种基于控制论的方法比固定偏置增量更鲁棒，能在专家数量扩展时保持稳定。

**负载均衡**：在 [[Loss-Free Load Balancing]] 的基础上，LongCat-Flash 将零计算专家作为额外的一组进行均衡，确保 FFN 专家与零计算专家的比例收敛到 $K_e : (K - K_e)$。

> [!example] 实际效果
> - 在匹配计算预算下，带零计算专家的模型（top-k=12，动态激活 4.2B-7.0B）比固定激活（top-k=8，固定 6B）有更低的验证损失
> - 训练约 20B token 后，平均激活专家数收敛到期望值，波动 <1%
> - 但标准差保持在较高水平（约 3），表明模型确实在不同 token 间分配了显著不同的计算资源

### Shortcut-Connected MoE (ScMoE)

ScMoE 通过跨层快捷连接重排执行流水线，扩大计算-通信重叠窗口。

> [!intuition] 为什么需要 ScMoE？
> MoE 模型的效率瓶颈在于**通信开销**：Expert Parallelism 要求先完成 all-to-all 通信将 token 路由到对应专家，再开始计算。传统 shared-expert 架构尝试用单个专家的计算时间掩盖通信，但窗口太小。
>
> ScMoE 的关键洞察：将前一个 block 的 dense FFN 计算与当前 MoE 层的 dispatch/combine 通信并行执行，创造更大的重叠窗口。

**质量验证**：在多种配置下（2.4B-16B MLA、3B-20B MHA、15B-193B GQA），ScMoE 与 baseline 的训练损失曲线几乎完全重合，证明这种重排不影响模型质量。

**效率收益**：
- **训练**：前一个 block 的计算可与 MoE 层的 dispatch/combine 完全并行
- **推理**：实现 Single Batch Overlap 流水线，理论 TPOT 比 [[DeepSeek-V3 (2024)]] 降低近 50%；NVLink 上的 TP 通信可与 RDMA 上的 EP 通信完全重叠

### Variance Alignment Design

LongCat-Flash 发现**方差不对齐**是小模型设计在大规模时失效的关键因素，提出两项修正：

**MLA Scale-Correction**：[[Multi-head Latent Attention]] 的低秩分解导致 query/key 各分量的方差与其来源维度成正比，但 RoPE 部分 $k_t^R$ 的方差与完整模型维度成正比。LongCat-Flash 引入缩放因子对齐方差：

$$\alpha_q = \sqrt{\frac{d_{\text{model}}}{d_q}}, \quad \alpha_{kv} = \sqrt{\frac{d_{\text{model}}}{d_{kv}}}$$

**Fine-grained Expert Variance Compensation**：采用 DeepSeek-MoE 的细粒度专家策略时，专家分割导致两个方差衰减来源：
1. **Gating 稀释**：softmax 概率分散到更多专家
2. **维度缩减**：每个专家的中间维度缩小

补偿因子 $\gamma = m$（$m$ 为分割倍数）应用于 MoE 输出：

$$\text{MoE}(x_t) = \gamma \left( \sum_{i=1}^{mN} g_i \cdot E_i(x_t) \right)$$

## 模型规格

| 参数 | 值 |
|------|-----|
| 总参数量 | 560B |
| 激活参数量 | 18.6B - 31.3B（平均 27B） |
| 层数 | 28（不含 MTP 层） |
| 隐藏维度 | 6144 |
| 注意力头数 | 64 |
| 每头维度 | 128 |
| KV 压缩维度 | 512 |
| Query 压缩维度 | 1536 |
| FFN 专家数 | 512 |
| 零计算专家数 | 256 |
| 每 token 激活专家数 | 12 |
| 词表大小 | 131,072 |
| 上下文长度 | 128k |

## 训练策略

### Hyperparameter Transfer

基于 width scaling 的超参数迁移策略，从小规模代理模型预测大模型的最优配置：

- 宽度缩放因子 $s = n_{\text{target}} / n_{\text{proxy}} = 8$
- 代理模型宽度 768
- 迁移规则：$\sigma^2_{\text{target}} = \sigma^2_{\text{proxy}} / s$，$\eta_{\text{target}} = \eta_{\text{proxy}} / s$

### Model Growth Initialization

采用 layer stacking 技术从半规模模型初始化：

$$L_{\text{target}} = \underbrace{L_{\text{small}} \circ L_{\text{small}} \circ \cdots \circ L_{\text{small}}}_{r}$$

- 先训练 14 层模型，再堆叠为 28 层
- 观察到特征性损失轨迹：初始上升后加速收敛，最终优于随机初始化
- 保留所有训练状态（样本计数器、学习率调度）

### Training Stability

**Router Stability**：监控两个指标防止负载均衡损失主导：
- Router Weight Similarity：专家权重向量的平均余弦相似度
- Gradient Norm Ratio $R_g$：建议保持 $R_g < 0.1$

**Hidden z-loss**：抑制大激活值，防止 BF16 训练中的数值错误：

$$\mathcal{L}_Z = \frac{\lambda}{T} \sum_{t=1}^{T} \left( \log \sum_{i=1}^{|z_t|} \exp(|z_t^i|) \right)^2$$

**Adam Epsilon**：大模型需要更小的 $\varepsilon$（LongCat-Flash 使用 1e-16），因为大规模模型的梯度 RMS 更小，默认 $\varepsilon$ 可能破坏自适应机制。

### 训练效率

- **数据量**：20T+ tokens
- **训练时间**：30 天
- **可用性**：98.48%（无需人工干预的故障恢复）
- **确定性计算**：保证实验可复现，支持 SDC 检测

## 性能对比

### Base Model

| Benchmark | DeepSeek-V3.1 | Llama-4-Maverick | Kimi-K2 | LongCat-Flash |
|-----------|---------------|------------------|---------|---------------|
| 总参数 | 671B | 402B | 1043B | **560B** |
| 激活参数 | 37B | 17B | 32B | **27B** |
| MMLU | 87.46 | 84.41 | 87.47 | 87.05 |
| MMLU-Pro | 59.29 | 63.90 | 68.36 | **70.32** |
| GPQA | 47.16 | 48.08 | 45.89 | **51.09** |
| BBH | 89.46 | 87.56 | 89.19 | **90.54** |

### Chat Model (Non-thinking)

| Benchmark | DeepSeek-V3.1 | Qwen3-MoE | Kimi-K2 | LongCat-Flash |
|-----------|---------------|-----------|---------|---------------|
| ArenaHard-V2 | 84.10 | **88.20** | 85.70 | 86.50 |
| IFEval | 86.69 | 88.54 | 88.91 | **89.65** |
| AIME25 | 49.27 | **68.33** | 50.66 | 61.25 |
| TerminalBench | 31.30 | 17.28 | 25.93 | **39.51** |
| $\tau^2$-Bench (telecom) | 38.50 | 22.50 | 67.50 | **73.68** |

> [!comparison] 与 DeepSeek-V3 的对比
> - **架构**：LongCat-Flash 采用 Zero-Computation Experts 实现动态计算，DeepSeek-V3 使用固定 top-k
> - **效率**：ScMoE 的 Single Batch Overlap 理论上比 DeepSeek-V3 的推理 TPOT 降低 50%
> - **规模**：LongCat-Flash 560B/27B vs DeepSeek-V3 671B/37B，更小的模型达到相近性能
> - **训练**：两者都采用 [[Multi-head Latent Attention]] 和 [[Multi-Token Prediction]]，但 LongCat-Flash 增加了 variance alignment 设计

## 局限性

> [!warning] 边界条件
> - **非 thinking 模型**：LongCat-Flash 是 non-thinking foundation model，在需要深度推理的任务（如 AIME）上不如 thinking 模型
> - **动态计算的可解释性**：虽然模型学会了差异化分配计算，但哪些 token 被分配更多计算、为什么，仍需进一步研究
> - **ScMoE 的硬件依赖**：效率收益依赖于 NVLink/RDMA 的异构通信能力

## 延伸阅读

**原始论文与相关材料**：
- [[Clippings/Paper/250901322v2/250901322v2|LongCat-Flash Technical Report 原文]]

**后续发展**：
- LongCat-Flash-Thinking — 基于 LongCat-Flash 的推理模型（独立论文）

**相关架构**：
- [[DeepSeek-V3 (2024)]] — 采用类似 MLA + MoE 架构的 671B 模型
- [[Kimi K2 (2025)]] — 1T 参数 MoE 模型，同样强调 Agentic 能力
- [[Qwen3 (2025)]] — 首创 thinking/non-thinking 双模式统一框架
