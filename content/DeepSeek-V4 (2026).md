---
type: paper
description: DeepSeek-V4 技术报告提出百万 token 上下文模型系列，通过 CSA/HCA 混合注意力、mHC、Muon、FP4 QAT 与 OPD 管线共同提升长上下文效率和推理能力
aliases:
  - DeepSeek-V4
  - DeepSeek V4
  - DeepSeek-V4 技术报告
prerequisites:
  - "[[Attention]]"
  - "[[Sparse Attention]]"
  - "[[Mixture of Experts]]"
  - "[[Multi-Token Prediction]]"
  - "[[Quantization]]"
  - "[[On-Policy Distillation]]"
tags:
  - architecture
  - attention
  - efficiency
  - scaling
  - pretraining
  - post-training
created: 2026-05-01
updated: 2026-05-01T12:44
---

# DeepSeek-V4 (2026)

DeepSeek-V4 是 DeepSeek-AI 在 2026 年发布的技术报告，目标是让开放模型原生支持百万 token 上下文，同时保持可训练、可部署、可后训练的工程效率。报告包含两个预览模型：DeepSeek-V4-Pro 有 1.6T 总参数、每 token 激活 49B；DeepSeek-V4-Flash 有 284B 总参数、每 token 激活 13B；二者都基于 [[Mixture of Experts|MoE]] 架构并支持 1M token context。

> [!paper] 论文出处
> 本笔记吸收自 [[Clippings/Paper/DeepSeek_V4/DeepSeek_V4|DeepSeek-V4 技术报告]]：DeepSeek-AI, 2026, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*。

## 核心问题

DeepSeek-V4 的主线可以概括为一个问题：当推理模型需要更长的上下文、更长的思考轨迹和更复杂的 agent 交互时，标准 [[Attention]] 的二次复杂度会把 test-time scaling 的收益卡在成本墙上。百万 token 上下文带来的挑战同时出现在三个层面：

- 架构层面：每个 query 访问完整 KV cache 会带来极高 FLOPs 和显存占用。
- 训练层面：MoE、超长序列、压缩注意力和新优化器会放大稳定性与并行实现难度。
- 后训练层面：RL、distillation、agent rollout 和工具环境都需要处理长轨迹与大规模状态。

DeepSeek-V4 的答案是一组协同设计：CSA/HCA 混合注意力降低长上下文注意力成本，[[Manifold-Constrained Hyper-Connections]] 稳定层间信号传播，[[Muon Optimizer]] 改善收敛和训练稳定性，FP4 QAT 降低部署成本，[[On-Policy Distillation]] 把多个领域专家合并到统一模型。

## 架构设计

DeepSeek-V4 继承了 [[DeepSeek-V3 (2024)|DeepSeek-V3]] 的两条重要路线：用 [[Mixture of Experts|MoE]] 解耦总参数量与每 token 计算量，用 [[Multi-Token Prediction]] 改善训练信号与推理效率。V4 的新增设计集中在长上下文注意力、残差连接和优化器。

> [!intuition] 设计直觉
> 百万 token 模型的瓶颈主要来自“每一层都为所有历史 token 保存和读取高精度 KV”。DeepSeek-V4 将历史 token 先压缩成更少的 KV entry，再让不同注意力分支承担不同粒度的信息读取：CSA 负责较精细的稀疏检索，HCA 负责更激进的全局压缩记忆。

### CSA：压缩后再稀疏选择

[[Compressed Sparse Attention]]（CSA）先把每 $m$ 个 token 的 KV 压缩成一个 entry，再用类似 [[Sparse Attention]] 的 top-$k$ 选择机制，让 query 只访问最相关的压缩 KV entry。报告中的 Pro/Flash 主要设置为 $m=4$，Pro 的 top-$k$ 为 1024，Flash 的 top-$k$ 为 512。

CSA 的核心流程是：

1. 对输入 hidden states 生成两组 KV entry 与对应压缩权重。
2. 用 softmax 权重把局部窗口内的多个 KV entry 聚合成一个 compressed KV entry。
3. 通过 lightning indexer 计算 query 与 compressed KV block 的相关性。
4. 对每个 query 选择 top-$k$ compressed KV entry 执行 core attention。
5. 额外保留一条 sliding window attention 分支，补足最近 token 的细粒度局部依赖。

这种设计让长距离信息访问从“扫描全部历史 token”变成“在压缩索引空间中检索少量候选块”。CSA 牺牲一部分逐 token 细节，换来 KV cache 与 attention FLOPs 的大幅下降。

### HCA：更重压缩的全局记忆

[[Heavily Compressed Attention]]（HCA）采用更大的压缩率 $m'$，报告中设置为 $m'=128$。HCA 把每 128 个 token 压缩成一个 KV entry，然后在压缩后的序列上执行 dense attention。它保留了全局覆盖能力，同时把序列长度缩小到原来的 $\frac{1}{128}$。

CSA 与 HCA 的分工可以这样理解：CSA 用稀疏选择保留较高分辨率的长程信息，HCA 用重压缩提供低成本的全局摘要。DeepSeek-V4 在层间交错使用二者，使模型同时具备细粒度检索和粗粒度全局建模能力。

### mHC：受约束的残差扩展

[[Manifold-Constrained Hyper-Connections]]（mHC）把残差流从 $d$ 维扩展为 $n_{\text{hc}}\times d$，在层输入、残差变换、层输出之间引入动态映射。普通 Hyper-Connections 容易在深层堆叠中带来数值不稳定；mHC 将残差映射矩阵 $B_l$ 约束到 doubly stochastic matrix 所在的 Birkhoff polytope：

$$
\mathcal{M}=\{M\in\mathbb{R}^{n\times n}\mid M\mathbf{1}_n=\mathbf{1}_n,\ \mathbf{1}_n^T M=\mathbf{1}_n^T,\ M\geq 0\}
$$

这个约束让残差映射的谱范数受控，从而降低层间信号放大风险。实现上，DeepSeek-V4 用 Sinkhorn-Knopp 迭代把原始矩阵投影到该流形上，报告中的 $t_{\text{max}}=20$。

### Muon：面向矩阵更新的优化器

[[Muon Optimizer]] 用于大多数参数，AdamW 继续用于 embedding、prediction head、RMSNorm 权重和部分 mHC 参数。Muon 的关键步骤是对梯度动量矩阵做近似正交化，再进行带权重衰减的更新。DeepSeek-V4 使用 hybrid Newton-Schulz iterations：前 8 步快速推近奇异值，后 2 步稳定到接近 1。

Muon 与 CSA/HCA 的组合也带来一个工程优势：DeepSeek-V4 在 attention query 和 compressed KV entry 上加入 RMSNorm，缓解 attention logits 爆炸，因此训练中无需使用 QK-Clip。

## 百万上下文的效率机制

DeepSeek-V4 的长上下文能力来自“少存、少读、低精度、可复用”四个层面的组合。

### KV cache 变小

CSA 把 KV 序列长度降到 $\frac{1}{m}$，HCA 把它降到 $\frac{1}{m'}$。在 1M context 场景下，报告称 DeepSeek-V4-Pro 相对 DeepSeek-V3.2 只需要 27% 的 single-token inference FLOPs 和 10% 的 KV cache；DeepSeek-V4-Flash 进一步降到 10% FLOPs 和 7% KV cache。相对常见的 BF16 GQA8 attention baseline，DeepSeek-V4 的 KV cache 在 1M context 设置下可降到约 2%。

### 低精度路径降低带宽压力

DeepSeek-V4 在 KV cache 中混合使用 BF16 与 FP8：RoPE 相关维度保持 BF16，其余维度用 FP8。报告还将 FP4 QAT 应用于 MoE expert weights 和 CSA indexer 的 QK path，使长上下文 top-$k$ selector 获得约 2 倍加速，同时保持 99.7% 的 KV entry recall。

这里与 [[Quantization]] 的关系很直接：量化承担部署效率任务，QAT 让模型提前适应低精度误差。DeepSeek-V4 在 rollout 和 inference 阶段直接使用真实 FP4 权重，使采样行为与线上部署保持一致。

### Contextual Parallelism 处理压缩边界

传统 context parallelism 沿序列维度切分 token。CSA/HCA 的压缩需要连续 $m$ 或 $m'$ 个 KV entry，压缩块可能跨 rank 边界，packed sequence 也会让各 rank 的压缩长度变化。DeepSeek-V4 使用两阶段通信：先把 rank 尾部未压缩 KV 发送给下一个 rank 以完成跨边界压缩，再 all-gather 压缩后的 KV 并用 fused select-and-pad 整理成固定布局。

这部分与 [[3D Parallelism]]、[[Expert Parallelism]] 的关系在于：百万上下文训练已经超出单一并行维度能处理的范围，模型并行、专家并行、序列并行和压缩注意力的通信模式需要一起设计。

### 推理侧缓存适配混合注意力

CSA/HCA、sliding window attention 和未完成压缩的 tail tokens 产生了多种 KV cache 形态。DeepSeek-V4 将 cache 分成 classical KV cache 与 state cache：前者存储压缩 KV，后者存储 SWA 与未压缩尾部状态。对于 shared-prefix serving，报告还设计了 on-disk KV cache：压缩 KV 直接落盘复用，SWA cache 则在 full caching、periodic checkpointing、zero caching 之间按部署场景选择。

## 训练与稳定性

DeepSeek-V4-Flash 在 32T tokens 上预训练，DeepSeek-V4-Pro 在 33T tokens 上预训练。训练从 4K sequence length 开始，逐步扩展到 16K、64K 和 1M。稀疏注意力先用 dense attention warmup，再在长序列阶段引入 lightning indexer 和 sparse attention。

MoE 大模型训练中，loss spike 被报告归因于 MoE layer outliers 与 routing 机制的相互放大。DeepSeek-V4 使用两项实用稳定化技巧：

- [[Anticipatory Routing]]：在 step $t$ 使用当前参数计算特征，同时用历史参数 $\theta_{t-\Delta t}$ 预先计算 routing indices。系统只在检测到 loss spike 时短期启用该模式，把额外开销控制在较低水平。
- [[SwiGLU Clamping]]：将 SwiGLU 的 linear component 限制在 $[-10,10]$，并将 gate component 上界设为 10，用数值裁剪抑制异常值。

报告明确指出，这两项方法的机制解释仍有待研究。这里的学习重点是：稳定性方案已经有效，但理论理解还处在经验阶段。

## Post-training：专家先分化，再统一蒸馏

DeepSeek-V4 的后训练采用两阶段范式。第一阶段训练多个领域专家，覆盖数学、代码、agent、instruction following 等方向；每个专家经过 SFT，再通过 [[GRPO|Group Relative Policy Optimization]] 等 RL 方法按领域 reward 优化。第二阶段用 multi-teacher [[On-Policy Distillation]] 将十多个专家模型合并成一个统一模型。

OPD 的目标写成：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)=\sum_{i=1}^{N}w_i\cdot D_{\mathrm{KL}}(\pi_\theta\|\pi_{E_i})
$$

这里 $\pi_\theta$ 是学生模型，$\pi_{E_i}$ 是第 $i$ 个专家教师，$w_i$ 是教师权重。训练轨迹来自学生自身，因此保持 on-policy；损失使用 reverse KL，让学生在自己的生成分布上贴近相关专家。

DeepSeek-V4 还采用 full-vocabulary logit distillation，直接计算完整词表上的 KL，减少 token-level KL 估计的高方差。为处理大词表与多教师成本，系统缓存教师最后一层 hidden states，训练时再通过对应 prediction head 重建 logits，并用专门的 TileLang kernel 计算 KL。

> [!comparison] 与 [[RLHF]] 的关系
> DeepSeek-V4 的 specialist 阶段仍包含 RL 和 [[Reward Model|reward signal]]，尤其面向可验证或半可验证任务；统一模型阶段转向 OPD，用专家教师的分布对齐来合并能力。GRM（Generative Reward Model）把生成与评价能力放在同一 actor 中训练，用少量 rubric-guided annotation 支持 hard-to-verify tasks 的评分。

## 评估信号

Base model 评估显示，DeepSeek-V4-Flash-Base 用更少 activated parameters 超过 DeepSeek-V3.2-Base 的多数指标，DeepSeek-V4-Pro-Base 在知识、推理、代码和长上下文上进一步提升。LongBench-V2 上，DeepSeek-V3.2-Base 为 40.2，V4-Flash-Base 为 44.7，V4-Pro-Base 为 51.5。

后训练模型中，DeepSeek-V4-Pro-Max 在多个开放模型基准上达到领先水平。报告中的关键结论包括：

- Knowledge：SimpleQA-Verified 和 Chinese-SimpleQA 上显著超过已有开放模型，但 Gemini-3.1-Pro 在部分知识指标仍领先。
- Reasoning：Pro-Max 在 LiveCodeBench、Codeforces、Apex Shortlist 等任务上表现强，Codeforces rating 报告为 3206。
- 1M Context：MRCR 1M 上 DeepSeek-V4-Pro-Max 为 83.5，低于 Claude Opus 4.6 的 92.9，高于 Gemini-3.1-Pro 的 76.3；CorpusQA 1M 上为 62.0，高于 Gemini-3.1-Pro 的 53.8，低于 Claude Opus 4.6 的 71.7。
- Agentic tasks：SWE Verified 接近闭源强模型，Terminal Bench、SWE Pro、BrowseComp 等仍显示与部分闭源前沿模型的差距。

> [!intuition] 如何读这些结果
> DeepSeek-V4 的主要贡献在效率-能力折中：Flash 用较低激活参数获得接近强模型的推理表现，Pro-Max 用更大参数和更高 reasoning effort 逼近闭源前沿。百万上下文能力在检索型长上下文任务上已经清晰体现，复杂 agent 与真实办公场景仍依赖系统、工具、评测框架和后训练数据共同决定效果。

## 边界与后续问题

> [!warning] 理解边界
> DeepSeek-V4 是预览版技术报告，许多结果来自内部评估或特定 harness，跨模型比较需要关注上下文长度、reasoning effort、工具权限、采样预算和评测实现。报告中的架构组合也较复杂：CSA、HCA、mHC、Muon、FP4 QAT、contextual parallelism、KV cache 管理共同作用，单个组件的独立贡献需要更多 ablation 才能完全拆清。

报告给出的未来方向主要有三类：

- 架构简化：当前设计保留了多种已验证组件和技巧，后续需要提炼出更本质的长上下文结构。
- 稳定性理论：Anticipatory Routing 与 SwiGLU Clamping 已被验证有效，但其原理仍需更系统解释。
- 在线学习与长程任务：百万上下文让模型可以保留更长的任务状态，为 test-time scaling、long-horizon agent 和 online learning 提供基础。

## 面试视角

> [!interview] DeepSeek-V4 的核心创新是什么？
> 可以从三层回答：架构上用 CSA/HCA 混合注意力把百万 token 的 KV cache 和 attention FLOPs 降下来；训练上用 mHC、Muon、Anticipatory Routing 和 SwiGLU Clamping 稳定超大 MoE 训练；后训练上用多专家 RL 加 OPD，把领域能力合并成统一模型。

> [!interview] CSA 和 HCA 怎么分工？
> CSA 先以较小压缩率保留较高分辨率，再通过 top-$k$ sparse selection 检索相关压缩块；HCA 采用更大压缩率，提供低成本的全局压缩记忆。交错使用二者后，模型同时获得细粒度长程检索和粗粒度全局覆盖。

> [!interview] DeepSeek-V4 的百万上下文为何可部署？
> 可部署性来自多项工程设计叠加：压缩 KV 降低 cache 长度，FP8/FP4 降低存储和计算带宽，contextual parallelism 解决训练切分问题，异构 KV cache layout 与 on-disk prefix cache 支持 serving 复用。
