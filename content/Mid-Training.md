---
type: concept
description: 在 pretraining 和 post-training 之间的关键桥梁阶段，通过高质量数据、学习率退火和双向能力平衡，将模型从记忆转向抽象，为下游任务创造有效的权重配置
aliases:
  - 中间训练
  - Annealing Phase
prerequisites:
  - "[[Pretraining]]"
  - "[[Post-training]]"
tags:
  - training
  - post-training
  - data
  - optimization
created: 2026-03-23
updated: 2026-03-23T00:31
---

# Mid-Training

Mid-Training（中间训练）是在 [[Pretraining]] 和 [[Post-training]] 之间的关键桥梁阶段，通过高质量数据混合、学习率退火和双向能力平衡，将模型的学习动态从记忆转向抽象，为后续的 [[SFT]] 和 [[RLHF]] 创造有效的权重配置。

> [!warning] Mid-Training vs Continued Pretraining
>
> **Mid-Training**：
> - 有意识的发展阶段，具有过渡意图
> - 混合 pre-training 数据比例，防止灾难性遗忘
> - 继承学习率动态，保持优化器状态
> - 双向能力平衡（前向传播专业能力 + 后向保留通用能力）
>
> **Continued Pretraining**：
> - 用领域特定数据扩展 pre-training
> - 不考虑原始优化器状态或分布保留
> - 可能导致通用能力丧失

## 动机

> [!intuition] 为什么需要 Mid-Training？
>
> **问题**：Pretraining 和 Post-training 之间存在"能力鸿沟"
> - Pretraining 使用海量但质量参差的 web 数据，模型学会广泛的语言模式但缺乏专业推理能力
> - 直接进入 Post-training（SFT + RL）跨度太大，模型权重配置不适合精细优化
> - 在低质量数据上继续训练边际收益递减，甚至可能导致性能停滞
>
> **Mid-Training 的核心洞察**：
> - 在两个阶段之间插入一个"桥梁"，用高质量数据重构权重空间
> - 通过学习率退火稳定优化，从"快速探索"转向"精细收敛"
> - 为下游任务（特别是 RL）创造一个"可优化的起点"

## 核心机制

Mid-Training 通过三个维度的协同作用实现从记忆到抽象的转变。

### 1. 数据分布调整

> [!intuition] 从广度到深度
>
> **Pretraining 数据特征**：
> - 海量（数万亿 tokens）
> - 多样但噪声大
> - 覆盖广泛但深度不足
>
> **Mid-Training 数据特征**：
> - 适中规模（数百亿 tokens）
> - 高质量、领域聚焦
> - 强调推理、事实性、结构化知识
>
> **数据类型**：
> - **高质量过滤 web 数据**：FineWeb-Edu、DCLM-baseline（保持泛化能力）
> - **代码和数学**：Stack、OpenWebMath、FineMath（符号推理）
> - **指令和 QA**：UltraChat、EvolInstruct（意图理解）
> - **合成教科书**：Cosmopedia（知识密集）
> - **推理和 CoT**：显式推理链（结构化思维）
> - **长上下文数据**：扩展上下文窗口能力

### 2. 学习率调度

> [!intuition] 从探索到收敛
>
> **Pretraining 阶段**：
> - 高学习率（快速探索参数空间）
> - 模型学习广泛的语言模式
>
> **Mid-Training 阶段**：
> - 学习率退火（annealing）
> - 稳定收敛，抑制梯度噪声
> - 精细吸收高质量 tokens
>
> **常见策略**：
> - Linear/Cosine decay
> - Multi-stage schedulers
> - Adaptive schemes

### 3. 长上下文扩展

> [!intuition] 从短文本到长文档
>
> **动机**：
> - Pretraining 通常限制在 4K-8K context
> - 下游任务需要更长的上下文（文档、对话历史）
>
> **方法**：
> - Position Interpolation
> - NTK-aware interpolation
> - YaRN
> - 渐进式引入长文本（curriculum）

### 训练流程

```
Pretraining (数万亿 tokens，广泛知识)
├── 高学习率
├── 多样但噪声数据
└── 短上下文（4K-8K）
    ↓
Mid-Training (数百亿 tokens，精选知识)
├── 学习率退火
├── 高质量领域数据
├── 上下文扩展（可选）
└── 多阶段 annealing（可选）
    ↓
Post-Training (SFT + RL)
├── 指令微调
└── 偏好对齐
```

## 理论基础

Mid-Training 的有效性可以从三个理论视角理解。

### 1. 梯度噪声尺度（Gradient Noise Scale, GNS）

> [!math] 梯度噪声尺度
>
> GNS 反映每次更新步骤中有用信号的数量：
> - **高质量数据** → 更大的梯度方差 → 更高的 GNS
> - **冗余/噪声数据** → 减少多样性 → 更低的 GNS
>
> **作用**：
> - 更大的 GNS 帮助模型逃离尖锐极小值
> - 避免过拟合到低质量数据的局部最优
> - 在训练后期改善优化（信号稀疏性可能导致停滞）

### 2. 信息瓶颈（Information Bottleneck, IB）

> [!intuition] 压缩与保留
>
> **核心思想**：
> - 神经网络学习 = 压缩内部状态 + 保留任务相关信息
> - 学习率退火阶段，模型逐步减少对噪声特征的依赖
> - 高质量监督信号提供更清晰、低熵的指导
>
> **效果**：
> - 从大规模记忆转向抽象、可泛化的表征
> - 识别语义结构，削弱虚假相关性

### 3. 课程学习（Curriculum Learning）

> [!intuition] 从简单到复杂
>
> **学习路径**：
> 1. **Pretraining**：多样化、噪声数据（建立广泛基础）
> 2. **Mid-Training**：逐步转向更具挑战性、信息量更大的样本
> 3. **Post-Training**：任务特定的精细调整
>
> **效果**：
> - 优化学习效率
> - 强化复杂技能（多步推理、代码生成）

## 与 RL 的交互

> [!warning] 关键发现：Mid-Training 是 RL 成功的必要条件
>
> **实验观察**（来自 PRISM）：
> - RL 对所有模型应用相同的权重更新
> - 但只在 mid-trained 模型上成功
> - 直接在 base model 上 RL：性能接近 0
>
> **机制解释**：
> - **Mid-Training**：密集重构 >90% 权重，创造"可优化的权重配置"
> - **RL**：稀疏调整 ~5% 权重，进行精细优化
> - RL 保持 mid-training 的表征几何结构（CKA >0.998）
>
> **类比**：
> - Mid-Training = 粗加工（塑造大致形状）
> - RL = 精加工（细节雕琢）
> - RL 需要在"正确的工件"上才能发挥作用

## 实践模式

### 常见配置模式

**单阶段 Annealing**：
```
Pretraining → Mid-Training (一次 annealing) → Post-Training
```
- 适用于：资源受限、目标明确
- 示例：SmolLM2 的 annealing stage

**多阶段 Annealing**：
```
Pretraining → Mid-Training Stage 1 → Stage 2 → Stage 3 → Post-Training
```
- 适用于：渐进式能力提升
- 示例：Qwen3（30T → 5T 高质量 → 长上下文）

**Annealing + 长上下文扩展**：
```
Pretraining → Annealing → Long-Context Extension → Post-Training
```
- 适用于：需要长上下文能力的模型
- 示例：Llama3 405B、Apple Foundation Model

**重复训练并平均**：
```
Pretraining → Mid-Training (多次运行，不同随机顺序) → 模型平均 → Post-Training
```
- 适用于：最大化泛化能力
- 示例：OLMo-2

### 数据混合策略

> [!example] 典型数据配比
>
> **保守配置**（保持泛化）：
> - 60-70% 高质量 web 数据
> - 15-20% 代码
> - 10-15% 数学
> - 5% 指令/合成数据
>
> **激进配置**（强化推理）：
> - 40-50% 高质量 web 数据
> - 20-25% 代码
> - 15-20% 数学
> - 10-15% 推理/CoT 数据
> - 5-10% 合成教科书
>
> **关键原则**：
> - 保留足够的通用数据（防止灾难性遗忘）
> - 上采样目标领域数据
> - 下采样低质量数据

### Token 预算

**经验规律**：
- **最小有效规模**：~27B tokens（PRISM）
- **典型规模**：50B-1T tokens
- **大规模**：1T-5T tokens（Qwen3、Llama3）

**权衡**：
- 更多 tokens → 更大收益，但成本更高
- 数据质量 > 数据量

## 局限性

> [!warning] Mid-Training 的边界条件
>
> **长上下文能力退化**：
> - 在短上下文（如 8K）进行 mid-training 可能降低长上下文能力
> - 解决方案：添加 context extension phase 或 model merging
>
> **灾难性遗忘风险**：
> - 过度聚焦领域数据可能导致通用能力退化
> - 解决方案：保留 30-40% 通用 web 数据
>
> **计算成本**：
> - Mid-training 需要额外的计算资源
> - 需要权衡成本与收益
>
> **调度器设计**：
> - 最优持续时间和衰减形状因模型而异
> - 缺乏统一的理论指导
>
> **评估范围**：
> - 现有研究主要聚焦推理任务
> - 对其他能力（创意写作、多语言）的影响未充分探索

## 何时使用 Mid-Training

**适用场景**：
- ✅ 需要提升特定领域能力（数学、代码、科学）
- ✅ 计划后续进行 RL 训练
- ✅ 有高质量领域数据可用
- ✅ 需要扩展上下文长度
- ✅ 预训练后性能停滞

**不适用场景**：
- ❌ 只需要通用对话能力（直接 SFT 即可）
- ❌ 缺乏高质量领域数据
- ❌ 预算极度受限
- ❌ 模型已经在目标领域表现良好

## 相关概念

**训练流程**：
- [[Pretraining]] — Mid-Training 的前置阶段，建立广泛知识基础
- [[Post-training]] — Mid-Training 的后续阶段，包含 SFT 和 RL
- [[SFT]] — Post-training 的第一步，教会模型遵循指令
- [[RLHF]] — Post-training 的核心，通过人类反馈优化模型

**理论基础**：
- [[Gradient Noise Scale]] — 解释高质量数据如何改善优化
- [[Information Bottleneck]] — 解释从记忆到抽象的转变
- [[Curriculum Learning]] — 解释渐进式数据分布调整

**数据相关**：
- [[Data Synthesis]] — 生成高质量训练数据
- [[Preference Data]] — RL 阶段使用的偏好数据

**优化相关**：
- [[Learning Rate Schedule]] — 学习率调度策略
- [[Annealing]] — 学习率退火技术

## 延伸阅读

**综述论文**：
- [[Mid-Training Survey (2024)]] — 首个 mid-training 综述，提出三维分类法

**实证研究**：
- [[PRISM (2026)]] — 系统研究 mid-training 的设计选择和 RL 交互

**代表性模型**：
- Qwen3 (2024) — 三阶段 mid-training
- Llama3 405B (2024) — Annealing + 长上下文扩展
- OLMo-2 (2024) — 重复训练并平均
- SmolLM2 (2024) — 小模型的 mid-training 实践

> [!interview] 面试视角
>
> **Q: Mid-Training 和 Continued Pretraining 有什么区别？**
> A: 本质上是同一概念的不同称呼。Mid-Training 强调其在训练流程中的位置（pretraining 和 post-training 之间），Continued Pretraining 强调其技术形式（继续预训练）。有些文献也称之为 Annealing Phase，强调学习率退火的作用。
>
> **Q: Mid-Training 的核心作用是什么？**
> A: 三个核心作用：1) **数据质量提升**：从噪声 web 数据转向高质量领域数据；2) **优化稳定**：通过学习率退火稳定收敛；3) **权重重构**：为下游任务（特别是 RL）创造有效的权重配置。本质是将模型从"记忆"转向"抽象"。
>
> **Q: 为什么 RL 需要 mid-training？**
> A: Mid-Training 密集重构 >90% 权重，创造一个"可优化的权重配置"。RL 对所有模型应用相同的权重更新，但只在 mid-training 创造的权重配置下才能有效改进性能。类比：RL 是精密工具，需要在正确的"工件"上才能发挥作用。
>
> **Q: Mid-Training 会不会导致灾难性遗忘？**
> A: 有风险，但可以通过数据混合策略缓解。关键是保留 30-40% 通用 web 数据，在提升领域能力的同时保持泛化能力。数据混合的平衡是关键，而非单纯使用领域数据。
>
> **Q: Mid-Training 的理论基础是什么？**
> A: 三个理论视角：1) **梯度噪声尺度**：高质量数据提升 GNS，帮助逃离尖锐极小值；2) **信息瓶颈**：压缩噪声特征，保留任务相关信息；3) **课程学习**：渐进式数据分布调整，从简单到复杂。这些理论共同解释了 mid-training 如何促进泛化和抽象。
