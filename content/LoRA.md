---
type: method
description: 通过低秩分解冻结预训练权重、只训练小规模增量矩阵，实现参数高效微调
aliases:
  - 低秩适应
  - Low-Rank Adaptation
prerequisites:
  - "[[Transformer]]"
  - "[[SFT]]"
tags:
  - fine-tuning
  - peft
  - efficiency
created: 2026-01-27
updated: 2026-03-01T22:13
---

# LoRA (Low-Rank Adaptation)

低秩适应（LoRA, Low-Rank Adaptation）是一种**参数高效微调**（[[PEFT]], Parameter-Efficient Fine-Tuning）方法，通过在预训练权重旁边添加低秩分解的增量矩阵，实现只训练极少量参数就能达到接近全量微调的效果。

> [!paper] 论文出处
> Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022

---

## 核心思想

> [!intuition] 直觉理解
> 微调大模型时，权重的变化量 $\Delta W$ 可能具有**低秩结构**——也就是说，虽然 $\Delta W$ 是一个大矩阵，但它的"有效维度"很低。
>
> LoRA 的洞察是：既然 $\Delta W$ 是低秩的，我们可以用两个小矩阵的乘积来近似它，从而大幅减少需要训练的参数量。

**核心公式：**

$$
W' = W_0 + \Delta W = W_0 + BA
$$

其中：
- $W_0 \in \mathbb{R}^{d \times k}$：预训练权重（**冻结**，不更新）
- $B \in \mathbb{R}^{d \times r}$：下投影矩阵（可训练）
- $A \in \mathbb{R}^{r \times k}$：上投影矩阵（可训练）
- $r \ll \min(d, k)$：秩（rank），通常 $r \in \{4, 8, 16, 32, 64\}$

**前向传播：**

$$
h = W_0 x + \frac{\alpha}{r} \cdot BAx
$$

其中 $\alpha$ 是缩放因子，$\frac{\alpha}{r}$ 用于稳定不同 rank 下的训练。

---

## 数学原理

### 为什么低秩有效？

> [!math] Intrinsic Dimension 假说
> Aghajanyan et al. (2020) 发现，预训练模型的微调过程存在一个**内在维度**（Intrinsic Dimension）：
>
> - 虽然模型有数十亿参数，但微调时的有效自由度远小于参数总数
> - 这意味着 $\Delta W$ 可以被投影到一个低维子空间而不损失太多信息
>
> 实验表明：对于 GPT-3 175B，内在维度可能只有几千维。

> [!intuition] 直觉解释
> 预训练模型已经学到了丰富的表示，微调只是在这个表示空间中做**小幅调整**。这种调整往往是"方向性"的，而不是"全方位"的，因此可以用低秩矩阵捕捉。

### 参数量分析

> [!math] 参数量对比
> 假设原始权重 $W_0 \in \mathbb{R}^{d \times k}$：
>
> | 方法 | 可训练参数量 |
> |------|-------------|
> | Full Fine-tuning | $d \times k$ |
> | LoRA (rank $r$) | $r \times (d + k)$ |
>
> **压缩比**：
> $$
> \text{Compression Ratio} = \frac{d \times k}{r \times (d + k)} \approx \frac{dk}{2rk} = \frac{d}{2r}
> $$
>
> 对于 $d = 4096$, $r = 8$：压缩比约为 **256 倍**。

---

## 实现细节

### 初始化策略

> [!definition] 标准初始化
> - **$A$**：使用 Kaiming 均匀分布初始化（或高斯初始化）
> - **$B$**：初始化为**零矩阵**
>
> 这样确保训练开始时 $\Delta W = BA = 0$，模型行为与预训练模型完全一致。

> [!intuition] 为什么是 B=0 而不是 A=0？
> 虽然 $A=0$ 或 $B=0$ 都能让初始 $\Delta W = 0$，但选择 B=0 是因为**梯度流**的考虑：
>
> **标准方式（A 随机，B=0）**：
> - $\frac{\partial L}{\partial B} = \frac{\partial L}{\partial \Delta W} \cdot A^T$ → B 立即接收梯度，可以开始学习
> - $\frac{\partial L}{\partial A} = B^T \cdot \frac{\partial L}{\partial \Delta W}$ → A 暂时不更新（因为 B=0）
>
> **反向方式（A=0，B 随机）**：
> - $\frac{\partial L}{\partial A}$ 有梯度，A 可以学习
> - $\frac{\partial L}{\partial B} = 0$ → **B 无法学习**，训练初期陷入梯度死区
>
> **直观理解**：
> - A 是降维投影（$d \to r$），B 是升维映射（$r \to d$）
> - 标准初始化：A 提供随机投影方向，B 学习"如何利用这个投影"
> - 反向初始化：A=0 意味着"没有任何输入"，B 的随机映射无法接收信号
>
> 因此 B=0 的方式让训练从第一步就能有效进行。

### Rank $r$ 的选择

> [!example] 经验指导
>
> | Rank | 适用场景 | 参数量（相对） |
> |------|----------|---------------|
> | 4-8 | 简单任务、资源受限 | 最少 |
> | 16-32 | 通用场景、平衡选择 | 中等 |
> | 64-128 | 复杂任务、追求效果 | 较多 |
>
> **经验法则**：
> - 任务越复杂，需要的 rank 越高
> - 但 rank 过高会失去 LoRA 的效率优势
> - 通常从 $r=8$ 或 $r=16$ 开始尝试

### Scaling Factor $\alpha$

> [!definition] $\alpha$ 参数
> $\alpha$ 是一个常数缩放因子，实际应用中 LoRA 的输出会乘以 $\frac{\alpha}{r}$：
>
> $$
> \Delta W \cdot x = \frac{\alpha}{r} \cdot BA \cdot x
> $$
>
> **作用**：
> - 当改变 rank $r$ 时，不需要重新调整学习率
> - 通常设置 $\alpha = r$ 或 $\alpha = 2r$
> - 有些实现固定 $\alpha = 16$ 或 $\alpha = 32$

> [!math] 为什么 $1/r$ 缩放使学习率与 rank 无关？
> 训练初期，$BA$ 可以看作 $r$ 个 rank-1 外积的和：$BA = \sum_{i=1}^r b_i a_i^T$。
>
> 由于 $a_i$, $b_i$ 的初始化不依赖于 rank，每个 $\Delta_i = b_i a_i^T$ 的期望更新相同。
> 因此 $(1/r)\sum_{i=1}^r \Delta_i$ 是 $r$ 个同分布项的平均，其期望不依赖于 rank。
>
> **实践意义**：不同 rank 的 LoRA 在训练初期有几乎相同的学习曲线，最优学习率也近似相同。

### 学习率设置

> [!example] LoRA vs FullFT 学习率关系
> Schulman et al. (2025) 的实验发现：
>
> | 训练长度 | LoRA 最优 LR / FullFT 最优 LR |
> |----------|------------------------------|
> | 短训练 | ~15x |
> | 长训练 | ~10x |
>
> 这一比例在 SFT 和 RL 中都成立，且与模型大小无关。
>
> **实践建议**：如果已知 FullFT 的最优学习率，LoRA 可以直接用 10x 作为起点。

### 应用到哪些层？

LoRA 通常应用于 [[Transformer]] 的特定层：

> [!example] 层选择策略
>
> | 层类型 | 是否应用 LoRA | 说明 |
> |--------|--------------|------|
> | **Query ($W_Q$)** | 通常应用 | 影响[[Multi-Head Attention|注意力]]模式 |
> | **Key ($W_K$)** | 可选 | 与 Query 配合 |
> | **Value ($W_V$)** | 通常应用 | 影响输出内容 |
> | **Output ($W_O$)** | 可选 | 注意力输出投影 |
> | **FFN Up/Down** | **推荐应用** | MLP 层对效果影响最大 |
> | **Embedding** | 通常不应用 | 参数量大，效果有限 |
> | **LM Head** | 通常不应用 | 与 Embedding 共享 |

> [!warning] MLP 层的重要性
> Schulman et al. (2025) 的实验表明：**MLP-only LoRA 显著优于 Attention-only LoRA**，即使参数量相同。
>
> - Attention-only (rank=256, 0.25B params) < MLP-only (rank=128, 0.24B params)
> - 在 Attention 基础上加 MLP 几乎没有额外收益，说明 MLP 层才是关键
> - 这一结论在 dense 模型（Llama-3.1-8B）和 MoE 模型（Qwen3-30B）上都成立
>
> **理论解释**：从 eNTK（empirical Neural Tangent Kernel）角度，参数量最多的层对 kernel 影响最大。MLP 层占据了大部分参数，因此 LoRA 应用到 MLP 层时，其 eNTK 才能近似 FullFT 的 eNTK。

**推荐配置**：
- **最小配置**：只对 $W_Q$, $W_V$ 应用（原论文推荐，**已过时**）
- **标准配置**：对所有 Attention 权重应用
- **最佳配置**：**所有层（Attention + MLP）**，尤其不要遗漏 MLP

---

## 与 Full Fine-tuning 的对比

> [!comparison] LoRA vs Full Fine-tuning

| 方面 | Full Fine-tuning | LoRA |
|------|------------------|------|
| **可训练参数** | 100% | 0.1% ~ 1% |
| **GPU 内存** | 高（需存储所有梯度） | 低（只存储 LoRA 梯度） |
| **训练速度** | 较慢 | 较快 |
| **每次 pass FLOPs** | 基准 | ~2/3（forward + backward） |
| **存储开销** | 每个任务一个完整模型 | 每个任务只需存储 LoRA 权重 |
| **效果** | 最优 | 接近最优（满足条件时等效） |
| **多任务部署** | 需要多个模型副本 | 共享基座 + 多个 LoRA adapter |
| **推理开销** | 无额外开销 | 可合并，无额外开销 |
| **最优学习率** | 基准 | 约为 FullFT 的 **10 倍** |

> [!intuition] LoRA 的核心优势
> 1. **内存效率**：不需要存储完整模型的优化器状态
> 2. **存储效率**：多个任务可以共享基座模型
> 3. **部署灵活**：可以动态加载/切换不同的 LoRA adapter
> 4. **无推理开销**：训练后可将 LoRA 权重合并到基座模型
> 5. **计算效率**：每次 forward-backward pass 只需约 2/3 的 FLOPs

### LoRA 与 FullFT 等效的条件

> [!definition] Low-Regret Regime
> Schulman et al. (2025) 的实验表明，满足以下条件时，LoRA 可以达到与 FullFT **相同的样本效率和最终性能**：
>
> 1. **应用到所有层**：尤其是 MLP/MoE 层（占据大部分参数）
> 2. **容量不受限**：可训练参数量 > 数据集信息量
>
> 对于典型的 post-training 场景（SFT、RLHF），这两个条件通常都能满足。

> [!warning] Batch Size 效应
> LoRA 对大 batch size 的容忍度低于 FullFT：
> - 随着 batch size 增大，LoRA 与 FullFT 的 loss 差距会扩大
> - 这是 $BA$ 乘积参数化的特性，与 rank 无关
> - 实践建议：使用较小的 batch size（如 32）可以消除这一差距

### 权重合并

> [!math] 推理时合并
> 训练完成后，可以将 LoRA 权重合并到原始权重：
>
> $$
> W_{\text{merged}} = W_0 + \frac{\alpha}{r} \cdot BA
> $$
>
> 合并后的模型与原始模型结构完全相同，**推理时没有任何额外开销**。

---

## 在 SFT 中的应用

LoRA 是 [[SFT]] 阶段最常用的参数高效微调方法：

> [!example] 典型 SFT + LoRA 配置
>
> ```python
> # 使用 PEFT 库
> from peft import LoraConfig, get_peft_model
>
> lora_config = LoraConfig(
>     r=16,                    # rank
>     lora_alpha=32,           # scaling factor
>     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
>     lora_dropout=0.05,
>     bias="none",
>     task_type="CAUSAL_LM"
> )
>
> model = get_peft_model(base_model, lora_config)
> ```

**为什么 SFT 适合用 LoRA？**

1. **SFT 的目标是"激活"能力**：不是学习新知识，而是调整输出格式
2. **变化量确实是低秩的**：从"续写"到"对话"的转变相对简单
3. **资源限制**：很多团队没有足够资源做全量微调

---

## 在强化学习中的应用

> [!intuition] RL 中 LoRA 的特殊优势
> Schulman et al. (2025) 发现：**在 RL 中，即使 rank=1 的 LoRA 也能完全匹配 FullFT 的性能**。
>
> 这一结果在 MATH、GSM、DeepMath 等数学推理任务上都得到验证。

**信息论解释**：

RL 与 SFT 的本质区别在于**每个样本提供的信息量**：

| 训练方式 | 每个样本的信息量 | 说明 |
|----------|------------------|------|
| SFT | $O(\text{tokens})$ bits | 每个 token 都提供监督信号 |
| Policy Gradient RL | $O(1)$ bits | 只有 episode 结束时的 reward |

> [!math] 容量需求估算
> 以 MATH 数据集为例：
> - 训练 ~10,000 问题 × 32 samples/问题 = 320,000 bits 信息
> - Rank-1 LoRA for Llama-3.1-8B 有 ~3M 参数
> - 容量远超需求，因此 rank=1 就足够
>
> **推论**：DeepSeek-R1-Zero（5.3M episodes）的训练信息量 < 低秩 LoRA 的容量，理论上可以用 LoRA 复现。

> [!example] RL + LoRA 的实践发现
> - LoRA 在 RL 中有**更宽的最优学习率范围**
> - 不同 rank 的 LoRA 和 FullFT 的学习曲线几乎完全重合
> - 模型同样能发展出 backtracking、self-verification 等高级推理行为

---

## LoRA 变体

### QLoRA (Quantized LoRA)

> [!definition] QLoRA
> Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023
>
> **核心思想**：将基座模型[[Quantization|量化]]到 4-bit，只在 LoRA 部分保持高精度
>
> **关键技术**：
> - **4-bit NormalFloat (NF4)**：专为正态分布权重设计的量化格式
> - **Double Quantization**：对量化常数再次量化
> - **Paged Optimizers**：利用 CPU 内存处理梯度峰值

> [!comparison] LoRA vs QLoRA
>
> | 方面 | LoRA | QLoRA |
> |------|------|-------|
> | 基座模型精度 | FP16/BF16 | 4-bit (NF4) |
> | 内存占用 | 较高 | 极低 |
> | 训练速度 | 较快 | 较慢（量化/反量化开销） |
> | 效果 | 基准 | 接近 LoRA |
>
> **意义**：QLoRA 使得在单张消费级 GPU 上微调 65B 模型成为可能。

### LoRA+

> [!definition] LoRA+
> Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models", 2024
>
> **核心发现**：$A$ 和 $B$ 矩阵应该使用**不同的学习率**
>
> - 原始 LoRA：$\eta_A = \eta_B$
> - LoRA+：$\eta_B = \lambda \cdot \eta_A$，其中 $\lambda \approx 16$
>
> **效果**：在相同训练步数下，LoRA+ 收敛更快、效果更好。

### DoRA (Weight-Decomposed Low-Rank Adaptation)

> [!definition] DoRA
> Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", 2024
>
> **核心思想**：将权重分解为**幅度**（magnitude）和**方向**（direction）两部分
>
> $$
> W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}
> $$
>
> 其中 $m$ 是可学习的幅度向量，$\|\cdot\|_c$ 是列归一化。
>
> **优势**：
> - 更接近 Full Fine-tuning 的学习模式
> - 在相同 rank 下效果更好
> - 额外参数量可忽略

### 其他变体

| 变体 | 核心改进 |
|------|----------|
| **AdaLoRA** | 自适应分配不同层的 rank |
| **LoRA-FA** | 冻结 $A$，只训练 $B$，进一步减少内存 |
| **VeRA** | 共享随机矩阵，只训练缩放向量 |
| **LoRA-XS** | 极小 rank + SVD 初始化 |

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: LoRA 的核心思想是什么？**
> A: LoRA 基于"微调时权重变化是低秩的"这一假设，用两个小矩阵 $B$ 和 $A$ 的乘积来近似权重变化 $\Delta W$，从而大幅减少可训练参数量。
>
> **Q2: 为什么 LoRA 有效？为什么低秩假设成立？**
> A: 预训练模型已经学到了丰富的表示，微调只是在这个表示空间中做小幅调整。研究表明，微调的"内在维度"远小于参数总数，因此低秩近似是合理的。
>
> **Q3: LoRA 的 $\alpha$ 参数有什么作用？**
> A: $\alpha$ 是缩放因子，实际输出乘以 $\frac{\alpha}{r}$。它的作用是让不同 rank 的配置可以使用相同的学习率，简化超参数调整。
>
> **Q4: LoRA 应该应用到哪些层？**
> A: 最新研究（Schulman et al., 2025）表明：**MLP 层比 Attention 层更重要**。MLP-only LoRA 显著优于 Attention-only LoRA，即使参数量相同。最佳实践是应用到所有层，尤其不要遗漏 MLP。
>
> **Q5: LoRA 和 QLoRA 的区别？**
> A: QLoRA 在 LoRA 基础上将基座模型量化到 4-bit，进一步降低内存需求。代价是训练速度略慢（量化/反量化开销），但效果接近。
>
> **Q6: LoRA 训练完成后如何部署？**
> A: 可以将 LoRA 权重合并到基座模型（$W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$），合并后推理没有任何额外开销。也可以保持分离，动态加载不同的 adapter。
>
> **Q7: LoRA 在 RL 中表现如何？**
> A: 在 RL（如 RLHF、数学推理）中，LoRA 即使 rank=1 也能完全匹配 FullFT。这是因为 policy gradient 每个 episode 只提供 O(1) bits 信息，远低于 LoRA 的容量。
>
> **Q8: LoRA 的学习率应该怎么设置？**
> A: LoRA 的最优学习率约为 FullFT 的 10 倍（短训练约 15 倍）。$1/r$ 缩放使得最优学习率近似与 rank 无关。

---

## 延伸阅读

- [[Adapter]] — 另一种参数高效微调方法
- [[Prefix Tuning]] — 在输入前添加可学习的 prefix
- [[Prompt Tuning]] — 只调整 soft prompt
- [[Full Fine-tuning]] — 传统的全参数微调

**深入材料**：
- [[Clippings/Article/LoRA Without Regret|LoRA Without Regret (Schulman et al., 2025)]] — 系统性研究 LoRA 与 FullFT 等效的条件
