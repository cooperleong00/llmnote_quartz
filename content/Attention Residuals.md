---
type: method
description: 用 softmax attention 替代固定残差累加，让每层通过学习的 pseudo-query 选择性聚合前面层的输出，解决 PreNorm 下隐藏状态 O(L) 增长导致的层贡献稀释问题
aliases:
  - AttnRes
  - 注意力残差
  - Block AttnRes
  - Block Attention Residuals
prerequisites:
  - "[[Attention]]"
  - "[[Transformer]]"
  - "[[Linear Attention]]"
tags:
  - architecture
  - transformer
  - depth
created: 2026-03-16
updated: 2026-03-16T19:01
---

# Attention Residuals

Attention Residuals（AttnRes）将标准残差连接中的固定权重累加替换为 softmax attention，让每层能选择性地聚合之前各层的输出。核心洞察是：标准残差连接本质上是深度维度的**线性 attention**，AttnRes 将其推广为 **softmax attention** -- 这与序列维度上从 RNN 到 [[Transformer]] 的转变完全对应。

> [!paper] Kimi Team (Moonshot AI), 2025
> *Technical Report of Attention Residuals*
> https://github.com/MoonshotAI/Attention-Residuals

## 动机：标准残差连接的三个局限

标准残差连接的更新规则是 $\boldsymbol{h}_l = \boldsymbol{h}_{l-1} + f_{l-1}(\boldsymbol{h}_{l-1})$，展开后得到：

$$\boldsymbol{h}_l = \boldsymbol{h}_1 + \sum_{i=1}^{l-1} f_i(\boldsymbol{h}_i)$$

每一层的输出以**固定的单位权重**累加到隐藏状态中。这个看似简洁的设计在深层网络中产生了三个根本问题：

1. **无选择性访问**：每层只能接收所有前序层输出的均匀混合，无法根据需要（比如 attention 层 vs MLP 层可能需要不同的信息组合）选择性地强调或抑制特定层的贡献
2. **不可逆的信息丢失**：一旦信息在累加过程中被稀释，深层无法选择性地恢复早期层的特定输出
3. **输出幅度 O(L) 增长**：在 [[PreNorm]] 下，隐藏状态的幅度随深度线性增长，后面的层必须产生越来越大的输出才能在已经膨胀的残差流中产生影响，导致训练不稳定

> [!intuition] 时间-深度对偶性
> 这三个问题与 RNN 在序列维度上面临的困境完全对应。RNN 将所有历史信息压缩到单个隐藏状态 $\boldsymbol{h}_t$ 中，[[Transformer]] 通过 [[Attention]] 机制让每个位置直接访问所有前序位置来解决这个问题。AttnRes 在深度维度上做了同样的事：让每层通过 attention 直接访问所有前序层的输出，而不是只能接收一个压缩的累加状态。

## 核心机制

### Full Attention Residuals

AttnRes 将固定累加 $\boldsymbol{h}_l = \sum_i \boldsymbol{v}_i$ 替换为加权聚合：

$$\boldsymbol{h}_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot \boldsymbol{v}_i$$

其中 $\boldsymbol{v}_0 = \boldsymbol{h}_1$（token embedding），$\boldsymbol{v}_i = f_i(\boldsymbol{h}_i)$（第 $i$ 层的输出）。权重 $\alpha_{i \to l}$ 通过 softmax attention 计算：

$$\alpha_{i \to l} = \frac{\phi(\boldsymbol{q}_l, \boldsymbol{k}_i)}{\sum_{j=0}^{l-1} \phi(\boldsymbol{q}_l, \boldsymbol{k}_j)}$$

关键设计选择：

- **Query**：$\boldsymbol{q}_l = \boldsymbol{w}_l$，每层一个**可学习的** $d$ 维向量（pseudo-query），不依赖输入
- **Key = Value**：$\boldsymbol{k}_i = \boldsymbol{v}_i$，直接使用层输出
- **核函数**：$\phi(\boldsymbol{q}, \boldsymbol{k}) = \exp(\boldsymbol{q}^\top \text{RMSNorm}(\boldsymbol{k}))$，RMSNorm 防止幅度大的层输出支配 attention 权重

> [!intuition] 为什么 query 不依赖输入？
> 这是一个刻意的设计权衡。input-dependent query（从隐藏状态投影得到）在消融实验中损失更低（1.731 vs 1.737），但引入了每层 $d \times d$ 的投影矩阵，且推理时需要顺序访问内存。pseudo-query 作为可学习参数，可以**并行计算**一组层的 attention 权重，这对 Block AttnRes 的高效实现至关重要。

**复杂度**：Full AttnRes 的计算量为 $O(L^2 d)$，内存为 $O(Ld)$。由于网络深度 $L$ 远小于序列长度（通常 $L < 1000$），计算开销很小。在 vanilla 训练中，$O(Ld)$ 的层输出本来就需要为反向传播保留，不增加额外内存。

**初始化**：所有 pseudo-query 向量**必须初始化为零**。这使得初始 attention 权重在所有源层上均匀分布，AttnRes 退化为等权重平均，避免训练初期的不稳定。

### Block Attention Residuals

在大规模训练中，activation recomputation 和 pipeline parallelism 使得 $O(Ld)$ 的内存和跨阶段通信成为瓶颈。Block AttnRes 通过分块来解决：

**思路**：将 $L$ 层分成 $N$ 个 block，每个 block 包含 $S = L/N$ 层。

**Block 内**（intra-block）：用标准残差累加，将 block 内所有层输出求和为单个表示：

$$\boldsymbol{b}_n = \sum_{j \in \mathcal{B}_n} f_j(\boldsymbol{h}_j)$$

**Block 间**（inter-block）：对 $N$ 个 block 级别的表示做 softmax attention，格式与 Full AttnRes 相同，只是源从 $L$ 个层变为 $N$ 个 block。

> [!math] Block AttnRes 的输入构造
> 对 block $n$ 中第 $i$ 层的输入，value 矩阵为：
> - 第一层（$i=1$）：$\mathbf{V} = [\boldsymbol{b}_0, \boldsymbol{b}_1, \dots, \boldsymbol{b}_{n-1}]^\top$
> - 后续层（$i \geq 2$）：$\mathbf{V} = [\boldsymbol{b}_0, \boldsymbol{b}_1, \dots, \boldsymbol{b}_{n-1}, \boldsymbol{b}_n^{i-1}]^\top$
>
> 其中 $\boldsymbol{b}_0 = \boldsymbol{h}_1$（token embedding 始终作为源），$\boldsymbol{b}_n^{i-1}$ 是当前 block 的部分累加。这样每层既能跨 block 选择信息，也能看到当前 block 内的最新状态。

**复杂度优势**：内存和通信从 $O(Ld)$ 降至 $O(Nd)$。实验中 $N \approx 8$（例如 54 层分 9 个 block），Block AttnRes 在最大规模上与 Full AttnRes 的损失差距仅为 0.001。

## 残差连接的统一视角：深度混合矩阵

论文提出了一个优美的统一框架：所有残差变体都可以表示为深度混合矩阵 $\mathbf{M} \in \mathbb{R}^{L \times L}$，其中 $\mathbf{M}_{i \to l}$ 是第 $l$ 层对第 $i$ 层输出的权重。不同方法的区别在于 $\mathbf{M}$ 的结构：

| 方法 | $\mathbf{M}$ 的结构 | Semiseparable Rank | 特点 |
|------|------|------|------|
| 标准残差 | 全 1 下三角 | 1 | 固定，input-independent |
| Highway | 1-semiseparable | 1 | input-dependent 门控 |
| (m)HC | $m$-semiseparable | $m$ | $m$ 个并行流，线性 attention |
| **Full AttnRes** | **Dense** | **$L$** | **input-dependent softmax** |
| **Block AttnRes** | 介于 $N$ 和 $N+S$ | $N \sim N+S$ | **分块 softmax** |

> [!intuition] 从线性到 softmax 的转变
> 标准残差连接和 Highway/HC 等变体本质上都是深度维度的**线性 attention**（不同的核函数产生不同的 $\mathbf{M}$ 结构）。AttnRes 将其推广为 **softmax attention**。这正是序列维度上从 [[Linear Attention|线性注意力]]（如 RNN、线性 Transformer）到 softmax attention（标准 [[Transformer]]）的转变在深度维度上的复现。

## 实验结果

### Scaling Laws

在五个模型规模（194M-528M activated params）上对比 Baseline / Block AttnRes / Full AttnRes：

- 三者的 scaling curve 斜率相似，但 AttnRes 在整个 compute range 上一致更低
- Block AttnRes 等价于 baseline 使用 **1.25x 更多 compute** 的效果
- Full AttnRes 与 Block AttnRes 的差距随规模增大而缩小

### 大规模验证

在 [[Kimi K2 (2025)|Kimi]] Linear 架构上验证（48B total / 3B activated，[[Mixture of Experts|MoE]] 架构），1.4T tokens：
- Block AttnRes（9 blocks + embedding = 10 个深度源）
- **所有评估任务都有提升**
- 训练动态分析显示：隐藏状态幅度在各层保持有界（不再 O(L) 增长），梯度范数分布更均匀

### 关键消融发现

| 设计选择 | 结论 |
|------|------|
| Input-dependent query | 损失更低（1.731），但参数和推理成本更高 |
| Input-independent 标量混合 | 显著差于 attention（1.749 vs 1.737），content-dependent 很重要 |
| softmax vs sigmoid | softmax 更好（1.737 vs 1.741），竞争性归一化促进更尖锐的选择 |
| Multi-head depth attention | 反而更差（1.752 vs 1.746），最优深度混合在各通道间大致一致 |
| 去掉 RMSNorm | 显著退化，尤其对 Block AttnRes（block 表示幅度差异更大） |
| DenseFormer（固定系数） | 无增益（1.767 vs 1.766），证明 input-dependent 权重是关键 |

### AttnRes 偏好更深的架构

在固定 compute/params 的架构搜索中：
- Baseline 最优 $d_\text{model}/L_b \approx 60$
- AttnRes 最优 $d_\text{model}/L_b \approx 45$（更深更窄）

AttnRes 能更有效地利用额外深度，因为每层可以选择性地聚合信息，不再受固定累加的稀释问题困扰。

## 学到的 Attention 模式

可视化分析揭示了三个有趣的规律：

1. **局部性保持**：每层最强地 attend 到直接前驱（对角线主导），同时出现选择性的长程跳跃连接
2. **层类型特化**：pre-attention 层保持更广的感受野（跨层路由信息），pre-MLP 层更依赖近期表示（局部操作）
3. **Embedding 持续性**：token embedding 在所有深度上都保持非零权重，尤其在 pre-attention 层中 -- 类似序列维度上的 [[Attention|attention sink]] 现象

## 局限性

> [!warning] 边界条件
> - **推理开销**：虽然训练开销可控，Block AttnRes 在推理时仍需维护 block 表示的缓存。论文报告推理延迟增加 <2%，但这取决于具体的 serving 架构
> - **Pipeline parallelism 约束**：Full AttnRes 在 pipeline parallel 下需要跨 stage 传输所有层输出，$O(Ld)$ 通信是当前硬件下的瓶颈（这也是 Block AttnRes 存在的原因）
> - **仅验证于 PreNorm**：论文的动机主要针对 PreNorm 下的隐藏状态增长问题。对于 PostNorm 或其他 normalization 方案，收益可能不同
> - **未探索极深网络**：当前实验的最大深度为 54 层（27 Transformer blocks）。对于更深的架构（如 100+ blocks），$O(L^2)$ 的 depth attention 成本可能变得显著，可能需要引入线性复杂度的深度 attention 变体

> [!interview] 面试视角
> **Q: Attention Residuals 解决了什么问题？**
> A: 标准残差连接以固定权重累加所有层输出，导致 PreNorm 下隐藏状态随深度 O(L) 增长，逐渐稀释每层贡献。AttnRes 用 softmax attention 替代固定累加，让每层能选择性地聚合前序层的输出。
>
> **Q: 为什么说标准残差是"深度维度的线性 attention"？**
> A: 展开残差递推 $\boldsymbol{h}_l = \boldsymbol{h}_1 + \sum f_i(\boldsymbol{h}_i)$，每层输出被赋予固定的单位权重（$\mathbf{M}_{i \to l} = 1$），这等价于核函数为常数的线性 attention。Highway/HC 等变体改变了核函数但仍是线性的。AttnRes 用 softmax 替代线性核，完成了深度维度上从线性到 softmax 的转变。
>
> **Q: Block AttnRes 相比 Full AttnRes 损失了什么？**
> A: Block 内的层输出被求和压缩为单个 block 表示，丧失了 block 内个体层的选择性访问。但实验表明这种损失很小（最大规模下差距仅 0.001），因为 block 内仍保留标准残差的局部信息流，而 block 间的选择性聚合捕获了跨深度的关键模式。
