---
type: concept
description: 通过多次 forward-backward 累积梯度再统一更新参数，用小显存模拟大 batch size 的训练技巧
aliases:
  - 梯度累积
prerequisites:
  - "[[Data Parallelism]]"
tags:
  - distributed-training
created: 2026-02-24
updated: 2026-02-24T14:38
---

# Gradient Accumulation

梯度累积（Gradient Accumulation）是一种用**时间换显存**的训练技巧：当单卡显存装不下目标 batch size 时，把大 batch 拆成多个 micro-batch 依次做 forward-backward，把梯度累加起来，最后统一做一次参数更新。数学上，这与直接用大 batch 训练**完全等价**。

## 动机

> [!intuition] 为什么需要梯度累积？
> 大 batch size 对训练很重要——梯度估计更稳定、收敛更快、学习率可以开更大。但 GPU 显存是有限的，一张卡可能只能塞下 batch size = 2 的数据。
>
> 怎么办？**攒着**。做 4 次 forward-backward，每次处理 2 个样本，把梯度加起来，效果等同于 batch size = 8。代价是训练变慢（串行了 4 次），但不需要更多显存。

这在 LLM 训练中尤其常见：模型本身已经占满大部分显存，留给 activation 的空间很少，micro-batch size 往往只有 1-2。

## 核心机制

### 数学等价性

设目标 effective batch size 为 $B$，micro-batch size 为 $b$，累积步数为 $K = B / b$。

标准 SGD 的梯度更新：

$$\theta \leftarrow \theta - \eta \cdot \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \mathcal{L}(x_i)$$

梯度累积将这 $B$ 个样本分成 $K$ 组，每组 $b$ 个：

$$\theta \leftarrow \theta - \eta \cdot \frac{1}{K} \sum_{k=1}^{K} \underbrace{\left( \frac{1}{b} \sum_{j=1}^{b} \nabla_\theta \mathcal{L}(x_{k,j}) \right)}_{\text{第 } k \text{ 个 micro-batch 的平均梯度}}$$

展开后与标准公式完全一致——$\frac{1}{K} \cdot \frac{1}{b} = \frac{1}{Kb} = \frac{1}{B}$。关键公式：

$$\text{effective\_batch\_size} = \text{accumulation\_steps} \times \text{micro\_batch\_size}$$

### 实现：Loss Scaling 的细节

等价性成立的前提是 loss 的归一化方式正确。大多数框架中 `loss = criterion(output, target)` 默认对 micro-batch 内取平均（`reduction='mean'`），所以每个 micro-batch 返回的梯度已经除以了 $b$。累积 $K$ 步后，还需要再除以 $K$ 才能得到正确的平均梯度：

> [!example] PyTorch 伪代码
> ```python
> optimizer.zero_grad()
> for i, (inputs, targets) in enumerate(micro_batches):
>     outputs = model(inputs)
>     loss = criterion(outputs, targets)
>     loss = loss / accumulation_steps  # 关键：除以 K
>     loss.backward()                   # 梯度累加到 .grad
>
>     if (i + 1) % accumulation_steps == 0:
>         optimizer.step()
>         optimizer.zero_grad()
> ```
>
> 如果忘了 `loss / accumulation_steps`，等效学习率会被放大 $K$ 倍，训练会发散。

注意这里 `loss.backward()` 不会清零梯度——PyTorch 默认将新梯度**累加**到 `.grad` 属性上，这正是梯度累积能工作的原因。只有显式调用 `optimizer.zero_grad()` 才会清零。

### 与 Mixed Precision Training 的配合

在使用 [[Mixed Precision Training]] 时，梯度累积需要配合 `GradScaler`：

```python
scaler = GradScaler()
for i, (inputs, targets) in enumerate(micro_batches):
    with autocast():
        loss = criterion(model(inputs), targets)
        loss = loss / accumulation_steps
    scaler.scale(loss).backward()  # scaled 梯度累加

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)     # unscale → clip → step
        scaler.update()
        optimizer.zero_grad()
```

`GradScaler` 的 scale/unscale 在累积过程中保持一致，不影响等价性。但如果某个 micro-batch 出现 inf/nan 梯度，整个累积窗口的更新都会被跳过。

## 与分布式训练的配合

### 与 Data Parallelism 组合

在 [[Data Parallelism|DDP]] 中，每张卡独立处理不同数据，梯度通过 AllReduce 同步。加上梯度累积后，effective batch size 变为三个因子的乘积：

$$\text{effective\_batch\_size} = \underbrace{N}_{\text{GPU 数}} \times \underbrace{b}_{\text{micro-batch}} \times \underbrace{K}_{\text{accumulation steps}}$$

> [!example] 实际计算
> 8 张 A100，micro-batch = 2，accumulation_steps = 4：
> effective batch size = 8 × 2 × 4 = 64

这是 LLM 训练中最常见的配置方式——GPU 数量和显存决定了 $N$ 和 $b$，然后用 $K$ 凑到目标 batch size。

### 梯度同步优化

DDP 默认每次 `backward()` 都做 AllReduce 同步梯度。但在梯度累积中，中间步骤的同步是浪费的——只有最后一步需要同步。PyTorch 提供了 `no_sync()` 上下文管理器来跳过中间同步：

```python
for i, (inputs, targets) in enumerate(micro_batches):
    # 中间步骤：跳过 AllReduce
    context = model.no_sync() if (i + 1) % K != 0 else nullcontext()
    with context:
        loss = criterion(model(inputs), targets) / K
        loss.backward()

    if (i + 1) % K == 0:
        optimizer.step()  # 这里才触发 AllReduce
        optimizer.zero_grad()
```

不用 `no_sync()` 也能得到正确结果（多次 AllReduce 的平均 = 一次 AllReduce 的平均），但通信开销会增大 $K$ 倍。

### 与 Pipeline Parallelism 的关系

[[Pipeline Parallelism]] 天然依赖梯度累积的思想：GPipe 将一个 mini-batch 切成 $m$ 个 micro-batch 送入 pipeline，所有 micro-batch 的梯度累积后做一次参数更新。$m$ 越大，pipeline bubble 越小——这里的 $m$ 本质上就是 accumulation steps。

## 注意事项与边界

### BatchNorm 的行为差异

BatchNorm 的 running mean/variance 是按**每次 forward 的 micro-batch** 更新的，而非按 effective batch 更新。这意味着：

- 统计量的估计基于 micro-batch size $b$，而非 effective batch size $B$
- $b$ 很小时（如 1-2），BatchNorm 的统计量噪声很大，可能影响训练稳定性
- 解决方案：使用 GroupNorm 或 LayerNorm 替代（LLM 中普遍使用 LayerNorm/RMSNorm，所以这个问题主要影响 CV 模型）

### 学习率调整

梯度累积改变的是 effective batch size，而学习率通常需要随 batch size 调整。常见策略：

- **线性缩放**（Linear Scaling Rule）：$\eta' = \eta \times \frac{B'}{B}$，batch size 翻倍则学习率翻倍
- **平方根缩放**：$\eta' = \eta \times \sqrt{B'/B}$，更保守
- 配合 warmup 使用，避免训练初期因大学习率发散

### 训练速度 Trade-off

梯度累积是**时间换显存**的策略：

| 方案 | 吞吐量 | 显存 |
|------|--------|------|
| 大 batch 直接训练 | 最高 | 需要大显存 |
| [[Data Parallelism\|数据并行]] | 高（$\times N$ 卡） | 每卡需完整模型 |
| 梯度累积 | 低（$\div K$ 步） | 最省 |
| DDP + 梯度累积 | 中 | 平衡 |

累积步数 $K$ 越大，每个 optimization step 的墙钟时间越长。但由于中间步骤不需要 optimizer state 更新和梯度同步，额外开销主要是 forward-backward 的计算本身。

> [!interview] 面试要点
> **Q: 梯度累积和数据并行有什么区别？**
> A: 两者都能增大 effective batch size，但数据并行用空间（多卡）换时间，梯度累积用时间（多步）换空间。数据并行 $N$ 卡同时算 $N$ 个 micro-batch，吞吐量 $\times N$；梯度累积串行算 $K$ 个 micro-batch，吞吐量不变。两者可以组合使用。
>
> **Q: 梯度累积时 loss 为什么要除以 accumulation_steps？**
> A: 因为 PyTorch 的 `backward()` 默认累加梯度。如果 loss 已经对 micro-batch 取了平均（`reduction='mean'`），累积 $K$ 步后梯度是 $K$ 个平均值的和，需要再除以 $K$ 才等价于对整个 effective batch 取平均。不除的话等效学习率放大 $K$ 倍。
>
> **Q: DDP 中梯度累积如何避免不必要的通信？**
> A: 用 `model.no_sync()` 跳过中间步骤的 AllReduce，只在最后一步同步。否则每个 micro-batch 都做一次 AllReduce，通信开销放大 $K$ 倍。

## 延伸阅读

**显存优化的其他手段**：
- [[Gradient Checkpointing]] — 用重计算换 activation 显存，与梯度累积互补
- [[ZeRO]] — 在数据并行中切分 optimizer states / gradients / parameters，减少冗余显存
