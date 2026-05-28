---
description: 通过旋转矩阵编码相对位置的方法，是 Llama/Qwen/Mistral 的标准方案
type: method
prerequisites:
  - "[[Positional Encoding]]"
  - "[[Attention]]"
tags:
  - transformer
  - positional-encoding
  - architecture
created: 2025-01-26
updated: 2026-02-01T01:10
---

# RoPE (Rotary Position Embedding)

旋转位置编码（RoPE, Rotary Position Embedding）是一种[[Positional Encoding|相对位置编码]]方法，通过旋转矩阵将位置信息编码到 [[Transformer]] 的 Query 和 Key 中。它是当前主流 LLM（Llama、Qwen、Mistral 等）的标准位置编码方案。

> [!paper] 论文出处
> Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021

---

## 核心思想

> [!intuition] 直觉理解
> RoPE 的核心洞察：**相对位置可以通过旋转来表示**。
>
> 想象二维平面上的向量：
> - 位置 $m$ 的向量旋转角度 $m\theta$
> - 位置 $n$ 的向量旋转角度 $n\theta$
> - 它们的点积只依赖于相对角度 $(m-n)\theta$
>
> 这样，[[Attention|注意力]]分数自然地编码了相对位置信息。

---

## 数学推导

### 二维情况

对于二维向量 $\mathbf{q}$ 和 $\mathbf{k}$，在位置 $m$ 和 $n$：

$$
\mathbf{q}_m = R_m \mathbf{q}, \quad \mathbf{k}_n = R_n \mathbf{k}
$$

其中旋转矩阵：

$$
R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
$$

它们的点积：

$$
\mathbf{q}_m^T \mathbf{k}_n = \mathbf{q}^T R_m^T R_n \mathbf{k} = \mathbf{q}^T R_{n-m} \mathbf{k}
$$

> [!math] 关键性质
> $R_m^T R_n = R_{n-m}$，点积只依赖于相对位置 $n-m$！

### 高维扩展

对于 $d$ 维向量，将其分成 $d/2$ 组，每组 2 维，分别应用不同频率的旋转：

> [!definition] RoPE 公式
> $$
> R_m = \begin{pmatrix}
> \cos m\theta_1 & -\sin m\theta_1 & & & \\
> \sin m\theta_1 & \cos m\theta_1 & & & \\
> & & \cos m\theta_2 & -\sin m\theta_2 & \\
> & & \sin m\theta_2 & \cos m\theta_2 & \\
> & & & & \ddots
> \end{pmatrix}
> $$
>
> 其中 $\theta_i = 10000^{-2i/d}$（与 Sinusoidal PE 相同的频率设计）

> [!intuition] 频率设计的含义
> **基频（Base）**：10000 是基数参数，控制频率的衰减速度。
>
> **高低频的定义**：
> - $\theta$ 是**角速度**（每个位置旋转的角度）
> - **大 $\theta$** → 旋转快 → **高频** → 周期短（$2\pi/\theta$ 小）
> - **小 $\theta$** → 旋转慢 → **低频** → 周期长（$2\pi/\theta$ 大）
>
> | $i$ | $\theta_i$ | 周期 $2\pi/\theta$ | 频率 |
> |-----|------------|-------------------|------|
> | 0 | 1 | ~6 位置 | **高频** |
> | 63 ($d$=128) | ~0.0001 | ~60000 位置 | **低频** |
>
> **为什么这样设计？**
> - **高频分量**（大 $\theta$，前面的维度）：周期短，对**局部**位置差异敏感
> - **低频分量**（小 $\theta$，后面的维度）：周期长，能编码**远距离**位置关系
> - 类似傅里叶变换：用不同频率的基函数组合表示位置信息

### 高效实现

不需要真的构造旋转矩阵，可以用逐元素操作：

$$
\begin{aligned}
q'_{2i} &= q_{2i} \cos m\theta_i - q_{2i+1} \sin m\theta_i \\
q'_{2i+1} &= q_{2i} \sin m\theta_i + q_{2i+1} \cos m\theta_i
\end{aligned}
$$

```python
def apply_rope(x, cos, sin):
    # x: (batch, seq, d)
    x1, x2 = x[..., ::2], x[..., 1::2]  # 奇偶分离
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
```

---

## RoPE 的优势

### 1. 相对位置编码

注意力分数只依赖于相对位置：

$$
\mathbf{q}_m^T \mathbf{k}_n = f(\mathbf{q}, \mathbf{k}, m-n)
$$

### 2. 远程衰减

> [!intuition] 自然的距离衰减
> 由于不同维度使用不同频率，高频分量（大 $\theta$）在远距离时会"抵消"，产生自然的距离衰减效果。

> [!math] 高频抵消的数学原理
> 考虑位置 $m$ 和 $n$，相对距离 $\Delta = n - m$：
>
> **高频分量**（$\theta$ 大，如 $\theta_0 = 1$）：
> - 旋转角度：$\Delta \cdot \theta_0 = \Delta$
> - 当 $\Delta$ 很大时，角度快速增长，$\cos(\Delta)$ 和 $\sin(\Delta)$ 快速振荡
> - 在多个高频维度上，这些振荡的正负值会相互抵消
>
> **低频分量**（$\theta$ 小，如 $\theta_{63} \approx 0.0001$）：
> - 旋转角度：$\Delta \cdot \theta_{63} \approx 0.0001 \Delta$
> - 即使 $\Delta = 1000$，角度也只有 $\sim 0.1$ 弧度，变化平缓
> - 低频分量在远距离时仍能保持稳定的相位关系
>
> **结果**：
> - 近距离：高频+低频都贡献 → 注意力强
> - 远距离：高频抵消，只有低频贡献 → 注意力自然衰减

### 3. 线性计算复杂度

RoPE 只需要对 Q 和 K 做逐元素操作，不增加额外的计算复杂度。

### 4. 无需额外参数

旋转角度是预定义的，不需要学习额外参数。

---

## [[长度外推]]问题

> [!warning] RoPE 的外推局限
> 虽然 RoPE 是相对位置编码，但在超出训练长度时仍会出现性能下降：
> - 训练时未见过的相对距离
> - 低频分量在长距离时的行为不稳定

> [!intuition] 为什么低频分量不稳定？
> **训练时的周期覆盖**：
> - 假设训练长度 $L_{train} = 2048$
>
> **高频分量**（$\theta_0 = 1$，周期 ~6 位置）：
> - 训练时见过 $2048 / 6 \approx 341$ 个完整周期
> - 外推时只是重复已见过的模式 → **稳定**
>
> **低频分量**（$\theta_{63} \approx 0.0001$，周期 ~60000 位置）：
> - 训练时只见过周期的 $2048 / 60000 \approx 3\%$
> - 外推到 $L_{infer} = 8192$ 时，进入了训练时未见过的相位区域
> - 模型不知道如何处理这些新的相位值 → **不稳定**
>
> **核心问题**：低频分量的周期太长，训练数据无法覆盖完整周期。

### 解决方案

#### 1. Position Interpolation (PI)

将位置索引线性缩放到训练范围内：

$$
m' = m \cdot \frac{L_{train}}{L_{target}}
$$

**优点**：简单有效
**缺点**：需要微调

#### 2. NTK-aware Scaling

调整频率基数，而非位置索引：

$$
\theta'_i = 10000^{-2i/d} \cdot \alpha^{-2i/d}
$$

**优点**：保持局部分辨率
**缺点**：需要选择合适的 $\alpha$

> [!intuition] 为什么 NTK 能保持局部分辨率？
> **Position Interpolation 的问题**：
> - PI 缩放位置索引 $m' = m \cdot s$（$s < 1$）
> - 所有频率分量都被等比例压缩
> - 高频分量也被压缩，损失局部位置的分辨率
>
> **NTK-aware 的策略**：
> - 增大基数：$10000 \to 10000 \cdot \alpha$
> - $\theta'_i = (10000 \cdot \alpha)^{-2i/d} = \theta_i \cdot \alpha^{-2i/d}$
> - 由于指数关系，**高频分量几乎不变，低频分量显著降低**
> - 例如 $\alpha = 2$：
>   - $\theta'_0 = 1 \cdot 2^0 = 1$（不变，高频保持）
>   - $\theta'_{63} = 0.0001 \cdot 2^{-0.98} \approx 0.00005$（减半，低频周期翻倍）
>
> **结果**：
> - 高频不变 → **局部分辨率保持**
> - 低频周期变长 → **能编码更远距离**
>
> 这类似于 Neural Tangent Kernel (NTK) 理论中的频谱调整策略。

#### 3. YaRN (Yet another RoPE extensioN)

结合 PI 和 NTK，对不同频率分量使用不同策略。

---

## 在主流模型中的应用

| 模型        | 位置编码 | 外推方案           |
| --------- | ---- | -------------- |
| Llama 1/2 | RoPE | 原始             |
| Llama 3   | RoPE | 扩展训练           |
| Qwen      | RoPE | NTK-aware      |
| Mistral   | RoPE | Sliding Window |
| GPT-NeoX  | RoPE | 原始             |

---

## 与其他位置编码的对比

| 方法         | 类型  | 外推能力 | 计算开销 | 额外参数 |
| ---------- | --- | ---- | ---- | ---- |
| Sinusoidal | 绝对  | 差    | 低    | 无    |
| 可学习 PE     | 绝对  | 无    | 低    | 有    |
| **RoPE**   | 相对  | 中等   | 低    | 无    |
| [[ALiBi]]  | 相对  | 好    | 低    | 无    |
| T5 Bias    | 相对  | 中等   | 中    | 有    |

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: RoPE 的核心思想是什么？**
> A: 用旋转矩阵编码位置，使得 Q 和 K 的点积只依赖于相对位置。位置 $m$ 的向量旋转 $m\theta$，点积自然编码相对距离。
>
> **Q2: RoPE 为什么是相对位置编码？**
> A: 因为 $R_m^T R_n = R_{n-m}$，Q 和 K 的点积只依赖于 $n-m$，即相对位置。
>
> **Q3: RoPE 中的高频和低频分量分别有什么作用？**
> A: $\theta$ 大是高频（周期短），$\theta$ 小是低频（周期长）。高频分量对局部位置敏感，低频分量能编码远距离关系。远距离时高频振荡抵消，产生自然的距离衰减。
>
> **Q4: 为什么 RoPE 的基数选择 10000？**
> A: 继承自 Sinusoidal PE 的设计。10000 使得最低频分量的周期约为 60000 位置，足够覆盖常见的序列长度，同时保持数值稳定性。
>
> **Q5: RoPE 的计算复杂度是多少？**
> A: $O(nd)$，只需要对 Q 和 K 做逐元素的旋转操作，不增加额外复杂度。
>
> **Q6: RoPE 的长度外推问题如何解决？**
> A: Position Interpolation（缩放位置索引）、NTK-aware Scaling（调整频率基数，保持局部分辨率）、YaRN（混合策略）。
>
> **Q7: 为什么 RoPE 成为主流？**
> A: 相对位置编码、无额外参数、计算高效、效果好。Llama 的成功推动了 RoPE 的广泛采用。

---

## 相关概念

- [[Positional Encoding]] — 位置编码的基础概念
- [[ALiBi]] — 另一种相对位置编码
- [[长度外推]] — RoPE 的核心挑战
