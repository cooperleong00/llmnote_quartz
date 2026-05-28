---
type: method
description: 通过解耦权重衰减修正 Adam 中 L2 正则化失效问题的优化器，是现代 LLM 训练的标准选择
aliases:
  - AdamW 优化器
  - Adam with Decoupled Weight Decay
prerequisites:
  - "[[Adam]]"
tags:
  - optimization
  - foundations
created: 2026-03-01
updated: 2026-03-01T23:05
---

# AdamW

AdamW（Adam with Decoupled Weight Decay）解决了 [[Adam]] 优化器中 L2 正则化与自适应学习率交互失效的问题。核心改进是将权重衰减从梯度更新中解耦，直接作用于参数本身，使正则化效果不再受自适应学习率影响。这一简单修正使 AdamW 成为现代 LLM 训练的标准优化器。

## 动机：Adam 中的 L2 正则化问题

> [!intuition] 为什么 Adam + L2 正则化会失效？
>
> 在 SGD 中，L2 正则化等价于权重衰减：
> - L2 正则化：在损失函数中加入 $\frac{\lambda}{2}\|\theta\|^2$
> - 权重衰减：更新时直接 $\theta \leftarrow (1-\lambda)\theta - \alpha \nabla f(\theta)$
>
> 但在 Adam 中，这个等价性被打破了。原因是 Adam 的自适应学习率会"稀释"梯度中的正则化项，导致：
> - 梯度大的参数（分母 $\sqrt{v_t}$ 大）→ 正则化被削弱
> - 梯度小的参数（分母 $\sqrt{v_t}$ 小）→ 正则化被放大
>
> 这与我们的期望相反——我们希望所有参数受到**统一强度**的正则化，而不是根据梯度大小变化。

### Adam 的更新规则回顾


标准 Adam 优化器（带 L2 正则化）的更新步骤：

$$
\begin{align}
g_t &= \nabla_\theta f(\theta_{t-1}) + \lambda \theta_{t-1} \quad \text{(梯度 + L2 正则项)} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(一阶动量)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(二阶动量)} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(偏差修正)} \\
\theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align}
$$

问题在于：L2 正则项 $\lambda \theta_{t-1}$ 被加入梯度 $g_t$，然后经过自适应学习率 $\frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}$ 的缩放。这导致正则化强度与梯度大小耦合。

## AdamW 的解决方案：解耦权重衰减

> [!intuition] 核心思想
>
> 既然问题是"正则化项被自适应学习率影响"，那就把它从梯度中拿出来，直接作用于参数：
>
> **Adam**：$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$（其中 $\hat{m}_t$ 包含 L2 项）
>
> **AdamW**：$\theta_t = (1-\lambda)\theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$（L2 项独立作用）
>
> 这样，权重衰减的强度由 $\lambda$ 直接控制，不再受梯度大小影响。

### AdamW 的完整更新规则

> [!math] 数学形式
>
> $$
> \begin{align}
> g_t &= \nabla_\theta f(\theta_{t-1}) \quad \text{(纯梯度，不含正则项)} \\
> m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
> v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
> \hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
> \theta_t &= (1-\alpha\lambda) \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \quad \text{(解耦的权重衰减)}
> \end{align}
> $$
>
> 注意最后一行：权重衰减 $(1-\alpha\lambda)$ 直接乘在 $\theta_{t-1}$ 上，与自适应学习率部分完全独立。


## 为什么 AdamW 是 LLM 训练的标准选择

> [!comparison] AdamW vs Adam
>
> | 维度 | Adam (with L2) | AdamW |
> |------|----------------|-------|
> | **正则化方式** | L2 正则化（加入损失函数） | 权重衰减（直接作用于参数） |
> | **正则化强度** | 受梯度大小影响（不均匀） | 统一强度（由 $\lambda$ 控制） |
> | **泛化性能** | 较差（正则化失效） | 更好（正则化有效） |
> | **超参数调优** | $\lambda$ 与学习率耦合 | $\lambda$ 与学习率解耦 |
> | **训练稳定性** | 一般 | 更稳定 |

### 实证优势

1. **更好的泛化**：在 ImageNet、BERT 等任务上，AdamW 的测试误差显著低于 Adam
2. **超参数解耦**：可以独立调整学习率和权重衰减，调优空间更清晰
3. **训练稳定**：大模型训练中（如 GPT、Llama），AdamW 的收敛更稳定
4. **工业标准**：几乎所有现代 LLM（GPT-3/4、Llama、GLM 等）都使用 AdamW

> [!example] 典型超参数
>
> LLM 训练中的常见配置：
> - $\beta_1 = 0.9$（一阶动量）
> - $\beta_2 = 0.95$ 或 $0.999$（二阶动量）
> - $\epsilon = 10^{-8}$
> - $\lambda = 0.01 \sim 0.1$（权重衰减系数）
> - 学习率：通常配合 warmup + cosine decay

## 实现细节

> [!warning] 常见误区
>
> **误区 1**：以为 AdamW 只是"Adam + weight decay"
> - **正解**：关键是"解耦"——weight decay 不经过自适应学习率
>
> **误区 2**：在 PyTorch 中使用 `torch.optim.Adam` + `weight_decay` 参数
> - **正解**：`torch.optim.Adam` 的 `weight_decay` 实现的是 L2 正则化（耦合版本），应使用 `torch.optim.AdamW`
>
> **误区 3**：对所有参数都应用 weight decay
> - **正解**：通常不对 bias 和 LayerNorm 参数应用 weight decay


### PyTorch 示例

```python
import torch

# ✅ 正确：使用 torch.optim.AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # 这里是真正的解耦 weight decay
)

# ❌ 错误：torch.optim.Adam 的 weight_decay 是 L2 正则化
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01  # 这是耦合的 L2 正则化，不是 AdamW
)

# 🔧 高级：对不同参数组使用不同的 weight decay
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-4)
```

## 历史与影响

> [!paper] 原始论文
>
> **Decoupled Weight Decay Regularization**
> - 作者：Ilya Loshchilov, Frank Hutter
> - 会议：ICLR 2019
> - 核心贡献：
>   1. 指出 Adam 中 L2 正则化与自适应学习率的耦合问题
>   2. 提出解耦权重衰减的 AdamW 算法
>   3. 实验证明 AdamW 在多个任务上优于 Adam

这篇论文的影响深远——它不仅修正了 Adam 的一个技术问题，更重要的是揭示了**优化器设计中的一个普遍原则**：正则化应该与学习率调整机制解耦，以保持其预期效果。

## 相关概念

- [[Adam]] — AdamW 的基础，理解 Adam 是理解 AdamW 的前提
- [[SGD]] — 在 SGD 中 L2 正则化与 weight decay 等价，AdamW 恢复了这一性质
- [[Mixed Precision Training]] — 现代 LLM 训练中，AdamW 通常与混合精度训练配合使用
- [[Learning Rate Scheduling]] — AdamW 常配合 warmup + cosine decay 使用

> [!interview] 面试要点
>
> **Q: AdamW 和 Adam 的区别是什么？**
> A: AdamW 将权重衰减从梯度更新中解耦。Adam 把 L2 正则项加入梯度，导致正则化强度受自适应学习率影响；AdamW 直接对参数做权重衰减，使正则化强度统一且可控。
>
> **Q: 为什么 Adam 中 L2 正则化会失效？**
> A: 因为 L2 正则项被加入梯度后，会经过 Adam 的自适应学习率缩放（除以 $\sqrt{v_t}$）。梯度大的参数正则化被削弱，梯度小的参数正则化被放大，导致正则化强度不均匀。
>
> **Q: PyTorch 的 `torch.optim.Adam` 设置 `weight_decay` 参数就是 AdamW 吗？**
> A: 不是。`torch.optim.Adam` 的 `weight_decay` 实现的是耦合的 L2 正则化，应该使用 `torch.optim.AdamW` 才是真正的解耦权重衰减。
>
> **Q: 什么时候应该用 AdamW 而不是 Adam？**
> A: 几乎所有需要正则化的场景都应该用 AdamW。特别是大模型训练（LLM、Vision Transformer 等），AdamW 已经是事实标准。
