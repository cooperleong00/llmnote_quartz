---
description: 通过 λ 参数平衡 bias-variance 的优势函数估计方法，是 PPO 的重要组件
type: concept
prerequisites:
  - "[[Actor-Critic]]"
  - "[[TD Learning]]"
tags:
  - reinforcement-learning
  - variance-reduction
created: 2025-01-25
updated: 2026-02-01T00:52
---

# GAE (Generalized Advantage Estimation)

广义优势估计（GAE, Generalized Advantage Estimation）是一种平衡 **bias** 和 **variance** 的优势函数估计方法。它是 [[PPO]] 等现代 RL 算法的重要组成部分，主要用于降低 [[Policy Gradient|策略梯度]] 估计的方差。

> [!paper] 论文出处
> Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR 2016

---

## 核心问题

在 [[Actor-Critic]] 中，我们需要估计优势函数 $A(s, a) = Q(s, a) - V(s)$。

两种极端方法：

| 方法 | 公式 | Bias | Variance |
|------|------|------|----------|
| **TD(0)** | $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | 高（依赖 V 准确性） | 低 |
| **Monte Carlo** | $\sum_{k=0}^{\infty} \gamma^k r_{t+k} - V(s_t)$ | 低 | 高 |

> [!intuition] 核心问题
> 如何在 bias 和 variance 之间找到平衡？

---

## GAE 公式

> [!definition] GAE 定义
> $$
> A^{GAE(\gamma, \lambda)}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
> $$
>
> 其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD error。

展开形式：
$$
A^{GAE}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \ldots
$$

---

## λ 参数的作用

$\lambda \in [0, 1]$ 控制 bias-variance 权衡：

| λ 值 | 效果 | Bias | Variance |
|------|------|------|----------|
| $\lambda = 0$ | $A_t = \delta_t$（TD(0)） | 高 | 低 |
| $\lambda = 1$ | Monte Carlo | 低 | 高 |
| $0 < \lambda < 1$ | 指数加权平均 | 中 | 中 |

> [!intuition] 直觉
> - $\lambda$ 小：更信任 Critic 的估计（$V$），但 Critic 可能不准
> - $\lambda$ 大：更信任实际采样的 reward，但方差大
> - 典型值：$\lambda = 0.95$

---

## 推导

### 从 n-step return 出发

n-step advantage：
$$
A^{(n)}_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)
$$

可以证明：
$$
A^{(n)}_t = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}
$$

### GAE 是指数加权平均

GAE 是所有 n-step advantage 的指数加权平均：
$$
A^{GAE}_t = (1-\lambda)\left(A^{(1)}_t + \lambda A^{(2)}_t + \lambda^2 A^{(3)}_t + \ldots\right)
$$

化简后得到：
$$
A^{GAE}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

---

## 实际计算

在有限长度轨迹中，从后向前递归计算：

$$
A^{GAE}_t = \delta_t + \gamma \lambda \cdot A^{GAE}_{t+1}
$$

```python
def compute_gae(rewards, values, gamma, lam):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages
```

---

## 在 PPO 中的应用

[[PPO]] 使用 GAE 来估计优势函数：

1. 用当前 Critic 计算所有状态的 $V(s)$
2. 计算 TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
3. 用 GAE 公式计算 $A^{GAE}_t$
4. 用 $A^{GAE}_t$ 作为 Actor 更新的权重

**Critic 的目标**：
$$
V_{target} = A^{GAE}_t + V(s_t)
$$

---

## 与 TD(λ) 的关系

GAE 和 [[TD Learning|TD(λ)]] 密切相关：

- **TD(λ)**：用于估计 $V(s)$
- **GAE**：用于估计 $A(s, a)$

两者都使用 $\lambda$ 参数来平衡 bias 和 variance。

---

## 面试要点

> [!interview] 常见问题
>
> **Q1: GAE 解决什么问题？**
> A: 平衡优势函数估计的 bias 和 variance。TD(0) 方差低但有偏，Monte Carlo 无偏但方差高，GAE 通过 λ 参数在两者之间权衡。
>
> **Q2: GAE 的公式是什么？**
> A: $A^{GAE}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$，是 TD error 的指数加权和。
>
> **Q3: λ 参数的作用？**
> A: 控制 bias-variance 权衡。λ=0 是 TD(0)（高 bias 低 variance），λ=1 是 Monte Carlo（低 bias 高 variance）。典型值 0.95。
>
> **Q4: GAE 在 PPO 中怎么用？**
> A: 用 GAE 估计优势函数，作为 Actor 更新的权重。Critic 的目标是 $A^{GAE} + V(s)$。
