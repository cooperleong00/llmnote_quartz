---
type: concept
description: 通过衰减而非截断保留边界 token 梯度的 RL 优化策略，解决 hard clipping 的探索损失问题
aliases:
  - Soft Clipping
  - 软裁剪
prerequisites:
  - "[[PPO]]"
  - "[[Importance Sampling]]"
tags:
  - post-training
  - reinforcement-learning
  - optimization
created: 2026-04-05
updated: 2026-04-05
---

# Soft Clipping

软裁剪（Soft Clipping）是一类通过**衰减（decay）而非截断（clip）**来处理重要性采样比率（IS ratio）边界 token 的 RL 优化策略，旨在解决 [[PPO]]/[[GRPO]] 中 hard clipping 导致的探索损失问题。

## 核心思想

> [!intuition] 从 Hard Clipping 到 Soft Clipping

**Hard Clipping（PPO/GRPO）**：
- 当 IS ratio $w = \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ 超出 $[1-\epsilon, 1+\epsilon]$ 范围时，**梯度被直接截断为零**
- 问题：边界外的 token（通常是低概率但重要的 token，如反思 token）被丢弃，限制探索

**Soft Clipping**：
- 对边界 token 应用**衰减的梯度权重**，而非直接丢弃
- 保留 token 的梯度贡献，同时控制其影响大小
- 平衡探索（保留梯度）与稳定性（衰减权重）

## 方法演进

| 方法 | 权重形式 | 左边界 | 右边界 | 主要问题 |
|------|----------|--------|--------|----------|
| **PPO/GRPO** | Hard clip | 0 | 0 | 丢弃边界 token |
| **CISPO** | Constant $\nabla\log\pi$ | 发散 | 发散 | 梯度无限增长 |
| **GPPO** | Constant $\nabla\log\pi$ | 发散 | 发散 | 同上 |
| **CE-GPPO** | Scaled constant | 缓解发散 | 缓解发散 | 未根本解决 |
| **ASPO** | Reversed ratio | 收敛 | 收敛 | 大偏差 |
| **DGPO** | 双边解耦衰减 | 多项式衰减 | 倒数根式衰减 | 最优平衡 |

## 数学统一视角

Soft clipping 方法可以统一表示为：

$$\mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{\sum_i |o_i|} \sum_i \sum_t \mathcal{W}(w_{i,t}) \hat{A}_i \nabla_\theta \log \pi_\theta(o_{i,t}|\cdot) \right]$$

或（DGPO 的 probability 梯度形式）：

$$\mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{\sum_i |o_i|} \sum_i \sum_t \mathcal{W}(w_{i,t}) \hat{A}_i \pi_\theta(o_{i,t}|\cdot) \right]$$

其中 $\mathcal{W}(w)$ 是 IS ratio 的权重函数：
- Hard clipping：$\mathcal{W}(w) = \mathbb{1}_{[1-\epsilon, 1+\epsilon]}(w)$
- Soft clipping：$\mathcal{W}(w)$ 在边界外是 $w$ 的连续衰减函数

## 关键权衡

> [!warning] Soft Clipping 的挑战

1. **梯度发散 vs 梯度丢弃**：保留边界梯度可能导致权重发散（如 CISPO/GPPO 在左边界）
2. **偏差 vs 方差**：Soft clipping 引入偏差，但可能减少方差
3. **探索 vs 稳定性**：衰减机制需要在两者之间取得平衡

## 相关概念

- [[PPO]] - Hard clipping 的原始实现
- [[GRPO]] - 将 hard clipping 应用于 group RL
- [[CISPO]] - Soft clipping 的早期实现
- [[DGPO]] - 从 probability 梯度角度解决 soft clipping 的发散问题
- [[Entropy Collapse]] - Soft clipping 试图解决的核心问题

## 延伸阅读

- [[2603.14389|Fu et al., 2026]] - DGPO 论文，深入分析 soft clipping 的理论问题
- [[MiniMax-M1 (2025)]] - CISPO 的来源
