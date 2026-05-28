---
type: method
description: 通过两层分组结构（episode-level + step-level）实现 multi-turn LLM agent 的细粒度信用分配，保持 group-based RL 的 critic-free 优势
aliases:
  - Group-in-Group Policy Optimization
  - 组内组策略优化
prerequisites:
  - "[[GRPO]]"
  - "[[RLOO]]"
  - "[[Credit Assignment]]"
  - "[[Value Function]]"
tags:
  - post-training
  - rlhf
  - agent
  - policy-optimization
created: 2026-03-06
updated: 2026-03-06T02:34
---

# GiGPO

GiGPO（Group-in-Group Policy Optimization）是一种专为 multi-turn LLM agent 训练设计的 group-based RL 算法。它通过**两层分组结构**实现细粒度的 [[Credit Assignment|信用分配]]，在保持 [[GRPO]] 的 critic-free、低内存、稳定收敛优势的同时，解决了长序列任务中的 per-step credit assignment 问题。

## 动机

> [!intuition] 为什么需要 GiGPO？
>
> **问题**：Multi-turn agent 任务（如 ALFWorld、WebShop）中，轨迹可能长达 50 步，奖励稀疏且延迟。[[GRPO]] 只提供 episode-level 的整体评价，无法区分单个动作的好坏——一个 50 步轨迹中，哪些步骤是关键决策？哪些是无效探索？
>
> **现有方案的困境**：
> - **Per-step rollout**（图 1 middle）：为每个状态额外采样多个动作，构建 step-level groups → 计算成本爆炸（需要大量额外 LLM forward passes）
> - **Trajectory-level GRPO**（如 RAGEN）：将整个 episode 拼接为单个 response → 在长 horizon 任务中扩展性差
>
> **GiGPO 的洞察**：在相同任务和初始状态下，多条轨迹会**自然地重复访问相同的环境状态**（如重复访问的网页、房间、游戏场景）。这些重复状态提供了"免费的"step-level groups——不需要额外 rollout，只需 retroactively 识别和分组。

## 核心机制

GiGPO 的核心是**两层分组结构**，分别提供 macro 和 micro 两个层次的信用分配信号。

### 1. Episode-Level Grouping（宏观信号）

与 [[GRPO]] 相同，GiGPO 首先采样一组完整轨迹 $\{\tau_1, \dots, \tau_N\}$，所有轨迹来自相同的任务 $x$ 和初始状态。

**Episode relative advantage** 衡量整条轨迹的相对质量：

$$
A^E(\tau_i) = \frac{R(\tau_i) - \mu_R}{\sigma_R}
$$

其中：
- $R(\tau_i) = \sum_{t=1}^T r_t^{(i)}$ 是轨迹的总回报
- $\mu_R, \sigma_R$ 是 group 内所有轨迹回报的均值和标准差

> [!intuition] Episode advantage 的作用
> 提供全局视角：这条轨迹整体上是成功还是失败？但它无法告诉我们"第 15 步的动作是好是坏"。

### 2. Step-Level Grouping（微观信号）

这是 GiGPO 的核心创新。通过 **anchor state grouping** 机制，retroactively 构建 step-level groups。

#### Anchor State Grouping

**关键观察**：在相同任务和初始状态下，多条轨迹会重复访问相同的环境状态 $\tilde{s}$（如重复访问的网页、房间）。

**构建过程**：
1. 收集所有唯一状态：$\mathcal{U} = \{\tilde{s}_1, \tilde{s}_2, \dots, \tilde{s}_U\}$
2. 对每个唯一状态 $\tilde{s}$，构建 step-level group：

$$
G^S(\tilde{s}) = \{(\boldsymbol{a}_t^{(i)}, r_t^{(i)}) \mid \boldsymbol{s}_t^{(i)} = \tilde{s}, \forall i \in [1, N], t \in [1, T]\}
$$

这个 group 包含所有"从状态 $\tilde{s}$ 出发的动作"及其即时奖励。

> [!example] 具体例子
> 在 WebShop 任务中：
> - 轨迹 1：访问商品页 A → 点击"加入购物车" → 失败
> - 轨迹 2：访问商品页 A → 点击"查看详情" → 成功
> - 轨迹 3：访问商品页 A → 点击"返回搜索" → 失败
>
> 这三条轨迹都访问了"商品页 A"这个状态，GiGPO 会将这三个动作分组，计算它们的相对优势。

#### Step Relative Advantage

对于 group $G^S(\tilde{s})$ 中的每个动作，计算其 **discounted return**（而非即时奖励，以捕捉长期影响）：

$$
R_t^{(i)} = \sum_{k=t}^T \gamma^{k-t} r_k^{(i)}
$$

然后计算 **step relative advantage**：

$$
A^S(\boldsymbol{a}_t^{(i)}) = \frac{R_t^{(i)} - \mu_{G^S(\tilde{s})}}{\sigma_{G^S(\tilde{s})}}
$$

其中 $\mu_{G^S(\tilde{s})}, \sigma_{G^S(\tilde{s})}$ 是该 step-level group 内所有 discounted returns 的均值和标准差。

> [!intuition] Step advantage 的作用
> 提供局部视角：在相同状态下，这个动作相比其他选择是更好还是更差？

### 3. 组合优势函数

最终的 advantage 是两层信号的加权组合：

$$
A(\boldsymbol{a}_t^{(i)}) = A^E(\tau_i) + \omega \cdot A^S(\boldsymbol{a}_t^{(i)})
$$

其中 $\omega \in \mathbb{R}_{\geq 0}$ 是平衡系数（论文中设为 1）。

> [!math] GiGPO 目标函数
>
> $$
> \begin{aligned}
> \mathcal{J}_{\text{GiGPO}}(\theta) = \mathbb{E} \left[ \frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} \min \left( \rho_\theta(\boldsymbol{a}_t^{(i)}) A(\boldsymbol{a}_t^{(i)}), \text{clip}(\rho_\theta(\boldsymbol{a}_t^{(i)}), 1 \pm \epsilon) A(\boldsymbol{a}_t^{(i)}) \right) \right] \\
> - \beta \mathbb{D}_{\text{KL}}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x))
> \end{aligned}
> $$
>
> 其中：
> - $\rho_\theta(\boldsymbol{a}_t^{(i)}) = \frac{\pi_\theta(\boldsymbol{a}_t^{(i)} \mid \boldsymbol{s}_t^{(i)}, x)}{\pi_{\theta_{\text{old}}}(\boldsymbol{a}_t^{(i)} \mid \boldsymbol{s}_t^{(i)}, x)}$ 是 importance sampling ratio
> - $\text{clip}(\cdot, 1 \pm \epsilon)$ 是 [[PPO]] 风格的 clipping 机制
> - $\beta$ 控制 KL penalty 强度

## 关键优势

> [!comparison] GiGPO vs 其他方法
>
> | 方法 | Episode-level | Step-level | 额外成本 | Critic-free |
> |------|---------------|------------|----------|-------------|
> | [[GRPO]] | ✅ | ❌ | 无 | ✅ |
> | Per-step rollout | ✅ | ✅ | 极高（额外 LLM forward） | ✅ |
> | [[PPO]] | ✅ | ✅ | 中（value network） | ❌ |
> | **GiGPO** | ✅ | ✅ | 极低（< 0.002%） | ✅ |

**核心优势**：
1. **细粒度信用分配**：同时提供 episode 和 step 两个层次的反馈
2. **零额外 rollout**：完全 offline，只需 hashmap-based grouping
3. **保持 group-based RL 优势**：critic-free、低内存、稳定收敛
4. **优雅降级**：如果没有重复状态（$A^S = 0$），自动退化为 [[GRPO]]

## 实验结果

**Benchmarks**：
- **ALFWorld**（embodied household tasks）：> 12% 提升 over GRPO
- **WebShop**（web-based shopping）：> 9% 提升 over GRPO
- **Search-augmented QA**：42.1% (Qwen2.5-3B), 47.2% (Qwen2.5-7B)

**计算成本**：
- 时间开销：< 0.002%（几乎可忽略）
- GPU 内存：与 GRPO 相同
- LLM rollout 次数：与 GRPO 相同

## 局限性

> [!warning] 边界条件
>
> **依赖状态匹配**：GiGPO 的 step-level grouping 依赖于识别重复状态。在高度复杂或噪声环境中，相同状态可能难以检测。
>
> **缓解方案**：
> - **Similarity-based grouping**：使用 longest matching subsequence，相似度 > 0.9 时分组
> - **优雅降级**：极端情况下（无重复状态），自动退化为 GRPO
>
> **未来方向**：
> - Embedding-based state representations
> - Domain-specific structural equivalence

## 与相关工作的关系

**基于 group-based RL**：
- [[GRPO]]：GiGPO 的 episode-level 部分与 GRPO 相同
- [[RLOO]]：同样是 critic-free 的 group-based 方法

**解决 agent 训练问题**：
- **RAGEN**：将整个 episode 拼接为单个 response，在长 horizon 任务中扩展性差
- **ArCHer, AgentQ**：需要额外的 value network 或 MCTS，计算开销大

**与 PPO 的关系**：
- 使用 PPO 风格的 clipping 机制
- 但无需 critic network（通过 group-based advantage 替代）

> [!paper] 原始论文
> **Group-in-Group Policy Optimization for LLM Agent Training**
> Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An (NTU & Skywork AI)
> arXiv:2505.10978, 2025
> GitHub: https://github.com/langfengQ/verl-agent

## 延伸阅读

**Group-based RL 方法**：
- [[GRPO]] — GiGPO 的 episode-level 基础
- [[RLOO]] — 另一种 critic-free group-based 方法
- [[PPO]] — 传统 actor-critic 方法

**相关概念**：
- [[Credit Assignment]] — GiGPO 要解决的核心问题
- [[Value Function]] — GiGPO 通过 group-based advantage 避免显式建模

**应用场景**：
- [[MOC - Post-training]] — LLM agent 训练是 post-training 的重要方向
