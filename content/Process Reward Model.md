---
type: concept
description: 对推理过程中每个中间步骤给出奖励信号的 Reward Model 变体，解决 Outcome RM 无法定位错误步骤的 credit assignment 问题
aliases:
  - PRM
  - 过程奖励模型
  - Process RM
  - 过程监督
prerequisites:
  - "[[Reward Model]]"
  - "[[Credit Assignment]]"
tags:
  - reward-model
  - reasoning
  - post-training
  - math
created: 2026-02-10
updated: 2026-02-11T00:03
---

# Process Reward Model

Process Reward Model（PRM，过程奖励模型）是 [[Reward Model]] 的一种变体，它对推理过程中的**每个中间步骤**给出奖励信号，而非仅评估最终结果。PRM 的核心价值在于解决 [[Credit Assignment]] 问题——当一个多步推理的最终答案错误时，传统的 Outcome Reward Model（ORM）无法告诉你**哪一步**出了问题，而 PRM 可以。

## 动机：为什么需要过程监督？

> [!intuition] 从阅卷老师的视角理解
> 想象你在批改一道数学证明题。一种方式是只看最终答案对不对（ORM）；另一种是逐步检查推导过程，在第一个出错的步骤旁标注红叉（PRM）。
>
> 后者提供的反馈信息量远大于前者——学生知道**哪里**出了问题，而不只是"答案错了"。

传统的 ORM 只提供 sequence-level 的 sparse reward $r_{ORM}(x, y) \in \mathbb{R}$，这在短回答任务中足够，但在**长链推理**中面临根本困难：

1. **Credit assignment 模糊**：一个 10 步推理链答案错了，ORM 给出低分，但模型无法知道是第 3 步的代数错误还是第 7 步的逻辑跳跃导致的
2. **学习效率低**：模型只能通过大量试错来隐式学习哪些步骤模式是好的，无法直接从错误步骤获得反馈
3. **容易被 hack**：模型可能学会在最后一步"猜"出正确答案的模式，而中间推理过程完全不合理（参见 [[Reward Hacking]]）

PRM 通过提供 **step-level 的 dense reward** 直接解决这些问题。

## 核心定义：PRM vs ORM

> [!definition] 形式化定义
> 给定输入 $x$ 和推理过程 $y = (y_1, y_2, \ldots, y_T)$（$T$ 个步骤）：
>
> - **ORM**：$r_{ORM}(x, y) \in \mathbb{R}$ — 只对完整序列给出一个分数
> - **PRM**：$r_{PRM}(x, y_{1:t}) \in \mathbb{R}$ for each $t \in \{1, \ldots, T\}$ — 对每个步骤给出分数

> [!comparison] PRM vs ORM 全面对比
>
> | 维度 | ORM（Outcome RM） | PRM（Process RM） |
> |------|-------------------|-------------------|
> | **评估粒度** | 整个序列 | 每个推理步骤 |
> | **信号密度** | Sparse（1 signal/sequence） | Dense（T signals/sequence） |
> | **Credit assignment** | 困难，需要 [[GAE]] 等方法 | 直接，每步有独立反馈 |
> | **Reward hacking** | 容易（只需最终结果看起来对） | 更难（每步都需要正确） |
> | **标注成本** | 低（只需判断最终答案） | 高（需要逐步标注） |
> | **"步骤"定义** | 不需要 | 需要明确定义步骤边界 |
> | **适用场景** | 通用任务 | 长链推理（数学、代码、逻辑） |

到这里，核心概念已经清楚：PRM 用更细粒度的监督换取更好的 credit assignment 和更强的 reward hacking 抵抗力。接下来的问题是——**PRM 的训练数据从哪来？**

## 训练方法：三种标注范式

PRM 训练的核心挑战在于获取 step-level 的标注数据。目前有三种主要方法，各有取舍：

### 1. 人工标注

最直接的方式：让人类标注员逐步检查推理过程，标记每步是否正确。

> [!paper] PRM800K (Lightman et al., 2023)
> "Let's Verify Step by Step" 提出了 PRM 的概念，并构建了 PRM800K 数据集——约 800K 个步骤级标注，覆盖 MATH 数据集的数学推理问题。这是 PRM 领域的奠基工作。

**优点**：标注质量最高，直接评估步骤正确性
**缺点**：成本极高，难以规模化；标注一致性依赖标注员的数学能力

### 2. Monte Carlo (MC) Estimation

用 completion model 自动估计每步的"正确性"：对于步骤 $y_t$，从该步骤出发做 $K$ 次独立 completion，统计最终到达正确答案的比例。

$$\hat{r}(x, y_{1:t}) = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}[\text{completion}_k(x, y_{1:t}) \text{ reaches correct answer}]$$

> [!warning] MC Estimation 的根本局限
> [[Lessons of Developing PRMs (2025)]] 揭示了一个深刻的洞察：MC estimation 本质上是在估计 **value function**（从当前状态出发的未来成功概率），而非 **step correctness**（当前步骤是否正确）。
>
> 这导致两类噪声：
> - **假阴性**：步骤正确，但 completion model 能力不足，后续全部做错 -- 被误标为"错误步骤"
> - **假阳性**：步骤错误，但 completion model 恰好"歪打正着" -- 被误标为"正确步骤"

> [!paper] Math-Shepherd (Wang et al., 2024)
> 提出了基于 MC estimation 的自动标注方法，大幅降低了 PRM 训练数据的获取成本。但如上所述，其标注质量存在系统性偏差。

### 3. LLM-as-a-Judge

用强大的 LLM（如 GPT-4、Qwen2.5-72B）直接判断每步是否正确。

**优点**：效果接近人工标注，成本远低于人工
**缺点**：依赖 judge LLM 的推理能力；对复杂数学步骤可能判断失误

### Consensus Filtering：整合多种方法

[[Lessons of Developing PRMs (2025)]] 提出了一种实用的整合策略：同时使用 MC estimation 和 LLM-as-a-judge 标注同一批数据，**只保留两者对错误步骤位置达成共识的样本**。由于两种方法的噪声模式不同（MC 的噪声来自 completion model 的不确定性，LLM-as-a-judge 的噪声来自推理能力局限），取交集可以有效过滤各自的假阳性/假阴性。实验表明，仅保留约 40% 的数据就能达到接近甚至更好的效果。

## 评估方法

PRM 的评估本身也是一个值得注意的问题。目前有两种主要评估范式：

### Best-of-N (BoN)

从 $N$ 个候选 response 中，用 PRM 打分选出最优的一个，看最终答案是否正确。这是 **response-level** 的评估。

> [!warning] BoN 评估的系统性偏差
> [[Lessons of Developing PRMs (2025)]] 发现 BoN 存在三重偏差：
> 1. **目标不对齐**：BoN 选的是"最可能正确"的 response，而非"过程最正确"的 response
> 2. **分数膨胀**：随着 N 增大，即使是随机选择也能提升准确率
> 3. **Process-to-Outcome Shift**：BoN 本质上是在做 outcome selection，无法区分"过程正确"和"结果碰巧正确"
>
> 这解释了一个反直觉的现象：MC estimation 训练的 PRM 在 BoN 上表现最好，但在 step-level 错误识别上表现最差。

### PROCESSBENCH

直接评估 PRM 识别错误步骤的能力（step-level），更直接地衡量 PRM 的核心能力。

**建议**：同时使用 BoN（衡量实用价值）和 PROCESSBENCH（衡量过程监督能力）进行评估，避免单一指标的误导。

## 应用场景

PRM 的价值不仅在于训练阶段，在推理阶段同样重要：

**训练阶段**：
- 在 [[GRPO]] 等 RL 算法中提供 step-level reward，实现更细粒度的策略优化
- 作为 [[Reward Hacking]] 的缓解手段——每步都需要正确，模型更难通过"捷径"获得高分

**推理阶段（Test-Time Compute Scaling）**：
- 与 MCTS / beam search 结合：PRM 作为每步的评估函数，引导搜索树的展开方向
- 在每个推理步骤进行 verify-then-continue，及时剪枝错误分支
- 这是 test-time compute scaling 的核心组件之一——通过在推理时投入更多计算来提升推理质量

## 局限性

> [!warning] PRM 的边界条件
> 1. **"步骤"的定义是模糊的**：自然语言推理不像数学证明那样有清晰的步骤边界。如何分割步骤、步骤粒度如何选择，目前没有统一标准
> 2. **标注成本仍然是瓶颈**：即使有 MC estimation 和 LLM-as-a-judge，高质量的 step-level 标注仍然比 outcome-level 标注昂贵得多
> 3. **可能引入新的 reward hacking 模式**：模型可能学会生成"看起来每步都正确"但实际上逻辑不连贯的推理链
> 4. **领域依赖性强**：PRM 在数学推理中效果显著，但在开放式生成（如创意写作）中，"步骤正确性"本身就难以定义
> 5. **与 [[DPO]] 不兼容**：DPO 的框架基于 sequence-level 的偏好对比，无法直接利用 step-level 的过程奖励信号

## 面试要点

> [!interview] 核心问题
> **Q: PRM 和 ORM 的本质区别是什么？**
> A: ORM 只评估最终结果（sparse reward），PRM 评估每个推理步骤（dense reward）。核心优势是解决 credit assignment——能定位具体哪一步出错。代价是标注成本更高。
>
> **Q: MC estimation 训练 PRM 有什么问题？**
> A: MC estimation 本质上是在估计 value function（未来成功概率），而非 step correctness（当前步骤是否正确）。这会引入假阳性（错误步骤被标为正确）和假阴性（正确步骤被标为错误）。
>
> **Q: PRM 在推理阶段有什么用？**
> A: 可以与 MCTS/beam search 结合，在每步评估后决定是否继续展开，实现 test-time compute scaling。

## 延伸阅读

**系统性经验总结**：
- [[Lessons of Developing PRMs (2025)]] -- Qwen 团队的 PRM 开发实践，深入分析了标注方法和评估偏差

**替代方案**：
- [[On-Policy Distillation]] -- 通过 token-level 的 dense supervision 绕过显式 PRM，直接提供每个 token 的监督信号
