---
type: paper
description: 首篇系统性综述 On-Policy Distillation (OPD) 的论文，提出统一 f-divergence 框架，按反馈信号、教师访问权限、损失粒度三维度分类 OPD 方法
aliases:
  - OPD Survey
  - On-Policy Distillation Survey
authors:
  - Mingyang Song
  - Mao Zheng
affiliations:
  - Tencent LLM Department
arxiv: "2604.00626"
year: 2026
tags:
  - post-training
  - distillation
  - survey
  - on-policy
prerequisites:
  - "[[Knowledge Distillation]]"
  - "[[Exposure Bias]]"
  - "[[f-divergence]]"
created: 2026-04-05
updated: 2026-04-05
---

# A Survey of On-Policy Distillation for Large Language Models (2026)

## 核心贡献

这篇综述是**首篇系统性梳理 On-Policy Distillation (OPD)** 的论文，填补了该领域缺乏统一理论框架的空白。

> [!paper] 六大核心贡献
> 1. **统一理论框架**：提出 f-divergence 统一框架，揭示 GKD、MiniLLM、DistiLLM 等方法是同一目标函数在不同参数选择下的实例
> 2. **三维分类法**：按反馈信号（logit/outcome/self-play）、教师访问权限（白盒/黑盒/无教师）、损失粒度（token/sequence/hybrid）正交分类
> 3. **连接白盒与黑盒**：系统比较需要完整 logits 的方法与仅需 API 访问的方法，分析信息差距
> 4. **识别被忽视的挑战**：计算-质量权衡、低置信度状态下的"回音室"效应、动态发散适配
> 5. **未来研究路线图**：提出蒸馏 Scaling Law、不确定性感知 OPD、课程驱动采样、Agent 级蒸馏等开放问题
> 6. **工业部署指南**：提供实用的方法选择决策框架和计算预算分配建议

## 动机：从 Off-Policy 到 On-Policy

传统蒸馏是 **off-policy** 的：学生在教师生成的静态数据上训练，但推理时必须基于自己生成的（可能错误的）前缀继续生成。这种 **train-test mismatch** 是 exposure bias 的实例。

> [!intuition] Exposure Bias 的代价
> - 模仿学习理论（DAgger）表明，学生在训练分布上的每步误差 $\epsilon$ 会导致推理时 $O(\epsilon T^2)$ 的累积误差
> - OPD 通过在策略分布 $d_{\pi_\theta}$ 而非数据分布 $d_{\mathcal{D}}$ 上训练，将误差降至 $O(\epsilon T)$

## 统一数学框架

### f-divergence 族

所有 OPD 方法可统一为：

$$
\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{y \sim \pi_{\text{mix}}} \left[ \sum_{t=1}^{|y|} \mathcal{D}_f(p_\mathcal{T}(\cdot|x, y_{<t}), p_\theta(\cdot|x, y_{<t})) \right]
$$

| 方法 | $\pi_{\text{mix}}$ | $\mathcal{D}_f$ | 关键创新 |
|------|-------------------|-----------------|---------|
| **GKD** | $(1-\lambda)p_\theta + \lambda \mathcal{D}$ | F-KL/R-KL/JSD | 统一 OPD 框架 |
| **MiniLLM** | $(1-\alpha)p_\theta + \alpha p_\mathcal{T}$ | $D_{\text{KL}}(p_\theta \| p_\mathcal{T})$ | Reverse KL + REINFORCE |
| **DistiLLM** | $p_\theta$ | Skewed KL (SKL+SRKL) | $\alpha$-smoothing 避免除零 |
| **ToDi** | $\mathcal{D}$ | 自适应 F-KL/R-KL | 每 token 根据置信度选择 divergence |
| **Entropy-Aware** | $p_\theta$ | 熵门控混合 | 高熵区用 F-KL，低熵区用 R-KL |

### Divergence 的几何直觉

> [!intuition] Mode-Covering vs Mode-Seeking
> - **Forward KL** (mode-covering): 学生必须覆盖教师的所有模式，但会在模式间的"幻觉区"放置概率质量
> - **Reverse KL** (mode-seeking): 学生集中于单一高概率模式，丢失多样性但避免平均
> - **JSD**: 平衡两者，适合通用场景

## 三维分类法

### 维度 1: Feedback Signal

| 类型 | 描述 | 代表方法 |
|------|------|---------|
| **Logit-Based** | 教师提供完整 token-level 概率分布 | GKD, MiniLLM, DistiLLM, ToDi |
| **Outcome-Based** | 教师提供标量奖励或偏好对 | RLKD, ORPO-Distill, DAIL |
| **Self-Play** | 学生自己生成对比信号 | SPIN, OPSD, GATES |

### 维度 2: Teacher Access

| 类型 | 信息 | 约束 | 代表方法 |
|------|------|------|---------|
| **White-Box** | 完整 logits | 需托管大模型 | GKD, DistiLLM, MiniLLM |
| **Black-Box** | 仅 API 输出 | 无法计算 KL | GAD, Lion, OVD |
| **Teacher-Free** | 无外部教师 | 完全自举 | SPIN, OPSD, SDPO |

### 维度 3: Loss Granularity

| 粒度 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **Token-Level** | 梯度稳定、收敛快 | 局部优化、短视 | 通用指令遵循 |
| **Sequence-Level** | 全局结构感知 | 方差高、计算贵 | 多步推理 |
| **Hybrid** | 平衡两者 | 架构复杂 | 推理任务 |

## 代表性方法详解

### White-Box Token-Level

> [!example] GKD (Agarwal et al., 2024)
> 定义混合策略 $\pi_{\text{mix}}$ 插值数据集和学生生成，可配置 F-KL、R-KL 或 JSD。$\lambda=1$ 时纯 on-policy。

> [!example] DistiLLM (Ko et al., 2024)
> 提出 **Skewed KL** 解决标准 KL 的大概率不匹配时的梯度不稳定：
> - SKL: $D_{\text{KL}}(p_\mathcal{T} \| \alpha p_\mathcal{T} + (1-\alpha)p_\theta)$ — 用于教师输出
> - SRKL: $D_{\text{KL}}(p_\theta \| (1-\alpha)p_\mathcal{T} + \alpha p_\theta)$ — 用于学生输出

> [!example] Entropy-Aware OPD (Jin et al., 2026)
> 根据教师熵动态选择 divergence：
> $$\alpha_t \propto \mathcal{H}(p_\mathcal{T}(\cdot|y_{<t}))$$
> 高熵区用 F-KL 保留多样性，低熵区用 R-KL 精确模仿。

### Black-Box Methods

> [!example] GAD (Ye et al., 2025)
> 将黑盒蒸馏建模为 minimax 博弈：学生生成器 $G$ 产生 on-policy 响应，判别器 $D$ 区分学生与教师输出，通过 Bradley-Terry 偏好模型优化。

> [!example] Lion (Jiang et al., 2023)
> 三阶段对抗循环：
> 1. 模仿：学生在教师响应上微调
> 2. 判别：教师识别学生薄弱环节
> 3. 生成：教师创建针对薄弱点的更难指令

### Self-Distillation

> [!example] SPIN (Chen et al., 2024)
> 将自蒸馏建模为两玩家游戏：迭代 $t+1$ 的模型区分迭代 $t$ 的生成与人类参考。理论保证收敛到 $p_\theta = p_{\text{data}}$，但无法超越数据质量天花板。

> [!example] OPSD (Zhao et al., 2026)
> 利用 **Privileged Information (PI)** 打破自玩天花板：
> - 教师：条件于问题和正确答案 $p(\cdot|x, y^*)$
> - 学生：仅条件于问题 $p(\cdot|x)$
> 学生生成 rollout，教师在其上提供 dense token-level 监督。

> [!warning] Self-Play 的饱和问题
> 纯自玩会迅速饱和甚至退化（"Ouroboros" 问题）。突破策略：
> 1. 外部验证器（代码执行、数学验证）提供非可微 truth 信号
> 2. KD-RL 循环：交替自蒸馏与 PPO，用独立训练的奖励模型维持多样性

## 推理蒸馏的特殊考量

### Chain-of-Thought OPD

$$
\mathcal{L}_{\text{CoT-OPD}} = \mathbb{E}_{r^S \sim P_\theta(\cdot|x)} \left[ \sum_{t=1}^{|r^S|} D_{\text{KL}}(P_\theta(\cdot|x, r^S_{<t}) \| P_\mathcal{T}(\cdot|x, r^S_{<t})) \right]
$$

使用 **Reverse KL** 确保学生强化自身参数化下高概率的推理路径。

### 推理压缩

> [!example] OPSDC (Sang et al., 2026)
> 同一模型作为教师（提示"简洁"）和学生（无约束生成），通过 on-policy Reverse KL 将 CoT token 数减少 **57-59%**，同时提升准确率 9-16%。

## 关键开放问题

| 问题 | 描述 | 已有探索 |
|------|------|---------|
| **Scaling Laws** | 学生大小 $N_S$、教师大小 $N_T$、on-policy 数据量 $D_{\text{on}}$ 的最优分配 | DeepSeek-R1 显示 1.5B→7B 增益最大，14B→32B 边际递减 |
| **Uncertainty-Aware** | 区分教师的认知不确定性（应减弱蒸馏）与偶然不确定性（应保留） | Entropy-Aware, GATES |
| **Dynamic Curriculum** | 根据学生能力边界动态调整 prompt 难度 | PACED (Beta kernel weighting) |
| **Latent Space Distillation** | 绕过词汇层，在隐藏状态空间匹配 | DSKD 双空间投影器 |
| **Agent-Level OPD** | 多轮交互、工具使用、环境反馈的蒸馏 | SCoRe, OEL, Privileged Info Distillation |
| **Multimodal OPD** | 跨模态（文本→视觉、语音）蒸馏 | VOLD, Video-OPD, X-OPD |
| **KD-RL Loop** | 交替蒸馏与 RL，避免静态蒸馏饱和 | KDRL, SDFT, Li et al. 统一框架 |

## 实践决策框架

### 何时用 Off-Policy vs On-Policy

> [!comparison] 选择指南
> - **Off-Policy SFT**：通用指令遵循和对话，成本效益最佳
> - **切换到 On-Policy 的信号**：
>   1. 学生在自己生成的前缀上表现明显更差（exposure bias）
>   2. 多步推理任务中错误累积严重
>   3. 需要在特定领域超越教师（reward-guided exploration）

### 计算预算分配（"蒸馏税"规则）

```
60-70% → Off-policy 预热（SFT on teacher data）
20-30% → On-policy logit 蒸馏
10%    → Reward-guided 细化
```

### 方法选择决策树

```
教师是白盒？
├─ 是 → 学生 < 7B？
│      ├─ 是 → GKD / DistiLLM (token-level)
│      └─ 否 → MiniLLM / Hybrid (sequence-level)
└─ 否 → 有 reward model/verifier？
       ├─ 是 → RLKD / AlignDistil (reward-guided)
       └─ 否 → GAD / Lion (adversarial) 或 SPIN / OPSD (self-distill)
```

## 与 Related Work 的连接

- [[Knowledge Distillation]] — 传统 off-policy KD 的基础
- [[Exposure Bias]] — OPD 解决的核心问题，源于模仿学习
- [[f-divergence]] — 统一框架的数学基础
- [[DAgger]] — OPD 的理论动机来源
- [[RLHF]] — 与 outcome-based OPD 的边界正在融合
- [[DeepSeek-R1]] — 大规模 off-policy 蒸馏的工业实践

## 延伸资源

**原始论文**：
- arXiv: [2604.00626](https://arxiv.org/abs/2604.00626)
- 本地 clipping: [[Clippings/Paper/2604.00626/2604.00626.md]]

**关键方法论文**：
- GKD (Agarwal et al., 2024) — on-policy distillation 奠基
- MiniLLM (Gu et al., 2024) — sequence-level Reverse KL
- DistiLLM (Ko et al., 2024) — skewed KL 稳定优化
- ToDi (Jung et al., 2025) — 自适应 divergence
- SPIN (Chen et al., 2024) — self-play 蒸馏
- OPSD (Zhao et al., 2026) — privileged information 自蒸馏
