---
type: concept
description: 通过让 agent 与自身或历史版本对弈来提升能力的训练范式，在 LLM 对齐中演化为自我批评、自我评估等多种形式
aliases:
  - 自博弈
  - Self-Play Training
prerequisites:
  - "[[RLHF]]"
  - "[[Reward Model]]"
tags:
  - post-training
  - reinforcement-learning
  - alignment
created: 2026-02-26
updated: 2026-02-26
---

# Self-Play

自博弈（Self-Play）的核心思想是：agent 不依赖外部对手或外部标注者，而是与自身（或自身的历史版本）交互来产生训练信号。这个思想从博弈论和棋类 AI 发展而来，在 LLM 时代演化出了多种变体——模型自我批评、自我评估、自我对弈——本质上都是用模型自身的能力替代人类反馈，实现可扩展的自我改进。

## 动机

> [!intuition] 为什么需要 Self-Play？
> [[RLHF]] 的瓶颈在于人类反馈：标注昂贵、速度慢、难以扩展到复杂任务。如果模型自己能充当"对手"或"评委"，就能突破这个瓶颈——训练信号从有限的人类标注变成几乎无限的自我交互。

具体来说，Self-Play 解决的问题是：

1. **标注瓶颈**：人类偏好标注成本高、速度慢，无法覆盖所有场景
2. **能力天花板**：如果训练信号完全来自人类，模型能力受限于人类标注者的水平
3. **对抗性不足**：静态数据集无法持续挑战不断进步的模型

## 经典案例：AlphaGo / AlphaZero

Self-Play 最经典的成功案例来自 DeepMind 的围棋 AI。理解这个案例有助于把握 Self-Play 的本质。

AlphaGo（2016）最初依赖人类棋谱进行监督学习，然后通过 Self-Play 强化学习超越人类水平。AlphaZero（2017）更进一步——完全抛弃人类棋谱，从随机初始化开始，纯粹通过自我对弈学习。

> [!intuition] 为什么 Self-Play 在棋类中如此有效？
> 棋类游戏有两个关键特性：(1) 结果可验证——赢/输/平是客观的；(2) 对手即自己——不需要外部对手就能产生有意义的训练信号。当模型变强时，对手（自己的历史版本）也在变强，形成持续的学习压力。

这揭示了 Self-Play 的核心机制：**自动课程学习**（automatic curriculum）。对手的难度随着自身能力自然提升，不需要人工设计训练课程。

## 在 LLM 中的应用形式

LLM 不像棋类有明确的胜负判定，因此 Self-Play 在语言模型中的应用需要创造性地重新定义"对弈"和"胜负"。

### 1. Constitutional AI 的 Self-Revision

[[Constitutional AI]] 是 Self-Play 思想在对齐中的早期应用。模型扮演两个角色：

- **生成者**：产生初始回答
- **批评者**：根据预定义的宪法原则（constitutional principles）批评自己的回答，然后修正

这不是传统意义上的"对弈"，而是**自我博弈的变体**——模型与自己的"不良倾向"对抗。批评-修正循环产生改进后的数据，用于后续的 [[RLAIF]] 训练。

### 2. SPIN（Self-Play Fine-Tuning）

SPIN（Zhu et al., 2024）将 Self-Play 形式化为一个**两人博弈**：

- **主玩家**（当前模型）：尝试生成与人类回答无法区分的文本
- **对手**（判别器，也是模型自身）：尝试区分模型生成的文本和人类写的文本

> [!math] SPIN 的博弈目标
> SPIN 的目标函数类似 GAN 的 minimax 博弈：
> $$\max_\theta \mathbb{E}_{x \sim \mathcal{D}} \left[ \log p_\theta(y_{\text{human}} | x) - \log p_\theta(y_{\text{model}} | x) \right]$$
> 当模型生成的文本与人类文本不可区分时，博弈达到纳什均衡，训练自然收敛。

SPIN 的巧妙之处在于：它不需要 [[Reward Model]]，也不需要偏好标注——只需要人类写的 SFT 数据。模型通过与自己的上一轮版本对弈来逐步逼近人类分布。

### 3. Self-Rewarding Language Models

Self-Rewarding（Yuan et al., 2024）让模型同时扮演**生成者**和**评委**：

1. 模型生成多个候选回答
2. 模型用 LLM-as-a-Judge 的方式给自己的回答打分
3. 用这些自生成的偏好数据做 [[DPO]] 训练
4. 训练后的模型成为更好的评委，进入下一轮迭代

> [!intuition] 自我奖励的递归改进
> 传统 RLHF 中，[[Reward Model]] 是固定的——它不会随着策略模型的进步而进步。Self-Rewarding 打破了这个限制：评委和选手是同一个模型，选手变强的同时评委也在变强，形成正向螺旋。

## 与 RLAIF 的关系

Self-Play 和 [[RLAIF]] 有密切的概念联系：

> [!comparison] Self-Play vs RLAIF
> - **RLAIF** 是一个宽泛的框架：用 AI 反馈替代人类反馈进行 RL 训练
> - **Self-Play** 是 RLAIF 的一种实现思路：当"提供 AI 反馈的模型"就是"被训练的模型自身"时，RLAIF 就变成了 Self-Play
>
> 换句话说：Self-Play $\subset$ RLAIF $\subset$ 对齐方法。不是所有 RLAIF 都是 Self-Play（可以用外部模型提供反馈），但 Self-Play 式的对齐天然属于 RLAIF。

[[Constitutional AI]] 是这种关系的典型例子——它既是 RLAIF（用 AI 反馈训练），也是 Self-Play（反馈来自模型自身）。

## 优势

1. **可扩展性**：摆脱人类标注瓶颈，训练信号几乎无限
2. **自动课程**：难度随模型能力自然提升，不需要人工设计
3. **超越人类水平的潜力**：AlphaZero 证明了 Self-Play 可以超越人类专家（在可验证任务中）
4. **成本效率**：大幅降低对齐训练的人力成本

## 局限性与风险

> [!warning] Self-Play 的核心风险
> **Self-Reinforcing Bias（自我强化偏见）**：模型的错误会被自己的反馈放大。如果模型对某类问题有系统性偏见，Self-Play 会强化而非纠正这个偏见——因为评委和选手共享同样的盲点。

**Mode Collapse（模式坍缩）**：模型可能收敛到一个狭窄的"自洽"策略——回答看起来合理但缺乏多样性。这在 [[Data Synthesis]] 中也是核心风险。

**能力天花板**：在不可验证的开放任务中（不像围棋有客观胜负），Self-Play 的改进上限受限于模型自身的判断能力。模型无法教会自己不知道的东西。

**评估困难**：当训练信号来自模型自身时，如何判断模型是真的在进步还是在"自欺欺人"？这需要独立的外部评估机制。

> [!interview] 面试要点
> **Q: Self-Play 在 LLM 中和在棋类 AI 中有什么本质区别？**
> A: 棋类有客观的胜负判定，Self-Play 的训练信号是可靠的。LLM 的任务（对话、写作）没有客观标准，Self-Play 的训练信号依赖模型自身的判断，因此存在 self-reinforcing bias 的风险。这也是为什么 LLM 的 Self-Play 通常需要额外的约束（如 Constitutional AI 的宪法原则）或外部验证。
>
> **Q: Self-Play 和 RLHF 是什么关系？**
> A: 不是替代关系，而是互补。RLHF 用人类反馈提供可靠但昂贵的信号，Self-Play 用自我反馈提供廉价但可能有偏的信号。实践中常结合使用——先用 RLHF 建立基础对齐，再用 Self-Play 扩展和精炼。

## 延伸阅读

**原始论文**：
- SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (Zhu et al., 2024)
- Self-Rewarding Language Models (Yuan et al., 2024)

**后续发展**：
- [[SPAG]] — Self-Play 在多轮对话中的应用
- [[Debate]] — 两个模型互相辩论，人类只需判断最终结果
