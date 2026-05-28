---
type: paper
description: Qwen-Scope 发布 Qwen 系列的 SAE 套件，并展示 sparse features 如何作为 steering、评测、数据构造和 post-training 的统一表示接口
aliases:
  - Qwen-Scope
  - Qwen Scope
  - Qwen-Scope 技术报告
  - SAE suite for Qwen
prerequisites:
  - "[[Sparse Autoencoder]]"
  - "[[Mechanistic Interpretability]]"
tags:
  - interpretability
  - sparse-autoencoder
  - qwen
  - post-training
  - evaluation
created: 2026-05-01
updated: 2026-05-01
---

# Qwen-Scope (2026)

Qwen-Scope 是 Qwen Team 发布的一套面向 Qwen3 / Qwen3.5 的 [[Sparse Autoencoder|sparse autoencoder]]（SAE）资源和应用报告。它的核心贡献是把模型内部 activation 分解成稀疏、可解释的 feature，再把这些 feature 用作模型开发接口：可以做 inference-time steering、评测集分析、数据分类、数据合成，以及 [[MOC - Post-training|post-training]] 中的 [[SFT]] 和 [[DAPO]] 改造。

> [!paper] 论文出处
> Qwen Team, *Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models*, 2026. 原始 clipping: [[Clippings/Paper/Qwen_Scope/Qwen_Scope|Qwen-Scope clipping]]。

## 为什么 SAE features 值得作为开发工具

大模型的行为通常只能从输入输出层面观察：某个 benchmark 得分升高、某类坏 case 变少、某个 safety 指标改善。Qwen-Scope 的切入点是把观察位置前移到 representation：当模型处理一个样本时，SAE 把 residual stream 的高维 activation 映射到少数激活的 latent features，每个 feature 可以近似看作模型内部状态中的一个方向。

> [!intuition] 表示层接口
> 如果 raw hidden state 是一团难解释的高维向量，SAE feature 就像给这团向量建立了一组可命名的坐标。一个样本激活了哪些坐标，可以作为它触发了哪些内部模式的指纹；调高或压低某个坐标，可以作为直接干预模型状态的旋钮。

这使 [[Mechanistic Interpretability]] 从事后解释延伸到工程工作流：开发者可以用 feature 找到坏行为的内部线索，用 feature coverage 估计评测集覆盖范围，用 feature descriptions 合成训练样本，也可以在训练目标中加入 feature-level 约束。

## Release Scope 与训练设置

Qwen-Scope 发布 14 组 SAE weights，覆盖 7 个 Qwen3 / Qwen3.5 backbone，包括 dense 和 mixture-of-experts（MoE）模型。每个 backbone 的每一层都训练单独 SAE，因此一个 released SAE 对应“某个模型、某一层 residual stream”的 feature dictionary。

核心训练设置：

- activation 来源：对应 backbone 的 residual-stream activations。
- latent 结构：overcomplete sparse representation。
- 稀疏化：Top-$k$ activation rule，只保留最大的 $k$ 个 latent activations，发布版本包含 Top-$k=50$ 或 $100$。
- dense backbone：SAE width 通常是 hidden size 的 $16\times$。
- MoE backbone：额外发布更宽的 SAE，最高到 hidden size 的 $64\times$，用于捕捉更细粒度结构。
- 稳定性处理：使用 auxiliary loss 降低 dead features，并过滤极大 $L_2$ norm activation outliers。

到这里，Qwen-Scope 建立了一个前提：同一套 layer-wise SAE features 可以在多个下游任务中复用。后续应用的共同模式是先把文本、评测样本或训练样本映射到 feature space，再在 feature space 中做选择、干预或验证。

## 应用一：Inference-Time Steering

Qwen-Scope 使用 [[Activation Steering]] 的基本假设：语言、风格、偏好、重复等行为可以对应到 representation space 中的方向。SAE 的作用是把这些方向变得更稀疏、更容易定位。

给定 hidden state $\mathbf{h}$、SAE feature direction $\mathbf{d}$ 和 steering coefficient $\alpha$，feature-level intervention 写作：

$$
\mathbf{h}' \leftarrow \mathbf{h} + \alpha \mathbf{d}
$$

$\alpha > 0$ 表示放大该 feature，$\alpha < 0$ 表示压低该 feature。模型随后使用 $\mathbf{h}'$ 继续 forward pass，因此生成行为会沿着这个 feature 对应的内部模式移动。

Qwen-Scope 展示了两类 [[Feature Steering]] 用法：

- 诊断与修复：英文 prompt 触发中文混入时，排序 SAE activations 可以找到高度激活的 Chinese-language feature；在生成时压低该 feature 可以减少 code-switching。
- 可控生成：现代中文续写任务中，放大 classical-Chinese feature 可以把输出推向文言风格，同时保留 prompt 的语义方向。

这个应用说明 SAE feature 可以充当“可解释控制点”。它的优势在于无需改模型权重；代价是每次推理都需要 intervention，并且 steering strength、层位置、feature 选择都会影响副作用。

## 应用二：Representation-Level Evaluation

评测集设计有两个常见问题：一个 benchmark 内部是否高度重复，两个 benchmarks 是否测了相同能力。直接做法需要跑多个模型和大量 subsets，成本随模型数和样本数增长。Qwen-Scope 提出用 SAE feature footprint 作为 [[Representation-Level Evaluation|representation-level evaluation]] 信号。

对 benchmark $\mathcal{D}=\{x_i\}_{i=1}^N$，先定义每个样本激活的 feature set：

$$
F(x_i)=\{j:z_j(x_i)>0\}
$$

整个 benchmark 的 feature footprint 是：

$$
F(\mathcal{D})=\bigcup_{i=1}^N F(x_i)
$$

如果随机加入少量样本后 feature coverage 很快饱和，说明很多样本触发相同内部模式，benchmark 在能力空间中存在较高 [[Benchmark Redundancy]]。论文用 feature coverage curve 的 AUC 加上 feature growth-rate correction 构造冗余指标，并在 17 个常用 benchmarks 上观察到它与 performance-based redundancy 有较高相关性，Spearman 相关约为 $0.85$。

跨 benchmark 比较则使用 feature overlap：

$$
\operatorname{overlap}(\mathcal{D}_1,\mathcal{D}_2)=
\frac{|F(\mathcal{D}_1)\cap F(\mathcal{D}_2)|}{|F(\mathcal{D}_1)|}
$$

这个指标是非对称的，因此能表达“$\mathcal{D}_1$ 的能力范围有多少被 $\mathcal{D}_2$ 覆盖”。例如 GSM8K 的许多 features 被 MATH 覆盖，反向覆盖比例低，符合“基础数学被竞赛数学包含一部分”的直觉。用于 evaluation suite design 时，低 overlap 的 benchmark 提供互补覆盖，高 overlap 的 benchmark 可以成为合并或抽样候选。

> [!warning] 使用边界
> Feature redundancy 适合辅助评测集裁剪和覆盖分析。高冗余并不等于低质量；在降低 variance 或覆盖某个固定领域时，冗余样本仍然有价值。这个指标主要服务于“用更低成本保持模型排序信号”的场景。

## 应用三：数据分类与数据合成

### Toxicity Classification

Qwen-Scope 把 multilingual toxicity classification 作为一个测试：固定 SAE 后，是否可以用少量 feature 直接构造分类器。流程很简单：

1. 在 toxic / clean selection split 上统计每个 SAE feature 的 firing frequency 差异。
2. 选择 toxic-biased features。
3. 在 held-out data 上使用 OR-rule：只要选中 feature 在任意 token position 激活，就判为 toxic。

这个分类器没有额外 classifier head，也没有额外 gradient fitting。实验中，英文 toxicity detection 的 held-out F1 可超过 $0.90$；跨语言结果显示，toxicity features 在中间层存在共享结构，英文发现的 features 能迁移到多种语言，迁移强度会随语言距离变化。

这里的教学重点是：SAE feature 可以直接从“解释对象”变成“预测变量”。每个 positive prediction 都能追溯到 feature、layer 和 token position，因此可审计性强于普通 dense classifier head。

### Feature-Driven Safety Data Synthesis

数据合成部分把问题从“还要采哪些 prompt”转成“哪些 safety-relevant internal directions 覆盖不足”。Qwen-Scope 先用 seed corpus 测量 target feature coverage，再用 feature descriptions 生成 prompt-completion pairs，最后用 representation-level verification 过滤样本。

流程可以概括为：

- target discovery：找出 safety-relevant 且当前 seed corpus 未覆盖或弱覆盖的 SAE features。
- prompt construction：根据 feature description 生成 vanilla prompt，并构造 adversarial variants。
- response construction：根据 harmful / benign label 生成拒答或正常 helpful response。
- verification：保留真正激活目标 feature 的样本。

论文报告在固定预算下，feature-driven synthesis 的 target feature coverage 达到 $99.74\%$，明显高于 natural sampling 和 random safety-related synthesis。下游 [[SFT]] 中，用 4k real safety examples 加 4k feature-driven synthetic examples，在 safety accuracy 上接近 120k safety-only 设置，并且优于同预算 random synthetic data。

这说明 feature coverage 可以作为 data-centric workflow 的优先级信号：合成数据的价值来自覆盖缺失内部方向，而只看文本表面多样性会漏掉 representation space 的长尾。

## 应用四：Post-Training 中的 SFT 与 RL

Qwen-Scope 把 SAE features 接入 [[MOC - Post-training|post-training]] 的方式分成两类：在 [[SFT]] 中加入辅助 feature loss，在 [[RLHF]] / online RL 类流程中改变 rollout distribution。

### SAE-Guided SFT：降低意外 Code-Switching

意外 code-switching 指模型在不该切换语言时突然生成目标外语言。Qwen-Scope 先用 SAE 找到 language-specific features，并观察到目标语言 feature 的 pre-activation 会在切换前逐步升高。进一步地，directional ablation 目标语言 feature 可以降低 code-switching ratio。

在训练中，Sparse Autoencoder-guided Supervised Fine-Tuning（SASFT）把这个观察转化为 auxiliary loss：标准 cross-entropy 继续学习目标 response，同时 $L_{\text{reduce}}$ 惩罚非目标语言数据中目标语言 features 的过高 pre-activation。

$$
L_{\text{training}} = L_{\text{cross-entropy}} + \lambda L_{\text{reduce}}
$$

实验显示，SASFT 在多种模型和中、俄、韩三个 target languages 上通常能把 code-switching ratio 降低超过 $50\%$，同时保持多语言 benchmark 能力基本稳定。这里的关键是把 feature-level 诊断变成 persistent parameter update，使修复效果进入模型权重。

### SAE-Guided DAPO：Rare Negative Augmentation

RL 部分聚焦 endless repetition 这种低频坏行为。标准 online RL 很少在 rollout 中采到这类失败样本，因此 correction signal 很弱。Qwen-Scope 的做法是先定位 repetition-related features，再在 [[DAPO]] rollout group 中额外生成一个 SAE-steered negative sample。

具体地，每组 rollout 中正常采样 $G-1$ 个输出，再用 repetition feature steering 采样一个倾向重复的输出 $o_G$。这个输出和正常 rollouts 一起进入 reward 与 advantage 计算，让 policy 看到原本稀有的负例。

这个设计比“用 SAE 生成高质量 positive rollout”更稳妥，因为多步推理的正确性很难靠 steering 直接制造，而诱发坏行为更容易，且坏样本的 fluency 副作用主要用于反向学习。实验中，SAE-guided rare negative augmentation 比 vanilla DAPO 更快、更强地降低 repetition ratio；通用 benchmark 结果整体可竞争，但提升不均匀。

> [!warning] Repetition feature 的语义边界
> 论文发现 repetition-related features 也会在正常重复场景中激活，例如复述用户问题或重现选择题选项。因此直接在训练中持续压低 repetition feature 可能伤害正常能力。Qwen-Scope 选择把这些 features 用于 rare negative synthesis，而将抑制目标交给 RL 信号学习。

## 这篇技术报告的核心贡献

1. 发布 Qwen 系列的 layer-wise SAE suite：14 组 SAE weights，覆盖 7 个 Qwen3 / Qwen3.5 dense 与 MoE backbones。
2. 展示 SAE features 可以作为 inference-time control handles，用于语言、风格和坏行为修复。
3. 提出用 feature coverage / overlap 分析 benchmark redundancy、capability coverage 和 evaluation suite overlap。
4. 展示 SAE features 可以直接支持透明的 toxicity classification，并具备一定跨语言迁移能力。
5. 提出 feature-driven safety data synthesis，用 representation-level verification 提升 safety data 覆盖效率。
6. 将 SAE signals 接入 [[SFT]] 和 [[DAPO]]，分别处理 code-switching 和 endless repetition。

## 边界与后续问题

Qwen-Scope 的价值在于把 SAE feature 从解释产物变成开发接口；这个接口的可靠性仍依赖 feature quality、feature interpretation、目标层选择和 causal validation。

需要继续追问的边界包括：

- SAE feature 是否稳定跨模型、跨训练阶段、跨语言迁移。
- feature description 由 judge model 生成或筛选时会引入多少语义偏差。
- feature coverage 与真实能力覆盖之间的关系在哪些任务上会失效。
- steering coefficient 与层位置如何系统选择，怎样避免意外副作用。
- safety feature synthesis 是否可能被误用来增强有害能力或绕过防护。
- post-training 后 feature dictionary 是否仍然对齐原模型的内部结构。

论文提出的未来方向包括 reasoning-model interpretability、internal monitoring and auditing、model diffing、interpretability-driven control and training，以及 data-centric interpretability。对知识库来说，Qwen-Scope 可以作为连接 [[Sparse Autoencoder]]、[[Feature Steering]]、[[Representation-Level Evaluation]] 和 [[MOC - Post-training|post-training]] 工程实践的入口笔记。
