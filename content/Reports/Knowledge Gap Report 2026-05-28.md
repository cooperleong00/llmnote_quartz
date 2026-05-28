---
updated: 2026-05-29T00:07
---
# Knowledge Gap Report 2026-05-28

本报告基于当前 worktree 的自动扫描生成。扫描范围覆盖根目录主笔记与 `interview/`，排除 `Clippings/`、`AGENTS.md`、`CLAUDE.md`、`Index.md`、`Obsidian CLI Guide.md` 与 `Reports/`。`Clippings/` 只作为链接解析目标和素材来源参与判断。

## 扫描证据

- 主笔记：193 篇
- Clippings：112 篇
- 主笔记 wikilink 出现次数：3538
- 无法解析的唯一链接目标：163 个，出现 273 次
- 缺失 heading 链接：1 个
- 孤立主笔记：12 篇
- 零出链主笔记：1 篇
- 低出链主笔记：12 篇
- MOC：7 篇
- tag：116 个
- 显式标记空缺命中：29 处
- 禁用句式命中：48 处
- 未被主笔记引用的 clippings：21 篇

## 最高优先级空缺

这些目标被至少 3 篇主笔记引用，优先处理会减少最多断链，并补齐多个知识路径的前置节点。

| 空缺 | 引用篇数 | 出现次数 | 主要来源 | 建议 |
|---|---:|---:|---|---|
| [[Hybrid Retrieval]] | 5 | 8 | [[BM25]], [[Embedding]], [[Multi-Query]], [[Query Expansion]], 另 1 篇 | 合并为混合检索主题，统一 alias |
| [[ReAct]] | 5 | 5 | [[AgentFold (2025)]], [[Agentic RL]], [[Context-Folding (2025)]], [[DeepAgent (2025)]], 另 1 篇 | 创建 concept/method 笔记 |
| [[RLVR]] | 5 | 9 | [[Near-Future Policy Optimization]], [[Nemotron 3 Super (2026)]], [[RLSD]], [[ROLL (2025)]], 另 1 篇 | 创建 concept/method 笔记 |
| [[Curriculum Learning]] | 4 | 4 | [[DAVINCI-LLM (2026)]], [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[Mid-Training]] | 创建 concept/method 笔记 |
| [[Mamba]] | 4 | 5 | [[DeltaNet]], [[Linear Attention]], [[Nemotron 3 Super (2026)]], [[Test-time Regression (2025)]] | 创建 concept/method 笔记 |
| [[Post-training]] | 4 | 6 | [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[Mid-Training]], [[PRISM (2026)]] | 创建 overview 或给 MOC 增加 alias |
| [[Pretraining]] | 4 | 6 | [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[Mid-Training]], [[PRISM (2026)]] | 创建 overview 或给 MOC 增加 alias |
| [[Information Retrieval]] | 3 | 4 | [[BM25]], [[Dense Retrieval]], [[Sparse Retrieval]] | 创建 concept/method 笔记 |
| [[Knowledge Distillation]] | 3 | 4 | [[A Survey of On-Policy Distillation for Large Language Models (2026)]], [[MOPD]], [[On-Policy Distillation]] | 创建 concept/method 笔记 |
| [[Learning Rate Schedule]] | 3 | 3 | [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[Mid-Training]] | 创建 concept/method 笔记 |
| [[Muon]] | 3 | 6 | [[GLM-5 (2026)]], [[Kimi K2 (2025)]], [[Open-Source LLM Landscape (2025-2026)]] | 创建优化器笔记，和 Muon Optimizer 统一 |
| [[Tool Use]] | 3 | 3 | [[Agentic RL]], [[Chain-of-Thought]], [[Context-Folding (2025)]] | 创建 concept/method 笔记 |

## 结构性空缺总表

### 概念/方法缺口

共 154 个唯一目标。

| 目标 | 引用篇数 | 出现次数 | 来源 |
|---|---:|---:|---|
| [[Hybrid Retrieval]] | 5 | 8 | [[BM25]], [[Embedding]], [[Multi-Query]], [[Query Expansion]], [[RAG]] |
| [[ReAct]] | 5 | 5 | [[AgentFold (2025)]], [[Agentic RL]], [[Context-Folding (2025)]], [[DeepAgent (2025)]], [[Towards Agentic RAG with Deep Reasoning (2025)]] |
| [[RLVR]] | 5 | 9 | [[Near-Future Policy Optimization]], [[Nemotron 3 Super (2026)]], [[RLSD]], [[ROLL (2025)]], [[SFT-then-RL]] |
| [[Curriculum Learning]] | 4 | 4 | [[DAVINCI-LLM (2026)]], [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[Mid-Training]] |
| [[Mamba]] | 4 | 5 | [[DeltaNet]], [[Linear Attention]], [[Nemotron 3 Super (2026)]], [[Test-time Regression (2025)]] |
| [[Post-training]] | 4 | 6 | [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[Mid-Training]], [[PRISM (2026)]] |
| [[Pretraining]] | 4 | 6 | [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[Mid-Training]], [[PRISM (2026)]] |
| [[Information Retrieval]] | 3 | 4 | [[BM25]], [[Dense Retrieval]], [[Sparse Retrieval]] |
| [[Knowledge Distillation]] | 3 | 4 | [[A Survey of On-Policy Distillation for Large Language Models (2026)]], [[MOPD]], [[On-Policy Distillation]] |
| [[Learning Rate Schedule]] | 3 | 3 | [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[Mid-Training]] |
| [[Muon]] | 3 | 6 | [[GLM-5 (2026)]], [[Kimi K2 (2025)]], [[Open-Source LLM Landscape (2025-2026)]] |
| [[Tool Use]] | 3 | 3 | [[Agentic RL]], [[Chain-of-Thought]], [[Context-Folding (2025)]] |
| [[Agentic RAG]] | 2 | 3 | [[GraphRAG]], [[RAG]] |
| [[Annealing]] | 2 | 2 | [[LLM Mid-Training Survey (2024)]], [[Mid-Training]] |
| [[DeepSeek-R1]] | 2 | 2 | [[A Survey of On-Policy Distillation for Large Language Models (2026)]], [[LongCat-Flash-Thinking (2025)]] |
| [[DualPipe]] | 2 | 2 | [[3D Parallelism]], [[Expert Parallelism]] |
| [[Fast Weight Programming]] | 2 | 2 | [[DeltaNet]], [[Test-time Regression (2025)]] |
| [[Gated DeltaNet]] | 2 | 2 | [[Linear Attention]], [[Test-time Regression (2025)]] |
| [[GLA]] | 2 | 3 | [[DeltaNet]], [[Test-time Regression (2025)]] |
| [[Gradient Noise Scale]] | 2 | 2 | [[Mid-Training Survey (2024)]], [[Mid-Training]] |
| [[Hybrid Search]] | 2 | 4 | [[Dense Retrieval]], [[Sparse Retrieval]] |
| [[In-Context Learning]] | 2 | 4 | [[CL-Bench]], [[Citation Generation]] |
| [[Long-Context Modeling]] | 2 | 3 | [[CL-Bench]], [[Context-Folding (2025)]] |
| [[Prioritized Experience Replay]] | 2 | 3 | [[DQN]], [[Experience Replay]] |
| [[Probability Distribution]] | 2 | 2 | [[Cross-Entropy Loss]], [[KL Divergence]] |
| [[Probability Theory]] | 2 | 2 | [[Importance Sampling]], [[Policy Gradient]] |
| [[SGD]] | 2 | 3 | [[Adam]], [[AdamW]] |
| [[Softmax]] | 2 | 2 | [[Cross-Entropy Loss]], [[Entropy]] |
| [[TF-IDF]] | 2 | 3 | [[BM25]], [[Sparse Retrieval]] |
| [[Variance Reduction]] | 2 | 2 | [[A3C]], [[Importance Sampling]] |
| [[Activation Steering]] | 1 | 1 | [[Qwen-Scope (2026)]] |
| [[Adafactor]] | 1 | 1 | [[Adam]] |
| [[Adapter]] | 1 | 1 | [[LoRA]] |
| [[Adaptive RAG]] | 1 | 1 | [[RAG]] |
| [[Agent]] | 1 | 1 | [[Chain-of-Thought]] |
| [[Agent Training]] | 1 | 1 | [[Rollout Routing Replay]] |
| [[AI Safety]] | 1 | 1 | [[Constitutional AI]] |
| [[Alignment Tax]] | 1 | 1 | [[InstructGPT]] |
| [[Alpaca]] | 1 | 1 | [[SFT]] |
| [[Anticipatory Routing]] | 1 | 1 | [[DeepSeek-V4 (2026)]] |
| [[AWQ]] | 1 | 1 | [[Quantization]] |
| [[AWR]] | 1 | 1 | [[RLTF (2025)]] |
| [[Backpropagation]] | 1 | 1 | [[Data Parallelism]] |
| [[Benchmark Redundancy]] | 1 | 1 | [[Qwen-Scope (2026)]] |
| [[BGE]] | 1 | 1 | [[Embedding]] |
| [[Catastrophic Forgetting]] | 1 | 1 | [[SFT]] |
| [[Chat Template]] | 1 | 1 | [[SFT]] |
| [[Compressed Sparse Attention]] | 1 | 1 | [[DeepSeek-V4 (2026)]] |
| [[Covariate Shift]] | 1 | 1 | [[DAgger]] |
| [[CursorBench]] | 1 | 1 | [[Composer 2 (2025)]] |
| [[Data Decontamination]] | 1 | 1 | [[LLM Mid-Training Survey (2024)]] |
| [[DCLM]] | 1 | 1 | [[The Synthetic Data Playbook (2026)]] |
| [[Debate]] | 1 | 1 | [[Self-Play]] |
| [[Decoupled RL Training]] | 1 | 1 | [[JACKPOT]] |
| [[Double DQN]] | 1 | 1 | [[DQN]] |
| [[Dueling DQN]] | 1 | 1 | [[DQN]] |
| [[E5]] | 1 | 1 | [[Embedding]] |
| [[Elo Rating]] | 1 | 2 | [[Bradley-Terry Model]] |
| [[Evol-Instruct]] | 1 | 1 | [[Data Synthesis]] |
| [[Expected Value]] | 1 | 1 | [[Importance Sampling]] |
| [[Expert Choice]] | 1 | 1 | [[Loss-Free Load Balancing]] |
| [[Exposure Bias]] | 1 | 2 | [[A Survey of On-Policy Distillation for Large Language Models (2026)]] |
| [[f-divergence]] | 1 | 2 | [[A Survey of On-Policy Distillation for Large Language Models (2026)]] |
| [[Feature Steering]] | 1 | 2 | [[Qwen-Scope (2026)]] |
| [[Feedback Descent]] | 1 | 1 | [[RLTF (2025)]] |
| [[Few-shot Learning]] | 1 | 2 | [[Chain-of-Thought]] |
| [[FineWeb]] | 1 | 1 | [[The Synthetic Data Playbook (2026)]] |
| [[Flare]] | 1 | 1 | [[RAG Survey (2023)]] |
| [[Forking Tokens]] | 1 | 1 | [[Credit Assignment]] |
| [[Full Fine-tuning]] | 1 | 1 | [[LoRA]] |
| [[Function Calling]] | 1 | 1 | [[Agentic RL]] |
| [[Generative Reward Model]] | 1 | 2 | [[Kimi K2.5 (2026)]] |
| [[GPTQ]] | 1 | 1 | [[Quantization]] |
| [[Gradient Descent]] | 1 | 1 | [[Policy Gradient]] |
| [[Grounding]] | 1 | 1 | [[Citation Generation]] |
| [[Hallucination]] | 1 | 1 | [[Citation Generation]] |
| [[Heavily Compressed Attention]] | 1 | 1 | [[DeepSeek-V4 (2026)]] |
| [[Hindsight Experience Replay]] | 1 | 1 | [[Experience Replay]] |
| [[HippoRAG]] | 1 | 1 | [[GraphRAG]] |
| [[HotpotQA]] | 1 | 1 | [[MEM1 (2025)]] |
| [[HybridFlow]] | 1 | 2 | [[ROLL (2025)]] |
| [[Information Bottleneck]] | 1 | 1 | [[Mid-Training]] |
| [[JS Divergence]] | 1 | 1 | [[KL Divergence]] |
| [[Kernel Regression]] | 1 | 1 | [[Test-time Regression (2025)]] |
| [[Kimi K1.5]] | 1 | 1 | [[Kimi K2 (2025)]] |
| [[Knowledge Graph]] | 1 | 1 | [[GraphRAG]] |
| [[LAMB]] | 1 | 1 | [[Adam]] |
| [[LatentMoE]] | 1 | 1 | [[Nemotron 3 Super (2026)]] |
| [[Learning Rate Scheduling]] | 1 | 1 | [[AdamW]] |
| [[Leiden Algorithm]] | 1 | 1 | [[GraphRAG]] |
| [[LLM]] | 1 | 1 | [[RAG Survey (2023)]] |
| [[LLM-as-a-Judge]] | 1 | 1 | [[GraphRAG]] |
| [[LLM-as-Judge]] | 1 | 1 | [[Data Synthesis]] |
| [[Longhorn]] | 1 | 2 | [[Test-time Regression (2025)]] |
| [[LPO]] | 1 | 1 | [[Open-Source LLM Landscape (2025-2026)]] |
| [[Manifold-Constrained Hyper-Connections]] | 1 | 2 | [[DeepSeek-V4 (2026)]] |
| [[Map-Reduce]] | 1 | 1 | [[GraphRAG]] |
| [[MCP]] | 1 | 1 | [[ASTRA]] |
| [[MCTS]] | 1 | 1 | [[Towards Agentic RAG with Deep Reasoning (2025)]] |
| [[Mechanistic Interpretability]] | 1 | 2 | [[Qwen-Scope (2026)]] |
| [[Mixed-Policy Methods]] | 1 | 3 | [[SFT-then-RL]] |
| [[Mode Connectivity]] | 1 | 1 | [[Model Merging]] |
| [[Momentum]] | 1 | 3 | [[Adam]] |
| [[Multi-Query Retrieval]] | 1 | 1 | [[Query Expansion]] |
| [[Multi-task Learning]] | 1 | 1 | [[Model Merging]] |
| [[Muon Optimizer]] | 1 | 2 | [[DeepSeek-V4 (2026)]] |
| [[Nadaraya-Watson Estimator]] | 1 | 1 | [[Test-time Regression (2025)]] |
| [[Naive RAG]] | 1 | 1 | [[GraphRAG]] |
| [[Nemotron 3 Nano]] | 1 | 1 | [[Nemotron 3 Super (2026)]] |
| [[OLMo]] | 1 | 1 | [[DAVINCI-LLM (2026)]] |
| [[OpenRLHF]] | 1 | 1 | [[ROLL (2025)]] |
| [[PEFT]] | 1 | 1 | [[LoRA]] |
| [[Perplexity]] | 1 | 1 | [[Cross-Entropy Loss]] |
| [[PivotRL]] | 1 | 1 | [[Nemotron 3 Super (2026)]] |
| [[Plackett-Luce Model]] | 1 | 2 | [[Bradley-Terry Model]] |
| [[Plan-and-Solve]] | 1 | 1 | [[DeepAgent (2025)]] |
| [[Prefix Tuning]] | 1 | 1 | [[LoRA]] |
| [[PreNorm]] | 1 | 1 | [[Attention Residuals]] |
| [[Probability]] | 1 | 1 | [[Markov Decision Process]] |
| [[Prompt Tuning]] | 1 | 1 | [[LoRA]] |
| [[QLoRA]] | 1 | 2 | [[Quantization]] |
| [[Query-Focused Summarization]] | 1 | 1 | [[GraphRAG]] |
| [[REALM]] | 1 | 1 | [[RAG Survey (2023)]] |
| [[Reasoning]] | 1 | 1 | [[Towards Agentic RAG with Deep Reasoning (2025)]] |
| [[Recursive Least Squares]] | 1 | 1 | [[Test-time Regression (2025)]] |
| [[Red Teaming]] | 1 | 1 | [[Constitutional AI]] |
| [[Representation-Level Evaluation]] | 1 | 2 | [[Qwen-Scope (2026)]] |
| [[RetNet]] | 1 | 1 | [[Linear Attention]] |
| [[Retrieval]] | 1 | 1 | [[RAG Survey (2023)]] |
| [[Reward Overoptimization]] | 1 | 1 | [[Reward Hacking]] |
| [[RL Infrastructure]] | 1 | 1 | [[interview/rl_infra|rl_infra]] |
| [[RMSNorm]] | 1 | 3 | [[Attention Sink]] |
| [[RMSprop]] | 1 | 3 | [[Adam]] |
| [[Scalable Oversight]] | 1 | 1 | [[Reward Hacking]] |
| [[Scaling Law]] | 1 | 1 | [[DAVINCI-LLM (2026)]] |
| [[Scaling Laws]] | 1 | 2 | [[The Art of Scaling RL Compute for LLMs (2025)]] |
| [[Self-Consistency]] | 1 | 1 | [[Chain-of-Thought]] |
| [[Self-Instruct]] | 1 | 1 | [[Data Synthesis]] |
| [[Self-RAG]] | 1 | 1 | [[RAG Survey (2023)]] |
| [[SmoothQuant]] | 1 | 2 | [[Quantization]] |
| [[SPAG]] | 1 | 1 | [[Self-Play]] |
| [[Sparse Autoencoder]] | 1 | 3 | [[Qwen-Scope (2026)]] |
| [[State Space Model]] | 1 | 1 | [[Test-time Regression (2025)]] |
| [[SWE-Universe]] | 1 | 3 | [[SWE-Bench]] |
| [[SwiGLU Clamping]] | 1 | 1 | [[DeepSeek-V4 (2026)]] |
| [[TDPO]] | 1 | 1 | [[DPO]] |
| [[Temperature Scaling]] | 1 | 1 | [[Entropy]] |
| [[Trajectory-Level KL Instability]] | 1 | 1 | [[TCOD]] |
| [[Tree-of-Thought]] | 1 | 1 | [[Chain-of-Thought]] |
| [[Truncated Importance Sampling]] | 1 | 1 | [[Rollout Routing Replay]] |
| [[TTT-Linear]] | 1 | 2 | [[Test-time Regression (2025)]] |
| [[Verifiable Reward]] | 1 | 1 | [[SWE-Bench]] |
| [[vLLM]] | 1 | 1 | [[ROLL (2025)]] |
| [[Widrow-Hoff Rule]] | 1 | 1 | [[DeltaNet]] |

### 论文年份链接策略

共 5 个唯一目标。

| 目标 | 引用篇数 | 出现次数 | 来源 |
|---|---:|---:|---|
| [[DPO (2023)]] | 3 | 3 | [[DPO]], [[IPO]], [[SimPO]] |
| [[DeepSeek-V3 (2025)]] | 1 | 1 | [[Mixture of Experts]] |
| [[FLAN (2022)]] | 1 | 1 | [[SFT]] |
| [[Mixtral (2024)]] | 1 | 1 | [[Mixture of Experts]] |
| [[Switch Transformer (2021)]] | 1 | 1 | [[Mixture of Experts]] |

### 中文 alias 或主题缺口

共 3 个唯一目标。

| 目标 | 引用篇数 | 出现次数 | 来源 |
|---|---:|---:|---|
| [[最大似然估计]] | 1 | 1 | [[Bradley-Terry Model]] |
| [[概率论]] | 1 | 1 | [[Bradley-Terry Model]] |
| [[概率论基础]] | 1 | 1 | [[Entropy]] |

### clipping 路径缺失

共 1 个唯一目标。

| 目标 | 引用篇数 | 出现次数 | 来源 |
|---|---:|---:|---|
| [[Clippings/Paper/1602.01783v1]] | 1 | 1 | [[A3C]] |

## 缺失 heading 链接

| 链接 | 来源 | 已解析到 |
|---|---|---|
| `[[DAPO#Dr. GRPO]]` | [[MemAgent (2025)]] | [[DAPO]] |

## 连接性空缺

### 孤立主笔记

| 笔记 | type | 出链 | 描述 |
|---|---|---:|---|
| [[CL-Bench]] | concept | 1 | 评估语言模型从复杂上下文中学习新知识并应用于推理的能力，揭示当前模型在 Context Learning 上的严重不足 |
| [[Reward Shaping]] | concept | 8 | 通过添加辅助奖励信号引导学习过程，在不改变最优策略的前提下加速收敛 |
| [[interview/rl_infra|rl_infra]] | interview | 25 | 考察 RL post-training 基础设施的系统理解，覆盖 slime 数据流、GRPO/PPO/CISPO、SGLang 指标、确定性与 fully async 训练 |
| [[interview/游戏 LLM Agent RL 长交互优化|游戏 LLM Agent RL 长交互优化]] | interview | 8 | 游戏场景（如自走棋）中 LLM Agent RL 的长交互优化：context 管理、reasoning faithfulness、conciseness |
| [[Open-Source LLM Landscape (2025-2026)]] | overview | 35 | 2025-2026 开源大模型技术全景图，深度分析 12 个旗舰模型的架构演进、训练范式、推理能力和效率优化的趋势与权衡 |
| [[AgentFold (2025)]] | paper | 4 | 通过双尺度折叠机制（granular condensation + deep consolidation）实现主动上下文管理的长时域 Web Agent，30B 参数达到 671B 模型性能 |
| [[Composer 2 (2025)]] | paper | 12 | Cursor 的 agentic coding 专用模型，展示了 continued pretraining + 大规模 RL 两阶段范式如何将开源 MoE 基座提升到 frontier 水平 |
| [[DAVINCI-LLM (2026)]] | paper | 3 | SII/SJTU/GAIR 的全开放预训练研究项目，通过 Data Darwinism 框架（L0-L9）、两阶段自适应课程和 200+ 控制消融实验，系统性研究预训练科学，3B 模型匹配 OLMo-3 7B |
| [[DeepAgent (2025)]] | paper | 5 | 端到端深度推理智能体，通过统一推理流程实现自主工具发现、调用和记忆管理，用 ToolPO 训练支持万级工具集的通用任务求解 |
| [[MEM1 (2025)]] | paper | 13 | 端到端 RL 框架，通过统一的内部状态实现推理与记忆整合，在长时程多轮任务中保持恒定内存 |
| [[MemEvolve (2025)]] | paper | 0 | 提出 MemEvolve 元进化框架，通过双层优化同时演进 Agent 的经验知识和记忆架构，实现跨任务的自适应记忆系统 |
| [[Qwen-Scope (2026)]] | paper | 4 | Qwen-Scope 发布 Qwen 系列的 SAE 套件，并展示 sparse features 如何作为 steering、评测、数据构造和 post-training 的统一表示接口 |

### 零出链与低出链

| 笔记 | 入链 | 出链 | type | 行数 |
|---|---:|---:|---|---:|
| [[MemEvolve (2025)]] | 0 | 0 | paper | 215 |
| [[SWE-Bench]] | 6 | 1 | concept | 181 |
| [[Adam]] | 5 | 1 | method | 182 |
| [[GraphRAG]] | 2 | 1 | method | 260 |
| [[TIES-Merging]] | 2 | 1 | method | 108 |
| [[interview/如何搭建 LOL 陪玩助手|如何搭建 LOL 陪玩助手]] | 1 | 1 | interview | 469 |
| [[interview/游戏知识库如何构建|游戏知识库如何构建]] | 1 | 1 | interview | 619 |
| [[CL-Bench]] | 0 | 1 | concept | 171 |
| [[RAG Survey (2023)]] | 7 | 2 | paper | 180 |
| [[Radix Attention]] | 5 | 2 | method | 222 |
| [[ROLL (2025)]] | 4 | 2 | paper | 181 |
| [[AdamW]] | 3 | 2 | method | 176 |
| [[A Survey of On-Policy Distillation for Large Language Models (2026)]] | 1 | 2 | paper | 197 |

## MOC 覆盖空缺

### `post-training`：MOC - Post-training.md

- tag 笔记：86，MOC 已覆盖：63，缺口：23，覆盖率：73.3%
- 未覆盖笔记：[[A2C]], [[EMPG (2025)]], [[GLM-4.5 (2025)]], [[GiGPO (2025)]], [[Kimi K2.5 (2026)]], [[Lessons of Developing PRMs (2025)]], [[Let It Flow (2025)]], [[Ling 2.0 (2025)]], [[LongCat-Flash-Thinking (2025)]], [[MEM1 (2025)]], [[MEMENTO]], [[MemAgent (2025)]], [[Mid-Training Survey (2024)]], [[MiniMax-M1 (2025)]], [[Nemotron 3 Super (2026)]], [[Qwen-Scope (2026)]], [[REINFORCE++]], [[RLOO]], [[RLTF (2025)]], [[ReMax]], [[Step 3.5 Flash (2025)]], [[interview/rl_infra|rl_infra]], [[interview/如何搭建 LOL 陪玩助手|如何搭建 LOL 陪玩助手]]

### `reinforcement-learning`：MOC - Reinforcement Learning.md

- tag 笔记：56，MOC 已覆盖：43，缺口：13，覆盖率：76.8%
- 未覆盖笔记：[[Composer 2 (2025)]], [[DGPO]], [[DeepAgent (2025)]], [[GLM-5 (2026)]], [[Kimi K2.5 (2026)]], [[MOPD]], [[Post-Training Phase Transition]], [[RLTF (2025)]], [[Reward Shaping]], [[Self-Play]], [[Soft Clipping]], [[interview/rl_infra|rl_infra]], [[interview/游戏 LLM Agent RL 长交互优化|游戏 LLM Agent RL 长交互优化]]

### `inference`：MOC - Inference.md

- tag 笔记：19，MOC 已覆盖：16，缺口：3，覆盖率：84.2%
- 未覆盖笔记：[[Chain-of-Thought]], [[MiniMax-M1 (2025)]], [[interview/rl_infra|rl_infra]]

### `rag`：MOC - RAG.md

- tag 笔记：17，MOC 已覆盖：15，缺口：2，覆盖率：88.2%
- 未覆盖笔记：[[interview/如何搭建 LOL 陪玩助手|如何搭建 LOL 陪玩助手]], [[interview/游戏知识库如何构建|游戏知识库如何构建]]

### `foundations`：MOC - Foundations.md

- tag 笔记：16，MOC 已覆盖：16，缺口：0，覆盖率：100.0%

### `distributed-training`：MOC - Distributed Training.md

- tag 笔记：14，MOC 已覆盖：12，缺口：2，覆盖率：85.7%
- 未覆盖笔记：[[A3C]], [[interview/rl_infra|rl_infra]]

### `attention`：MOC - Attention.md

- tag 笔记：12，MOC 已覆盖：10，缺口：2，覆盖率：83.3%
- 未覆盖笔记：[[DeepSeek-V4 (2026)]], [[MiniMax-M1 (2025)]]

## 缺少 MOC 的高频 tag

这些 tag 已经形成局部领域，缺少专门导航会降低复习和扩展效率。

| tag | 笔记数 | 示例 |
|---|---:|---|
| `optimization` | 34 | [[A3C]], [[Adam]], [[AdamW]], [[Context Compression]], [[Continuous Batching]], [[DGPO]], [[Data Parallelism]], [[Dr. GRPO]] |
| `architecture` | 27 | [[ALiBi]], [[Attention Residuals]], [[Attention Sink]], [[Attention]], [[DeepSeek-V3 (2024)]], [[DeepSeek-V4 (2026)]], [[DeltaNet]], [[GLM-4.5 (2025)]] |
| `rlhf` | 24 | [[Credit Assignment]], [[DGPO]], [[DORA]], [[DeepSeek-V3 (2024)]], [[Dr. GRPO]], [[Entropy Collapse]], [[GLM-4.5 (2025)]], [[GRPO]] |
| `efficiency` | 24 | [[DORA]], [[Data Parallelism]], [[DeepSeek-V3 (2024)]], [[DeepSeek-V4 (2026)]], [[Grouped-Query Attention]], [[LoRA]], [[LongCat-Flash (2025)]], [[Loss-Free Load Balancing]] |
| `reasoning` | 23 | [[AgentFold (2025)]], [[CISPO]], [[CL-Bench]], [[Chain-of-Thought]], [[DAPO]], [[DeepAgent (2025)]], [[GLM-4.5 (2025)]], [[Lessons of Developing PRMs (2025)]] |
| `alignment` | 22 | [[Constitutional AI]], [[DAgger]], [[DPO]], [[Data Synthesis]], [[IPO]], [[InstructGPT]], [[KTO]], [[LLM RL Algorithm Evolution]] |
| `agent` | 14 | [[ARLArena (2026)]], [[ASTRA]], [[AgentFold (2025)]], [[Agentic RL Survey (2025)]], [[Agentic RL]], [[DeepAgent (2025)]], [[EMPG (2025)]], [[EMPG]] |
| `moe` | 13 | [[DeepSeek-V3 (2024)]], [[Expert Parallelism]], [[GLM-5 (2026)]], [[IcePop]], [[Kimi K2 (2025)]], [[Ling 2.0 (2025)]], [[LongCat-Flash (2025)]], [[Loss-Free Load Balancing]] |
| `agentic` | 13 | [[AgentFold (2025)]], [[Agentic RL Survey (2025)]], [[Composer 2 (2025)]], [[DAgger]], [[GLM-4.5 (2025)]], [[GLM-5 (2026)]], [[Kimi K2 (2025)]], [[Kimi K2.5 (2026)]] |
| `transformer` | 10 | [[ALiBi]], [[Attention Residuals]], [[Attention]], [[Grouped-Query Attention]], [[Multi-Head Attention]], [[Multi-Query Attention]], [[Multi-head Latent Attention]], [[Positional Encoding]] |
| `pretraining` | 7 | [[DAVINCI-LLM (2026)]], [[DeepSeek-V3 (2024)]], [[DeepSeek-V4 (2026)]], [[Kimi K2 (2025)]], [[Multi-Token Prediction]], [[Qwen3 (2025)]], [[Tokenization]] |
| `data` | 7 | [[Data Synthesis]], [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[Mid-Training]], [[PRISM (2026)]], [[Preference Data]], [[The Synthetic Data Playbook (2026)]] |
| `technical-report` | 6 | [[GLM-5 (2026)]], [[Kimi K2.5 (2026)]], [[Ling 2.0 (2025)]], [[MiMo-V2-Flash (2025)]], [[Open-Source LLM Landscape (2025-2026)]], [[Step 3.5 Flash (2025)]] |
| `survey` | 6 | [[A Survey of On-Policy Distillation for Large Language Models (2026)]], [[Agentic RL Survey (2025)]], [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[RAG Survey (2023)]], [[Towards Agentic RAG with Deep Reasoning (2025)]] |
| `efficient-attention` | 6 | [[Attention Sink]], [[DeltaNet]], [[Linear Attention]], [[Sliding Window Attention]], [[Sparse Attention]], [[Test-time Regression (2025)]] |
| `serving` | 5 | [[Continuous Batching]], [[MEMENTO]], [[MOC - Inference]], [[Paged Attention]], [[Radix Attention]] |
| `scaling` | 5 | [[DeepSeek-V4 (2026)]], [[Mixture of Experts]], [[Ring-1T (2025)]], [[The Art of Scaling RL Compute for LLMs (2025)]], [[The Synthetic Data Playbook (2026)]] |
| `retrieval` | 5 | [[Embedding]], [[MOC - RAG]], [[RAG Survey (2023)]], [[RAG]], [[Reranking]] |
| `math` | 5 | [[Bradley-Terry Model]], [[Cross-Entropy Loss]], [[KL Divergence]], [[Lessons of Developing PRMs (2025)]], [[Process Reward Model]] |
| `long-context` | 5 | [[CL-Bench]], [[Length Extrapolation]], [[MemAgent (2025)]], [[Sparse Attention]], [[interview/游戏 LLM Agent RL 长交互优化|游戏 LLM Agent RL 长交互优化]] |
| `distillation` | 5 | [[A Survey of On-Policy Distillation for Large Language Models (2026)]], [[MOPD]], [[On-Policy Distillation]], [[Post-Training Phase Transition]], [[TIP]] |
| `direct-alignment` | 5 | [[DPO]], [[IPO]], [[KTO]], [[ORPO]], [[SimPO]] |

## Tag 体系空缺

| 变体组 | 建议 |
|---|---|
| `pre-training`(1), `pretraining`(7) | 选一个主 tag，其余迁移为 alias 或停用 |
| `rl`(3), `reinforcement-learning`(57), `rlhf`(24), `rl-for-llm`(1) | 选一个主 tag，其余迁移为 alias 或停用 |
| `agent`(14), `agentic`(13), `agentic-rl`(3) | 选一个主 tag，其余迁移为 alias 或停用 |
| `efficient-inference`(1), `inference-efficiency`(1), `inference`(20) | 选一个主 tag，其余迁移为 alias 或停用 |
| `reward-model`(2), `preference-modeling`(1) | 选一个主 tag，其余迁移为 alias 或停用 |
| `training`(3), `training-dynamics`(2), `training-stability`(3), `training-efficiency`(1) | 选一个主 tag，其余迁移为 alias 或停用 |
| `data`(7), `data-centric`(1), `data-preparation`(1), `data-synthesis`(1), `synthetic-data`(1) | 选一个主 tag，其余迁移为 alias 或停用 |

## 显式标记空缺

这些空缺来自正文中的 `待创建`、`需要补充`、`待探索`、`有待研究` 等标记。它们补充了 wikilink 图谱扫描，因为部分未来主题没有写成 wikilink。

### 仍然缺失的显式目标

| 目标 | 来源 |
|---|---|
| [[Hallucination]] | `Citation Generation.md:193` |
| [[Grounding]] | `Citation Generation.md:194` |
| [[GLA]] | `DeltaNet.md:309` |
| [[Mamba]] | `DeltaNet.md:310` |
| [[Fast Weight Programming]] | `DeltaNet.md:317` |
| [[Widrow-Hoff Rule]] | `DeltaNet.md:318` |
| [[Expert Choice]] | `Loss-Free Load Balancing.md:223` |
| [[Switch Transformer (2021)]] | `Mixture of Experts.md:314` |
| [[Mixtral (2024)]] | `Mixture of Experts.md:315` |
| [[DeepSeek-V3 (2025)]] | `Mixture of Experts.md:316` |

### 未链接或研究型未来主题

- `Agentic RL Survey (2025).md:232`
- `Data Synthesis.md:316`
- `DeepSeek-V4 (2026).md:119`
- `MemAgent (2025).md:246`
- `RLSD.md:224`
- `VAPO (2025).md:147`

### 陈旧 `待创建` 标记

以下目标已经能解析到主笔记，相关 MOC 或文末列表仍标记为 `待创建`。这属于维护空缺。

| 来源 | 已存在目标 |
|---|---|
| `Data Parallelism.md:198` | [[ZeRO]] |
| `Data Parallelism.md:199` | [[Tensor Parallelism]] |
| `Data Parallelism.md:200` | [[Pipeline Parallelism]] |
| `Data Parallelism.md:201` | [[FSDP]] |
| `DeltaNet.md:308` | [[Linear Attention]] |
| `Loss-Free Load Balancing.md:219` | [[Expert Parallelism]] |
| `Loss-Free Load Balancing.md:220` | [[DeepSeek-V3]] |
| `Mixture of Experts.md:310` | [[Sparse Attention]] |
| `Mixture of Experts.md:311` | [[Expert Parallelism]] |
| `ZeRO.md:238` | [[FSDP]] |
| `ZeRO.md:239` | [[Tensor Parallelism]] |
| `ZeRO.md:240` | [[Pipeline Parallelism]] |
| `ZeRO.md:241` | [[Activation Checkpointing]] |

## 写作规范空缺

知识库当前规则禁止“先否定一个说法，再用转折词肯定另一个说法”的句式。扫描发现以下命中，建议按批次改写为直接陈述、条件句或并列对比。

`3D Parallelism.md:56`, `ARLArena (2026).md:176`, `Actor-Critic.md:284`, `AgentFold (2025).md:231`, `Agentic RL Survey (2025).md:59`, `Agentic RL Survey (2025).md:81`, `Agentic RL Survey (2025).md:83`, `Agentic RL Survey (2025).md:260`, `Composer 2 (2025).md:48`, `Cross-Entropy Loss.md:142`, `DAVINCI-LLM (2026).md:22`, `DAVINCI-LLM (2026).md:138`, `DAVINCI-LLM (2026).md:183`, `DQN.md:138`, `DeepAgent (2025).md:66`, `Dr. GRPO.md:199`, `EMPG (2025).md:129`, `EMPG.md:240`, `Experience Replay.md:75`, `Experience Replay.md:139`, `Flash Attention.md:24`, `Kimi K2.5 (2026).md:110`, `Kimi K2.5 (2026).md:220`, `LLM RL Algorithm Evolution.md:181`, `LLM RL Algorithm Evolution.md:315`, `LLM RL Algorithm Evolution.md:554`, `Let It Flow (2025).md:304`, `LoRA.md:277`, `Markov Decision Process.md:163`, `Mixed Precision Training.md:157`, `Open-Source LLM Landscape (2025-2026).md:138`, `RLAIF.md:105`, `Ring-1T (2025).md:27`, `SAMPO.md:30`, `SFT.md:117`, `Self-Play.md:55`, `Self-Play.md:118`, `Speculative Decoding.md:26`, `TD Learning.md:114`, `Test-time Regression (2025).md:57`, `The Art of Scaling RL Compute for LLMs (2025).md:126`, `Tokenization.md:146`, `Tokenization.md:294`, `Training-Inference Mismatch.md:148`, `VC-PPO.md:23`, `interview/rl_infra.md:229`, `interview/rl_infra.md:260`, `interview/rl_infra.md:693`

## 素材吸收空缺

`Clippings/` 中有 21 篇素材没有被主笔记引用；其中 6 篇 Article、15 篇 Paper。它们代表两类任务：有主笔记但缺少出处链接的溯源空缺，以及尚未吸收到主笔记的内容空缺。

### Article clipping

- `Clippings/Article/Approximating KL Divergence.md`
- `Clippings/Article/From Traditional RL to LLM RL 理论推导与工程改进.md`
- `Clippings/Article/Kimi K2.5 技术报告阅读笔记.md`
- `Clippings/Article/LLM训练-pretrain.md`
- `Clippings/Article/Learning from Mixed Rollouts Logit Fusion as a Bridge Between Imitation and Exploration  Notion.md`
- `Clippings/Article/Minimax M2.1.md`

### Paper clipping

| clipping | 摘要 |
|---|---|
| `Clippings/Paper/1707.06347/1707.06347.md` | PPO 原始论文素材 |
| `Clippings/Paper/231006770v3/231006770v3.md` | SWE-bench |
| `Clippings/Paper/231010505v4/231010505v4.md` | ReMax |
| `Clippings/Paper/240606484v6/240606484v6.md` | DeltaNet |
| `Clippings/Paper/240815664v1/240815664v1.md` | Loss-Free Balancing |
| `Clippings/Paper/2505.10978/2505.10978.md` | GiGPO |
| `Clippings/Paper/250720534v1/250720534v1.md` | Kimi K2 / MuonClip |
| `Clippings/Paper/251224601v2/251224601v2.md` | RLM 长上下文推理 |
| `Clippings/Paper/2601.05242/2601.05242.md` | GDPO |
| `Clippings/Paper/2602.21320/2602.21320.md` | 未补 description 的 paper clipping |
| `Clippings/Paper/260202361v1/260202361v1.md` | SWE-Universe |
| `Clippings/Paper/260203587v1/260203587v1.md` | CL-Bench |
| `Clippings/Paper/2604.23747/2604.23747.md` | SFT-then-RL / mixed-policy 推理 |
| `Clippings/Paper/Attention_Residuals/Attention_Residuals.md` | Attention Residuals |
| `Clippings/Paper/NVIDIA-Nemotron-3-Super-Technical-Report/NVIDIA-Nemotron-3-Super-Technical-Report.md` | Nemotron 3 Super |

## Metadata 空缺

### 缺少 aliases（26）

[[ALiBi]], [[Actor-Critic]], [[Attention]], [[Bradley-Terry Model]], [[Context-Folding (2025)]], [[DPO]], [[Flash Attention]], [[GAE]], [[Grouped-Query Attention]], [[KV Cache]], [[LLM Mid-Training Survey (2024)]], [[Mid-Training Survey (2024)]], [[Multi-Head Attention]], [[Multi-Query Attention]], [[PPO]], [[PRISM (2026)]], [[Paged Attention]], [[RLHF]], [[Radix Attention]], [[Reward Hacking]], [[Reward Model]], [[RoPE]], [[TRPO]], [[interview/如何搭建 LOL 陪玩助手|如何搭建 LOL 陪玩助手]], [[interview/游戏 LLM Agent RL 长交互优化|游戏 LLM Agent RL 长交互优化]], [[interview/游戏知识库如何构建|游戏知识库如何构建]]

### 缺少 type（1）

[[Context-Folding (2025)]]

### 缺少 description（1）

[[Context-Folding (2025)]]

### 缺少 tags（1）

[[Context-Folding (2025)]]

### 缺少 created（1）

[[Context-Folding (2025)]]

## 深度空缺候选

以下是启发式扫描结果：篇幅、示例、边界、动机、论文出处和出链。它们需要人工复核，优先从高入链笔记开始。

| 笔记 | type | 入链 | 出链 | 行数 | 触发项 |
|---|---|---:|---:|---:|---|
| [[RLHF]] | method | 59 | 11 | 192 | 示例不足 |
| [[Reward Model]] | concept | 35 | 10 | 184 | 示例不足 |
| [[DPO]] | method | 32 | 12 | 258 | 示例不足 |
| [[Flash Attention]] | method | 25 | 5 | 183 | 边界/局限不足, 示例不足, 论文出处不足 |
| [[KL Divergence]] | concept | 26 | 5 | 166 | 边界/局限不足, 示例不足 |
| [[Importance Sampling]] | concept | 28 | 8 | 239 | 示例不足 |
| [[Policy Gradient]] | concept | 27 | 9 | 213 | 示例不足 |
| [[Attention]] | concept | 20 | 6 | 156 | 边界/局限不足, 示例不足 |
| [[GAE]] | concept | 22 | 4 | 149 | 示例不足 |
| [[GSPO]] | method | 17 | 6 | 205 | 示例不足 |
| [[Value Function]] | concept | 14 | 11 | 159 | 示例不足 |
| [[Training-Inference Mismatch]] | concept | 14 | 11 | 287 | 示例不足 |
| [[Expert Parallelism]] | concept | 10 | 8 | 115 | 边界/局限不足, 示例不足 |
| [[REINFORCE]] | method | 12 | 11 | 231 | 示例不足 |
| [[Bradley-Terry Model]] | concept | 16 | 3 | 165 | 示例不足 |
| [[RAG]] | concept | 12 | 10 | 210 | 示例不足 |
| [[Actor-Critic]] | concept | 13 | 8 | 291 | 示例不足 |
| [[Speculative Decoding]] | method | 13 | 6 | 268 | 论文出处不足 |
| [[Process Reward Model]] | concept | 11 | 8 | 145 | 示例不足 |
| [[Tensor Parallelism]] | concept | 10 | 7 | 154 | 示例不足 |
| [[Sliding Window Attention]] | method | 10 | 6 | 210 | 论文出处不足 |
| [[RLAIF]] | method | 9 | 7 | 151 | 示例不足 |
| [[IPO]] | method | 8 | 9 | 178 | 示例不足 |
| [[TD Learning]] | concept | 9 | 6 | 196 | 示例不足 |
| [[SAMPO]] | method | 7 | 10 | 286 | 示例不足 |
| [[RLOO]] | method | 7 | 10 | 119 | 示例不足 |
| [[SimPO]] | method | 8 | 7 | 181 | 示例不足 |
| [[RoPE]] | method | 9 | 5 | 276 | 示例不足 |
| [[Preference Data]] | concept | 5 | 13 | 222 | 示例不足 |
| [[ASTRA]] | method | 9 | 5 | 152 | 示例不足 |
| [[SAPO]] | method | 8 | 6 | 310 | 示例不足 |
| [[Radix Attention]] | method | 5 | 2 | 222 | 论文出处不足, 出链不足 |
| [[Mixed Precision Training]] | concept | 8 | 6 | 242 | 示例不足 |
| [[MIS-PO]] | method | 5 | 11 | 148 | 示例不足 |
| [[Cross-Entropy Loss]] | concept | 8 | 5 | 223 | 示例不足 |
| [[Adam]] | method | 5 | 1 | 182 | 论文出处不足, 出链不足 |
| [[Soft Clipping]] | concept | 2 | 7 | 62 | 篇幅偏浅, 示例不足 |
| [[Quantization]] | method | 7 | 7 | 285 | 论文出处不足 |
| [[Linear Attention]] | concept | 6 | 7 | 197 | 示例不足 |
| [[EMPG]] | method | 6 | 7 | 257 | 示例不足 |
| [[TRPO]] | method | 8 | 3 | 139 | 示例不足 |
| [[Positional Encoding]] | concept | 7 | 5 | 180 | 边界/局限不足 |
| [[FSDP]] | method | 6 | 6 | 189 | 示例不足 |
| [[A2C]] | method | 6 | 6 | 178 | 示例不足 |
| [[SAC]] | concept | 5 | 8 | 167 | 示例不足 |
| [[Paged Attention]] | method | 7 | 4 | 226 | 边界/局限不足 |
| [[VC-PPO]] | method | 5 | 7 | 144 | 示例不足 |
| [[DGPO]] | method | 4 | 9 | 218 | 示例不足 |
| [[TIES-Merging]] | method | 2 | 1 | 108 | 示例不足, 出链不足 |
| [[SWE-Bench]] | concept | 6 | 1 | 181 | 出链不足 |
| [[ReMax]] | method | 2 | 9 | 134 | 示例不足 |
| [[JACKPOT]] | method | 2 | 9 | 162 | 示例不足 |
| [[DORA]] | method | 3 | 7 | 127 | 示例不足 |
| [[A3C]] | method | 4 | 5 | 231 | 示例不足 |
| [[RLSD]] | method | 3 | 6 | 210 | 示例不足 |
| [[Near-Future Policy Optimization]] | method | 2 | 8 | 220 | 示例不足 |
| [[DeltaNet]] | method | 3 | 5 | 327 | 示例不足 |
| [[Continuous Batching]] | concept | 3 | 5 | 256 | 示例不足 |
| [[SFT-then-RL]] | method | 2 | 7 | 101 | 示例不足 |
| [[Context-Folding (2025)]] | unknown | 2 | 6 | 59 | 篇幅偏浅 |
| [[AdamW]] | method | 3 | 2 | 176 | 出链不足 |
| [[TIP]] | method | 1 | 6 | 135 | 示例不足 |
| [[TCOD]] | method | 1 | 6 | 147 | 示例不足 |
| [[Model Merging]] | method | 2 | 4 | 231 | 示例不足 |
| [[Attention Residuals]] | method | 1 | 5 | 145 | 示例不足 |
| [[GraphRAG]] | method | 2 | 1 | 260 | 出链不足 |
| [[CL-Bench]] | concept | 0 | 1 | 171 | 出链不足 |

## 高中心度节点

这些节点是后续加深和链接修复的高杠杆位置。

| 笔记 | 总度数 | 入链 | 出链 | type |
|---|---:|---:|---:|---|
| [[GRPO]] | 85 | 72 | 13 | method |
| [[MOC - Post-training]] | 83 | 8 | 75 | moc |
| [[PPO]] | 78 | 68 | 10 | method |
| [[MOC - Reinforcement Learning]] | 73 | 5 | 68 | moc |
| [[RLHF]] | 70 | 59 | 11 | method |
| [[MOC - Foundations]] | 56 | 4 | 52 | moc |
| [[Reward Model]] | 45 | 35 | 10 | concept |
| [[DPO]] | 44 | 32 | 12 | method |
| [[Mixture of Experts]] | 42 | 34 | 8 | concept |
| [[LLM RL Algorithm Evolution]] | 39 | 4 | 35 | overview |
| [[SFT]] | 36 | 30 | 6 | method |
| [[Importance Sampling]] | 36 | 28 | 8 | concept |
| [[Policy Gradient]] | 36 | 27 | 9 | concept |
| [[MOC - Distributed Training]] | 35 | 6 | 29 | moc |
| [[Open-Source LLM Landscape (2025-2026)]] | 35 | 0 | 35 | overview |
| [[Credit Assignment]] | 32 | 21 | 11 | concept |
| [[KL Divergence]] | 31 | 26 | 5 | concept |
| [[KV Cache]] | 31 | 24 | 7 | concept |
| [[MOC - RAG]] | 31 | 10 | 21 | moc |
| [[Flash Attention]] | 30 | 25 | 5 | method |
| [[Transformer]] | 30 | 20 | 10 | concept |
| [[MOC - Attention]] | 30 | 5 | 25 | moc |
| [[Agentic RL]] | 28 | 22 | 6 | method |
| [[MOC - Inference]] | 28 | 6 | 22 | moc |
| [[On-Policy Distillation]] | 27 | 19 | 8 | method |
| [[GAE]] | 26 | 22 | 4 | concept |
| [[DAPO]] | 26 | 21 | 5 | method |
| [[Attention]] | 26 | 20 | 6 | concept |
| [[Reward Hacking]] | 25 | 20 | 5 | concept |
| [[Value Function]] | 25 | 14 | 11 | concept |
| [[Training-Inference Mismatch]] | 25 | 14 | 11 | concept |
| [[interview/rl_infra|rl_infra]] | 25 | 0 | 25 | interview |
| [[GSPO]] | 23 | 17 | 6 | method |
| [[REINFORCE]] | 23 | 12 | 11 | method |
| [[RAG]] | 22 | 12 | 10 | concept |
| [[ARLArena (2026)]] | 22 | 7 | 15 | paper |
| [[DeepSeek-V3 (2024)]] | 21 | 18 | 3 | paper |
| [[Kimi K2 (2025)]] | 21 | 13 | 8 | paper |
| [[Actor-Critic]] | 21 | 13 | 8 | concept |
| [[Data Parallelism]] | 21 | 11 | 10 | concept |
| [[InstructGPT]] | 21 | 10 | 11 | paper |
| [[CISPO]] | 20 | 14 | 6 | method |
| [[Entropy Collapse]] | 20 | 13 | 7 | concept |
| [[Constitutional AI]] | 20 | 12 | 8 | method |
| [[Bradley-Terry Model]] | 19 | 16 | 3 | concept |
| [[Speculative Decoding]] | 19 | 13 | 6 | method |
| [[Process Reward Model]] | 19 | 11 | 8 | concept |
| [[Grouped-Query Attention]] | 18 | 13 | 5 | method |
| [[Ring-1T (2025)]] | 18 | 12 | 6 | paper |
| [[Multi-Token Prediction]] | 18 | 12 | 6 | method |
| [[Multi-Head Attention]] | 18 | 12 | 6 | concept |
| [[ZeRO]] | 18 | 11 | 7 | method |
| [[Pipeline Parallelism]] | 18 | 11 | 7 | concept |
| [[Sparse Attention]] | 18 | 10 | 8 | concept |
| [[Reranking]] | 18 | 10 | 8 | concept |
| [[Expert Parallelism]] | 18 | 10 | 8 | concept |
| [[KTO]] | 18 | 9 | 9 | method |
| [[Preference Data]] | 18 | 5 | 13 | concept |
| [[Tensor Parallelism]] | 17 | 10 | 7 | concept |
| [[Multi-head Latent Attention]] | 17 | 10 | 7 | method |

## 命名冲突

| 名称或 alias | 目标 |
|---|---|
| `icepop` | [[IcePop]], [[Ring-1T (2025)]] |

## 建议生长路径

1. 先处理高引用结构空缺：`Hybrid Retrieval`、`ReAct`、`RLVR`、`Pretraining`、`Post-training`、`Curriculum Learning`、`Mamba`、`Knowledge Distillation`、`Muon`。
2. 统一链接策略：把 `DPO (2023)`、`IPO (2024)`、`KTO (2024)` 等论文年份链接改为方法笔记链接或 clipping 出处；把 `Hybrid Search` 并入 `Hybrid Retrieval` alias。
3. 把孤立的新前沿笔记接入 MOC：`Open-Source LLM Landscape (2025-2026)`、`AgentFold (2025)`、`Composer 2 (2025)`、`DAVINCI-LLM (2026)`、`DeepAgent (2025)`、`MEM1 (2025)`、`MemEvolve (2025)`、`Qwen-Scope (2026)`。
4. 更新 MOC 覆盖率最低的导航：`MOC - Post-training`、`MOC - Reinforcement Learning`、`MOC - Attention`。
5. 补齐高中心度笔记的示例和边界：`RLHF`、`Reward Model`、`Flash Attention`、`DPO`、`KL Divergence`、`Policy Gradient`、`Importance Sampling`。
6. 收敛 tag 体系：先统一 `pre-training/pretraining`、`agent/agentic/agentic-rl`、`rl/reinforcement-learning/rlhf/rl-for-llm`、`efficient-inference/inference-efficiency/inference`。
