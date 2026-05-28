---
type: paper
description: 通过双尺度折叠机制（granular condensation + deep consolidation）实现主动上下文管理的长时域 Web Agent，30B 参数达到 671B 模型性能
aliases:
  - AgentFold
  - 主动上下文管理
prerequisites:
  - "[[Agentic RL]]"
tags:
  - agent
  - agentic
  - reasoning
  - context-management
created: 2026-03-06
updated: 2026-03-06T18:48
---

# AgentFold (2025)

AgentFold 是一个专为长时域任务设计的 Web Agent 范式，核心创新在于**主动上下文管理**（Proactive Context Management）。它将上下文视为动态认知工作区，通过双尺度"折叠"操作在保留关键细节和控制上下文膨胀之间取得平衡。在 BrowseComp 上，AgentFold-30B-A3B 达到 36.2%，超越 DeepSeek-V3.1-671B（30.0%）和 OpenAI o4-mini（28.3%），且在 100 轮交互后上下文仅 7k tokens。

> [!paper] 论文信息
> **标题**: AgentFold: Long-Horizon Web Agents with Proactive Context Management
> **作者**: Rui Ye et al. (Tongyi Lab, Alibaba Group)
> **发表**: arXiv 2510.24699, 2025
> **链接**: https://tongyi-agent.github.io/blog
> **代码**: https://github.com/Alibaba-NLP/DeepResearch

## 动机：上下文管理的困境

当前 LLM-based Web Agent 在长时域任务上面临根本性权衡：

**ReAct-based agents 的问题**：
- 采用 append-only 策略，累积完整的 reasoning-action-observation 历史
- 保留了信息完整性，但遭受**上下文饱和**（context saturation）
- 原始 Web 数据噪声淹没关键信号，导致次优决策

**固定摘要方法的问题**（如 MEM1, MemAgent）：
- 每步机械地总结全部历史
- 保持上下文简洁，但**不可逆地丢失关键细节**
- 刚性的逐步压缩策略无法适应复杂任务

> [!intuition] 人类认知的启发
> 人类解决问题时，既不会记住所有细节，也不会每步都总结。而是在关键点进行**回顾性整合**（retrospective consolidation）：
> - 丢弃无关步骤
> - 提炼中间发现
> - 抽象关键洞察
>
> AgentFold 将上下文视为**动态工作区**（dynamic workspace），而非被动日志。

## 核心机制：双尺度折叠

### 上下文结构

AgentFold 的上下文 $C_t$ 由四部分组成：

$$C_t = (Q, T, S_{t-2}, I_{t-1})$$

| 组件 | 作用 | 类比 |
|------|------|------|
| $Q$ | 用户问题（不变） | 目标锚点 |
| $T$ | 可用工具列表（不变） | 行动能力 |
| $S_{t-2}$ | **多尺度状态摘要**（动态） | 长期记忆 |
| $I_{t-1}$ | **最新交互**（完整记录） | 工作记忆 |

**多尺度状态摘要** $S_t$ 是关键创新：

$$S_t = (s_{x_1, y_1}, s_{x_2, y_2}, \ldots, s_{x_m, y_m})$$

- 每个 $s_{x,y}$ 是步骤 $x$ 到 $y$ 的文本摘要
- $s_{x,x}$：单步摘要（细粒度）
- $s_{x,y}$ (where $y > x$)：多步整合（粗粒度）

**最新交互** $I_{t-1}$ 保留完整细节：

$$I_{t-1} = (e_{t-1}, a_{t-1}, o_{t-1})$$

- $e_{t-1}$: 解释（explanation）
- $a_{t-1}$: 动作（action）
- $o_{t-1}$: 观察（observation）

### 响应结构

每步生成四部分响应：

$$R_t = \text{AgentFold}(C_t; \theta) \rightarrow (th_t, f_t, e_t, a_t)$$

| 组件 | 作用 |
|------|------|
| $th_t$ | **思考过程**：分析上下文，权衡折叠和动作选项 |
| $f_t$ | **折叠指令**：管理历史轨迹 |
| $e_t$ | **解释**：阐述动作动机 |
| $a_t$ | **动作**：工具调用或最终答案 |

### 折叠指令的双尺度操作

折叠指令 $f_t$ 采用统一格式：

$$f_t = \{\text{"range"}: [k, t-1], \text{"summary"}: "\sigma_t"\}$$

支持两种操作模式：

#### 1. Granular Condensation（细粒度压缩）

**条件**: $k = t-1$（仅折叠最新交互）

**作用**: 将 $I_{t-1}$ 压缩为新的细粒度摘要 $s_{t-1, t-1}$，追加到 $S_{t-2}$

**适用场景**: 增量步骤，需要保留高分辨率历史

**示例**:
```
[Compressed Step 5] Found a new candidate XYZ that needs further exploration.
```

#### 2. Deep Consolidation（深度整合）

**条件**: $k < t-1$（折叠多个历史步骤）

**作用**: 将 $I_{t-1}$ 与步骤 $[k, t-2]$ 的所有摘要融合为单个粗粒度摘要 $s_{k, t-1}$

**适用场景**: 子任务完成，中间细节不再关键

**示例**:
```
[Compressed Step 5 to 9] Confirmed that XYZ does not fit all criteria after checking several sources.
```

> [!intuition] 为什么双尺度有效？
> - **Granular Condensation** 保护关键细节免受全历史总结的"信息损失累积"
> - **Deep Consolidation** 外科手术式地修剪无关历史，避免 ReAct 的"上下文饱和"
> - **延迟整合**：等到子任务结果明确后再整合，避免过早压缩导致的短视决策

### 操作循环

AgentFold 的运行循环：

```
1. 感知 (Perceive): 读取 C_t = (Q, T, S_{t-2}, I_{t-1})
2. 推理 (Reason): 生成 (th_t, f_t, e_t, a_t)
3. 折叠 (Fold): 应用 f_t，更新 S_{t-2} → S_{t-1}
4. 行动 (Act): 执行 a_t，获得 o_t
5. 更新工作记忆: I_t = (e_t, a_t, o_t)
6. 构建下一步上下文: C_{t+1} = (Q, T, S_{t-1}, I_t)
```

## 训练方法

### 数据生成：Fold-Generator

挑战：现有 LLM 无法通过 prompt engineering 可靠生成 AgentFold 的结构化多部分响应。

解决方案：
1. 使用强大的开源 LLM 作为数据生成器
2. 采用**拒绝采样**（rejection sampling）：
   - 丢弃格式不符的步骤
   - 丢弃环境错误过多的轨迹
3. 生成高质量交互对 $\{(C_t, R_t^*)\}_N$

### 监督微调（SFT）

在 Qwen3-30B-A3B-Instruct 上进行 SFT：
- 将"生成-过滤"策略蒸馏到模型权重
- 将"折叠"能力从脆弱的 prompt 依赖转化为内化技能
- 推理时比通用模型更高效

> [!warning] 训练限制
> 本文采用简单 SFT，未使用持续预训练或 RL。未来方向是用 RL 让 Agent 自主发现最优折叠策略。

## 实验结果

### 主要基准测试

| 模型 | BrowseComp | BrowseComp-ZH | WideSearch | GAIA |
|------|------------|---------------|------------|------|
| **Proprietary** |
| Claude-4-Opus | 18.8 | 37.4 | - | - |
| OpenAI o4-mini | 28.3 | 44.3 | - | - |
| OpenAI o3 | 49.7 | 58.1 | 60.0 | 70.5 |
| **Open-Source** |
| WebSailor-72B | 12.0 | 30.1 | - | 55.4 |
| GLM-4.5-355B-A32B | 26.4 | 37.5 | - | 66.0 |
| DeepSeek-V3.1-671B-A37B | 30.0 | 49.2 | - | 63.1 |
| **AgentFold-30B-A3B** | **36.2** | **47.3** | **62.1** | **67.0** |

**关键发现**：
- 以 30B 参数超越 671B 模型（DeepSeek-V3.1）
- 在 WideSearch 上超越所有闭源模型（包括 o3）
- 证明**架构创新**可以弥补模型规模差距

### 上下文动态分析

**Token 增长**（200 条 BrowseComp 轨迹）：
- 100 轮后：3.5k → 7k tokens（亚线性增长）
- 模型容量：128k tokens
- **利用率仅 5.5%**，潜力巨大

**与 ReAct 对比**（100 轮）：
- AgentFold: ~7k tokens
- ReAct: ~91k tokens
- **节省 84k tokens（92%）**
- 内存节省：~7GB per instance

**Block 数量增长**：
- ReAct: 线性增长（每步 +1）
- AgentFold: 亚线性增长（Deep Consolidation 合并多步）
- 结构复杂度显著降低

### 极限测试：500 轮交互

- 上下文大多保持在 20k tokens 以下
- **非单调增长**：Agent 可识别死胡同并深度整合失败子轨迹
- 展示自我修正的上下文管理能力

> [!example] 信息损失的数学分析
> 假设全历史总结每次有 1% 概率丢失关键细节：
> - 100 步后：$0.99^{100} \approx 36.6\%$ 存活率
> - 500 步后：$0.99^{500} \approx 0.66\%$ 存活率
>
> AgentFold 的 Granular Condensation 将关键细节保存在独立 block 中，避免重复处理导致的累积损失。

## 与相关方法的对比

| 方法 | 上下文策略 | 优势 | 劣势 |
|------|-----------|------|------|
| **ReAct** | Append-only | 信息完整 | 上下文饱和，噪声淹没信号 |
| **MEM1/MemAgent** | 每步全历史总结 | 上下文简洁 | 不可逆信息损失，刚性策略 |
| **AgentFold** | 双尺度折叠 | 自适应，细节保护 + 噪声修剪 | 需要训练折叠能力 |

**AgentFold 的独特优势**：
1. **灵活的回顾机制**：不是每步都压缩，而是在关键点回顾性评估
2. **多尺度管理**：可选择保留细节或深度抽象
3. **延迟整合**：等子任务结果明确后再整合，避免短视决策

## 局限性与未来方向

**当前限制**：
- 仅使用 SFT，未充分优化折叠策略
- 数据生成依赖拒绝采样，效率有提升空间
- 未探索 RL 训练折叠策略

**未来方向**：
1. **强化学习优化折叠**：直接优化任务成功率，发现非显而易见的折叠策略
2. **扩展到更长时域**：利用剩余 context 容量（95%未使用）
3. **跨领域泛化**：从 Web 搜索扩展到其他 agentic 任务

## 相关概念

**核心依赖**：
- [[Agentic RL]] — AgentFold 的训练范式
- [[ReAct]] — AgentFold 对比的基线范式（需创建）

**相关工作**：
- [[ASTRA]] — 自动化合成 agentic 轨迹的框架
- [[ARLArena (2026)]] — 分析 Agentic RL 训练稳定性

**应用场景**：
- [[Towards Agentic RAG with Deep Reasoning (2025)]] — RAG 与推理集成的演进

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/2510.24699/2510.24699]] — 完整论文提取

**相关基准**：
- BrowseComp (Wei et al., 2025) — 长时域 Web 信息搜索
- WideSearch (Wong et al., 2025) — 广度搜索能力评估
