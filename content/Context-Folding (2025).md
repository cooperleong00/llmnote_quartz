---
updated: 2026-03-06T18:48
---
## 局限性与未来方向

> [!warning] 当前局限
>
> **1. 单层分支限制**
> - 当前实现禁止嵌套分支（分支内不能再创建分支）
> - 对于需要递归分解的复杂任务可能不够灵活
>
> **2. 依赖外部判断器**
> - Out-of-scope penalty 需要 GPT-5-nano 判断分支是否越界
> - 增加训练成本和复杂度
>
> **3. 固定分支数量限制**
> - 最多 10 个分支的硬限制
> - 对于需要更多子任务分解的场景可能不足
>
> **4. 上下文压缩的信息损失**
> - 折叠后只保留结果摘要，中间推理细节丢失
> - 如果后续需要回溯分支内的细节，无法恢复

**未来方向**：
- 支持嵌套分支（hierarchical context folding）
- 自动学习 out-of-scope 判断（无需外部模型）
- 动态分支数量限制
- 可选的分支历史保留机制（trade-off 压缩率和可回溯性）

## 核心贡献

1. **Context-Folding 机制**：通过 branch-fold 操作让 Agent 主动管理工作上下文，实现 90% 以上压缩率

2. **FoldGRPO 训练框架**：基于 [[GRPO]] 扩展，引入 token-level process rewards 引导正确的分支行为

3. **长时域任务突破**：在 BrowseComp-Plus 和 [[SWE-Bench]] Verified 上显著超越长上下文 ReAct 和 summarization 基线

4. **效率与性能兼得**：32K 工作上下文处理 300K+ tokens 交互，同时保持高准确率

## 相关概念

**前置知识**：
- [[GRPO]] — FoldGRPO 的基础算法
- [[Agentic RL]] — Agent 训练的 RL 范式
- [[ReAct]] — 传统 Agent 框架（Context-Folding 的对比基线）

**相关方法**：
- [[Tool Use]] — Agent 的核心能力
- [[Long-Context Modeling]] — 长上下文建模的替代方案

**评估基准**：
- [[SWE-Bench]] — Agentic coding 评估标准

## 延伸阅读

**原始论文**：
- [[Clippings/Paper/2510.11967/2510.11967|Context-Folding 论文全文]]

**相关工作**：
- [[GLM-5 (2026)]] — 提出 Hierarchical Context Management 解决类似问题
- [[Kimi K2 (2025)]] — 在 SWE-Bench 上的 SOTA agentic 性能
- [[GiGPO (2025)]] — 另一种解决 multi-turn agent 信用分配的方法
