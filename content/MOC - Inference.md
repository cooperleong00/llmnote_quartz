---
description: Inference 优化导航：内存管理、计算加速、Serving 系统
type: moc
tags:
  - inference
  - optimization
  - serving
created: 2026-01-27
updated: 2026-01-31T22:42
---

# MOC - Inference

推理优化（Inference Optimization）是 LLM 落地的关键环节。本 MOC 组织推理相关的所有概念，包括内存优化、计算加速和 Serving 系统。

---

## 概览

```
LLM Inference 优化
├── 内存优化
│   ├── [[KV Cache]] — 避免重复计算 ✅
│   ├── [[Paged Attention]] — 分页内存管理 (vLLM) ✅
│   └── [[Radix Attention]] — 前缀共享 (SGLang) ✅
│
├── 计算优化
│   ├── [[Flash Attention]] — IO-aware 算法 ✅
│   ├── [[Speculative Decoding]] — 投机解码
│   └── [[Mixture of Experts]] — MoE 稀疏激活
│
├── Serving 优化
│   ├── [[Continuous Batching]] — 连续批处理
│   ├── [[Prefix Caching]] — 前缀缓存
│   └── [[Chunked Prefill]] — 分块预填充
│
└── 量化与压缩
    ├── [[Quantization]] — 量化基础
    ├── [[GPTQ]] — 训练后量化
    └── [[AWQ]] — 激活感知量化
```

> 注：✅ 表示已有笔记，其余为待创建

---

## 内存优化

### KV Cache 管理

自回归生成的核心优化是 [[KV Cache]]，避免每次生成都重新计算历史 token 的 Key/Value。但 KV Cache 带来新的挑战：

| 问题 | 描述 | 解决方案 |
|------|------|----------|
| 内存占用大 | 长序列 KV Cache 占用大量显存 | [[Grouped-Query Attention\|GQA]] 减少 KV heads |
| 内存碎片 | 预分配导致浪费 | [[Paged Attention]] 分页管理 |
| 重复存储 | 相同前缀重复缓存 | [[Radix Attention]] 前缀共享 |

### Paged Attention (vLLM)

[[Paged Attention]] 借鉴操作系统的虚拟内存思想：
- 将 KV Cache 分成固定大小的 block
- 按需分配，避免预分配浪费
- 支持 Copy-on-Write，高效处理 beam search

### Radix Attention (SGLang)

[[Radix Attention]] 使用 Radix Tree 管理 KV Cache：
- 自动识别和共享公共前缀
- 跨请求复用，适合多轮对话和批量推理
- LRU 淘汰策略管理缓存

---

## 计算优化

### Flash Attention

[[Flash Attention]] 解决 Attention 计算的 HBM 带宽瓶颈：
- **核心技术**：Tiling + Online Softmax + Recomputation
- **效果**：2-4x 加速，内存从 $O(N^2)$ 降到 $O(N)$
- **特点**：精确算法，结果与标准 Attention 完全相同

### Speculative Decoding

[[Speculative Decoding]]（投机解码）通过小模型加速大模型推理：
- 小模型（draft model）快速生成多个候选 token
- 大模型（target model）并行验证
- 数学上保证输出分布与原模型一致

### Mixture of Experts

[[Mixture of Experts]]（MoE）通过稀疏激活提升效率：
- 每个 token 只激活部分专家（如 8 选 2）
- 参数量大但计算量可控
- 代表模型：Mixtral、DeepSeek-V2

---

## Serving 优化

### Continuous Batching

[[Continuous Batching]]（连续批处理）解决传统静态 batching 的效率问题：
- 传统方式：等最长序列完成，短序列空等
- 连续批处理：序列完成即退出，新请求随时加入
- 显著提升吞吐量和 GPU 利用率

### 其他 Serving 技术

- [[Prefix Caching]] — 缓存常用前缀（system prompt）
- [[Chunked Prefill]] — 分块处理长 prompt，避免阻塞
- [[Disaggregated Serving]] — Prefill 和 Decode 分离部署

---

## 量化与压缩

量化是降低推理成本的重要手段：

| 方法 | 类型 | 精度 | 特点 |
|------|------|------|------|
| [[Quantization]] | 基础 | INT8/INT4 | 基本概念 |
| [[GPTQ]] | PTQ | 4-bit | 基于 Hessian 的权重量化 |
| [[AWQ]] | PTQ | 4-bit | 保护重要权重通道 |
| [[SmoothQuant]] | PTQ | INT8 | 平滑激活分布 |
| [[QLoRA]] | QAT | 4-bit | 量化 + LoRA 微调 |

---

## 推理框架对比

| 框架 | 核心技术 | 特点 |
|------|----------|------|
| vLLM | [[Paged Attention]] | 高吞吐，内存高效 |
| SGLang | [[Radix Attention]] | 前缀共享，结构化生成 |
| TensorRT-LLM | 图优化 + 量化 | NVIDIA 官方，极致性能 |
| llama.cpp | CPU 推理 + 量化 | 轻量级，本地部署 |

---

## 学习路径建议

**入门路线**：
1. [[KV Cache]] — 理解自回归推理的核心优化
2. [[Flash Attention]] — 理解计算层优化
3. [[Continuous Batching]] — 理解 Serving 基础

**进阶路线**：
1. [[Paged Attention]] / [[Radix Attention]] — 内存管理进阶
2. [[Speculative Decoding]] — 加速技术
3. [[Quantization]] — 量化基础与实践

**面试重点**：
- KV Cache 的内存占用计算
- Flash Attention 的核心思想
- Continuous Batching vs Static Batching
- 常见量化方法对比

---

## 相关 MOC

- [[MOC - Attention]] — Attention 机制
- [[MOC - Foundations]] — Transformer 基础
- [[MOC - Post-training]] — 后训练方法
