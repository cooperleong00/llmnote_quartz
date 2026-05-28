---
description: Inference 优化导航：KV Cache、attention 架构、serving 调度、推理加速与量化
type: moc
tags:
  - inference
  - optimization
  - serving
created: 2026-01-27
updated: 2026-05-29T00:00
---

# MOC - Inference

Inference 优化关注如何在给定延迟、吞吐、成本和质量约束下运行 LLM。核心问题包括 KV Cache 显存、prefill/decode 调度、attention 计算、量化、长上下文和 reasoning token 开销。

---

## 概览

```
LLM Inference
├── KV Cache 与内存管理
│   ├── [[KV Cache]] — 自回归生成缓存
│   ├── [[Paged Attention]] — 分页管理 KV block
│   └── [[Radix Attention]] — 前缀树复用公共前缀
│
├── 架构层减负
│   ├── [[Multi-Query Attention]] — 共享 K/V
│   ├── [[Grouped-Query Attention]] — 分组共享 K/V
│   └── [[Multi-head Latent Attention]] — latent KV 压缩
│
├── 计算与解码加速
│   ├── [[Flash Attention]] — attention kernel 优化
│   ├── [[Speculative Decoding]] — draft/target 并行验证
│   └── [[Multi-Token Prediction]] — 同时预测多个未来 token
│
├── Serving 调度
│   └── [[Continuous Batching]] — 迭代级动态 batching
│
└── 长上下文与压缩
    ├── [[Sliding Window Attention]] / [[Length Extrapolation]]
    ├── [[Context Compression]]
    └── [[MEMENTO]]
```

---

## KV Cache 与内存管理

[[KV Cache]] 是自回归推理的核心优化：历史 token 的 Key/Value 不再重复计算，decode 阶段只处理新增 token。它把计算节省转化为显存压力，尤其在长上下文、多并发和多 head 架构下显著。

| 问题 | 表现 | 解决路径 |
|------|------|----------|
| KV Cache 占用大 | 长上下文和高并发迅速吃满显存 | [[Multi-Query Attention]], [[Grouped-Query Attention]], [[Multi-head Latent Attention]] |
| 内存碎片 | 请求长度不同导致 block 利用率低 | [[Paged Attention]] |
| 前缀重复 | system prompt、few-shot 示例跨请求重复 | [[Radix Attention]] |

[[Paged Attention]] 借鉴虚拟内存分页，将 KV Cache 切成固定 block 并按需分配。[[Radix Attention]] 用 radix tree 管理前缀缓存，适合多轮对话、结构化 generation 和共享 prompt 场景。

---

## 架构层减负

推理效率可以在模型架构里提前设计。MQA、GQA、MLA 都通过减少或压缩 K/V 来降低 decode 阶段显存带宽压力。

| 方法 | K/V 处理方式 | 典型取舍 |
|------|--------------|----------|
| [[Multi-Query Attention]] | 所有 query head 共享同一组 K/V | 显存最省，质量可能受影响 |
| [[Grouped-Query Attention]] | query head 分组共享 K/V | 质量和效率平衡 |
| [[Multi-head Latent Attention]] | 将 K/V 压缩到 latent 表示 | 压缩更激进，实现更复杂 |

[[Mixture of Experts]] 也属于架构层效率设计。它让每个 token 只激活部分 expert，降低每 token 计算量；对应的训练和并行问题见 [[MOC - Distributed Training]]。

---

## 计算与解码加速

### Flash Attention

[[Flash Attention]] 通过 IO-aware kernel 设计减少 HBM 访问，在 prefill 阶段尤其重要。对于长 prompt，prefill 的 attention 计算可能成为主要瓶颈。

### Speculative Decoding

[[Speculative Decoding]] 用小模型快速生成候选 token，再由大模型并行验证。它在数学上保持目标模型输出分布一致，收益取决于 draft model 速度、接受率和系统实现。

[[Multi-Token Prediction]] 让模型训练时同时预测多个未来 token，可用于推理中的多 token proposal，与 speculative decoding 的加速目标相连。

---

## Serving 调度

[[Continuous Batching]] 将调度粒度从请求级变为 iteration 级：某个序列结束后可以立刻退出 batch，新请求可以随时加入。这解决了静态 batching 中短请求等待长请求的问题。

Serving 系统通常还会区分 prefill 和 decode 两类负载：

| 阶段 | 主要瓶颈 | 常见优化 |
|------|----------|----------|
| Prefill | 大矩阵计算与长 prompt attention | [[Flash Attention]]、chunked prefill、prefix cache |
| Decode | KV Cache 读写和小 batch 低利用率 | [[Continuous Batching]]、KV cache 管理、speculative decoding |

---

## 量化与压缩

[[Quantization]] 将权重或激活映射到低精度表示，降低显存和带宽需求。推理中最常见的是 INT8/INT4 权重量化；训练和后训练阶段还会出现 FP8、QAT、LoRA + quantization 等组合。

[[Context Compression]] 和 [[MEMENTO]] 从输入或推理链长度入手降低 token 开销。前者常用于 RAG，把检索上下文压缩到更短的 evidence；后者把长推理链分段摘要化，并结合 KV Cache 遮蔽减少峰值显存。

---

## 长上下文推理

长上下文推理同时挑战位置编码、attention 复杂度和 KV Cache 显存：

- [[Length Extrapolation]] 处理超出训练长度后的性能退化。
- [[Sliding Window Attention]] 用局部窗口把 attention 成本从 $O(n^2)$ 降到 $O(nw)$。
- [[Attention Sink]] 解释局部或稀疏 attention 中部分 token 吸收过多权重的现象。
- [[Gated Attention]] 用 gating 改善 attention sink 和训练稳定性。

---

## 推理框架视角

| 框架/方向 | 核心机制 | 适合关注 |
|-----------|----------|----------|
| vLLM | [[Paged Attention]] | 高吞吐 serving、KV block 管理 |
| SGLang | [[Radix Attention]] | 前缀复用、结构化调用 |
| llama.cpp | 量化与本地部署 | 低资源推理 |
| TensorRT-LLM | kernel 与图优化 | NVIDIA GPU 极致性能 |

---

## 学习路径建议

**基础路线**：
1. [[KV Cache]] — 先理解自回归推理为什么需要缓存。
2. [[Continuous Batching]] — 理解 serving 调度如何提升吞吐。
3. [[Quantization]] — 理解显存和带宽如何被低精度缓解。

**架构效率路线**：
1. [[Multi-Query Attention]] → [[Grouped-Query Attention]] → [[Multi-head Latent Attention]] — 理解 K/V 压缩。
2. [[Flash Attention]] — 理解 attention kernel 的 IO 瓶颈。
3. [[Speculative Decoding]] → [[Multi-Token Prediction]] — 理解解码加速。

**长上下文路线**：
1. [[Length Extrapolation]] → [[Sliding Window Attention]] — 理解长度和复杂度挑战。
2. [[Paged Attention]] → [[Radix Attention]] — 理解服务层缓存管理。
3. [[Context Compression]] → [[MEMENTO]] — 理解压缩式长推理。

---

## 相关 MOC

- [[MOC - Attention]] — Attention 架构、KV Cache 与高效注意力
- [[MOC - Distributed Training]] — 显存计算、并行策略与训练系统
- [[MOC - RAG]] — 检索上下文压缩与知识增强生成
- [[MOC - Post-training]] — reasoning model 带来的推理 token 与 serving 压力
