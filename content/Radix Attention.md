---
description: SGLang 的 Radix Tree KV Cache 管理，实现多请求间的自动前缀共享
type: method
prerequisites:
  - "[[KV Cache]]"
  - "[[Paged Attention]]"
tags:
  - inference
  - attention
  - optimization
  - serving
created: 2025-01-26
updated: 2026-01-31T22:42
---

# Radix Attention

Radix Attention 是 SGLang 提出的 KV Cache 管理方法，使用 Radix Tree（基数树）数据结构实现多请求间的自动前缀共享，特别适合有大量共享前缀的场景。

---

## 动机：前缀共享问题

> [!intuition] 直觉理解
> 很多 LLM 应用中，不同请求有相同的前缀（如 system prompt、few-shot examples）。传统方法为每个请求独立存储 KV Cache，造成大量重复。Radix Attention 自动识别和共享这些公共前缀。

### 典型场景

**场景 1：System Prompt**
```
请求 A: [System: You are a helpful assistant...][User: What is 2+2?]
请求 B: [System: You are a helpful assistant...][User: Tell me a joke]
请求 C: [System: You are a helpful assistant...][User: Write code for...]
         ↑ 相同的 system prompt，重复存储 3 次
```

**场景 2：Few-shot Learning**
```
请求 A: [Example 1][Example 2][Example 3][Query A]
请求 B: [Example 1][Example 2][Example 3][Query B]
         ↑ 相同的 few-shot examples
```

**场景 3：Multi-turn Conversation**
```
Turn 1: [System][User 1][Assistant 1]
Turn 2: [System][User 1][Assistant 1][User 2][Assistant 2]
Turn 3: [System][User 1][Assistant 1][User 2][Assistant 2][User 3]
         ↑ 每轮都包含之前的历史
```

---

## 核心数据结构：Radix Tree

### Radix Tree 简介

Radix Tree（基数树，也叫 Patricia Trie）是一种压缩的前缀树：
- 每个节点存储一个**字符串片段**（而非单个字符）
- 共享前缀的字符串共享路径
- 空间效率高于普通 Trie

### KV Cache 的 Radix Tree 组织

```
                    [Root]
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
    [System Prompt]      [Other Prefix]
      (KV Block 0-2)
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
 [User A] [User B] [User C]
 (Block 3) (Block 4) (Block 5)
```

**节点内容**：
- Token 序列的哈希（用于匹配）
- 对应的 KV Cache Block 引用
- 引用计数（用于 LRU 淘汰）

---

## 工作流程

### 请求到达时

```python
def process_request(tokens):
    # 1. 在 Radix Tree 中查找最长匹配前缀
    matched_prefix, matched_blocks = radix_tree.prefix_match(tokens)

    # 2. 复用匹配的 KV Cache
    kv_cache = matched_blocks  # 无需重新计算

    # 3. 只计算未匹配部分的 KV
    new_tokens = tokens[len(matched_prefix):]
    new_kv = compute_kv(new_tokens)

    # 4. 将新计算的 KV 插入 Radix Tree
    radix_tree.insert(tokens, kv_cache + new_kv)

    return kv_cache + new_kv
```

### 示例

```
初始状态: 空树

请求 A: "System prompt | Query A"
→ 无匹配，计算全部 KV，插入树
树结构: [System prompt] → [Query A]

请求 B: "System prompt | Query B"
→ 匹配 "System prompt"，复用其 KV
→ 只计算 "Query B" 的 KV
树结构: [System prompt] → [Query A]
                       → [Query B]

请求 C: "System prompt | Query A | Follow-up"
→ 匹配 "System prompt | Query A"
→ 只计算 "Follow-up" 的 KV
树结构: [System prompt] → [Query A] → [Follow-up]
                       → [Query B]
```

---

## 缓存管理

### LRU 淘汰策略

当内存不足时，使用 LRU（Least Recently Used）淘汰：

```
每个节点维护:
- last_access_time: 最后访问时间
- ref_count: 引用计数

淘汰策略:
1. 从叶子节点开始（没有子节点依赖）
2. 选择 last_access_time 最早的节点
3. 释放其 KV Cache Block
4. 从树中删除该节点
```

### 引用计数

```
请求 A 使用 [System prompt]:
  ref_count([System prompt]) = 1

请求 B 也使用 [System prompt]:
  ref_count([System prompt]) = 2

请求 A 结束:
  ref_count([System prompt]) = 1  # 不能淘汰

请求 B 结束:
  ref_count([System prompt]) = 0  # 可以淘汰（但可能保留用于未来请求）
```

---

## 与 Paged Attention 的对比

> [!comparison] 对比
> | 特性 | Paged Attention | Radix Attention |
> |------|-----------------|-----------------|
> | **核心目标** | 减少内存碎片 | 跨请求共享前缀 |
> | **数据结构** | Block Table | Radix Tree |
> | **内存分配** | 按需分配 Block | 按需分配 + 前缀复用 |
> | **适用场景** | 通用 | 大量共享前缀 |
> | **代表系统** | vLLM | SGLang |
>
> **关系**：Radix Attention 可以与 Paged Attention 结合使用——用 Paged Attention 管理单个请求的内存分配，用 Radix Tree 管理跨请求的前缀共享。

### 性能对比

| 场景 | vLLM (Paged) | SGLang (Radix) |
|------|--------------|----------------|
| 无共享前缀 | 相当 | 相当 |
| 共享 system prompt | 1x | 1.5-2x |
| Few-shot (长前缀) | 1x | 3-5x |
| Multi-turn 对话 | 1x | 2-4x |

---

## 实现细节

### Token 序列的哈希

为了高效匹配，使用 rolling hash：

```python
def compute_hash(tokens):
    # 增量计算，O(1) 更新
    hash_value = 0
    for token in tokens:
        hash_value = (hash_value * BASE + token) % MOD
    return hash_value
```

### 并发控制

多个请求可能同时访问/修改 Radix Tree：
- 读操作：无锁并发
- 写操作：细粒度锁（per-node）
- 淘汰操作：后台异步执行

---

## 面试要点

> [!interview] 面试要点
> **Q: Radix Attention 解决什么问题？**
> A: 多请求间的 KV Cache 前缀共享问题。当多个请求有相同前缀（如 system prompt）时，传统方法重复存储和计算，Radix Attention 自动识别并复用。
>
> **Q: 为什么用 Radix Tree 而不是普通 Trie？**
> A: Radix Tree 是压缩的 Trie，将只有一个子节点的路径压缩成单个节点，空间效率更高，查找也更快。
>
> **Q: Radix Attention 和 Paged Attention 的关系？**
> A: 两者解决不同问题：Paged Attention 解决单请求内的内存碎片，Radix Attention 解决跨请求的前缀共享。可以结合使用。
>
> **Q: 什么场景下 Radix Attention 收益最大？**
> A: 大量请求共享长前缀的场景：(1) 固定 system prompt；(2) Few-shot learning；(3) Multi-turn 对话；(4) 批量处理相似请求。

---

## 参考资料

- SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2024)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang Blog: RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
