---
type: concept
description: 将原始文本转换为离散 token ID 序列的过程，是 LLM 理解和生成语言的第一步，vocabulary 设计直接影响模型的多语言能力、数学推理和训练效率
aliases:
  - 分词
  - Tokenizer
  - 分词器
prerequisites:
  - "[[Transformer]]"
tags:
  - foundations
  - architecture
  - pretraining
created: 2026-02-25
updated: 2026-03-01T22:34
---

# Tokenization

Tokenization（分词）是将原始文本映射为离散 token ID 序列的过程。LLM 不能直接处理字符串——它需要一个固定大小的词表（vocabulary），把每段文本拆成词表中的 token，再通过 [[Embedding]] 层转换为连续向量送入 [[Transformer]]。这个看似简单的预处理步骤，实际上深刻影响着模型的能力边界：数学推理、多语言效率、代码理解，都与 tokenizer 的设计密切相关。

## 动机：为什么不直接用字符或单词？

在设计 tokenizer 之前，先理解两个极端方案的问题：

**字符级（character-level）**：词表极小（几百个字符），但序列极长。"transformer" 变成 11 个 token，模型需要跨越很多步才能捕捉词义，计算成本高且语义碎片化。

**单词级（word-level）**：语义完整，但词表爆炸（英语几十万词），且无法处理未登录词（OOV）。"unhappiness" 如果不在词表中就完全无法表示。

> [!intuition] 核心权衡
> Tokenization 的本质是在**词表大小**和**序列长度**之间找平衡。子词（subword）方法通过把常见词保留为整体、罕见词拆成更小的片段，同时解决了 OOV 问题和序列长度问题。

## 核心算法

### BPE (Byte Pair Encoding)

BPE 是目前最主流的 tokenization 算法，GPT 系列、Llama 系列都采用它。核心思想是**自底向上合并**：从最小单元开始，反复合并最频繁的相邻对。

#### 训练阶段详细步骤

假设训练语料：
```
"low" 出现 5 次
"lower" 出现 2 次
"newest" 出现 6 次
"widest" 出现 3 次
```

**步骤 1：初始化**

将每个单词拆成字符序列，并在末尾加上特殊标记 `</w>` 表示词尾：

```
l o w </w>      : 5
l o w e r </w>  : 2
n e w e s t </w>: 6
w i d e s t </w>: 3
```

初始词表：`{l, o, w, e, r, n, s, t, i, d, </w>}`

**步骤 2：统计所有相邻 token 对的频率**

```
(e, s): 6 + 3 = 9
(s, t): 6 + 3 = 9
(t, </w>): 6 + 3 = 9
(e, w): 6 + 2 = 8
(l, o): 5 + 2 = 7
(o, w): 5 + 2 = 7
(w, </w>): 5
...
```

**步骤 3：合并频率最高的 pair**

假设 `(e, s)` 频率最高（9 次），合并为 `es`：

```
l o w </w>      : 5
l o w e r </w>  : 2
n e w es t </w> : 6  ← 合并
w i d es t </w> : 3  ← 合并
```

词表更新：`{l, o, w, e, r, n, s, t, i, d, </w>, es}`

**步骤 4：重复迭代**

- 第 2 轮：`(es, t)` 合并为 `est`
- 第 3 轮：`(l, o)` 合并为 `lo`
- 第 4 轮：`(lo, w)` 合并为 `low`
- ...持续迭代直到词表达到目标大小（如 50,000）

#### 编码阶段

给定新文本 "lowest"，按训练时的合并顺序应用规则：

```
初始: l o w e s t
第 1 轮: l o w es t     (应用 e+s → es)
第 2 轮: l o w est      (应用 es+t → est)
第 3 轮: lo w est       (应用 l+o → lo)
第 4 轮: low est        (应用 lo+w → low)
```

最终编码：`[low, est]`

> [!warning] 关键点
> 必须按训练时的合并顺序（优先级）应用规则，而非贪心选择最长匹配。合并顺序决定了 tokenization 的结果。

#### 解码阶段

直接拼接 token 对应的文本片段：`[low, est]` → "lowest"，无损还原。

### WordPiece

WordPiece 与 BPE 结构相似，但合并策略不同。BPE 选择**频率最高**的 pair，WordPiece 选择**合并后语言模型似然增益最大**的 pair：

$$\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \cdot \text{freq}(y)}$$

这意味着 WordPiece 倾向于合并那些"经常一起出现但各自不太常见"的 pair，而非单纯的高频 pair。BERT 使用 WordPiece。

> [!comparison] BPE vs WordPiece
> - BPE：纯频率驱动，简单高效，GPT/Llama 的选择
> - WordPiece：似然驱动，更倾向于保留有语义意义的子词，BERT 的选择
> - 实践中两者差异不大，BPE 因为简单而成为主流

### Unigram Language Model

与 BPE 的自底向上不同，Unigram 采用**自顶向下剪枝**策略：

1. 初始化一个很大的候选词表（包含所有常见子串）
2. 为每个 token 分配概率，构建 unigram 语言模型：$P(x) = \prod_{i=1}^{n} P(x_i)$
3. 用 EM 算法优化 token 概率
4. 移除对整体似然贡献最小的 token
5. 重复步骤 3-4，直到词表缩减到目标大小

**编码时**，Unigram 使用 Viterbi 算法找到概率最高的分词方式，而非 BPE 的贪心合并。这意味着同一个词可能有多种分词方式，模型选择概率最优的那种。

> [!intuition] Unigram 的独特优势
> Unigram 天然支持**多种分词方式的采样**（subword regularization），训练时可以对同一句话用不同的分词方式，起到数据增强的效果，提升模型鲁棒性。

### SentencePiece：语言无关的框架

SentencePiece 不是一种分词算法，而是一个**预处理框架**。它的关键设计是：

- **将空格视为普通字符**（用 `▁` 替代），不依赖语言特定的预分词规则
- 支持 BPE 和 Unigram 两种算法
- 直接从原始文本训练，不需要预先分词——这对中文、日文等没有空格分隔的语言至关重要

Llama、T5 等模型使用 SentencePiece + BPE 的组合。

## Byte-level BPE

从 GPT-2 开始，OpenAI 引入了 Byte-level BPE：不以字符为基础单元，而以 **byte（256 个）** 为基础单元。

> [!intuition] 为什么用 byte？
> 字符级 BPE 需要一个 Unicode 基础词表（几千个字符），且遇到罕见 Unicode 字符仍会 OOV。Byte-level 的基础词表只有 256 个，**理论上可以编码任何文本**，彻底消除 OOV 问题。

### 训练阶段详细步骤

假设训练语料包含 "hello"：

**步骤 1：UTF-8 编码**

```
"hello" → UTF-8 bytes: [0x68, 0x65, 0x6C, 0x6C, 0x6F]
```

**步骤 2：Byte 到可打印字符的映射**

为了避免控制字符（0x00-0x1F）和特殊字符（0x7F-0x9F）在显示时出问题，GPT-2 使用了一个映射表，将 256 个 byte 映射到 256 个可打印的 Unicode 字符：

```python
def bytes_to_unicode():
    """
    构建 byte → unicode 映射
    - 可打印 ASCII (33-126) 直接映射
    - 扩展拉丁字符 (161-172, 174-255) 直接映射
    - 其他 byte 映射到私有区域
    """
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
```

假设映射后：
```
0x68 → 'h'
0x65 → 'e'
0x6C → 'l'
0x6F → 'o'
```

**步骤 3：BPE 合并**

初始序列：`h e l l o`

统计 pair 频率并合并（与字符级 BPE 相同）：
```
(l, l): 1 → 合并为 'll'
结果: h e ll o

(e, ll): 1 → 合并为 'ell'
结果: h ell o
```

最终可能得到：`[h, ell, o]` 或 `[hello]`（取决于训练数据和词表大小）。

### 编码阶段

给定新文本 "hello"：

1. **UTF-8 编码**：`[0x68, 0x65, 0x6C, 0x6C, 0x6F]`
2. **Byte-to-unicode 映射**：`h e l l o`
3. **应用 BPE 合并规则**：`h ell o`（假设训练时学到了这个合并）
4. **转换为 token ID**：`[1234, 5678, 9012]`（假设这些是词表中的 ID）

### 解码阶段

1. **Token ID 转回字符序列**：`[1234, 5678, 9012]` → `h ell o`
2. **Unicode-to-byte 逆映射**：`[0x68, 0x65, 0x6C, 0x6C, 0x6F]`
3. **UTF-8 解码**：`"hello"`

### 多语言示例

中文 "你好"：

1. **UTF-8 编码**：`[0xE4, 0xBD, 0xA0, 0xE5, 0xA5, 0xBD]`（每个汉字 3 bytes）
2. **Byte-to-unicode 映射**：假设映射为 `ä ½   å ¥ ½`（6 个字符）
3. **BPE 合并**：
   - 如果训练数据中中文常见，可能合并为 `你好`（单个 token）
   - 如果中文罕见，可能保持为 6 个 byte-level token

### 字符级 vs Byte-level 对比

| 特性 | 字符级 BPE | Byte-level BPE |
|------|-----------|----------------|
| 基础单元 | Unicode 字符（几千个） | Byte（256 个） |
| OOV 问题 | 罕见字符会 OOV | 理论上无 OOV |
| 序列长度 | 较短 | 初始较长，合并后接近 |
| 多语言 | 需要大基础词表 | 统一处理所有语言 |
| 实现复杂度 | 简单 | 需要 byte-unicode 映射 |

**代价**：byte 粒度更细，序列更长。但通过 BPE 合并，常见的多 byte 序列会被合并为单个 token，实际序列长度与字符级 BPE 接近。

**纯 byte-level 模型**（不做 BPE 合并）面临序列过长的挑战。[[Multi-Token Prediction]] 对此有显著帮助——8-byte prediction 配合 self-speculative decoding 可实现 6.4 倍推理加速，完全补偿 byte-level 序列更长的开销。

## Vocabulary Size 的权衡

词表大小是 tokenizer 设计中最重要的超参数之一：

| 词表太大 | 词表太小 |
|----------|----------|
| Embedding 参数量大（vocab × hidden_dim） | 序列过长，计算成本高 |
| 低频 token 训练不充分，embedding 质量差 | 语义碎片化，模型难以学习 |
| LM Head 输出层计算量大 | 多语言 token 效率低 |

**主流模型的词表大小**：

| 模型      | 词表大小     | 算法                        | 备注               |
| ------- | -------- | ------------------------- | ---------------- |
| GPT-2   | 50,257   | Byte-level BPE            | 开创 byte-level 方案 |
| GPT-4   | ~100,000 | Byte-level BPE (cl100k)   | 更大词表提升多语言效率      |
| Llama 2 | 32,000   | SentencePiece BPE         | 相对保守             |
| Llama 3 | 128,256  | Byte-level BPE (tiktoken) | 大幅扩展，提升多语言和代码    |
| BERT    | 30,522   | WordPiece                 | 经典选择             |

> [!intuition] 词表大小的趋势
> 近年来主流模型倾向于更大的词表（100k+），因为：
> 1. 大词表缩短序列长度，在长上下文场景下节省计算
> 2. 多语言和代码场景需要更多 token 覆盖
> 3. 模型参数量已经很大，embedding 层的额外开销相对可接受

## Tokenization 对模型能力的影响

Tokenizer 不只是预处理工具，它直接塑造了模型的能力边界。

### 数学与数字推理

早期 tokenizer 对数字的处理很不一致：`"123"` 可能被编码为一个 token，`"1234"` 却被拆成 `"123" + "4"`。这导致模型难以学习数字的位值关系和算术规律。

> [!warning] 数字 tokenization 的陷阱
> - `"380"` 和 `"381"` 可能有完全不同的 token 分解
> - 模型看到的不是"数字"，而是"子词片段"
> - 这是 LLM 算术能力弱的重要原因之一
>
> 改进方案：逐位分词（每个数字一个 token）、固定位数分词

### 多语言效率

不同语言的 token 效率差异巨大。以 GPT-2 tokenizer 为例，表达相同语义：

- 英语：~1 token/word
- 中文：~2-3 tokens/字（因为中文字符需要多个 byte）
- 日文：更高

这意味着中文用户在相同 context window 下能输入的信息量更少，推理成本也更高。GPT-4 和 Llama 3 通过扩大词表、增加中文 token 来缓解这个问题。

### 代码处理

代码中的空格和缩进对语义至关重要（尤其是 Python），但朴素的 tokenizer 可能把每个空格都编码为独立 token。现代 tokenizer 通常会将连续空格合并为单个 token（如 `"    "` 四个空格 → 一个 token），大幅提升代码场景的效率。

## 现代工具

| 工具 | 开发者 | 特点 |
|------|--------|------|
| **tiktoken** | OpenAI | GPT 系列使用，Rust 实现，极快 |
| **tokenizers** | HuggingFace | 支持 BPE/WordPiece/Unigram，Rust 核心 |
| **SentencePiece** | Google | C++ 实现，语言无关，Llama/T5 使用 |

三者都提供 Python 接口，训练自定义 tokenizer 通常只需几行代码和一份语料。

## 局限性

> [!warning] Tokenization 的根本局限
> - **信息瓶颈**：tokenization 是有损的——不同的文本可能映射到相同的 token 序列（尤其在 byte-level 方案中罕见，但 subword 级别存在歧义）
> - **训练后固定**：tokenizer 一旦训练完成就不再更新，无法适应新词汇（如新出现的术语）。模型只能通过子词组合来"拼凑"新词
> - **语言偏见**：tokenizer 在哪种语言的语料上训练，就对哪种语言更高效。这是多语言模型的系统性不公平来源
> - **与下游任务的耦合**：tokenizer 的选择在预训练前就固定了，但它影响所有下游任务的表现。换 tokenizer 意味着重新预训练

> [!interview] 面试要点
> **Q: 为什么 LLM 做算术不好？和 tokenization 有什么关系？**
> A: 数字被 tokenizer 拆分成不一致的子词片段，模型看到的是"子词"而非"数字"。`"380"` 和 `"381"` 可能有完全不同的 token 分解，模型难以学习位值关系。改进方案包括逐位分词和专门的数字 token。
>
> **Q: BPE 和 WordPiece 的核心区别？**
> A: 合并策略不同。BPE 选频率最高的 pair，WordPiece 选似然增益最大的 pair。WordPiece 更倾向于保留语义子词，但实践中差异不大，BPE 因简单而成为主流。
>
> **Q: 为什么 Llama 3 把词表从 32k 扩到 128k？**
> A: 三个原因：大词表缩短序列长度（省计算）、提升多语言 token 效率（中文等语言受益大）、更好的代码处理。代价是 embedding 层参数增加，但相对于模型总参数量可接受。
