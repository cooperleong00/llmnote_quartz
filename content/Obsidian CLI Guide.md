---
updated: 2026-03-03T16:57
---
# Obsidian CLI Guide

> **官方文档**: https://help.obsidian.md/cli

本文档说明如何使用 Obsidian CLI 进行所有 vault 交互操作。

---

## 核心原则

**何时使用 Obsidian CLI**：

1. **搜索笔记内容** → `obsidian search` 或 `obsidian search:context`
2. **读取笔记** → `obsidian read`
3. **创建笔记** → `obsidian create`
4. **追加内容** → `obsidian append`
5. **管理 properties** → `obsidian property:*` 系列命令
6. **查询 backlinks** → `obsidian backlinks`
7. **查询 tags** → `obsidian tags`
8. **列出文件** → `obsidian files`

**何时使用传统工具**：

- **结构化搜索**（如搜索标题）→ `rg "^#" file.md`
- **批量文件操作** → `rg` + `find`
- **复杂文本处理** → 传统 Unix 工具

**通用规则**：
- 使用 `format=json` 获取可解析的输出
- 使用 `file=<name>` 按文件名匹配（类似 wikilink）
- 使用 `path=<path>` 按精确路径匹配
- 带空格的值需要引号：`name="My Note"`

---

## 常用命令速查

### 1. 搜索（Search）

**基础搜索**（返回匹配文件列表）：
```bash
obsidian search query="关键词" format=json
```

**上下文搜索**（返回匹配行及上下文）：
```bash
obsidian search:context query="关键词" format=json
```

**限制搜索范围**：
```bash
# 限制到特定文件夹
obsidian search query="DPO" path="Clippings/Article" format=json

# 限制返回数量
obsidian search query="RLHF" limit=10 format=json

# 大小写敏感
obsidian search query="PPO" case format=json
```

**返回匹配数量**：
```bash
obsidian search query="post-training" total
```

### 2. 读取（Read）

**读取笔记内容**：
```bash
# 按文件名（wikilink 风格）
obsidian read file="DPO"

# 按精确路径
obsidian read path="DPO.md"
```

### 3. 创建（Create）

**创建新笔记**：
```bash
# 基础创建
obsidian create name="New Note" content="Initial content"

# 使用模板
obsidian create name="New Note" template="Concept Template"

# 覆盖已存在文件
obsidian create name="Note" content="Content" overwrite

# 创建后打开
obsidian create name="Note" content="Content" open
```

### 4. 追加内容（Append）

**追加到笔记末尾**：
```bash
# 追加一行
obsidian append file="DPO" content="New paragraph"

# 追加不换行
obsidian append file="DPO" content="inline text" inline
```

**前置内容（Prepend）**：
```bash
obsidian prepend file="DPO" content="New first line"
```

### 5. Properties 管理

**列出所有 properties**：
```bash
# 列出 vault 中所有 properties
obsidian properties format=json

# 列出特定文件的 properties
obsidian properties file="DPO" format=yaml

# 统计某个 property 的使用次数
obsidian properties name="type" counts
```

**读取 property 值**：
```bash
obsidian property:read name="type" file="DPO"
```

**设置 property**：
```bash
# 设置文本
obsidian property:set name="description" value="DPO simplifies RLHF" file="DPO"

# 设置列表
obsidian property:set name="tags" value="post-training,rlhf" type=list file="DPO"

# 设置日期
obsidian property:set name="updated" value="2026-03-03" type=date file="DPO"
```

**删除 property**：
```bash
obsidian property:remove name="old_field" file="DPO"
```

### 6. Backlinks 查询

**列出反向链接**：
```bash
# 基础列表
obsidian backlinks file="DPO" format=json

# 包含链接次数
obsidian backlinks file="DPO" counts format=json

# 只返回数量
obsidian backlinks file="DPO" total
```

### 7. Tags 管理

**列出所有 tags**：
```bash
# 列出 vault 中所有 tags
obsidian tags format=json

# 包含使用次数
obsidian tags counts format=json

# 按使用次数排序
obsidian tags counts sort=count format=json

# 列出特定文件的 tags
obsidian tags file="DPO" format=json
```

**查询特定 tag**：
```bash
# 查看 tag 信息
obsidian tag name="post-training" verbose
```

### 8. 文件与文件夹操作

**列出文件**：
```bash
# 列出所有文件
obsidian files

# 限制到特定文件夹
obsidian files folder="Clippings/Paper"

# 按扩展名过滤
obsidian files ext=md

# 只返回数量
obsidian files total
```

**文件信息**：
```bash
obsidian file file="DPO"
```

**移动/重命名**：
```bash
# 移动文件
obsidian move file="Old Note" to="Archive/"

# 重命名
obsidian rename file="Old Name" name="New Name"
```

**删除文件**：
```bash
# 移到回收站
obsidian delete file="Temp Note"

# 永久删除
obsidian delete file="Temp Note" permanent
```

### 9. Links 分析

**列出出链**：
```bash
obsidian links file="DPO" total
```

**列出未解析链接**（broken links）：
```bash
# 列出所有未解析链接
obsidian unresolved format=json

# 包含链接次数
obsidian unresolved counts format=json

# 包含来源文件
obsidian unresolved verbose format=json
```

**列出孤立笔记**（orphans）：
```bash
obsidian orphans total
```

**列出死胡同**（dead ends，无出链）：
```bash
obsidian deadends total
```

### 10. Outline（大纲）

**查看笔记大纲**：
```bash
# 树形结构
obsidian outline file="DPO" format=tree

# Markdown 格式
obsidian outline file="DPO" format=md

# JSON 格式
obsidian outline file="DPO" format=json
```

---

## 输出格式（format 参数）

大多数命令支持 `format` 参数：

| 格式 | 说明 | 适用场景 |
|------|------|----------|
| `json` | JSON 格式，易于解析 | 需要程序化处理输出 |
| `tsv` | Tab 分隔 | 表格数据 |
| `csv` | 逗号分隔 | 导出到 Excel |
| `yaml` | YAML 格式 | Properties 输出 |
| `text` | 纯文本 | 人类阅读 |

### ⚠️ 重要：处理 CLI 警告信息

Obsidian CLI 会在输出前打印 2 行警告信息（关于更新 installer），这些警告会**混入 stdout**，导致 JSON 解析失败。

**解决方案：使用 `sed` 跳过前 2 行**

```bash
# ❌ 错误：jq 会因为警告信息而失败
obsidian search query="DPO" format=json | jq '.matches[].path'
# Error: jq: parse error: Invalid numeric literal at line 1, column 11

# ✅ 正确：跳过前 2 行警告
obsidian search query="DPO" format=json 2>&1 | sed '1,2d' | jq '.matches[].path'

# ✅ TSV 格式同样需要跳过警告
obsidian tags counts format=tsv 2>&1 | sed '1,2d' | head -10
```

**推荐模式**：

```bash
# 创建辅助函数（添加到 ~/.zshrc 或 ~/.bashrc）
obs() {
  obsidian "$@" 2>&1 | sed '1,2d'
}

# 使用辅助函数
obs search query="DPO" format=json | jq '.matches[].path'
obs tags counts format=tsv | head -10
```

---

## 实战场景

### 场景 1：查找包含特定内容的笔记

```bash
# 推荐：使用 ripgrep（更快更可靠）
rg -l "reward model" *.md

# 或使用 Obsidian CLI（理解 Obsidian 语义，但可能较慢）
obsidian search query="reward model" format=text 2>&1 | sed '1,2d'

# 注意：search 的 JSON 格式在某些情况下可能不稳定，优先使用 text 格式或 ripgrep
```

### 场景 2：批量更新 properties

```bash
# 为所有 post-training 相关笔记添加 updated 字段
obsidian search query="post-training" format=json 2>&1 | sed '1,2d' | jq -r '.matches[].path' | while read file; do
  obsidian property:set name="updated" value="2026-03-03" type=date path="$file"
done
```

### 场景 3：分析笔记连接

```bash
# 找出被引用最多的笔记
obsidian files 2>&1 | sed '1,2d' | while read file; do
  count=$(obsidian backlinks file="$file" total 2>&1 | tail -1)
  echo "$count $file"
done | sort -rn | head -10
```

### 场景 4：检查笔记质量

```bash
# 找出没有 description 的笔记
obsidian files ext=md 2>&1 | sed '1,2d' | while read file; do
  desc=$(obsidian property:read name="description" path="$file" 2>&1 | tail -1)
  if [ -z "$desc" ]; then
    echo "$file"
  fi
done
```

### 场景 5：创建带完整 frontmatter 的笔记

```bash
# 创建新笔记
obsidian create name="New Concept" content="# New Concept\n\nContent here"

# 设置 properties
obsidian property:set name="type" value="concept" file="New Concept"
obsidian property:set name="description" value="A new concept" file="New Concept"
obsidian property:set name="tags" value="post-training" type=list file="New Concept"
obsidian property:set name="created" value="2026-03-03" type=date file="New Concept"
```

---

## 与传统工具的配合

Obsidian CLI 与传统工具各有优势，应根据场景选择：

| 任务 | Obsidian CLI | 传统工具 |
|------|--------------|----------|
| 搜索笔记内容 | `obsidian search` | `rg "pattern" *.md` |
| 搜索标题 | `obsidian outline` | `rg "^#" file.md` |
| 读取文件 | `obsidian read` | `cat file.md` |
| 查询 backlinks | `obsidian backlinks` | `rg "\[\[Note\]\]"` |
| 管理 properties | `obsidian property:*` | 手动编辑 YAML |
| 批量操作 | 循环调用 CLI | `find` + `xargs` |

**推荐策略**：
- **单文件操作** → Obsidian CLI（自动处理 wikilink、properties）
- **批量搜索** → 传统工具（更快）
- **结构化查询** → Obsidian CLI（理解 Obsidian 语义）

---

## 常见问题

### Q: file vs path 参数的区别？

- `file=<name>`：按文件名匹配（类似 wikilink），不需要扩展名
- `path=<path>`：精确路径匹配，需要完整路径

```bash
# 两者等价
obsidian read file="DPO"
obsidian read path="DPO.md"
```

### Q: 如何处理带空格的文件名？

使用引号：
```bash
obsidian read file="My Note"
obsidian create name="New Note" content="Content"
```

### Q: 如何在 content 中使用换行？

使用 `\n`：
```bash
obsidian create name="Note" content="Line 1\nLine 2\nLine 3"
```

### Q: format=json 的输出如何解析？

使用 `jq`：
```bash
# 提取所有匹配文件的路径
obsidian search query="DPO" format=json | jq -r '.matches[].path'

# 提取 backlinks 数量
obsidian backlinks file="DPO" format=json | jq '.backlinks | length'
```

### Q: 如何指定 vault？

使用 `vault=<name>` 参数：
```bash
obsidian search query="test" vault="My Vault"
```

---

## 性能考虑

1. **搜索大型 vault**：
   - `obsidian search` 会索引整个 vault，首次可能较慢
   - 对于简单文本搜索，`rg` 可能更快

2. **批量操作**：
   - 避免在循环中频繁调用 CLI（每次调用有启动开销）
   - 考虑使用 `obsidian files` 一次性获取文件列表

3. **format=json**：
   - JSON 解析有额外开销，但便于程序化处理
   - 如果只需要简单输出，使用默认格式

---

## 参考资源

- **官方文档**: https://help.obsidian.md/cli
- **命令列表**: `obsidian help`
- **特定命令帮助**: `obsidian help <command>`
