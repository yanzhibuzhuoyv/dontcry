# RAG System

本地 RAG（检索增强生成）系统：文档摄取 → 向量检索 → LLM 问答，附带锚点记忆功能。

## 快速开始

### 1. 配置环境

```bash
cd rag-system
cp .env.template .env
# 编辑 .env，填入你的 LLM API Key
```

`.env` 文件只需设置一个值：

```
RAG_LLM_API_KEY=sk-your-deepseek-api-key
```

嵌入默认使用本地模型（BGE-small-zh），零费用。

### 2. 安装

```bash
pip install -e .
```

首次运行时自动下载本地嵌入模型（约 100MB），仅需一次。

### 3. 使用

```bash
# 摄取文档
python -m rag_system ingest ./docs/
python -m rag_system ingest document.pdf

# 查看索引统计
python -m rag_system info

# 单次问答
python -m rag_system query "文档里讲了什么？"

# 交互式对话
python -m rag_system chat
```

## CLI 命令

### 文档检索

| 命令 | 说明 |
|------|------|
| `ingest <path> [--pattern **/*] [--force]` | 摄取文件或目录 |
| `query "<问题>" [--top-k 5] [--no-sources]` | 单次问答 |
| `chat` | 交互式对话 |
| `info [--store-dir <dir>]` | 查看索引统计 |

### 锚点记忆

| 命令 | 说明 |
|------|------|
| `memory end "<内容>" [--slug <标识>]` | 结束会话，保存记忆 |
| `memory recall "<关键词>" [--deep]` | 检索历史会话 |
| `memory list` | 列出所有会话 |
| `memory versions` | 查看提示词历史版本 |
| `memory rollback <文件名>` | 回退到指定版本 |

## Python API

```python
from rag_system import RAGSystem, load_rag_config

config = load_rag_config()
rag = RAGSystem(config)

# 文档检索
result = rag.ingest("./docs/")
answer = rag.query("文档里讲了什么？")

# 会话记忆
from rag_system.session_memory import SessionMemory
from pathlib import Path

memory = SessionMemory(base_dir=Path("."))
result = memory.end_session(content="对话内容...", slug="专题讨论")
recall = memory.recall("关键词")
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RAG_LLM_API_KEY` | (必需) | LLM API Key |
| `RAG_LLM_MODEL` | `deepseek-chat` | LLM 模型名 |
| `RAG_LLM_BASE_URL` | `https://api.deepseek.com/v1` | LLM API 地址 |
| `RAG_EMBEDDING_PROVIDER` | `local` | `local` 或 `api` |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型名 |
| `RAG_EMBEDDING_BASE_URL` | — | 嵌入 API 地址（仅 api 模式） |
| `RAG_EMBEDDING_API_KEY` | — | 嵌入 API Key（仅 api 模式） |
| `RAG_CHUNK_SIZE` | `512` | 文本块大小 |
| `RAG_CHUNK_OVERLAP` | `128` | 块重叠大小 |
| `RAG_TOP_K` | `5` | 检索块数 |
| `RAG_VECTOR_STORE_DIR` | `./rag_index` | 索引存储目录 |

## 架构

```
┌─────────────────────────────────────────┐
│  文档检索                                │
│  Documents (PDF/Word/Excel/PPT/HTML/MD)  │
│    → markitdown 解析                     │
│    → TextSplitter (中文标点感知)          │
│    → Embedder (本地/API)                 │
│    → FAISS 向量存储                      │
│    → 检索 → LLM 生成                     │
├─────────────────────────────────────────┤
│  锚点记忆                                │
│  对话内容                                │
│    → LLM 提取提示词                      │
│    → 与上次合并去重                       │
│    → prompts/ + sessions/ 存档           │
│    → prompt_index/ FAISS 索引            │
│    → 两级检索（提示词 → 全文）            │
│    → 支持回退历史版本                     │
└─────────────────────────────────────────┘
```

## 项目结构

```
rag-system/
├── rag_system/
│   ├── config.py           # 环境变量配置（.env 自动加载）
│   ├── embeddings.py       # 嵌入提供者 (Local/API)
│   ├── splitter.py         # 文本分割器
│   ├── documents.py        # 文档加载 (markitdown)
│   ├── vector_store.py     # FAISS 向量存储
│   ├── llm.py              # LLM 客户端
│   ├── retriever.py        # 检索管线
│   ├── rag.py              # 编排器
│   ├── session_memory.py   # 锚点记忆
│   └── cli.py              # CLI
├── .claude/rules/
│   └── memory-anchor.md    # 锚点记忆规则
├── prompts/                # 锚点提示词
├── sessions/               # 会话存档
├── tests/
├── .env.template           # 配置模板
├── .gitignore
├── requirements.txt
└── pyproject.toml
```
