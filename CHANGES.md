# dontcry 修改总结（CHANGES.md）

> 基于评审报告 `dontcry-code-review.md` 对 `rag_system/` 源码进行的修复
> 修改日期：2026-07-09

## P0 修复（正确性 / 核心功能）

### 1. `splitter.py` — chunk_overlap 真正生效
- `_merge_splits` 主路径实现 overlap carry-over：新 chunk 开始时从上一 chunk 末尾逆序取 piece 累加到 `overlap_prefix`，直到达到 `chunk_overlap` 预算。
- `chunk_overlap=0` 时短路跳过，无额外开销。
- 运行验证：`chunk_size=24, overlap=12` 切分中文文本，相邻 chunk 确实共享末尾句子（句4、句5 在相邻 chunk 间重复）。

### 2. `rag.py` — chat 多轮上下文 + 接入锚点记忆
- 新增 `_DEFAULT_CHAT_HISTORY_WINDOW = 6`，chat 维护 `history`，每轮构建 `[system+context] + history[-6:] + [current_user]`，支持"那它的价格呢？"这类追问。
- chat 退出（`/exit`、EOF、Ctrl-C）及 `/save` 命令调用 `_maybe_save_memory(history)`，询问后保存到 SessionMemory，让锚点记忆在日常 chat 中可用。
- `_memory_prompted` 标志避免 `/save` 后 `/exit` 重复询问。

### 3. `vector_store.py` — save 原子写
- `save()` 改为：两个临时文件都写成功后再 `os.replace` 改名；`try/finally` 清理失败的临时文件。
- rename 顺序：先 metadata 后 index（崩溃后 metadata 是 index 的超集，search 安全）。

### 4. `rag.py` — ingest 统计不再重复切分
- 引入 `total_chunks` 累加器，步骤 3/4 切分时累加，返回时直接用，不再二次切分。

## P1 修复（性能 / 健壮性）

### 5. `vector_store.py` — 归一化契约显式化
- `add_documents` 内 `faiss.normalize_L2(vectors)`，`search` 内 `faiss.normalize_L2(q)`，双侧归一化，不再隐式依赖 embedder。

### 6. `vector_store.py` — replace_source 批量重建
- `reconstruct_n(0, max_id+1)` 批量取向量替代逐个 `reconstruct` Python 循环；NumPy 布尔 mask 过滤。

### 7. `vector_store.py` — file_hash O(1)
- 新增 `_source_hashes: dict[str,str]`，`add_documents`/`replace_source` 同步维护；`file_hash` 直接查表。`load` 支持从旧版 metadata 回填。

### 8. `session_memory.py` — reindex 增量化
- `_reindex_prompts` 改为：load 已有 store，仅 embed `has_source` 为 False 的新 merged 文件并 append。O(n²) → O(n)。

### 9. `session_memory.py` — recall 缓存 embedder/store
- 新增 `_get_embedder` / `_get_prompt_store` 缓存句柄，避免每次 recall 重载 ~100MB 本地模型。

### 10. `session_memory.py` — 删除 importlib.reload 反模式
- `_get_generator` 失败直接返回 None 走 fallback，删除无意义的模块 reload。

### 11. `llm.py` — 流式输出 + token 截断
- 新增 `stream()` 流式生成器；retry 只作用于连接阶段，开始 yield 后失败立即抛出（避免重复输出）。
- `build_rag_prompt` 按 score 顺序累加 chunk，`_estimate_tokens` 估算（UTF-8 字节/3），超 `max_context_tokens=4000` 截断。
- `chat()` 接入 `stream()` 逐字打印，流式失败 fallback 到 `generate()`。

## P2 修复（工程化）

### 12. `session_memory.py` — 文件名解析
- `_SESSION_FILENAME_RE = ^(\d{4}-\d{2}-\d{2})-(.+)-(\d{6})$` 精确解析三段，slug 不再混入时间戳。

### 13. `cli.py` — 硬编码文件名 + base_dir bug
- `_cmd_info` 用 `VectorStore.INDEX_FILENAME/META_FILENAME` 常量。
- `_cmd_memory` 的 `base_dir` 默认 `Path.cwd()`（原从 `__file__` 推导，pip install 后指向 site-packages），新增 `--base-dir` 覆盖。

### 14. `config.py` — import 副作用 + 转换错误
- `_load_dotenv` 改为函数 + `_dotenv_loaded` 守卫，在 `load_rag_config()` 内调用，模块 import 无副作用。
- 新增 `_env_int` / `_env_float`，try/except 转 `ConfigurationError`，给出友好提示。

### 15. 全项目引入 logging
- `rag.py` / `cli.py` / `documents.py` / `session_memory.py` 加 `logger = logging.getLogger(__name__)`。
- 诊断输出（ingest 进度、跳过文件、失败警告）改 `logger.info/warning`；用户面 REPL 输出保留 `print`。

### 16. 函数内标准库 import 提到顶部
- `rag.py` 的 `hashlib`、`documents.py` 的 `re`（并预编译三个正则）、`cli.py` 的 `json`、`vector_store.py` 的 `os`/`tempfile`、`session_memory.py` 的 `Counter` 全部提到模块顶部。

## P3 修复

### 17. `session_memory.py` — rollback 语义澄清 + 缓存失效
- `rollback_to` 文档说明"恢复副本为当前版本"语义；调用后置 `self._prompt_store = None` 使下次 recall 看到重新索引的数据。

### 18. `session_memory.py` — CJK 正则扩展
- `_fallback_prompts` 用 `[\u4e00-\u9fff\u3400-\u4dbf]` 覆盖 CJK 扩展 A 区。

## 验证
- 全部 12 个 `.py` 文件 `py_compile` 通过。
- splitter overlap 运行时验证通过（相邻 chunk 共享末尾句子）。
- 子代理独立检阅确认所有 P0/P1 项修复到位，未发现阻塞性回归（详见 `subagent-review.md`）。

---

# 第二轮：功能增强（rerank / hybrid / source 管理 / embedder retry / 测试）

## 新增功能

### 19. `reranker.py`（新增）— cross-encoder 重排
- 基于 sentence-transformers `CrossEncoder`（默认 BAAI/bge-reranker-base），lazy load。
- `rerank(query, chunks, top_k)` 用 cross-encoder 重新打分并截取 top_k；predict 返回值加 `np.asarray().ravel()` 防御未来 2D 返回。

### 20. `bm25.py`（新增）— 零依赖 Okapi BM25
- 纯 Python 实现，中文用字符 bigram 分词（无需 jieba），ASCII 按词。
- IDF 带 +1 平滑保证非负；`avgdl=0` 防 0 除；空语料/空查询安全。

### 21. `retriever.py`（重写）— pipeline 集成 hybrid + rerank
- 流程：向量召回 `top_k * candidate_multiplier` 个候选 → 可选 hybrid 融合 → 可选 rerank → top_k。
- **`_hybrid_fuse` 在 BM25 全零时跳过融合**（保留原始向量分，避免无信号时把所有分乘 0.7 导致失真）。
- rerank 失败非致命，fallback 到向量序。
- 全部通过 `RetrievalConfig` 开关控制，默认关闭，不影响原有行为。

### 22. `config.py` — RetrievalConfig
- 新增 `hybrid_enabled`/`hybrid_alpha`/`reranker_enabled`/`reranker_model`/`candidate_multiplier` + `_env_bool`。
- `__post_init__` 校验 alpha∈[0,1]、multiplier>=1。

### 23. Source 管理 CLI
- `vector_store.remove_source(source)`：删除某 source 所有 chunk 并重建索引。
- `rag.list_sources()` / `rag.remove_source(source)`（含持久化）。
- `cli` 新增 `list-sources`（带 `--store-dir`）和 `remove <source>` 命令。

### 24. `embeddings.py` — API embedder retry + 截断
- `embed_documents` 加指数退避 retry（3 次），`_is_retryable_embedding_error` 遍历异常链查 status_code（与 LLMGenerator 一致）。
- `_truncate_text` 按 16000 字符截断输入，防超长被 API 拒。

### 25. `memory_utils.py`（新增）+ `__init__.py` 懒加载
- 抽出 `merge_dedup`/`make_slug`/`parse_session_filename` 到独立模块，可脱网单元测试。
- `__init__.py` RAGSystem 改 PEP 562 `__getattr__` 懒加载，直接 import 子模块（splitter/bm25/config）不拉 numpy/faiss。

## 单元测试（新增 39 个，全通过）
- `test_splitter.py`（8）：overlap 正确性、中文标点、空输入、强制切分。
- `test_bm25.py`（7）：bigram 分词、精确匹配、空语料/空查询、正分断言。
- `test_memory_utils.py`（11）：merge_dedup 顺序/归一化、slug、文件名解析（含带短横 slug）。
- `test_vector_store.py`（7）：add/search/remove、多块删除+重加、save_load 往返、file_hash O(1)。
- `test_retriever.py`（6）：候选数、空索引、rerank multiplier、rerank fallback、**BM25 全零保留向量分**、BM25 有信号重排。
- `conftest.py`：`collect_ignore` 排除 test_session_memory/stress_100/agnes_e2e/quality_check 等 e2e 脚本，pytest 只跑纯单元测试。

## 子代理两轮审查
- 第 1 轮指出 3 个 P1：hybrid 全零压低向量分、缺 test_retriever、test_session_memory 依赖网络 → 全部修复。
- 第 2 轮确认 P1 修复到位、本轮 P2/P3 修复无新风险，结论"可以结束迭代"。

