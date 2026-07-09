# 子代理独立检阅报告

> 审查对象：`exchange/rag_system/` 修改后的源码
> 审查基准：`dontcry-code-review.md` 评审报告
> 审查方式：只读，未修改任何文件

---

## 一、逐项核对结论表

### 1. 正确性 Bug

| # | 报告项 | 文件 | 结论 | 佐证 |
|---|--------|------|------|------|
| 1.1 | chunk_overlap 失效 | splitter.py | ✅ 已修复 | `_merge_splits` 主路径实现 overlap carry-over，新 chunk 携带上一 chunk 末尾 overlap 预算。`chunk_overlap=0` 短路跳过。 |
| 1.2 | ingest 重复切分 | rag.py | ✅ 已修复 | `total_chunks` 累加器，步骤 3/4 累加，返回直接用，不再二次切分。 |
| 1.3 | 归一化契约 | vector_store.py | ✅ 已修复 | `add_documents` 与 `search` 双侧 `faiss.normalize_L2`，对已归一化 embedder 幂等安全。 |
| 1.4 | 文件名解析 | session_memory.py | ✅ 已修复 | 正则 `^(\d{4}-\d{2}-\d{2})-(.+)-(\d{6})$` 精确匹配三段，legacy fallback 保留。 |
| 1.6 | 硬编码文件名 | cli.py | ✅ 已修复 | `_cmd_info` 用 `VectorStore.INDEX_FILENAME/META_FILENAME`。 |
| — | _cmd_memory base_dir bug | cli.py | ✅ 已修复 | 默认 `Path.cwd()`，新增 `--base-dir`。 |

### 2. 性能问题

| # | 报告项 | 文件 | 结论 | 佐证 |
|---|--------|------|------|------|
| 2.1 | replace_source 全量重建 | vector_store.py | ✅ 已修复 | `reconstruct_n` 批量取向量 + NumPy 布尔 mask 过滤，常数大幅降低。 |
| 2.2 | reindex 全量重建 | session_memory.py | ✅ 已修复 | 增量：`has_source` 判断，只 embed 新文件 append。 |
| 2.3 | recall 重载 embedder | session_memory.py | ✅ 已修复 | `_get_embedder`/`_get_prompt_store` 缓存 + `_embedder_init_failed` 防重试。 |
| 2.4 | importlib.reload 反模式 | session_memory.py | ✅ 已修复 | `_get_generator` 失败直接返回 None。 |
| 2.5 | file_hash 线性扫描 | vector_store.py | ✅ 已修复 | `_source_hashes` dict，O(1) 查找，`load` 回填兼容旧版。 |

### 3. 架构与设计

| # | 报告项 | 文件 | 结论 | 佐证 |
|---|--------|------|------|------|
| 3.1 | chat 与记忆割裂 | rag.py | ✅ 已修复 | 退出路径调用 `_maybe_save_memory`，`except` 兜底不阻断退出。 |
| 3.2 | chat 无多轮上下文 | rag.py | ✅ 已修复 | `history` + `[system]+history[-6:]+[user]`，插入位置正确。 |
| 3.3 | 无流式输出 | llm.py / rag.py | ✅ 已修复 | `stream()` 实现 + `chat()` 接入流式打印，失败 fallback `generate()`。stream retry 仅限连接阶段。 |
| 3.4 | 无 token 截断 | llm.py | ✅ 已修复 | `build_rag_prompt` 按 score 累加，`_estimate_tokens` 估算，超 4000 截断。 |

### 4. 健壮性

| # | 报告项 | 文件 | 结论 | 佐证 |
|---|--------|------|------|------|
| 4.1 | save 非原子写 | vector_store.py | ✅ 已修复 | 双临时文件 + `os.replace` + `try/finally` 清理。rename 顺序安全。 |
| 4.3 | import 时副作用 | config.py | ✅ 已修复 | `_load_dotenv` 延迟到 `load_rag_config()`，模块级无副作用。 |
| 4.4 | int/float 未捕获 | config.py | ✅ 已修复 | `_env_int`/`_env_float` 转 `ConfigurationError`。 |

### 5. 代码质量

| # | 报告项 | 文件 | 结论 | 佐证 |
|---|--------|------|------|------|
| 5.1 | print 替换 logging | 多文件 | ✅ 已修复 | 诊断输出用 `logger`，用户面 REPL 保留 print。 |
| 5.2 | 函数内 import | 多文件 | ✅ 已修复 | 标准库（hashlib/re/json/os/tempfile/Counter）全部提到顶部。 |

---

## 二、检阅中发现的次要问题（已记录，非阻塞）

1. **splitter 带 overlap 的新 chunk 可能略超 chunk_size**（P3）——LangChain 也有此行为，影响可忽略。
2. **session_memory 增量 reindex 不处理已删除的 merged 文件**（P2）——手动删除 prompts 文件时 store 残留旧向量；当前使用场景（文件名带时间戳、不主动删除）下不触发。
3. **vector_store.replace_source 的 keep_mask 假设 ID 连续**（P3）——当前 remap 维持了连续不变式，正确；建议未来加断言。

均不影响现有功能正确性，可作为后续迭代项。

---

## 三、总体评价

- **P0（4 项）、P1（5 项）、P2（3 项）、P3（2 项）全部修复 ✅**
- 修改注释清晰说明每处改动意图，边界条件（空 index、overlap=0、embedder 失败、维度不匹配）均有处理，向后兼容（`load` 回填 `_source_hashes`）考虑周到。
- 未发现阻塞性回归。运行时验证 splitter overlap 生效、全部文件 `py_compile` 通过。
- **结论：可以合并。**
