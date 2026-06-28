"""CLI argument parsing and command dispatch for the RAG system."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors="replace")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors="replace")
    except Exception:
        pass  # Fall back to default if reconfiguration fails

from .config import load_rag_config
from .errors import RAGSystemError
from .rag import RAGSystem


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="rag-system",
        description="本地 RAG 系统：摄取文档 → 提问 → 对话",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_parser = sub.add_parser("ingest", help="摄取文档到向量存储")
    ingest_parser.add_argument("path", help="文件或目录路径")
    ingest_parser.add_argument(
        "--pattern", default="**/*",
        help="glob 模式（默认 **/*）",
    )
    ingest_parser.add_argument(
        "--force", action="store_true",
        help="强制重新摄取已索引的文件",
    )
    ingest_parser.add_argument(
        "--store-dir", default=None,
        help="向量存储目录（覆盖环境变量）",
    )

    # query
    query_parser = sub.add_parser("query", help="单次 RAG 问答")
    query_parser.add_argument("question", help="要问的问题")
    query_parser.add_argument(
        "--top-k", type=int, default=None,
        help="检索的块数",
    )
    query_parser.add_argument(
        "--store-dir", default=None,
        help="向量存储目录",
    )
    query_parser.add_argument(
        "--no-sources", action="store_true",
        help="不显示参考来源",
    )

    # chat
    chat_parser = sub.add_parser("chat", help="交互式对话")
    chat_parser.add_argument(
        "--store-dir", default=None,
        help="向量存储目录",
    )

    # info
    info_parser = sub.add_parser("info", help="显示索引统计")
    info_parser.add_argument(
        "--store-dir", default=None,
        help="向量存储目录",
    )

    # ---- memory ----
    memory_parser = sub.add_parser("memory", help="会话锚点记忆管理")
    memory_sub = memory_parser.add_subparsers(dest="memory_cmd", required=True)

    m_end = memory_sub.add_parser("end", help="结束会话并保存记忆")
    m_end.add_argument("content", help="会话内容文本（或文件路径，前缀 file:）")
    m_end.add_argument("--slug", default=None, help="会话简短标识")
    m_end.add_argument("--date", default=None, help="日期 (YYYY-MM-DD)")

    m_recall = memory_sub.add_parser("recall", help="检索会话记忆")
    m_recall.add_argument("query", help="搜索关键词")
    m_recall.add_argument("--top-k", type=int, default=5)
    m_recall.add_argument("--deep", action="store_true", help="全文深度检索")

    memory_sub.add_parser("list", help="列出所有会话")
    memory_sub.add_parser("versions", help="列出提示词历史版本")

    m_rollback = memory_sub.add_parser("rollback", help="回退提示词版本")
    m_rollback.add_argument("file", help="提示词文件名")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse args, load config, dispatch to command handler."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Override store dir from CLI flag
    if hasattr(args, "store_dir") and args.store_dir:
        os.environ["RAG_VECTOR_STORE_DIR"] = args.store_dir

    try:
        config = load_rag_config()
    except RAGSystemError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        print("请检查 RAG_LLM_API_KEY 等环境变量。", file=sys.stderr)
        sys.exit(1)

    try:
        if args.command == "ingest":
            _cmd_ingest(config, args)
        elif args.command == "query":
            _cmd_query(config, args)
        elif args.command == "chat":
            _cmd_chat(config, args)
        elif args.command == "info":
            _cmd_info(config, args.store_dir if hasattr(args, "store_dir") and args.store_dir else config.vector_store_dir)
        elif args.command == "memory":
            _cmd_memory(args)
    except RAGSystemError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        sys.exit(130)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_ingest(config, args) -> None:
    path = args.path
    if not os.path.exists(path):
        print(f"错误: 路径不存在: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"摄取: {path}")
    print(f"向量存储: {config.vector_store_dir}")
    print(f"嵌入: {config.embedding.provider}/{config.embedding.model}")

    rag = RAGSystem(config)
    result = rag.ingest(path, pattern=args.pattern, force=args.force)

    skipped_msg = ""
    if result.get("skipped"):
        skipped_msg = f" (跳过 {result['skipped']} 个已索引文件)"
    print(
        f"\n完成: {result['files']} 个文件, {result['chunks']} 个块{skipped_msg}"
    )


def _cmd_query(config, args) -> None:
    rag = RAGSystem(config)

    if rag.vector_store.count == 0:
        print("错误: 尚未索引任何文档。请先运行 'ingest'。", file=sys.stderr)
        sys.exit(1)

    print(f"问题: {args.question}")
    print("检索中...")
    answer = rag.query(
        args.question,
        top_k=args.top_k,
        include_sources=not args.no_sources,
    )
    print(f"\n{answer}")


def _cmd_chat(config, args) -> None:
    rag = RAGSystem(config)

    if rag.vector_store.count == 0:
        print("错误: 尚未索引任何文档。请先运行 'ingest'。", file=sys.stderr)
        sys.exit(1)

    rag.chat()


def _cmd_info(config, store_dir: str) -> None:
    store_path = Path(store_dir)
    index_file = store_path / "index.faiss"
    meta_file = store_path / "metadata.json"

    print(f"向量存储: {store_dir}")
    print(f"  index.faiss: {'存在' if index_file.exists() else '不存在'}")
    print(f"  metadata.json: {'存在' if meta_file.exists() else '不存在'}")

    if not index_file.exists():
        print("\n尚未创建索引。运行 'ingest' 开始。")
        return

    try:
        from .vector_store import VectorStore

        store = VectorStore.load(store_dir)
        print(f"\n索引统计:")
        print(f"  文件数: {len(store.sources)}")
        print(f"  块数: {store.count}")
        print(f"  嵌入维度: {store.dimension}")
        print(f"\n配置:")
        print(f"  嵌入: {config.embedding.provider}/{config.embedding.model}")
        print(f"  LLM: {config.llm.model}")
        print(f"  块大小: {config.chunking.chunk_size}")
        print(f"  块重叠: {config.chunking.chunk_overlap}")
        print(f"  Top-K: {config.top_k}")
    except RAGSystemError as exc:
        print(f"\n读取索引失败: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Memory command handlers
# ---------------------------------------------------------------------------


def _cmd_memory(args) -> None:
    """Dispatch memory subcommands."""
    import json

    from .session_memory import SessionMemory

    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    memory = SessionMemory(base_dir=base_dir)

    if args.memory_cmd == "end":
        # Load content from file if prefixed with "file:"
        content = args.content
        if content.startswith("file:"):
            file_path = content[5:]
            content = Path(file_path).read_text(encoding="utf-8")

        result = memory.end_session(
            content=content,
            slug=args.slug,
            session_date=args.date,
        )
        print(f"[锚点记忆] 已保存会话记忆:")
        print(f"  提示词: {len(result['prompts'])} 个")
        print(f"  合并后: {result['merged_count']} 个")
        print(f"  会话文件: {result['session_file']}")
        if result["prompts"]:
            print(f"\n锚点提示词:")
            for p in result["prompts"]:
                print(f"  - {p}")

    elif args.memory_cmd == "recall":
        if args.deep:
            result = memory.recall_deep(args.query, top_k=args.top_k)
        else:
            result = memory.recall(args.query, top_k=args.top_k)

        if result["found"]:
            print(f"[锚点记忆] 找到 {len(result['results'])} 条匹配 ({result['method']}):\n")
            for r in result["results"]:
                if result["method"] == "prompt_index":
                    print(f"  [{r['score']:.2f}] {r['text']}")
                    print(f"         source: {r['source']}\n")
                else:
                    print(f"  [{r['score']}] {r['file']}")
                    print(f"         {r['snippet'][:200]}...\n")
        else:
            print(f"[锚点记忆] 未在提示词索引中找到与 '{args.query}' 相关的记忆。")
            print("如果笃定存在，尝试: python -m rag_system memory recall \"" +
                  args.query + "\" --deep")

    elif args.memory_cmd == "list":
        sessions = memory.list_sessions()
        if not sessions:
            print("[锚点记忆] 暂无会话记录。")
        else:
            print(f"[锚点记忆] 共 {len(sessions)} 条会话:\n")
            for s in sessions:
                print(f"  {s.date_str}  {s.session_file}  ({s.word_count} 字)")

    elif args.memory_cmd == "versions":
        versions = memory.list_prompt_versions()
        if not versions:
            print("[锚点记忆] 暂无提示词版本。")
        else:
            print(f"[锚点记忆] 共 {len(versions)} 个版本:\n")
            for v in versions:
                prompts_count = len(v["prompts"].split("\n"))
                print(f"  {v['file']}  ({prompts_count} 个提示词)")

    elif args.memory_cmd == "rollback":
        ok = memory.rollback_to(args.file)
        if ok:
            print(f"[锚点记忆] 已回退到 {args.file}，已重新索引。")
        else:
            print(f"[锚点记忆] 文件不存在: {args.file}", file=sys.stderr)
