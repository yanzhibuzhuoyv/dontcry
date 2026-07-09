"""RAG orchestrator: ingest documents, answer questions, interactive chat."""

import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import RAGConfig
from .documents import DocumentLoader
from .embeddings import create_embedder
from .errors import (
    ConfigurationError,
    IngestionError,
    RAGSystemError,
    RetrievalError,
)
from .llm import LLMGenerator
from .retriever import Retriever
from .splitter import create_splitter
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Number of most recent chat turns (user+assistant pairs × 2 messages) kept
# as multi-turn context. Older turns are dropped to bound prompt length.
_DEFAULT_CHAT_HISTORY_WINDOW = 6


class RAGSystem:
    """Top-level RAG orchestrator. Create one instance per index directory.

    Usage::

        config = load_rag_config()
        rag = RAGSystem(config)
        rag.ingest("./docs/")
        answer = rag.query("What is this about?")
        rag.chat()
    """

    def __init__(self, config: RAGConfig):
        self._config = config
        self._embedder = create_embedder(config.embedding)
        self._splitter = create_splitter(config.chunking)
        self._loader = DocumentLoader()
        self._generator = LLMGenerator(config.llm)

        # Load or create vector store
        store_dir = Path(config.vector_store_dir)
        index_file = store_dir / VectorStore.INDEX_FILENAME
        if index_file.exists():
            self._store = VectorStore.load(store_dir)
            embed_dim = self._embedder.dimension
            if self._store.dimension != embed_dim:
                raise ConfigurationError(
                    f"vector store dimension ({self._store.dimension}) "
                    f"does not match embedder dimension ({embed_dim}). "
                    "Delete the store directory and re-ingest, or change embedding model."
                )
        else:
            self._store = VectorStore(self._embedder.dimension)

        # Optional cross-encoder reranker (lazy-loaded on first use).
        reranker = None
        if config.retrieval.reranker_enabled:
            from .reranker import Reranker

            reranker = Reranker(
                model_name=config.retrieval.reranker_model,
                device=config.embedding.device,
            )

        self._retriever = Retriever(
            self._store,
            self._embedder,
            retrieval=config.retrieval,
            reranker=reranker,
        )

    @property
    def config(self) -> RAGConfig:
        return self._config

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    def list_sources(self) -> list[str]:
        """Return a sorted list of ingested source paths."""
        return sorted(self._store.sources)

    def remove_source(self, source: str) -> int:
        """Remove a source and all its chunks from the index.

        Persists the change to disk. Returns the number of chunks removed.
        """
        removed = self._store.remove_source(source)
        if removed:
            self._store.save(self._config.vector_store_dir)
        return removed

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        path: str,
        pattern: str = "**/*",
        force: bool = False,
    ) -> dict[str, object]:
        """Ingest documents from a file or directory.

        Returns summary dict with keys: files, chunks, skipped, updated.
        """
        # 1. Load documents
        try:
            documents = self._loader.load_directory(path, pattern)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"document loading failed: {exc}") from exc

        if not documents:
            return {"files": 0, "chunks": 0, "skipped": 0, "updated": 0}

        # 2. Classify documents: new / changed / unchanged
        new_docs: list = []
        updated_docs: list = []
        skipped = 0

        for doc in documents:
            content_bytes = doc.content.encode("utf-8")
            doc_hash = hashlib.sha256(content_bytes).hexdigest()

            # Attach hash to metadata for storage
            new_meta = dict(doc.metadata)
            new_meta["content_hash"] = doc_hash
            new_doc = doc.__class__(path=doc.path, content=doc.content, metadata=new_meta)

            if not self._store.has_source(doc.path):
                new_docs.append(new_doc)
            elif force:
                stored_hash = self._store.file_hash(doc.path)
                if stored_hash != doc_hash:
                    updated_docs.append(new_doc)
                else:
                    skipped += 1
            else:
                skipped += 1

        if not new_docs and not updated_docs:
            return {"files": 0, "chunks": 0, "skipped": skipped, "updated": 0}

        total_chunks = 0

        # 3. Replace updated documents (remove old chunks, add new)
        for doc in updated_docs:
            chunks = self._splitter.split_documents([doc])
            if not chunks:
                continue
            logger.info("replacing %s: %d chunks", doc.path, len(chunks))
            embeddings = self._embedder.embed_documents([c.text for c in chunks])
            self._store.replace_source(doc.path, chunks, embeddings)
            total_chunks += len(chunks)

        # 4. Add new documents
        if new_docs:
            chunks = self._splitter.split_documents(new_docs)
            if chunks:
                logger.info("embedding %d new chunks...", len(chunks))
                embeddings = self._embedder.embed_documents([c.text for c in chunks])
                self._store.add_documents(chunks, embeddings)
                total_chunks += len(chunks)

        # 5. Persist
        self._store.save(self._config.vector_store_dir)

        total = len(new_docs) + len(updated_docs)
        return {
            "files": total,
            "chunks": total_chunks,
            "skipped": skipped,
            "updated": len(updated_docs),
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
    ) -> str:
        """Answer a question using RAG.

        Raises RetrievalError if no index exists.
        """
        if self._store.count == 0:
            raise RetrievalError("no documents indexed. Run 'ingest' first.")

        k = top_k if top_k is not None else self._config.top_k

        result = self._retriever.retrieve(question, top_k=k)
        if not result.chunks:
            return "未在索引中找到与问题相关的内容。"

        messages = self._generator.build_rag_prompt(question, result.chunks)
        answer = self._generator.generate(messages)

        if include_sources and result.chunks:
            sources = {c.source for c in result.chunks}
            source_list = "\n".join(f"  - {s}" for s in sorted(sources))
            answer += f"\n\n---\n参考来源:\n{source_list}"

        return answer

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self) -> None:
        """Interactive chat REPL loop with multi-turn context.

        Each turn carries the last ``_DEFAULT_CHAT_HISTORY_WINDOW`` messages
        as conversation history so follow-up questions ("那它的价格呢？")
        can resolve pronouns against prior turns. On exit the user is offered
        to persist the conversation into anchor-point session memory.
        """
        print("=" * 60)
        print("  RAG 对话模式")
        print(f"  索引: {self._config.vector_store_dir}")
        print(
            f"  文件数: {len(self._store.sources)}  "
            f"块数: {self._store.count}"
        )
        print(
            f"  嵌入: {self._config.embedding.provider}"
            f"/{self._config.embedding.model}"
        )
        print(f"  LLM: {self._config.llm.model}")
        print()
        print("  命令: /exit /clear /sources /stats /save")
        print("  直接输入问题开始对话（支持多轮上下文）")
        print("=" * 60)

        last_sources: list[str] = []
        history: list[dict[str, str]] = []

        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                self._maybe_save_memory(history)
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                handled = self._handle_command(user_input, last_sources, history)
                if handled is None:
                    self._maybe_save_memory(history)
                    break
                last_sources = handled
                continue

            try:
                result = self._retriever.retrieve(
                    user_input, top_k=self._config.top_k
                )
                if not result.chunks:
                    print("\n未在索引中找到与问题相关的内容。")
                    last_sources = []
                    continue

                # Build messages: system(+context) + recent history + current user.
                base_messages = self._generator.build_rag_prompt(
                    user_input, result.chunks
                )
                messages = (
                    [base_messages[0]]
                    + history[-_DEFAULT_CHAT_HISTORY_WINDOW:]
                    + [base_messages[1]]
                )
                # Stream the answer token-by-token for responsiveness; fall
                # back to blocking generate if the endpoint lacks streaming.
                print()
                answer_parts: list[str] = []
                try:
                    for token in self._generator.stream(messages):
                        print(token, end="", flush=True)
                        answer_parts.append(token)
                    answer = "".join(answer_parts)
                except RAGSystemError:
                    if answer_parts:
                        # Keep partial output rather than re-generating and
                        # duplicating what was already printed.
                        answer = "".join(answer_parts)
                    else:
                        answer = self._generator.generate(messages)
                        print(answer, end="")
                print()

                # Record this turn for future context.
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": answer})
                last_sources = [c.source for c in result.chunks]
            except RAGSystemError as exc:
                print(f"\n错误: {exc}")
                last_sources = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_command(
        self, cmd: str, sources: list[str], history: list[dict[str, str]]
    ) -> Optional[list[str]]:
        cmd_lower = cmd.lower().strip()

        if cmd_lower in ("/exit", "/quit", "/q"):
            print("再见。")
            return None

        if cmd_lower == "/clear":
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            return sources

        if cmd_lower == "/sources":
            if sources:
                print("\n上次回答的参考来源:")
                for s in sorted(set(sources)):
                    print(f"  - {s}")
            else:
                print("\n暂无参考来源。")
            return sources

        if cmd_lower == "/stats":
            print(f"\n索引统计:")
            print(f"  目录: {self._config.vector_store_dir}")
            print(f"  文件数: {len(self._store.sources)}")
            print(f"  块数: {self._store.count}")
            print(f"  嵌入维度: {self._store.dimension}")
            print(f"  对话轮数: {len(history) // 2}")
            return sources

        if cmd_lower == "/save":
            self._maybe_save_memory(history)
            return sources

        print(f"未知命令: {cmd}")
        print("可用命令: /exit /clear /sources /stats /save")
        return sources

    def _maybe_save_memory(self, history: list[dict[str, str]]) -> None:
        """Offer to persist the current conversation into anchor memory.

        Wired into chat() exit paths so the anchor-memory feature is actually
        reachable from normal usage, instead of requiring a separate
        ``memory end`` invocation with manually pasted content. Only prompts
        once per chat session (e.g. ``/save`` then ``/exit`` won't ask twice).
        """
        if not history:
            return
        if getattr(self, "_memory_prompted", False):
            return
        self._memory_prompted = True
        try:
            from .session_memory import SessionMemory

            memory = SessionMemory(base_dir=Path.cwd())
            if not memory.ask_enable_memory():
                return
            content = "\n".join(
                f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
                for m in history
            )
            result = memory.end_session(content=content)
            print(
                f"[锚点记忆] 已保存: {result.get('merged_count', 0)} 个提示词，"
                f"会话文件 {result.get('session_file', '')}"
            )
        except Exception as exc:
            logger.warning("failed to save session memory: %s", exc)
            print(f"[锚点记忆] 保存失败: {exc}")
