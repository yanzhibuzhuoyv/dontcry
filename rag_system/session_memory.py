"""Anchor-point session memory for the RAG system.

Workflow per session:
1. Start  → ask user whether to enable memory
2. End    → generate prompt-words, merge-dedup with previous session,
            save markdown, re-index prompts into prompt_index
3. Recall → search prompt_index first; if miss, full-text search sessions/*.md
"""

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from .config import load_rag_config
from .errors import RAGSystemError
from .llm import LLMGenerator
from .vector_store import VectorStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPTS_DIR = "prompts"
_SESSIONS_DIR = "sessions"
_PROMPT_INDEX_DIR = "prompt_index"
_MAX_PROMPTS = 15
_MIN_PROMPT_LEN = 4


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionRecord:
    """Metadata for one stored session."""

    date_str: str
    slug: str
    prompt_file: str
    session_file: str
    prompt_count: int
    word_count: int


@dataclass
class SessionMemory:
    """Manages the anchor-point memory lifecycle.

    *base_dir*: root of the rag-system project.
    """

    base_dir: Path
    _generator: Optional[LLMGenerator] = None

    # ------------------------------------------------------------------
    # Directory properties
    # ------------------------------------------------------------------

    @property
    def prompts_dir(self) -> Path:
        return self.base_dir / _PROMPTS_DIR

    @property
    def sessions_dir(self) -> Path:
        return self.base_dir / _SESSIONS_DIR

    @property
    def prompt_index_dir(self) -> Path:
        return self.base_dir / _PROMPT_INDEX_DIR

    # ------------------------------------------------------------------
    # end_session — called when a conversation finishes
    # ------------------------------------------------------------------

    def end_session(
        self,
        content: str,
        session_date: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> dict[str, object]:
        """Generate prompts, save session, merge with previous, re-index.

        Returns: {"prompts": [...], "merged_count": int, "session_file": str}
        """
        if not content.strip():
            return {"prompts": [], "merged_count": 0, "session_file": ""}

        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate prompt-words
        raw_prompts = self._generate_prompts(content)
        if not raw_prompts:
            return {"prompts": [], "merged_count": 0, "session_file": ""}

        # 2. Filenames
        today = session_date or date.today().isoformat()
        safe_slug = slug or _make_slug(content)

        session_file = self.sessions_dir / f"{today}-{safe_slug}.md"
        prompt_file = self.prompts_dir / f"{today}-{safe_slug}.prompts.txt"
        merged_file = self.prompts_dir / f"{today}-{safe_slug}.merged.txt"

        # 3. Save session markdown
        header = f"# Session: {today} — {safe_slug}\n\n"
        session_file.write_text(header + content, encoding="utf-8")

        # 4. Load previous merged prompts
        prev_prompts = self._load_previous_merged()

        # 5. Merge-dedup
        merged = _merge_dedup(raw_prompts, prev_prompts)

        # 6. Write prompt files
        prompt_file.write_text("\n".join(raw_prompts), encoding="utf-8")
        merged_file.write_text("\n".join(merged), encoding="utf-8")

        # 7. Re-index
        try:
            self._reindex_prompts()
        except RAGSystemError:
            pass

        return {
            "prompts": raw_prompts,
            "merged_count": len(merged),
            "session_file": str(session_file),
        }

    # ------------------------------------------------------------------
    # recall — search prompts first, fallback to full-text
    # ------------------------------------------------------------------

    # Minimum cosine similarity score (normalized vectors, range [-1, 1]).
    # 0.25 is roughly "no semantic relation". Adjust based on embedding model.
    _MIN_RECALL_SCORE: float = 0.35

    def recall(self, query: str, top_k: int = 5) -> dict[str, object]:
        """Search prompt index. Returns {found, method, results}."""
        prompt_results = self._search_prompts(query, top_k=top_k)
        # Filter by minimum relevance score
        relevant = [
            r for r in prompt_results
            if r["score"] >= self._MIN_RECALL_SCORE
        ]
        if relevant:
            return {"found": True, "method": "prompt_index", "results": relevant}
        return {"found": False, "method": "none", "results": []}

    def recall_deep(self, query: str, top_k: int = 8) -> dict[str, object]:
        """Full-text search session markdown files."""
        results = self._search_sessions_fulltext(query, top_k=top_k)
        if results:
            return {"found": True, "method": "full_text", "results": results}
        return {"found": False, "method": "none", "results": []}

    def get_session_file(self, filename: str) -> str:
        """Read a specific session markdown file."""
        path = self.sessions_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"session file not found: {filename}")
        return path.read_text(encoding="utf-8")

    def list_sessions(self) -> list[SessionRecord]:
        """List all stored sessions."""
        records: list[SessionRecord] = []
        if not self.sessions_dir.exists():
            return records
        for f in sorted(self.sessions_dir.glob("*.md"), reverse=True):
            content = f.read_text(encoding="utf-8")
            date_str, slug = _parse_session_filename(f.stem)
            records.append(
                SessionRecord(
                    date_str=date_str,
                    slug=slug,
                    prompt_file=f"{f.stem}.prompts.txt",
                    session_file=f.name,
                    prompt_count=0,
                    word_count=len(content),
                )
            )
        return records

    def list_prompt_versions(self) -> list[dict[str, str]]:
        """List all merged prompt file versions for rollback."""
        versions: list[dict[str, str]] = []
        if not self.prompts_dir.exists():
            return versions
        for f in sorted(self.prompts_dir.glob("*.merged.txt"), reverse=True):
            versions.append({
                "file": f.name,
                "prompts": f.read_text(encoding="utf-8"),
            })
        return versions

    def rollback_to(self, prompt_filename: str) -> bool:
        """Rollback: set a specific merged file as active and re-index."""
        src = self.prompts_dir / prompt_filename
        if not src.exists():
            return False
        today = date.today().isoformat()
        rollback_file = self.prompts_dir / f"{today}-rollback.merged.txt"
        rollback_file.write_text(src.read_text(encoding="utf-8"))
        self._reindex_prompts()
        return True

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def ask_enable_memory(self) -> bool:
        """Prompt: enable anchor memory for this session?"""
        try:
            answer = input(
                "\n[锚点记忆] 是否启用本次会话记忆？(y/n): "
            ).strip().lower()
            return answer in ("y", "yes", "是")
        except (EOFError, KeyboardInterrupt):
            return False

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _generate_prompts(self, content: str) -> list[str]:
        """Use LLM to extract 5-10 anchor prompt-words from content."""
        generator = self._get_generator()
        if generator is None:
            return self._fallback_prompts(content)

        truncated = content[-8000:] if len(content) > 8000 else content

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个对话摘要助手。从以下对话中提取 5-10 个关键提示词/短语（锚点），"
                    "用于日后检索。\n"
                    "规则：\n"
                    "1. 每个提示词 4-15 个字，简洁、具体、唯一\n"
                    "2. 覆盖主要话题、决策、技术要点\n"
                    "3. 每行一个提示词，不要编号\n"
                    "4. 直接输出提示词"
                ),
            },
            {"role": "user", "content": f"对话内容:\n\n{truncated}"},
        ]

        try:
            response = generator.generate(messages)
            lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
            lines = [
                re.sub(r"^[\d\.\-\*\s]+", "", ln).strip() for ln in lines
            ]
            return [
                ln for ln in lines
                if _MIN_PROMPT_LEN <= len(ln) <= 40 and not ln.startswith("#")
            ][:_MAX_PROMPTS]
        except RAGSystemError:
            return self._fallback_prompts(content)

    def _fallback_prompts(self, content: str) -> list[str]:
        """Keyword extraction fallback when LLM unavailable."""
        lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
        candidates: list[str] = []
        for ln in lines:
            if 10 <= len(ln) <= 60 and not ln.startswith("#"):
                candidates.append(ln)
        seen: set[str] = set()
        unique: list[str] = []
        for c in candidates:
            key = c[:20].lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique[:_MAX_PROMPTS]

    # ------------------------------------------------------------------
    # Prompt merging
    # ------------------------------------------------------------------

    def _load_previous_merged(self) -> list[str]:
        """Load the most recent .merged.txt file."""
        if not self.prompts_dir.exists():
            return []
        for mf in sorted(self.prompts_dir.glob("*.merged.txt"), reverse=True):
            text = mf.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return [ln.strip() for ln in text.split("\n") if ln.strip()]
        return []

    # ------------------------------------------------------------------
    # Re-index prompts
    # ------------------------------------------------------------------

    def _reindex_prompts(self) -> None:
        """Ingest all .merged.txt files into prompt_index vector store."""
        merged_files = sorted(self.prompts_dir.glob("*.merged.txt"))
        if not merged_files:
            return

        from .documents import Document

        documents: list[Document] = []
        for mf in merged_files:
            text = mf.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append(
                    Document(
                        path=str(mf),
                        content=text,
                        metadata={"type": "prompts", "session": mf.stem},
                    )
                )
        if not documents:
            return

        config = load_rag_config()
        from .embeddings import create_embedder
        from .splitter import create_splitter

        embedder = create_embedder(config.embedding)
        splitter = create_splitter(config.chunking)
        chunks = splitter.split_documents(documents)
        if not chunks:
            return

        embeddings = embedder.embed_documents([c.text for c in chunks])
        store = VectorStore(embedder.dimension)
        store.add_documents(chunks, embeddings)
        store.save(str(self.prompt_index_dir))

        # Save session sources index for full-text fallback
        sources_path = self.prompt_index_dir / "sources.json"
        session_files = sorted(self.sessions_dir.glob("*.md"))
        sources_data = {
            f.stem: f.read_text(encoding="utf-8")
            for f in session_files[-20:]
        }
        sources_path.write_text(
            json.dumps(sources_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_prompts(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        """RAG search the prompt index."""
        index_file = self.prompt_index_dir / VectorStore.INDEX_FILENAME
        if not index_file.exists():
            return []

        config = load_rag_config()
        from .embeddings import create_embedder

        embedder = create_embedder(config.embedding)
        store = VectorStore.load(str(self.prompt_index_dir))
        query_vec = embedder.embed_query(query)
        results = store.search(query_vec, k=top_k)

        return [
            {"text": r.text, "score": r.score, "source": r.source}
            for r in results
        ]

    def _search_sessions_fulltext(
        self, query: str, top_k: int = 8
    ) -> list[dict[str, object]]:
        """Full-text grep over session markdown files."""
        if not self.sessions_dir.exists():
            return []

        keywords = [
            kw.lower() for kw in re.split(r"\s+", query) if len(kw) >= 2
        ]
        results: list[dict[str, object]] = []

        for mf in sorted(self.sessions_dir.glob("*.md"), reverse=True):
            try:
                content = mf.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            score = sum(content.lower().count(kw) for kw in keywords)
            if score > 0:
                first_match = -1
                for kw in keywords:
                    idx = content.lower().find(kw)
                    if idx >= 0:
                        first_match = idx
                        break
                start = max(0, first_match - 150) if first_match >= 0 else 0
                snippet = content[start : start + 500]
                results.append({
                    "file": mf.name,
                    "score": score,
                    "snippet": snippet,
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # LLM access
    # ------------------------------------------------------------------

    def _get_generator(self) -> Optional[LLMGenerator]:
        if self._generator is not None:
            return self._generator
        try:
            config = load_rag_config()
            self._generator = LLMGenerator(config.llm)
            return self._generator
        except RAGSystemError:
            return None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _merge_dedup(new_prompts: list[str], old_prompts: list[str]) -> list[str]:
    """Merge new prompts with old, deduplicate by normalized text.

    New prompts appear first.
    """
    seen: set[str] = set()
    merged: list[str] = []

    def _norm(p: str) -> str:
        return re.sub(r"\s+", "", p.lower())

    for p in new_prompts:
        key = _norm(p)
        if key not in seen:
            seen.add(key)
            merged.append(p)

    for p in old_prompts:
        key = _norm(p)
        if key not in seen:
            seen.add(key)
            merged.append(p)

    return merged


def _make_slug(content: str, max_len: int = 30) -> str:
    """Create a filename-safe slug from content."""
    first_line = content.strip().split("\n")[0][:80]
    slug = re.sub(r"[^\w一-鿿]+", "-", first_line.lower())
    slug = slug.strip("-")
    return slug[:max_len] if len(slug) > max_len else slug


def _parse_session_filename(stem: str) -> tuple[str, str]:
    """Parse '2026-06-28-my-slug' → ('2026-06-28', 'my-slug')."""
    if "-" in stem[:10] and len(stem) > 11:
        return stem[:10], stem[11:]
    return "", stem
