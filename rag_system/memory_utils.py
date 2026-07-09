"""Pure helper functions for session memory.

Split out from ``session_memory`` so they can be unit-tested without pulling
in numpy/faiss (which ``session_memory`` imports transitively via
``vector_store``). All functions here depend only on the standard library.
"""

import re

# Session filename shape: YYYY-MM-DD-slug-HHMMSS
_SESSION_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)-(\d{6})$")


def merge_dedup(new_prompts: list[str], old_prompts: list[str]) -> list[str]:
    """Merge new prompts with old, deduplicating by normalized text.

    New prompts appear first. Normalisation strips whitespace and lowercases
    so that "RAG 系统" and "rag系统" are treated as the same anchor.
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


def make_slug(content: str, max_len: int = 30) -> str:
    """Create a filename-safe slug from the first line of *content*."""
    first_line = content.strip().split("\n")[0][:80]
    slug = re.sub(r"[^\w一-鿿]+", "-", first_line.lower())
    slug = slug.strip("-")
    return slug[:max_len] if len(slug) > max_len else slug


def parse_session_filename(stem: str) -> tuple[str, str]:
    """Parse a session stem into (date, slug).

    Filenames are produced as ``YYYY-MM-DD-slug-HHMMSS``. Returns the clean
    date and slug (without the trailing timestamp). Falls back gracefully for
    legacy/malformed names.
    """
    m = _SESSION_FILENAME_RE.match(stem)
    if m:
        return m.group(1), m.group(2)
    # Legacy fallback: best-effort date prefix.
    if len(stem) >= 10 and stem[:4].isdigit() and stem[4] == "-":
        return stem[:10], stem[11:]
    return "", stem
