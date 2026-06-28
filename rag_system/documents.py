"""Document loading via markitdown CLI subprocess.

Converts PDF, Word, Excel, PPT, HTML, Markdown, and plain text files
to clean text suitable for embedding.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

from .errors import IngestionError


# File suffixes that markitdown can handle
_SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".html",
        ".htm",
        ".md",
        ".markdown",
        ".txt",
        ".csv",
        ".json",
        ".xml",
        ".epub",
        ".rtf",
    }
)

# Timeout per file in seconds
_FILE_TIMEOUT = 30


@dataclass(frozen=True)
class Document:
    """Immutable document with path and extracted text content."""

    path: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A text chunk with provenance tracking."""

    text: str
    source: str
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)


class DocumentLoader:
    """Converts documents to text via markitdown CLI."""

    SUPPORTED_SUFFIXES: ClassVar[frozenset[str]] = _SUPPORTED_SUFFIXES

    def __init__(self, markitdown_path: str = "markitdown"):
        """*markitdown_path*: path or command name for the markitdown CLI."""
        self._markitdown = markitdown_path

    def load(self, path: str | Path) -> Optional[Document]:
        """Load a single file. Returns None for unsupported file types.

        Raises IngestionError if conversion fails.
        """
        path = Path(path)
        if not path.exists():
            raise IngestionError(f"file not found: {path}")
        if not path.is_file():
            raise IngestionError(f"not a file: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            return None

        # Plain text: read directly (faster, no subprocess)
        if suffix in (".txt", ".md", ".markdown", ".csv", ".json", ".xml"):
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="gbk", errors="replace")
        else:
            content = self._convert_via_markitdown(str(path))

        if not content or not content.strip():
            return None

        return Document(
            path=str(path),
            content=_strip_markup(content),
            metadata={"filename": path.name},
        )

    def load_directory(self, path: str | Path, pattern: str = "**/*") -> list[Document]:
        """Walk directory with glob pattern, load all supported files.

        Logs warnings for files that fail; does not abort on individual failures.
        """
        path = Path(path)
        if not path.exists():
            raise IngestionError(f"path not found: {path}")

        if path.is_file():
            doc = self.load(path)
            return [doc] if doc else []

        documents: list[Document] = []
        for file_path in sorted(path.glob(pattern)):
            if not file_path.is_file():
                continue
            try:
                doc = self.load(file_path)
                if doc:
                    documents.append(doc)
            except IngestionError as exc:
                import sys

                print(f"  [WARN] skipping {file_path.name}: {exc}", file=sys.stderr)

        return documents

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _convert_via_markitdown(self, file_path: str) -> str:
        """Run `markitdown <file_path>` subprocess, return stdout text.

        Raises IngestionError on subprocess failure.
        """
        try:
            result = subprocess.run(
                [self._markitdown, file_path],
                capture_output=True,
                text=True,
                timeout=_FILE_TIMEOUT,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            raise IngestionError(
                f"markitdown CLI not found at '{self._markitdown}'. "
                "Install it: pip install markitdown"
            )
        except subprocess.TimeoutExpired:
            raise IngestionError(
                f"markitdown timed out after {_FILE_TIMEOUT}s: {file_path}"
            )

        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else "unknown error"
            raise IngestionError(f"markitdown failed on {file_path}: {stderr}")

        return result.stdout


def _strip_markup(text: str) -> str:
    """Remove common markdown artifacts for cleaner embedding text."""
    import re

    # Remove image links: ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove link URLs but keep link text: [text](url) -> text
    text = re.sub(r"\[([^\]]*?)\]\(.*?\)", r"\1", text)
    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
