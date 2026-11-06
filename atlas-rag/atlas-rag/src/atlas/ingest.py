"""Document loading and the ingestion pipeline.

Loaders are deliberately optional-dependency: PDF and DOCX support activates if
pypdf / python-docx are installed and degrades with a clear message if not,
rather than making every user of the markdown path install a PDF stack.

Ingestion is idempotent. Chunk IDs are derived from `doc_id` plus ordinal, and
the Qdrant point ID is a UUID5 of the chunk ID, so re-running ingestion over a
changed corpus updates in place instead of accumulating duplicates. Duplicate
chunks are the most common cause of a RAG system quietly returning the same
passage five times and wasting the whole context window.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .chunking import Chunk, build_chunks
from .config import Settings, get_settings
from .lexical import BM25Index
from .providers import Embedder
from .store import VectorStore
from .telemetry import span

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}
HTML_SUFFIXES = {".html", ".htm"}


@dataclass
class LoadedDoc:
    doc_id: str
    title: str
    text: str
    source: str
    page: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def _slug(path: Path, root: Path | None = None) -> str:
    try:
        rel = path.relative_to(root) if root else path
    except ValueError:
        rel = path
    return re.sub(r"[^a-z0-9]+", "-", str(rel.with_suffix("")).lower()).strip("-")


def _title_from(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped and not stripped.startswith(("<", "---")):
            return stripped[:120]
    return fallback


def strip_html(html: str) -> str:
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Turn block-level tags into paragraph breaks so structure survives.
    html = re.sub(r"</(p|div|li|tr|h[1-6]|section|article)>", "\n\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<h([1-6])[^>]*>", lambda m: "\n\n" + "#" * int(m.group(1)) + " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = (
        html.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    html = re.sub(r"[ \t]+", " ", html)
    return re.sub(r"\n{3,}", "\n\n", html).strip()


def load_pdf(path: Path, root: Path | None = None) -> list[LoadedDoc]:
    """One LoadedDoc per page, so citations can carry a page number -- the first
    thing anyone asks for when the answer comes from a 200-page policy PDF."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            f"{path.name}: PDF support needs `pip install pypdf`"
        ) from exc

    reader = PdfReader(str(path))
    base = _slug(path, root)
    docs: list[LoadedDoc] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        docs.append(
            LoadedDoc(
                doc_id=f"{base}-p{i:03d}",
                title=path.stem.replace("-", " ").replace("_", " ").title(),
                text=text,
                source=str(path),
                page=i,
                metadata={"filetype": "pdf"},
            )
        )
    return docs


def load_docx(path: Path, root: Path | None = None) -> list[LoadedDoc]:
    try:
        import docx  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"{path.name}: DOCX support needs `pip install python-docx`"
        ) from exc

    document = docx.Document(str(path))
    lines: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            lines.append("")
            continue
        style = (para.style.name or "").lower()
        if style.startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "1"
            lines.append(f"\n{'#' * min(int(level), 6)} {text}\n")
        else:
            lines.append(text)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return [
        LoadedDoc(
            doc_id=_slug(path, root),
            title=_title_from(body, path.stem),
            text=body,
            source=str(path),
            metadata={"filetype": "docx"},
        )
    ]


def load_file(path: Path, root: Path | None = None) -> list[LoadedDoc]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return [
            LoadedDoc(
                doc_id=_slug(path, root),
                title=_title_from(text, path.stem),
                text=text,
                source=str(path),
                metadata={"filetype": suffix.lstrip(".")},
            )
        ]
    if suffix in HTML_SUFFIXES:
        text = strip_html(path.read_text(encoding="utf-8", errors="replace"))
        return [
            LoadedDoc(
                doc_id=_slug(path, root),
                title=_title_from(text, path.stem),
                text=text,
                source=str(path),
                metadata={"filetype": "html"},
            )
        ]
    if suffix == ".pdf":
        return load_pdf(path, root)
    if suffix == ".docx":
        return load_docx(path, root)
    return []


def discover(root: str | Path) -> Iterator[Path]:
    root_path = Path(root)
    if root_path.is_file():
        yield root_path
        return
    supported = TEXT_SUFFIXES | HTML_SUFFIXES | {".pdf", ".docx"}
    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            yield path


@dataclass
class IngestStats:
    files: int = 0
    documents: int = 0
    chunks: int = 0
    embedded: int = 0
    skipped: list[str] = field(default_factory=list)
    content_hash: str = ""

    def summary(self) -> str:
        line = (
            f"{self.files} files -> {self.documents} docs -> {self.chunks} chunks "
            f"({self.embedded} embedded)"
        )
        if self.skipped:
            line += f"; skipped {len(self.skipped)}"
        return line


@dataclass
class Ingestor:
    store: VectorStore
    embedder: Embedder
    settings: Settings = field(default_factory=get_settings)
    batch_size: int = 64

    def chunk_docs(self, docs: Sequence[LoadedDoc]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(
                build_chunks(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    cfg=self.settings.chunking,
                    source=doc.source,
                    title=doc.title,
                    page=doc.page,
                    metadata=doc.metadata,
                )
            )
        return chunks

    def index(
        self, docs: Sequence[LoadedDoc], bm25: BM25Index | None = None
    ) -> tuple[list[Chunk], BM25Index, IngestStats]:
        stats = IngestStats(documents=len(docs))
        chunks = self.chunk_docs(docs)
        stats.chunks = len(chunks)

        index = bm25 or BM25Index()
        self.store.ensure_collection(self.embedder.dim)

        hasher = hashlib.blake2b(digest_size=16)
        with span("ingest.embed", chunks=len(chunks)):
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start : start + self.batch_size]
                texts = [c.embedding_text() for c in batch]
                vectors = self.embedder.embed(texts)
                self.store.upsert(batch, vectors)
                stats.embedded += len(batch)
                for chunk, text in zip(batch, texts):
                    # BM25 indexes the same text that was embedded, so the two
                    # arms of the hybrid search never disagree about content.
                    index.add(chunk.chunk_id, text)
                    hasher.update(chunk.chunk_id.encode())
                    hasher.update(text.encode())

        stats.content_hash = hasher.hexdigest()
        return chunks, index, stats

    def ingest_path(self, root: str | Path) -> tuple[list[Chunk], BM25Index, IngestStats]:
        docs: list[LoadedDoc] = []
        skipped: list[str] = []
        files = 0
        for path in discover(root):
            files += 1
            try:
                docs.extend(load_file(path, Path(root) if Path(root).is_dir() else None))
            except Exception as exc:
                skipped.append(f"{path.name}: {exc}")
        chunks, index, stats = self.index(docs)
        stats.files = files
        stats.skipped = skipped
        return chunks, index, stats
