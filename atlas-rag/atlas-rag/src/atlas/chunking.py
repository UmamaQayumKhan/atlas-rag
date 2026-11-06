"""Three chunking strategies, swappable by env var, plus the measurements that
justify the default.

Chunking is the highest-leverage knob in a RAG system and the thing most
tutorials skip. Measured by `python eval/run_eval.py --sweep` on the 41-question
golden set, offline embeddings, 5-document corpus:

    strategy    | chunks | recall@1 |  MRR  | nDCG@5 | answer-in-context
    ------------|--------|----------|-------|--------|------------------
    fixed       |   12   |  0.889   | 0.944 | 0.957  |      0.972
    recursive   |   10   |  0.889   | 0.944 | 0.957  |      1.000
    structural  |   46   |  0.889   | 0.940 | 0.945  |      1.000

Read honestly, this corpus is too small for the ranking metrics to separate the
three: with five documents, document-level recall@5 is saturated. What does
separate them is `answer-in-context` -- whether the expected answer text
survived the chunk boundary and reached the prompt -- where `fixed` loses 2.8
points by cutting through sentences.

`structural` is the default for two reasons the table does not show: it produces
4.6x more granular chunks, so a citation points at "Billing > Refunds > Annual
plans" instead of at a whole page, and it never merges text across a section
boundary, so a refund rule cannot be blended with a tax rule in one chunk.
The cost is that chunks are small enough to need neighbour expansion, which
`retrieve.py` handles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .config import ChunkConfig


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    ordinal: int
    source: str = ""
    title: str = ""
    section_path: tuple[str, ...] = ()
    page: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def _section_without_title(self) -> tuple[str, ...]:
        """The H1 usually *is* the document title, so including both renders a
        citation as "Billing Policy | Billing Policy > Refunds"."""
        path = self.section_path
        if path and self.title and path[0].strip().lower() == self.title.strip().lower():
            return path[1:]
        return path

    @property
    def citation_label(self) -> str:
        bits = [self.title or self.doc_id]
        section = self._section_without_title()
        if section:
            bits.append(" > ".join(section))
        if self.page is not None:
            bits.append(f"p.{self.page}")
        return " | ".join(bits)

    def embedding_text(self) -> str:
        """What actually gets embedded. Prefixing the heading path is a cheap
        trick that lifts recall on short chunks a long way, because an isolated
        sentence like "This does not apply to annual plans" is meaningless
        without "Billing > Refunds" in front of it."""
        header = " > ".join(
            part for part in ((self.title,) + self._section_without_title()) if part
        ).strip(" >")
        return f"{header}\n\n{self.text}" if header else self.text


def _tokens(text: str) -> list[str]:
    return text.split()


def _detok(tokens: Iterable[str]) -> str:
    return " ".join(tokens).strip()


def chunk_fixed(text: str, cfg: ChunkConfig) -> list[str]:
    """Baseline: fixed windows with overlap. Fast, context-blind, included so
    the eval table has an honest floor to compare against."""
    toks = _tokens(text)
    step = max(1, cfg.chunk_tokens - cfg.overlap_tokens)
    out: list[str] = []
    for start in range(0, len(toks), step):
        window = toks[start : start + cfg.chunk_tokens]
        if not window:
            break
        # A short final window is kept, never dropped: with overlap it is
        # usually already covered, but relying on that would make coverage
        # depend on where the document happens to end.
        out.append(_detok(window))
        if start + cfg.chunk_tokens >= len(toks):
            break
    return [c for c in out if c]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def chunk_recursive(text: str, cfg: ChunkConfig) -> list[str]:
    """Split on the largest natural boundary that fits: paragraph, then
    sentence, then hard token window as a last resort."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if buffer:
            chunks.append(_detok(" ".join(buffer).split()))
            buffer, buffer_len = [], 0

    for para in paragraphs:
        plen = len(_tokens(para))
        if plen > cfg.chunk_tokens:
            flush()
            sentences = _SENT_SPLIT.split(para)
            sent_buf: list[str] = []
            sent_len = 0
            for sent in sentences:
                slen = len(_tokens(sent))
                if sent_len + slen > cfg.chunk_tokens and sent_buf:
                    chunks.append(_detok(" ".join(sent_buf).split()))
                    # carry an overlap tail so a split sentence pair stays linked
                    tail = _tokens(" ".join(sent_buf))[-cfg.overlap_tokens :]
                    sent_buf, sent_len = [_detok(tail)], len(tail)
                sent_buf.append(sent)
                sent_len += slen
            if sent_buf:
                chunks.append(_detok(" ".join(sent_buf).split()))
            continue

        if buffer_len + plen > cfg.chunk_tokens:
            flush()
        buffer.append(para)
        buffer_len += plen

    flush()
    return _merge_undersized(chunks, cfg)


def _merge_undersized(chunks: list[str], cfg: ChunkConfig) -> list[str]:
    """Fold chunks below `min_chunk_tokens` into their neighbour instead of
    discarding them.

    Dropping them is the obvious implementation and it silently deletes
    document content -- a trailing "This does not apply to annual plans."
    paragraph is short, important, and exactly the kind of sentence a user asks
    about. `tests/test_ingest_coverage.py` asserts on total token coverage
    specifically to catch this.
    """
    if not chunks:
        return []
    out: list[str] = []
    for chunk in chunks:
        if out and len(_tokens(chunk)) < cfg.min_chunk_tokens:
            candidate = f"{out[-1]}\n\n{chunk}"
            # Only merge if the result stays within a reasonable budget;
            # otherwise keep the short chunk rather than build a giant one.
            if len(_tokens(candidate)) <= cfg.chunk_tokens * 1.5:
                out[-1] = candidate
                continue
        out.append(chunk)
    # A single short chunk is still returned: better a small chunk than none.
    return [c for c in out if c.strip()]


_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _parse_sections(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Walk markdown headings and return (heading_path, body) pairs."""
    sections: list[tuple[tuple[str, ...], str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def emit() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((tuple(name for _, name in stack), content))

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            emit()
            body = []
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        else:
            body.append(line)
    emit()
    if not sections:
        return [((), text.strip())]
    return sections


def chunk_structural(text: str, cfg: ChunkConfig) -> list[tuple[tuple[str, ...], str]]:
    """Heading-aware: never merges across a section boundary, and splits
    oversized sections recursively while keeping the heading path attached."""
    out: list[tuple[tuple[str, ...], str]] = []
    for path, body in _parse_sections(text):
        if len(_tokens(body)) <= cfg.chunk_tokens:
            if body.strip():
                out.append((path, body.strip()))
            continue
        for piece in chunk_recursive(body, cfg):
            out.append((path, piece))
    return out


def build_chunks(
    *,
    doc_id: str,
    text: str,
    cfg: ChunkConfig,
    source: str = "",
    title: str = "",
    page: int | None = None,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    strategy = cfg.strategy.lower()
    pieces: list[tuple[tuple[str, ...], str]]

    if strategy == "structural":
        pieces = chunk_structural(text, cfg)
    elif strategy == "recursive":
        pieces = [((), p) for p in chunk_recursive(text, cfg)]
    elif strategy == "fixed":
        pieces = [((), p) for p in chunk_fixed(text, cfg)]
    else:
        raise ValueError(f"unknown CHUNK_STRATEGY: {cfg.strategy}")

    return [
        Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}::{i:04d}",
            text=body,
            ordinal=i,
            source=source,
            title=title,
            section_path=path,
            page=page,
            metadata=dict(metadata or {}),
        )
        for i, (path, body) in enumerate(pieces)
        if body.strip()
    ]
