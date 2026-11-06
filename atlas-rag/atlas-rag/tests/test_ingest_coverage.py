"""Ingestion coverage: proves no document content is silently dropped.

Silent content loss is the nastiest RAG failure mode. The system keeps
answering, confidently, from a corpus with holes in it, and nothing in the logs
says so -- you only find out when a user asks about the paragraph that vanished.
These tests assert on token coverage per strategy so a chunking change cannot
quietly start eating text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.chunking import build_chunks
from atlas.config import ChunkConfig

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"


def _coverage(strategy: str) -> tuple[float, int, int]:
    cfg = ChunkConfig(
        strategy=strategy, chunk_tokens=320, overlap_tokens=64, min_chunk_tokens=40
    )
    src_tokens = kept_tokens = n_chunks = 0
    for path in sorted(CORPUS.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = build_chunks(doc_id=path.stem, text=text, cfg=cfg)
        src_tokens += len(text.split())
        kept_tokens += sum(len(c.text.split()) for c in chunks)
        n_chunks += len(chunks)
    return kept_tokens / src_tokens, n_chunks, src_tokens


def test_recursive_preserves_every_token():
    coverage, _, _ = _coverage("recursive")
    assert coverage == pytest.approx(1.0, abs=0.01)


def test_fixed_duplicates_only_by_its_overlap():
    """Coverage above 100% is the overlap, and it must stay bounded: runaway
    duplication wastes the context window on repeated text."""
    coverage, _, _ = _coverage("fixed")
    assert 1.0 <= coverage <= 1.25


def test_structural_body_loss_is_only_heading_lines():
    """`structural` keeps heading text in `section_path` rather than in the
    chunk body, so body coverage is slightly under 100%. The gap must be small
    and must be fully accounted for by heading lines -- anything larger would
    be real content loss."""
    coverage, _, _ = _coverage("structural")
    # The whole heading line disappears from the body, the '#' marker token
    # included -- so the floor counts every token on the line. Getting this
    # off by one marker per heading is what made this test first fail against
    # correct code; the residual was 11 '#' tokens, not missing prose.
    heading_tokens = sum(
        len(line.split())
        for path in CORPUS.glob("*.md")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    )
    total = sum(len(p.read_text(encoding="utf-8").split()) for p in CORPUS.glob("*.md"))
    expected_floor = 1.0 - (heading_tokens / total) - 0.01

    assert coverage >= expected_floor, (
        f"structural coverage {coverage:.3f} is below the heading-only floor "
        f"{expected_floor:.3f} -- real body content is being dropped"
    )
    assert coverage < 1.0


def test_heading_text_survives_in_metadata():
    """The counterpart to the test above: content that left the body must still
    be reachable, or the previous test is just licensing data loss."""
    cfg = ChunkConfig(strategy="structural", chunk_tokens=320, overlap_tokens=64)
    text = (CORPUS / "billing-policy.md").read_text(encoding="utf-8")
    chunks = build_chunks(doc_id="billing-policy", text=text, cfg=cfg, title="Billing")
    all_paths = {part for c in chunks for part in c.section_path}
    assert "Refunds" in all_paths
    assert "Annual plans" in all_paths
    # and it reaches the embedding input
    annual = next(c for c in chunks if "Annual plans" in c.section_path)
    assert "Annual plans" in annual.embedding_text()


def test_structural_produces_more_granular_chunks_than_recursive():
    """Granularity is the point of `structural`: tighter chunks mean a citation
    points at a section instead of a page."""
    _, structural_chunks, _ = _coverage("structural")
    _, recursive_chunks, _ = _coverage("recursive")
    assert structural_chunks > recursive_chunks * 2


def test_every_chunk_is_within_size_budget():
    for strategy in ("fixed", "recursive", "structural"):
        cfg = ChunkConfig(strategy=strategy, chunk_tokens=320, overlap_tokens=64)
        for path in CORPUS.glob("*.md"):
            for chunk in build_chunks(
                doc_id=path.stem, text=path.read_text(encoding="utf-8"), cfg=cfg
            ):
                size = len(chunk.text.split())
                assert size <= 320 * 1.6, f"{strategy}/{path.stem}: chunk of {size} tokens"
