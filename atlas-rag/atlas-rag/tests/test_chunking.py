from __future__ import annotations

import pytest

from atlas.chunking import (
    build_chunks,
    chunk_fixed,
    chunk_recursive,
    chunk_structural,
)
from atlas.config import ChunkConfig

DOC = """# Billing Policy

Intro paragraph about billing in general terms.

## Refunds

Monthly subscriptions are not refundable at any time.

### Annual plans

Annual plans are refundable pro-rata within 30 days.

## Taxes

Prices exclude VAT.
"""


@pytest.fixture
def cfg() -> ChunkConfig:
    return ChunkConfig(strategy="structural", chunk_tokens=60, overlap_tokens=10, min_chunk_tokens=3)


def test_structural_keeps_heading_path(cfg):
    pieces = chunk_structural(DOC, cfg)
    paths = [p for p, _ in pieces]
    assert ("Billing Policy", "Refunds", "Annual plans") in paths
    assert ("Billing Policy", "Taxes") in paths


def test_structural_never_merges_across_sections(cfg):
    for path, body in chunk_structural(DOC, cfg):
        if path and path[-1] == "Taxes":
            assert "refundable" not in body.lower()


def test_embedding_text_prefixes_heading_path(cfg):
    chunks = build_chunks(doc_id="billing", text=DOC, cfg=cfg, title="Billing Policy")
    annual = next(c for c in chunks if c.section_path and c.section_path[-1] == "Annual plans")
    embedded = annual.embedding_text()
    # The heading path is what rescues an otherwise context-free sentence.
    assert "Refunds" in embedded and "Annual plans" in embedded
    assert embedded.endswith(annual.text)


def test_fixed_respects_overlap():
    cfg = ChunkConfig(strategy="fixed", chunk_tokens=10, overlap_tokens=4, min_chunk_tokens=1)
    text = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_fixed(text, cfg)
    assert len(chunks) > 1
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-4:] == second[:4]


def test_recursive_splits_oversized_paragraph():
    cfg = ChunkConfig(strategy="recursive", chunk_tokens=20, overlap_tokens=4, min_chunk_tokens=1)
    long_para = " ".join(["This is a sentence number %d." % i for i in range(20)])
    chunks = chunk_recursive(long_para, cfg)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 40 for c in chunks)


def test_chunk_ids_are_stable_and_ordered(cfg):
    a = build_chunks(doc_id="billing", text=DOC, cfg=cfg)
    b = build_chunks(doc_id="billing", text=DOC, cfg=cfg)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.ordinal for c in a] == sorted(c.ordinal for c in a)
    assert a[0].chunk_id == "billing::0000"


def test_citation_label_includes_page_and_section(cfg):
    chunks = build_chunks(
        doc_id="policy", text=DOC, cfg=cfg, title="Billing Policy", page=4
    )
    label = chunks[-1].citation_label
    assert "Billing Policy" in label and "p.4" in label


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown CHUNK_STRATEGY"):
        build_chunks(doc_id="x", text="hello", cfg=ChunkConfig(strategy="magic"))


def test_no_empty_chunks(cfg):
    chunks = build_chunks(doc_id="x", text="# A\n\n\n\n## B\n\n\n", cfg=cfg)
    assert all(c.text.strip() for c in chunks)
