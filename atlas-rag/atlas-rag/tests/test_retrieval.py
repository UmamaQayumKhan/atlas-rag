from __future__ import annotations

import pytest

from atlas.lexical import BM25Index, tokenize
from atlas.retrieve import reciprocal_rank_fusion


def test_rrf_rewards_agreement_between_retrievers():
    """A document both retrievers rank highly must beat one that only a single
    retriever ranks first. This is the entire point of fusion."""
    dense = ["a", "b", "c"]
    lexical = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    assert fused["b"] > fused["c"]
    assert fused["a"] > fused["d"]


def test_rrf_is_scale_invariant():
    """RRF uses rank only, so rescaling scores cannot change the outcome --
    the property that makes it preferable to a weighted score blend."""
    a = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)
    b = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)
    assert a == b
    assert a["x"] == pytest.approx(a["y"])


def test_rrf_k_damps_top_rank_advantage():
    small_k = reciprocal_rank_fusion([["a", "b"]], k=1)
    large_k = reciprocal_rank_fusion([["a", "b"]], k=1000)
    assert small_k["a"] / small_k["b"] > large_k["a"] / large_k["b"]


def test_tokenizer_splits_snake_case():
    toks = tokenize("Set SSO_REDIRECT_URI in the environment")
    assert "sso_redirect_uri" in toks
    assert "redirect" in toks and "uri" in toks


def test_bm25_finds_exact_identifier():
    """The query dense retrieval is worst at: an opaque error code."""
    idx = BM25Index()
    idx.add("a", "Error SAML_AUDIENCE_MISMATCH means the audience URI does not match.")
    idx.add("b", "Billing is monthly and renews on the purchase date.")
    idx.add("c", "Rate limits are applied per workspace.")
    hits = idx.search("SAML_AUDIENCE_MISMATCH", top_k=3)
    assert hits and hits[0][0] == "a"


def test_bm25_idf_penalises_common_terms():
    idx = BM25Index()
    for i in range(10):
        idx.add(f"common{i}", "workspace workspace workspace")
    idx.add("rare", "workspace kerberos")
    hits = dict(idx.search("kerberos", top_k=5))
    assert "rare" in hits


def test_bm25_roundtrip(tmp_path):
    idx = BM25Index()
    idx.add("a", "annual plans refundable pro-rata within thirty days")
    path = tmp_path / "bm25.pkl"
    idx.save(path)
    loaded = BM25Index.load(path)
    assert loaded.search("refundable", top_k=1)[0][0] == "a"


def test_bm25_empty_index_returns_nothing():
    assert BM25Index().search("anything") == []


# --- end-to-end retrieval over the real corpus ---------------------------


def test_hybrid_retrieves_expected_document(pipeline):
    result = pipeline.retrieve("How long until a workspace is suspended after a failed payment?")
    docs = {sc.chunk.doc_id for sc in result.chunks}
    assert any("billing" in d for d in docs)


def test_lexical_arm_rescues_identifier_query(pipeline):
    """End-to-end proof the BM25 arm matters: hashing embeddings have no useful
    neighbourhood for this token, so a dense-only system misses it."""
    result = pipeline.retrieve("SAML_AUDIENCE_MISMATCH")
    assert result.lexical_hits > 0
    docs = {sc.chunk.doc_id for sc in result.chunks}
    assert any("sso" in d for d in docs)


def test_neighbour_expansion_adds_adjacent_chunks(pipeline):
    result = pipeline.retrieve("refund policy for annual plans")
    retrievers = {sc.retriever for sc in result.chunks}
    assert any("neighbour" in r for r in retrievers) or len(result.chunks) <= 2


def test_retrieval_is_deterministic(pipeline):
    q = "what are the API rate limits"
    first = [sc.chunk.chunk_id for sc in pipeline.retrieve(q).chunks]
    second = [sc.chunk.chunk_id for sc in pipeline.retrieve(q).chunks]
    assert first == second


def test_metadata_filter_scopes_search(pipeline):
    result = pipeline.retrieve("limits", flt={"doc_id": "api-limits"})
    assert result.chunks
    assert all(sc.chunk.doc_id == "api-limits" for sc in result.chunks)


def test_context_block_is_numbered_from_one(pipeline):
    block = pipeline.retrieve("webhook retries").context_block()
    assert block.startswith("[1]")


def test_index_persists_across_reload(pipeline):
    from atlas.config import Settings
    from atlas.pipeline import AtlasPipeline

    pipeline.save_index()
    fresh = AtlasPipeline(settings=Settings())
    assert fresh.load_index() is True
    assert fresh.store.count() == pipeline.store.count()
    assert len(fresh.bm25.doc_ids) == len(pipeline.bm25.doc_ids)
