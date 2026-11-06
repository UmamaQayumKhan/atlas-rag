from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(pipeline, monkeypatch):
    from atlas import api

    monkeypatch.setitem(api._state, "pipeline", pipeline)
    # Bypass lifespan: the pipeline is injected directly so the test does not
    # re-ingest the corpus for every request.
    return TestClient(api.app)


def test_healthz_reports_index_state(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 0
    assert body["bm25_docs"] > 0
    assert body["chunk_strategy"] == "structural"


def test_ask_returns_sources_and_flags(client):
    resp = client.post("/ask", json={"query": "When is a workspace suspended?"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"answer", "sources", "grounded", "refused", "retrieval"}
    assert body["retrieval"]["chunks_in_prompt"] > 0


def test_retrieve_endpoint_exposes_both_arms(client):
    body = client.post("/retrieve", json={"query": "SAML_AUDIENCE_MISMATCH"}).json()
    assert body["lexical_hits"] > 0
    assert body["chunks"]
    assert "retriever" in body["chunks"][0]


def test_blank_query_is_rejected(client):
    assert client.post("/ask", json={"query": "   "}).status_code == 422


def test_too_short_query_is_rejected(client):
    assert client.post("/ask", json={"query": "a"}).status_code == 422


def test_top_k_is_bounded(client):
    assert client.post("/ask", json={"query": "refunds", "top_k": 99}).status_code == 422


def test_filters_scope_retrieval(client):
    body = client.post(
        "/retrieve", json={"query": "limits", "filters": {"doc_id": "api-limits"}}
    ).json()
    assert all(c["doc_id"] == "api-limits" for c in body["chunks"])


def test_stream_emits_sources_then_tokens_then_final(client):
    with client.stream(
        "POST", "/ask/stream", json={"query": "What are the API rate limits?"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())

    assert events[0] == "sources"
    assert "token" in events
    assert events[-1] == "done"
    assert "final" in events


def test_stream_final_payload_is_valid_json(client):
    with client.stream("POST", "/ask/stream", json={"query": "refund policy"}) as resp:
        raw = "".join(resp.iter_text())
    finals = [
        block.split("data:", 1)[1].strip()
        for block in raw.split("\n\n")
        if "event: final" in block
    ]
    assert finals
    payload = json.loads(finals[0])
    assert "answer" in payload and "sources" in payload


def test_metrics_endpoint_shape(client):
    client.post("/ask", json={"query": "webhook retry schedule"})
    body = client.get("/metrics").json()
    assert "latency_ms" in body and "p95" in body["latency_ms"]
    assert body["llm_calls"] >= 1


def test_reindex_requires_admin_token(client, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert client.post("/admin/reindex").status_code == 403
