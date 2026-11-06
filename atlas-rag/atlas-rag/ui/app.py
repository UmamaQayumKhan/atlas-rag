"""Streamlit chat UI for AtlasRAG.

Talks to the FastAPI service over SSE rather than importing the pipeline, so the
UI exercises the same public API an external client would. Importing the
pipeline directly would make the UI pass while the API was broken.

    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import httpx
import streamlit as st

API_URL = os.getenv("ATLAS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="AtlasRAG", page_icon="📚", layout="wide")


def api_get(path: str) -> dict[str, Any] | None:
    try:
        resp = httpx.get(f"{API_URL}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def stream_answer(query: str, history: list[dict[str, str]], top_k: int) -> Iterator[tuple[str, Any]]:
    payload = {"query": query, "history": history[-4:], "top_k": top_k}
    with httpx.stream(
        "POST", f"{API_URL}/ask/stream", json=payload, timeout=120
    ) as resp:
        event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event:
                raw = line.split(":", 1)[1].strip()
                try:
                    yield event, json.loads(raw)
                except json.JSONDecodeError:
                    continue


# --- sidebar -------------------------------------------------------------

with st.sidebar:
    st.title("AtlasRAG")
    st.caption("Hybrid-retrieval RAG over enterprise documentation")

    health = api_get("/healthz")
    if health is None:
        st.error(f"API unreachable at {API_URL}")
        st.code("uvicorn atlas.api:app --reload", language="bash")
    else:
        ok = health.get("status") == "ok"
        (st.success if ok else st.warning)(f"API {health.get('status')}")
        c1, c2 = st.columns(2)
        c1.metric("Chunks", health.get("indexed_chunks", 0))
        c2.metric("BM25 docs", health.get("bm25_docs", 0))
        st.caption(
            f"**Model** {health.get('llm_model')}  \n"
            f"**Embeddings** {health.get('embedder')}  \n"
            f"**Store** {health.get('vector_backend')}  \n"
            f"**Chunking** {health.get('chunk_strategy')}  \n"
            f"**Reranker** {health.get('reranker')}"
        )
        if health.get("offline_mode"):
            st.info("No API key set — answers come from the offline stub.")

    st.divider()
    top_k = st.slider("Passages in context", 1, 12, 5)
    show_debug = st.toggle("Show retrieval debug", value=False)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    metrics = api_get("/metrics")
    if metrics and metrics.get("llm_calls"):
        st.divider()
        st.caption("**This session**")
        st.caption(
            f"calls {metrics['llm_calls']} · "
            f"p95 {metrics['latency_ms']['p95']} ms · "
            f"${metrics['total_cost_usd']:.4f}"
        )

# --- chat ----------------------------------------------------------------

st.session_state.setdefault("messages", [])

if not st.session_state.messages:
    st.markdown("#### Ask about the indexed documentation")
    cols = st.columns(2)
    examples = [
        "When is a workspace suspended after a failed payment?",
        "Why am I getting SAML_AUDIENCE_MISMATCH?",
        "How many writes per minute can a Growth workspace sustain?",
        "Is Northwind HIPAA eligible?",
    ]
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, use_container_width=True, key=f"ex{i}"):
            st.session_state.pending = example
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"{len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    st.markdown(f"**[{src['n']}]** {src['label']}  \n`{src['retriever']}` · score {src['score']}")

prompt = st.chat_input("Ask a question…") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        sources: list[dict[str, Any]] = []
        final: dict[str, Any] = {}
        buffer = ""

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        try:
            for event, data in stream_answer(prompt, history, top_k):
                if event == "sources":
                    sources = data
                elif event == "token":
                    buffer += data.get("text", "")
                    placeholder.markdown(buffer + "▌")
                elif event == "final":
                    final = data
                elif event == "error":
                    st.error(f"{data.get('error')}: {data.get('detail')}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")

        answer_text = final.get("answer") or buffer or "_no response_"
        placeholder.markdown(answer_text)

        shown = final.get("sources") or sources
        if shown:
            with st.expander(f"{len(shown)} source(s)"):
                for src in shown:
                    st.markdown(
                        f"**[{src['n']}]** {src['label']}  \n"
                        f"`{src['retriever']}` · score {src['score']}"
                    )

        badges = []
        if final.get("grounded"):
            badges.append("✅ grounded")
        if final.get("refused"):
            badges.append("🚫 refused — not in the documents")
        guard = final.get("guardrail") or {}
        if guard.get("injection_found"):
            badges.append(f"⚠️ injection neutralised ({', '.join(guard['injection_kinds'])})")
        if final.get("unknown_citations"):
            badges.append("❗ invalid citations — answer withheld")
        if badges:
            st.caption(" · ".join(badges))

        if show_debug and final.get("retrieval"):
            st.json(final["retrieval"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "sources": shown}
        )
