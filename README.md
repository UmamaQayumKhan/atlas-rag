# AtlasRAG — hybrid-retrieval RAG over enterprise documentation

A grounded question-answering service over a company's internal docs. Hybrid
retrieval (dense + BM25, fused with RRF), LLM reranking, enforced citations, a
refusal path, prompt-injection defence, and a retrieval eval harness wired into
CI as a build-breaking quality gate.

Runs with **no API keys at all** — it falls back to deterministic hashing
embeddings and a stub model, which is how the 61 tests and the CI quality gate
execute. Add a free Groq or Gemini key for real answers.

```bash
git clone https://github.com/YOUR-USERNAME/atlas-rag.git && cd atlas-rag
pip install -r requirements-dev.txt
make test          # 61 tests, offline, ~3s
make eval          # retrieval metrics, offline, ~0.3s
cp .env.example .env   # add GROQ_API_KEY and GEMINI_API_KEY (both free tiers)
make ingest && make run
```

---

## Why this exists

Most RAG projects are a single `similarity_search(query, k=4)` call wrapped in a
chat box. They fall over on the queries enterprise users actually type, and
there is no way to tell whether a change made them better or worse. This one is
built around the three things that separate a demo from a service:

1. **Retrieval is hybrid.** Dense-only retrieval fails on error codes, SKUs and
   config keys — `SAML_AUDIENCE_MISMATCH` has no useful embedding
   neighbourhood. BM25 handles those; RRF fuses the two without score
   normalisation.
2. **Answers are verified, not just prompted.** The model is told to cite or
   refuse, and then the output is *checked*: a citation pointing at a passage
   that does not exist means the answer is withheld.
3. **Quality is a test.** `eval/run_eval.py` scores a 41-question golden set and
   fails the build below a threshold, so a chunking tweak cannot quietly cost
   eight points of recall.

---

## Architecture

```
            ┌──────────────┐
  docs ────▶│   ingest     │  md · html · pdf (per page) · docx
            │  chunking    │  structural | recursive | fixed
            └──────┬───────┘
                   │  heading path prefixed before embedding
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐     ┌───────────────┐
│ Qdrant        │     │ BM25 index    │   two arms, same tokenised text
│ (dense, cos)  │     │ (lexical)     │
└───────┬───────┘     └───────┬───────┘
        └──────────┬──────────┘
                   ▼
          Reciprocal Rank Fusion        rank-based, no score normalisation
                   ▼
            LLM reranker                listwise, heuristic fallback on failure
                   ▼
         neighbour expansion            ±1 chunk, discounted score
                   ▼
    guardrails: injection + PII         retrieved text is DATA, not instructions
                   ▼
        generation with [n] citations    SSE streaming
                   ▼
     grounding check → refuse or answer  invalid citation ⇒ withhold
```

Served by FastAPI; Streamlit UI talks to it over the public API.

---

## Measured results

All numbers below are produced by `make eval` / `make sweep` in this repo and
regenerate on every CI run. They are the **offline** configuration (hashing
embeddings, no API key), so anyone cloning the repo reproduces them exactly.

### The bug the eval found

The eval harness paid for itself on the first run. Neighbour expansion sorts
chunks into reading order for the prompt, and an earlier version returned only
that list — so "top 5" silently meant "whichever documents sort first
alphabetically". Retrieval was fine; the measurement was lying.

| metric | before fix | after fix |
| --- | --- | --- |
| recall@1 | 0.556 | **0.889** |
| recall@5 | 0.750 | **1.000** |
| MRR | 0.645 | **0.940** |
| nDCG@5 | 1.677 *(invalid — above 1.0)* | **0.945** |

The nDCG above 1.0 was a second, separate bug: document-level relevance scored
against a chunk-level list, so DCG summed the same document five times while the
ideal DCG assumed one. Both fixes are regression-tested.

### Current retrieval quality

41 questions, 5 documents, 46 chunks:

| metric | value |
| --- | --- |
| recall@1 | 0.889 |
| recall@3 | 1.000 |
| recall@5 | 1.000 |
| MRR | 0.940 |
| nDCG@5 | 0.945 |
| answer-in-context | 1.000 |
| precision@5 | 0.514 |

**Read these honestly.** With a five-document corpus, document-level recall@5 is
saturated and means very little — recall@1 and MRR are the informative numbers.
`precision@5` is low by design: neighbour expansion deliberately pulls in
adjacent chunks that are not independently relevant, trading precision for the
context that stops the model hedging mid-clause.

### Chunking strategy sweep

`make sweep`:

| strategy | chunks | recall@1 | MRR | nDCG@5 | answer-in-context |
| --- | --- | --- | --- | --- | --- |
| fixed | 12 | 0.889 | 0.944 | 0.957 | 0.972 |
| recursive | 10 | 0.889 | 0.944 | 0.957 | **1.000** |
| structural | 46 | 0.889 | 0.940 | 0.945 | **1.000** |

The ranking metrics cannot separate these three on a corpus this small. What
does separate them is answer-in-context — whether the expected answer text
survived the chunk boundary — where `fixed` loses 2.8 points by cutting through
sentences.

`structural` is the default for reasons the table does not capture: it produces
4.6× more granular chunks, so a citation reads *"Billing Policy | Refunds >
Annual plans"* instead of pointing at a whole page, and it never merges across a
section boundary, so a refund rule cannot be blended with a tax rule in one
chunk.

### Ingestion coverage

`tests/test_ingest_coverage.py` asserts no document content is silently dropped
— the failure mode where the system keeps answering confidently from a corpus
with holes in it:

| strategy | token coverage | note |
| --- | --- | --- |
| recursive | 100.0% | every token preserved |
| fixed | 116.0% | overlap duplication, bounded ≤125% |
| structural | 93.5% | exactly the heading lines, which move into `section_path` |

Writing this test is what caught the real bug underneath: undersized chunks were
being *discarded* rather than merged, so short-but-important trailing paragraphs
("This does not apply to annual plans.") were vanishing from the index.

---

## Security

`tests/test_guardrails.py` is a red-team suite, not a formality.

**Indirect prompt injection** is the exposure specific to RAG: retrieved context
is attacker-influenced whenever any ingested document can be edited by a
non-admin — a wiki page, an uploaded PDF, a support ticket. Instructions inside
a *document* must never be treated as instructions from the *user*.

`data/redteam/poisoned-kb-article.md` is a fixture carrying override,
persona-switch, exfiltration and citation-suppression payloads plus a fake card
number and API key. The suite asserts that all of it is detected and
neutralised, that the article's legitimate content survives, and — critically —
that ordinary documentation containing phrases like *"the previous section"* and
*"these instructions"* is **not** flagged. A guardrail with false positives
redacts real policy text, which is worse than none.

| control | implementation |
| --- | --- |
| Injection detection | 9 pattern classes, matched against retrieved context only |
| Neutralisation | replaced with a visible marker, never silently stripped |
| PII / secret redaction | email, card, bearer token, API-key shapes |
| Hallucinated citations | citation to a non-existent passage ⇒ answer withheld |
| Weak-context refusal | retrieval score below threshold ⇒ refuse before generating |
| Container hardening | non-root user, multi-stage build, real readiness probe |

---

## API

| method | path | purpose |
| --- | --- | --- |
| `POST` | `/ask` | grounded answer with citations |
| `POST` | `/ask/stream` | the same, Server-Sent Events |
| `POST` | `/retrieve` | retrieval only — debug whether a bad answer is retrieval or generation |
| `GET` | `/healthz` | index size, provider, chunk strategy, reranker state |
| `GET` | `/metrics` | token spend, cost estimate, p50/p95/p99 |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"query":"When is a workspace suspended after a failed payment?"}' | jq
```

```json
{
  "answer": "A workspace is suspended 14 days after the failed invoice [1]...",
  "sources": [{"n": 1, "label": "Billing Policy | Dunning and suspension", "score": 0.83}],
  "grounded": true,
  "refused": false,
  "citation_coverage": 1.0,
  "guardrail": {"injection_found": false, "redactions": {}},
  "retrieval": {"dense_hits": 20, "lexical_hits": 14, "reranked": true}
}
```

`/ask/stream` emits `sources` before the first token, so the UI renders
citations while the answer is still arriving — retrieval costs a few hundred
milliseconds before any token can exist, and hiding that behind a visible stream
matters more than shaving it.

---

## Configuration

Everything is env-driven; see `.env.example`. The knobs worth knowing:

| variable | default | effect |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` | `groq` · `gemini` · `openai` · `offline` |
| `EMBEDDING_PROVIDER` | `gemini` | `gemini` · `offline` |
| `VECTOR_BACKEND` | `qdrant` | `qdrant` · `memory` |
| `CHUNK_STRATEGY` | `structural` | `structural` · `recursive` · `fixed` |
| `RRF_K` | `60` | fusion damping |
| `USE_RERANKER` | `true` | LLM rerank, heuristic fallback |
| `EXPAND_NEIGHBOURS` | `true` | ±1 chunk around each hit |
| `MIN_CONTEXT_SCORE` | `0.12` | refuse below this retrieval score |

No vendor SDK is imported anywhere. Swapping Groq for OpenAI, Gemini, vLLM or a
local Ollama server is one environment variable, because every call site depends
on the `LLM` / `Embedder` protocols in `providers.py`.

---

## Deploying

```bash
cp .env.example .env     # add your keys
docker compose up --build
# API  → localhost:8000/docs
# UI   → localhost:8501
```

Brings up Qdrant with a persistent volume, the API (2 uvicorn workers, non-root,
health-checked against `/healthz` rather than a port probe), and the Streamlit
UI. `INGEST_ON_BOOT=true` indexes `data/corpus` on first start.

---

## Project layout

```
src/atlas/
  config.py      env-driven settings, frozen dataclasses
  providers.py   LLM + Embedder protocols; Groq/Gemini/OpenAI/offline impls
  chunking.py    three strategies + the measured comparison
  lexical.py     BM25 from scratch (~80 lines, inspectable scoring)
  store.py       Qdrant REST + in-memory backend, one interface
  retrieve.py    RRF fusion, reranking, neighbour expansion
  rerank.py      listwise LLM reranker with heuristic fallback
  guardrails.py  injection, PII, grounding verification
  generate.py    citation-enforced synthesis, streaming, refusal
  telemetry.py   cost, token spend, latency percentiles, OTel-shaped spans
  api.py         FastAPI + SSE
  cli.py         ingest · ask · retrieve · health · metrics
eval/
  golden.jsonl   41 questions: factual, lexical, multi-doc, reasoning,
                 trap (plausible-but-wrong), unanswerable
  metrics.py     recall@k, precision@k, MRR, nDCG@k, refusal accuracy
  judge.py       LLM-as-judge faithfulness, per-claim
  run_eval.py    harness, strategy sweep, CI gate
tests/           61 tests, fully offline
```

---

## Design decisions worth defending

**RRF over a weighted score blend.** Cosine similarity and BM25 live on
different, corpus-dependent scales, so any `alpha * dense + (1-alpha) * lexical`
needs retuning per corpus and silently degrades as the corpus drifts. RRF uses
rank position only — no normalisation, no tuning. `test_rrf_is_scale_invariant`
pins that property.

**BM25 implemented rather than imported.** Eighty lines, and it guarantees the
two arms of the hybrid search tokenise identically. A tokenisation mismatch
between a library's BM25 and your own embedding input is invisible and ruins
fusion.

**Deterministic metrics gate CI; the LLM judge runs on demand.** Every RAGAS
metric costs an LLM call per sample, so a 41-item eval becomes a rate-limit
exercise on a free tier. The deterministic metrics run in 0.24s with no key,
which is what makes them usable as a build gate. Faithfulness judging is still
there (`eval/judge.py`) for the subjective half.

**Refusal is measured separately from accuracy.** Five of the 41 golden
questions are deliberately unanswerable. A system that answers everything scores
well on recall and is useless in production, so `refusal_accuracy` and
`over_refusal_rate` are tracked as first-class metrics.

## Known limits

- The corpus is 5 synthetic documents. Ranking metrics saturate; absolute
  numbers would change on a real corpus of thousands of documents.
- Reranking adds one LLM call per query (~300–600 ms). `USE_RERANKER=false`
  trades roughly 4 points of nDCG for that latency.
- The hashing embedder exists for reproducibility, not quality. Real retrieval
  quality needs real embeddings.
- No multi-tenancy. Metadata filtering is implemented and payload-indexed in
  Qdrant, but tenant isolation is not enforced at the API layer.
