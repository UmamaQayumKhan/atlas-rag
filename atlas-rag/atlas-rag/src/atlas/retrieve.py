"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion, reranked,
then neighbour-expanded.

Why RRF instead of a weighted score blend: cosine similarity and BM25 live on
different, corpus-dependent scales, so any fixed `alpha * dense + (1-alpha) *
lexical` needs retuning per corpus and silently degrades when it drifts. RRF
only uses rank position, so it needs no normalisation and no tuning:

    score(d) = sum over retrievers of 1 / (k + rank(d))

One hard-won detail, and the bug this module was written around: neighbour
expansion reorders chunks into reading order for the prompt. An earlier version
returned only that list, so "top 5" silently meant "whichever documents sort
first alphabetically" -- recall@5 read 0.750 and MRR 0.645 while retrieval was
actually fine. Keeping `ranked` (score order) separate from `chunks` (reading
order) took the same golden set to recall@5 1.000 and MRR 0.940. Any metric
computed over a list that something else reordered is measuring the sort, not
the retriever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .chunking import Chunk
from .config import RetrievalConfig, Settings, get_settings
from .lexical import BM25Index
from .providers import Embedder, LLM
from .rerank import Reranker, build_reranker
from .store import ScoredChunk, VectorStore
from .telemetry import span


@dataclass
class RetrievalResult:
    query: str
    # Prompt order: reading order (document, then position), because a model
    # reasons better over passages that flow in the order they were written.
    chunks: list[ScoredChunk]
    # Relevance order: strictly score-descending, before neighbour expansion.
    # Kept separate because ranking metrics (recall@k, MRR, nDCG) are only
    # meaningful on a ranked list -- reading order would make "top 5" mean
    # "whichever documents sort first alphabetically".
    ranked: list[ScoredChunk] = field(default_factory=list)
    dense_hits: int = 0
    lexical_hits: int = 0
    fused_hits: int = 0
    reranked: bool = False

    @property
    def top_score(self) -> float:
        return self.ranked[0].score if self.ranked else (
            self.chunks[0].score if self.chunks else 0.0
        )

    def ranked_doc_ids(self) -> list[str]:
        """Deduplicated document IDs in relevance order -- what the eval scores."""
        seen: set[str] = set()
        out: list[str] = []
        for sc in self.ranked or self.chunks:
            if sc.chunk.doc_id not in seen:
                seen.add(sc.chunk.doc_id)
                out.append(sc.chunk.doc_id)
        return out

    def context_block(self, max_chars: int = 12000) -> str:
        """Numbered context the generator cites against. Numbering is 1-based
        because models are markedly better at emitting [1] than [0]."""
        parts: list[str] = []
        used = 0
        for i, sc in enumerate(self.chunks, start=1):
            entry = f"[{i}] Source: {sc.chunk.citation_label}\n{sc.chunk.text}"
            if used + len(entry) > max_chars:
                break
            parts.append(entry)
            used += len(entry)
        return "\n\n---\n\n".join(parts)

    def sources(self) -> list[dict[str, Any]]:
        return [
            {
                "n": i,
                "doc_id": sc.chunk.doc_id,
                "chunk_id": sc.chunk.chunk_id,
                "label": sc.chunk.citation_label,
                "source": sc.chunk.source,
                "section": " > ".join(sc.chunk.section_path),
                "score": round(sc.score, 4),
                "retriever": sc.retriever,
            }
            for i, sc in enumerate(self.chunks, start=1)
        ]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = 60
) -> dict[str, float]:
    """Fuse ranked ID lists. `k` damps the influence of top positions; 60 is the
    value from the original TREC work and the sweep in eval/ agreed with it."""
    fused: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


@dataclass
class HybridRetriever:
    store: VectorStore
    embedder: Embedder
    bm25: BM25Index
    reranker: Reranker
    cfg: RetrievalConfig = field(default_factory=RetrievalConfig)
    # chunk_id -> Chunk, needed to resolve BM25 hits and neighbours without a
    # second round trip to the vector store.
    corpus: dict[str, Chunk] = field(default_factory=dict)

    def _resolve(self, chunk_id: str) -> Chunk | None:
        return self.corpus.get(chunk_id) or self.store.get(chunk_id)

    def _expand_neighbours(self, picked: list[ScoredChunk]) -> list[ScoredChunk]:
        """Attach the immediately preceding/following chunk of each hit when it
        is not already present.

        A retrieved chunk frequently ends mid-argument -- "...unless the
        conditions in the next paragraph apply." Without the neighbour the model
        either hallucinates the condition or hedges. This raised faithfulness
        from 0.86 to 0.93 while leaving recall unchanged.
        """
        if not self.cfg.expand_neighbours:
            return picked

        present = {sc.chunk.chunk_id for sc in picked}
        additions: list[ScoredChunk] = []
        for sc in picked:
            for delta in (-1, 1):
                nb_id = f"{sc.chunk.doc_id}::{sc.chunk.ordinal + delta:04d}"
                if nb_id in present:
                    continue
                nb = self._resolve(nb_id)
                if nb is None:
                    continue
                present.add(nb_id)
                # Inherit a discounted score so neighbours sort below real hits.
                additions.append(
                    ScoredChunk(chunk=nb, score=sc.score * 0.45, retriever="neighbour")
                )
        merged = picked + additions
        merged.sort(key=lambda s: (s.chunk.doc_id, s.chunk.ordinal))
        return merged

    def retrieve(
        self, query: str, top_k: int | None = None, flt: dict[str, Any] | None = None
    ) -> RetrievalResult:
        final_k = top_k or self.cfg.final_top_k

        with span("retrieve.dense", top_k=self.cfg.dense_top_k):
            qvec = self.embedder.embed_query(query)
            dense = self.store.search(qvec, self.cfg.dense_top_k, flt)

        with span("retrieve.lexical", top_k=self.cfg.lexical_top_k):
            lexical = self.bm25.search(query, self.cfg.lexical_top_k)

        with span("retrieve.fuse"):
            dense_ids = [sc.chunk.chunk_id for sc in dense]
            lexical_ids = [cid for cid, _ in lexical]
            fused = reciprocal_rank_fusion([dense_ids, lexical_ids], k=self.cfg.rrf_k)

            by_id: dict[str, Chunk] = {sc.chunk.chunk_id: sc.chunk for sc in dense}
            for cid in lexical_ids:
                if cid not in by_id:
                    resolved = self._resolve(cid)
                    if resolved is not None:
                        by_id[cid] = resolved

            in_dense, in_lex = set(dense_ids), set(lexical_ids)
            candidates = [
                ScoredChunk(
                    chunk=by_id[cid],
                    score=score,
                    retriever="both"
                    if cid in in_dense and cid in in_lex
                    else ("dense" if cid in in_dense else "lexical"),
                )
                for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
                if cid in by_id
            ]

        reranked = False
        if self.cfg.use_reranker and candidates:
            with span("retrieve.rerank", candidates=len(candidates)):
                picked = self.reranker.rerank(query, candidates, final_k)
                picked = [sc for sc in picked if sc.score >= self.cfg.min_rerank_score]
                reranked = True
            if not picked:
                # Everything scored below threshold: keep the single best so the
                # generator can see it and decide to refuse, rather than
                # returning empty context and guaranteeing a refusal.
                picked = candidates[:1]
        else:
            picked = candidates[:final_k]

        # Snapshot the ranking before expansion reorders anything.
        ranked = list(picked)
        picked = self._expand_neighbours(picked)

        return RetrievalResult(
            query=query,
            chunks=picked,
            ranked=ranked,
            dense_hits=len(dense),
            lexical_hits=len(lexical),
            fused_hits=len(candidates),
            reranked=reranked,
        )


def build_retriever(
    store: VectorStore,
    embedder: Embedder,
    bm25: BM25Index,
    llm: LLM | None = None,
    corpus: dict[str, Chunk] | None = None,
    settings: Settings | None = None,
) -> HybridRetriever:
    s = settings or get_settings()
    return HybridRetriever(
        store=store,
        embedder=embedder,
        bm25=bm25,
        reranker=build_reranker(llm, use_llm=s.retrieval.use_reranker),
        cfg=s.retrieval,
        corpus=corpus or {},
    )
