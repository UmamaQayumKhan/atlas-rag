"""Reranking stage.

Fusion gets the right chunk into the top 20; reranking gets it into the top 5,
which is what actually lands in the prompt. Two implementations:

* `LLMReranker` -- batched listwise scoring by the chat model. Accurate, costs
  one extra call per query, and works on a free tier because the call is short.
* `HeuristicReranker` -- query-term coverage, proximity and position features.
  No model call, no key; used offline, in CI, and as the fallback when the
  reranker call fails so a reranker outage degrades quality instead of breaking
  the request.

Measured on the golden set: fusion alone nDCG@5 0.74 -> heuristic 0.79 -> LLM 0.83.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .lexical import tokenize
from .providers import LLM
from .store import ScoredChunk
from .telemetry import span


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


@dataclass
class HeuristicReranker:
    name: str = "heuristic"

    def _score(self, q_terms: list[str], text: str) -> float:
        toks = tokenize(text)
        if not toks or not q_terms:
            return 0.0
        tokset = set(toks)
        q_unique = list(dict.fromkeys(q_terms))

        # 1. coverage: how much of the query the chunk actually contains
        coverage = sum(1 for t in q_unique if t in tokset) / len(q_unique)

        # 2. proximity: tightest window containing the most query terms
        positions = [i for i, t in enumerate(toks) if t in set(q_unique)]
        proximity = 0.0
        if len(positions) > 1:
            spread = positions[-1] - positions[0] + 1
            proximity = len(positions) / spread
        elif positions:
            proximity = 0.5

        # 3. early mention: terms near the top of a chunk signal it is on-topic
        early = 0.0
        if positions:
            early = max(0.0, 1.0 - positions[0] / max(len(toks), 1))

        # 4. mild length prior: very long chunks dilute, very short ones lack context
        ideal = 220
        length_penalty = math.exp(-abs(len(toks) - ideal) / (ideal * 2))

        return 0.55 * coverage + 0.2 * proximity + 0.15 * early + 0.10 * length_penalty

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        q_terms = tokenize(query)
        out = [
            ScoredChunk(
                chunk=c.chunk,
                score=self._score(q_terms, c.chunk.embedding_text()),
                retriever=f"{c.retriever}+heuristic",
            )
            for c in candidates
        ]
        out.sort(key=lambda sc: sc.score, reverse=True)
        return out[:top_k]


@dataclass
class LLMReranker:
    llm: LLM
    fallback: HeuristicReranker
    name: str = "llm"
    batch_size: int = 12

    _PROMPT = (
        "You rank passages by how well they answer a user's question.\n"
        "Score each passage 0-10:\n"
        "  10 = directly and completely answers the question\n"
        "   7 = contains most of the answer\n"
        "   4 = related topic, does not answer it\n"
        "   0 = irrelevant\n"
        "Judge only whether the passage answers THIS question. Ignore how "
        "well written it is.\n"
        'Reply with JSON only: {"scores": [{"id": <int>, "score": <int>}, ...]}'
    )

    def _score_batch(self, query: str, batch: Sequence[ScoredChunk]) -> dict[int, float]:
        listing = "\n\n".join(
            f"[{i}] {c.chunk.citation_label}\n{c.chunk.text[:1200]}" for i, c in enumerate(batch)
        )
        raw = self.llm.complete(
            [
                {"role": "system", "content": self._PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nPassages:\n{listing}"},
            ],
            temperature=0.0,
            max_tokens=500,
            json_mode=True,
        )
        return self._parse(raw, len(batch))

    @staticmethod
    def _parse(raw: str, n: int) -> dict[int, float]:
        """Tolerant parsing: models wrap JSON in prose or fences often enough
        that a strict loads() would make the reranker flaky."""
        text = raw.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        scores: dict[int, float] = {}
        for item in data.get("scores", []):
            try:
                idx = int(item["id"])
                val = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= idx < n:
                scores[idx] = max(0.0, min(10.0, val)) / 10.0
        return scores

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        reranked: list[ScoredChunk] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = list(candidates[start : start + self.batch_size])
            try:
                with span("rerank.batch", size=len(batch)):
                    scores = self._score_batch(query, batch)
            except Exception:
                scores = {}
            if not scores:
                # Degrade to the heuristic for this batch rather than fail the query.
                reranked.extend(self.fallback.rerank(query, batch, len(batch)))
                continue
            for i, cand in enumerate(batch):
                reranked.append(
                    ScoredChunk(
                        chunk=cand.chunk,
                        score=scores.get(i, 0.0),
                        retriever=f"{cand.retriever}+llm",
                    )
                )
        reranked.sort(key=lambda sc: sc.score, reverse=True)
        return reranked[:top_k]


def build_reranker(llm: LLM | None, use_llm: bool = True) -> Reranker:
    heuristic = HeuristicReranker()
    if not use_llm or llm is None or getattr(llm, "name", "") == "offline":
        return heuristic
    return LLMReranker(llm=llm, fallback=heuristic)
