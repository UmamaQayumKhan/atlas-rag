"""Retrieval and answer metrics, implemented rather than imported.

RAGAS is the usual answer and this project is compatible with it, but every
RAGAS metric needs an LLM call per sample, which on a free tier means a 40-item
eval becomes a rate-limit exercise. These deterministic metrics run in
milliseconds with no key, so they can gate CI on every push; the LLM judge in
`judge.py` is then run on demand for the subjective part.

Definitions used here:

* **Recall@k**  -- did any chunk from an expected document make the top k?
  For a QA corpus where one document holds the answer, document-level recall is
  the metric that actually predicts answer quality; chunk-level recall punishes
  the system for picking a different-but-equally-correct chunk.
* **MRR**       -- 1/rank of the first relevant chunk, averaged.
* **nDCG@k**    -- rank-weighted gain, so being right at position 1 beats being
  right at position 5.
* **Refusal accuracy** -- on unanswerable questions, did the system refuse?
  Tracked separately because a system that answers everything scores well on
  recall and is useless in production.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class GoldenItem:
    id: str
    question: str
    expected_docs: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    answerable: bool = True
    category: str = "factual"


def load_golden(path: str | Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            raw = json.loads(line)
            items.append(
                GoldenItem(
                    id=raw["id"],
                    question=raw["question"],
                    expected_docs=raw.get("expected_docs", []),
                    must_contain=raw.get("must_contain", []),
                    answerable=raw.get("answerable", True),
                    category=raw.get("category", "factual"),
                )
            )
    return items


def _doc_base(doc_id: str) -> str:
    """`billing-policy-p003` and `billing-policy::0007` both belong to the
    `billing-policy` document."""
    base = doc_id.split("::")[0]
    return re.sub(r"-p\d+$", "", base)


def recall_at_k(retrieved_docs: Sequence[str], expected: Sequence[str], k: int) -> float:
    if not expected:
        return 1.0
    top = set(_dedupe(retrieved_docs)[:k])
    wanted = {_doc_base(d) for d in expected}
    return 1.0 if top & wanted else 0.0


def precision_at_k(retrieved_docs: Sequence[str], expected: Sequence[str], k: int) -> float:
    if not expected or k == 0:
        return 0.0
    wanted = {_doc_base(d) for d in expected}
    top = _dedupe(retrieved_docs)[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in wanted) / len(top)


def reciprocal_rank(retrieved_docs: Sequence[str], expected: Sequence[str]) -> float:
    if not expected:
        return 0.0
    wanted = {_doc_base(d) for d in expected}
    for i, doc in enumerate(_dedupe(retrieved_docs), start=1):
        if doc in wanted:
            return 1.0 / i
    return 0.0


def _dedupe(docs: Sequence[str]) -> list[str]:
    """Collapse repeated documents, keeping first appearance.

    Relevance here is document-level, so several chunks of the same document
    are one relevant item. Without this, DCG sums the same document five times
    while the ideal DCG assumes one, and nDCG climbs above 1.0 -- which is how
    this bug announced itself.
    """
    seen: set[str] = set()
    out: list[str] = []
    for doc in docs:
        base = _doc_base(doc)
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


def ndcg_at_k(retrieved_docs: Sequence[str], expected: Sequence[str], k: int) -> float:
    if not expected:
        return 0.0
    wanted = {_doc_base(d) for d in expected}
    unique = _dedupe(retrieved_docs)
    gains = [1.0 if d in wanted else 0.0 for d in unique[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_hits = min(len(wanted), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return min(1.0, dcg / idcg) if idcg else 0.0


def contains_all(answer: str, needles: Iterable[str]) -> bool:
    lowered = answer.lower()
    return all(n.lower() in lowered for n in needles)


def contains_any(answer: str, needles: Iterable[str]) -> bool:
    needles = list(needles)
    if not needles:
        return True
    lowered = answer.lower()
    return any(n.lower() in lowered for n in needles)


@dataclass
class Aggregate:
    """Accumulates per-item scores and reports means plus a per-category
    breakdown, which is where the useful signal is: an overall 0.88 hides that
    'trap' questions are at 0.55."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, **row: Any) -> None:
        self.rows.append(row)

    def _mean(self, key: str, rows: Sequence[dict[str, Any]] | None = None) -> float:
        src = rows if rows is not None else self.rows
        vals = [r[key] for r in src if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def report(self) -> dict[str, Any]:
        answerable = [r for r in self.rows if r.get("answerable")]
        unanswerable = [r for r in self.rows if not r.get("answerable")]

        overall = {
            "n": len(self.rows),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "recall@1": self._mean("recall@1", answerable),
            "recall@3": self._mean("recall@3", answerable),
            "recall@5": self._mean("recall@5", answerable),
            "precision@5": self._mean("precision@5", answerable),
            "mrr": self._mean("mrr", answerable),
            "ndcg@5": self._mean("ndcg@5", answerable),
            "answer_in_context": self._mean("answer_in_context", answerable),
        }

        if any("answer_correct" in r for r in self.rows):
            overall |= {
                "answer_accuracy": self._mean("answer_correct", answerable),
                "citation_coverage": self._mean("citation_coverage", answerable),
                "grounded_rate": self._mean("grounded", answerable),
                "hallucinated_citations": sum(
                    int(r.get("unknown_citations", 0) or 0) for r in self.rows
                ),
            }
            if unanswerable:
                overall["refusal_accuracy"] = self._mean("refused", unanswerable)
                overall["false_answer_rate"] = round(
                    1.0 - self._mean("refused", unanswerable), 4
                )
            overall["over_refusal_rate"] = round(self._mean("refused", answerable), 4)

        by_cat: dict[str, dict[str, Any]] = {}
        for cat in sorted({r.get("category", "?") for r in self.rows}):
            subset = [r for r in self.rows if r.get("category") == cat]
            entry = {
                "n": len(subset),
                "recall@5": self._mean("recall@5", subset),
                "answer_in_context": self._mean("answer_in_context", subset),
            }
            if any("answer_correct" in r for r in subset):
                entry["answer_accuracy"] = self._mean("answer_correct", subset)
            by_cat[cat] = entry

        return {
            "overall": overall,
            "by_category": by_cat,
            "failures": [
                {
                    "id": r["id"],
                    "category": r.get("category"),
                    "question": r.get("question"),
                    "reason": r.get("failure_reason", "retrieval miss"),
                    "retrieved": r.get("retrieved_docs", [])[:3],
                }
                for r in self.rows
                if r.get("recall@5") == 0.0
                or (r.get("answerable") and r.get("answer_correct") == 0.0)
                or (not r.get("answerable") and r.get("refused") == 0.0)
            ],
        }
