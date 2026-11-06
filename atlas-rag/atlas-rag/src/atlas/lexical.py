"""BM25 lexical index, implemented directly rather than pulled from a library.

The dense arm of a hybrid search fails on exactly the queries enterprise users
type most: error codes, SKUs, product names, config keys. "SSO_REDIRECT_URI" has
no useful embedding neighbourhood, but BM25 nails it. Keeping the
implementation here (~80 lines) means the scoring is inspectable and there is no
hidden tokenisation mismatch between the two arms.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9_]+")

# Dropped before indexing. Deliberately short: aggressive stoplists destroy
# phrase queries like "how to cancel".
_STOP = frozenset(
    "a an the of to in for on at by is are was were be been am do does did and or "
    "but if then than that this these those it its as with from".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word, and additionally split camelCase and
    snake_case so `SSO_REDIRECT_URI` matches a query for `redirect uri`."""
    raw = _TOKEN.findall(text.lower())
    out: list[str] = []
    for tok in raw:
        out.append(tok)
        if "_" in tok:
            out.extend(part for part in tok.split("_") if part)
    return [t for t in out if t not in _STOP and len(t) > 1]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    doc_ids: list[str] = field(default_factory=list)
    doc_tokens: list[Counter] = field(default_factory=list)
    doc_lens: list[int] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    avg_len: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        counts = Counter(toks)
        self.doc_ids.append(doc_id)
        self.doc_tokens.append(counts)
        self.doc_lens.append(len(toks))
        for term in counts:
            self.df[term] += 1
        self.avg_len = sum(self.doc_lens) / max(1, len(self.doc_lens))

    def add_many(self, items: list[tuple[str, str]]) -> None:
        for doc_id, text in items:
            self.add(doc_id, text)

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        # Robertson/Sparck-Jones idf with +1 smoothing to stay non-negative.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        if not q_terms or not self.doc_ids:
            return []
        scores: list[tuple[str, float]] = []
        for i, counts in enumerate(self.doc_tokens):
            dl = self.doc_lens[i] or 1
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                score += self._idf(term) * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((self.doc_ids[i], score))
        scores.sort(key=lambda kv: kv[1], reverse=True)
        return scores[:top_k]

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} is not a BM25Index")
        return obj
