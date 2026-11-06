"""Vector store behind one interface, with a Qdrant backend for real
deployments and a pure-Python backend for tests and small corpora.

Qdrant is the production choice here: it does metadata filtering and payload
indexing server-side, runs in one container with no managed-service bill, and
supports the named-vector layout this project uses. The in-memory backend
implements the same four methods so the whole test suite runs with no services.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    retriever: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 5), "retriever": self.retriever, **asdict(self.chunk)}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    payload = asdict(chunk)
    payload["section_path"] = list(chunk.section_path)
    return payload


def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    data = dict(payload)
    data["section_path"] = tuple(data.get("section_path") or ())
    data.setdefault("metadata", {})
    allowed = set(Chunk.__dataclass_fields__)
    return Chunk(**{k: v for k, v in data.items() if k in allowed})


class VectorStore(Protocol):
    def ensure_collection(self, dim: int) -> None: ...

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self, vector: Sequence[float], top_k: int, flt: dict[str, Any] | None = None
    ) -> list[ScoredChunk]: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def count(self) -> int: ...


class MemoryStore:
    """Brute-force cosine search. O(n) per query, which is fine up to a few tens
    of thousands of chunks and removes every external dependency from CI."""

    def __init__(self, persist_path: str | None = None) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._chunks: dict[str, Chunk] = {}
        self._persist = Path(persist_path) if persist_path else None
        if self._persist and self._persist.exists():
            self.load()

    def ensure_collection(self, dim: int) -> None:  # noqa: ARG002 - nothing to create
        return None

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for chunk, vec in zip(chunks, vectors):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = list(vec)
        if self._persist:
            self.save()

    @staticmethod
    def _matches(chunk: Chunk, flt: dict[str, Any] | None) -> bool:
        if not flt:
            return True
        for key, want in flt.items():
            got = getattr(chunk, key, None)
            if got is None:
                got = chunk.metadata.get(key)
            if isinstance(want, (list, tuple, set)):
                if got not in want:
                    return False
            elif got != want:
                return False
        return True

    def search(
        self, vector: Sequence[float], top_k: int, flt: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=self._chunks[cid], score=cosine(vector, vec), retriever="dense")
            for cid, vec in self._vectors.items()
            if self._matches(self._chunks[cid], flt)
        ]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        return sorted(self._chunks.values(), key=lambda c: (c.doc_id, c.ordinal))

    def save(self) -> None:
        if not self._persist:
            return
        self._persist.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist, "w", encoding="utf-8") as fh:
            for cid, chunk in self._chunks.items():
                fh.write(
                    json.dumps({"chunk": _chunk_to_payload(chunk), "vector": self._vectors[cid]})
                    + "\n"
                )

    def load(self) -> None:
        if not self._persist or not self._persist.exists():
            return
        with open(self._persist, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                chunk = _payload_to_chunk(row["chunk"])
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = row["vector"]


class QdrantStore:
    """Talks to Qdrant over its REST API with httpx.

    Using the HTTP API rather than the SDK keeps the dependency surface small
    and makes the wire format visible -- useful when debugging why a metadata
    filter is not matching.
    """

    def __init__(self, url: str, collection: str, api_key: str = "", timeout: float = 30.0):
        import httpx  # imported lazily so MemoryStore users need no httpx

        self.collection = collection
        self.base = url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["api-key"] = api_key
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def _req(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        resp = self._client.request(method, f"{self.base}{path}", **kw)
        if resp.status_code >= 400:
            raise RuntimeError(f"qdrant {method} {path} [{resp.status_code}]: {resp.text[:300]}")
        return resp.json()

    def ensure_collection(self, dim: int) -> None:
        try:
            self._req("GET", f"/collections/{self.collection}")
            return
        except RuntimeError:
            pass
        self._req(
            "PUT",
            f"/collections/{self.collection}",
            json={"vectors": {"size": dim, "distance": "Cosine"}},
        )
        # Payload indexes make metadata filtering cheap instead of a full scan.
        for field_name, schema in (("doc_id", "keyword"), ("source", "keyword")):
            try:
                self._req(
                    "PUT",
                    f"/collections/{self.collection}/index",
                    json={"field_name": field_name, "field_schema": schema},
                )
            except RuntimeError:
                pass

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        # Qdrant wants a UUID or unsigned int; derive one deterministically so
        # re-ingesting the same chunk updates in place instead of duplicating.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        points = [
            {
                "id": self._point_id(c.chunk_id),
                "vector": list(v),
                "payload": _chunk_to_payload(c),
            }
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), 128):
            self._req(
                "PUT",
                f"/collections/{self.collection}/points",
                params={"wait": "true"},
                json={"points": points[i : i + 128]},
            )

    @staticmethod
    def _to_filter(flt: dict[str, Any] | None) -> dict[str, Any] | None:
        if not flt:
            return None
        must = []
        for key, want in flt.items():
            if isinstance(want, (list, tuple, set)):
                must.append({"key": key, "match": {"any": list(want)}})
            else:
                must.append({"key": key, "match": {"value": want}})
        return {"must": must}

    def search(
        self, vector: Sequence[float], top_k: int, flt: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        body: dict[str, Any] = {
            "vector": list(vector),
            "limit": top_k,
            "with_payload": True,
        }
        qfilter = self._to_filter(flt)
        if qfilter:
            body["filter"] = qfilter
        res = self._req("POST", f"/collections/{self.collection}/points/search", json=body)
        return [
            ScoredChunk(
                chunk=_payload_to_chunk(hit["payload"]),
                score=float(hit["score"]),
                retriever="dense",
            )
            for hit in res.get("result", [])
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        try:
            res = self._req(
                "GET", f"/collections/{self.collection}/points/{self._point_id(chunk_id)}"
            )
        except RuntimeError:
            return None
        payload = (res.get("result") or {}).get("payload")
        return _payload_to_chunk(payload) if payload else None

    def count(self) -> int:
        res = self._req(
            "POST", f"/collections/{self.collection}/points/count", json={"exact": True}
        )
        return int((res.get("result") or {}).get("count", 0))


def build_store(settings: Any = None) -> VectorStore:
    from .config import get_settings

    s = settings or get_settings()
    if s.vector_backend.lower() in {"memory", "inmemory", "local"}:
        return MemoryStore(persist_path=str(Path(s.index_dir) / "vectors.jsonl"))
    return QdrantStore(
        url=s.qdrant_url,
        collection=s.collection,
        api_key=s.qdrant_api_key,
        timeout=s.request_timeout,
    )
