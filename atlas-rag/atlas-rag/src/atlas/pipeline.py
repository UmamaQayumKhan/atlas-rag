"""The façade the API, the UI and the eval harness all share.

One class owns wiring so there is exactly one definition of "the system" and the
eval numbers in the README describe the same code path a user hits. Duplicating
this wiring between an API and an eval script is how projects end up reporting
scores for a pipeline nobody is actually serving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .chunking import Chunk
from .config import Settings, get_settings
from .generate import Answer, Generator
from .ingest import Ingestor, IngestStats
from .lexical import BM25Index
from .providers import Embedder, LLM, build_embedder, build_llm
from .retrieve import HybridRetriever, RetrievalResult, build_retriever
from .store import MemoryStore, VectorStore, build_store
from .telemetry import TELEMETRY, span


@dataclass
class AtlasPipeline:
    settings: Settings = field(default_factory=get_settings)
    store: VectorStore | None = None
    embedder: Embedder | None = None
    llm: LLM | None = None
    bm25: BM25Index | None = None
    corpus: dict[str, Chunk] = field(default_factory=dict)
    retriever: HybridRetriever | None = None
    generator: Generator | None = None

    # -- construction ----------------------------------------------------
    def __post_init__(self) -> None:
        self.store = self.store or build_store(self.settings)
        self.embedder = self.embedder or build_embedder(self.settings)
        self.llm = self.llm or build_llm(self.settings)
        self.bm25 = self.bm25 or BM25Index()
        self.generator = Generator(
            llm=self.llm,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_output_tokens,
            min_context_score=self.settings.min_context_score,
        )
        self._rebuild_retriever()

    def _rebuild_retriever(self) -> None:
        self.retriever = build_retriever(
            store=self.store,
            embedder=self.embedder,
            bm25=self.bm25,
            llm=self.llm,
            corpus=self.corpus,
            settings=self.settings,
        )

    @property
    def bm25_path(self) -> Path:
        return Path(self.settings.index_dir) / "bm25.pkl"

    # -- ingestion -------------------------------------------------------
    def ingest(self, root: str | Path | None = None) -> IngestStats:
        ingestor = Ingestor(store=self.store, embedder=self.embedder, settings=self.settings)
        with span("ingest.total"):
            chunks, index, stats = ingestor.ingest_path(root or self.settings.data_dir)
        self.bm25 = index
        self.corpus = {c.chunk_id: c for c in chunks}
        self._rebuild_retriever()
        self.save_index()
        return stats

    def save_index(self) -> None:
        """BM25 lives outside the vector DB, so it has to be persisted
        separately or every restart silently loses the lexical arm -- and the
        system keeps answering, just worse. That failure mode is quiet enough
        that it is worth an explicit save/load path."""
        Path(self.settings.index_dir).mkdir(parents=True, exist_ok=True)
        self.bm25.save(self.bm25_path)
        if isinstance(self.store, MemoryStore):
            self.store.save()

    def load_index(self) -> bool:
        if not self.bm25_path.exists():
            return False
        self.bm25 = BM25Index.load(self.bm25_path)
        if isinstance(self.store, MemoryStore):
            self.store.load()
            self.corpus = {c.chunk_id: c for c in self.store.all_chunks()}
        self._rebuild_retriever()
        return True

    # -- query -----------------------------------------------------------
    def retrieve(
        self, query: str, top_k: int | None = None, flt: dict[str, Any] | None = None
    ) -> RetrievalResult:
        return self.retriever.retrieve(query, top_k=top_k, flt=flt)

    def ask(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        flt: dict[str, Any] | None = None,
    ) -> Answer:
        with span("query.total", query_chars=len(query)):
            result = self.retrieve(query, top_k=top_k, flt=flt)
            return self.generator.answer(query, result, history=history)

    def stream(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        flt: dict[str, Any] | None = None,
    ) -> Iterator[tuple[str, Any]]:
        result = self.retrieve(query, top_k=top_k, flt=flt)
        # Sources are emitted before the first token so the UI can render
        # citations while the answer is still streaming.
        yield "sources", result.sources()
        yield from self.generator.stream_answer(query, result, history=history)

    # -- ops -------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        try:
            indexed = self.store.count()
            store_ok = True
        except Exception as exc:
            indexed, store_ok = 0, f"error: {exc}"
        return {
            "status": "ok" if store_ok is True and indexed > 0 else "degraded",
            "indexed_chunks": indexed,
            "bm25_docs": len(self.bm25.doc_ids) if self.bm25 else 0,
            "vector_backend": self.settings.vector_backend,
            "store_reachable": store_ok,
            "llm_provider": getattr(self.llm, "name", "unknown"),
            "llm_model": getattr(self.llm, "model", "unknown"),
            "embedder": getattr(self.embedder, "name", "unknown"),
            "offline_mode": self.settings.offline,
            "chunk_strategy": self.settings.chunking.strategy,
            "reranker": "llm" if self.settings.retrieval.use_reranker else "off",
        }

    def metrics(self) -> dict[str, Any]:
        return TELEMETRY.summary()


def build_pipeline(settings: Settings | None = None, load: bool = True) -> AtlasPipeline:
    pipe = AtlasPipeline(settings=settings or get_settings())
    if load:
        pipe.load_index()
    return pipe
