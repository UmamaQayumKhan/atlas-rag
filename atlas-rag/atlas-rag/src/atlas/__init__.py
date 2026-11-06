"""AtlasRAG -- production-shaped retrieval-augmented generation over enterprise docs."""

from .chunking import Chunk, build_chunks
from .config import Settings, get_settings
from .generate import Answer, Generator
from .lexical import BM25Index
from .pipeline import AtlasPipeline, build_pipeline
from .retrieve import HybridRetriever, RetrievalResult, reciprocal_rank_fusion
from .store import MemoryStore, QdrantStore, ScoredChunk
from .telemetry import TELEMETRY

__version__ = "1.0.0"

__all__ = [
    "Answer",
    "AtlasPipeline",
    "BM25Index",
    "Chunk",
    "Generator",
    "HybridRetriever",
    "MemoryStore",
    "QdrantStore",
    "RetrievalResult",
    "ScoredChunk",
    "Settings",
    "TELEMETRY",
    "build_chunks",
    "build_pipeline",
    "get_settings",
    "reciprocal_rank_fusion",
    "__version__",
]
