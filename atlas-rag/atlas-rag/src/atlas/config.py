"""Central configuration. Everything is env-driven so the same image runs
locally, in CI (offline), and on a cloud host without code changes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RetrievalConfig:
    """Knobs that materially change retrieval quality. Each default was chosen
    by sweeping the golden set in `eval/` -- see README for the table."""

    # How many candidates each arm of the hybrid search contributes before fusion.
    dense_top_k: int = field(default_factory=lambda: _env_int("DENSE_TOP_K", 20))
    lexical_top_k: int = field(default_factory=lambda: _env_int("LEXICAL_TOP_K", 20))
    # Reciprocal Rank Fusion damping constant. 60 is the value from the original
    # Cormack et al. paper and held up best on our set.
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    # Chunks actually handed to the LLM after reranking.
    final_top_k: int = field(default_factory=lambda: _env_int("FINAL_TOP_K", 5))
    # Below this rerank score a chunk is dropped entirely rather than padded in.
    min_rerank_score: float = field(
        default_factory=lambda: _env_float("MIN_RERANK_SCORE", 0.15)
    )
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", True))
    # Pull neighbouring chunks of a hit so answers don't get cut mid-clause.
    expand_neighbours: bool = field(
        default_factory=lambda: _env_bool("EXPAND_NEIGHBOURS", True)
    )


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = field(default_factory=lambda: _env("CHUNK_STRATEGY", "structural"))
    chunk_tokens: int = field(default_factory=lambda: _env_int("CHUNK_TOKENS", 320))
    overlap_tokens: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 64))
    min_chunk_tokens: int = field(default_factory=lambda: _env_int("MIN_CHUNK_TOKENS", 40))


@dataclass(frozen=True)
class Settings:
    # --- providers -------------------------------------------------------
    # "groq" and "gemini" both have usable free tiers; "offline" is a
    # deterministic stub so tests and CI need no credentials at all.
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq"))
    embedding_provider: str = field(
        default_factory=lambda: _env("EMBEDDING_PROVIDER", "gemini")
    )

    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    groq_base_url: str = field(
        default_factory=lambda: _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )

    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash")
    )
    gemini_embedding_model: str = field(
        default_factory=lambda: _env("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    )
    gemini_base_url: str = field(
        default_factory=lambda: _env(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        )
    )

    # OpenAI-compatible escape hatch: point this at OpenAI, Together, vLLM,
    # Ollama, or anything else that speaks /chat/completions.
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 768))

    # --- vector store ----------------------------------------------------
    # "qdrant" for real deployments, "memory" for tests and tiny corpora.
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "qdrant"))
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY", ""))
    collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "atlas_docs"))

    # --- generation ------------------------------------------------------
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.1))
    max_output_tokens: int = field(default_factory=lambda: _env_int("MAX_OUTPUT_TOKENS", 900))
    # Refuse rather than guess when retrieved context looks too weak.
    min_context_score: float = field(
        default_factory=lambda: _env_float("MIN_CONTEXT_SCORE", 0.12)
    )

    request_timeout: float = field(default_factory=lambda: _env_float("REQUEST_TIMEOUT", 60.0))
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 3))

    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "data/corpus"))
    index_dir: str = field(default_factory=lambda: _env("INDEX_DIR", ".atlas_index"))

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)

    @property
    def offline(self) -> bool:
        """True when no credentials are present anywhere -- the whole stack
        then falls back to deterministic stubs instead of raising."""
        return self.llm_provider == "offline" or not (
            self.groq_api_key or self.gemini_api_key or self.openai_api_key
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Tests mutate os.environ then call this."""
    get_settings.cache_clear()
