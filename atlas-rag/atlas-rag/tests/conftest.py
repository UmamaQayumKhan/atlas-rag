from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))


@pytest.fixture(autouse=True)
def offline_env(monkeypatch, tmp_path):
    """Every test runs fully offline: hashing embeddings, in-memory store, stub
    LLM. No network, no keys, no services -- so CI is deterministic and a
    contributor can run `pytest` immediately after cloning."""
    from atlas.config import reset_settings_cache

    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("VECTOR_BACKEND", "memory")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def corpus_dir() -> Path:
    return ROOT / "data" / "corpus"


@pytest.fixture
def pipeline(corpus_dir):
    from atlas.config import Settings
    from atlas.pipeline import AtlasPipeline

    settings = Settings()
    pipe = AtlasPipeline(settings=settings)
    pipe.ingest(corpus_dir)
    return pipe
