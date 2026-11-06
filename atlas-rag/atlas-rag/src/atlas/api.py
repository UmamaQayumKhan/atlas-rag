"""FastAPI service: /ask, /ask/stream (SSE), /retrieve, /healthz, /metrics.

Design notes that come up in interviews:

* The pipeline is built once in the lifespan handler, not per request. Rebuilding
  it per request would re-read the BM25 index from disk on every call.
* `/ask/stream` is Server-Sent Events rather than WebSockets. SSE is one-way,
  survives proxies, needs no connection upgrade, and is what the UI actually
  needs; WebSockets would be extra failure surface for nothing.
* Streaming runs in a thread via `iterate_in_threadpool` because the provider
  clients are sync. Yielding a blocking generator straight into an async route
  would stall the event loop for every other request.
* Errors return a shaped JSON body, never a bare 500 with a stack trace.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import iterate_in_threadpool

from .config import get_settings
from .pipeline import AtlasPipeline, build_pipeline

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    pipe = build_pipeline(settings, load=True)
    if pipe.store.count() == 0 and os.getenv("INGEST_ON_BOOT", "false").lower() == "true":
        pipe.ingest()
    _state["pipeline"] = pipe
    yield
    _state.clear()


app = FastAPI(
    title="AtlasRAG",
    version="1.0.0",
    description="Hybrid-retrieval RAG service over enterprise documentation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_pipeline() -> AtlasPipeline:
    pipe = _state.get("pipeline")
    if pipe is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    return pipe


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[Turn] = Field(default_factory=list, max_length=10)
    # Metadata filter, e.g. {"doc_id": "billing-policy"} to scope the search.
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)[:400]},
    )


@app.get("/healthz")
def healthz(pipe: AtlasPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    return pipe.health()


@app.get("/metrics")
def metrics(pipe: AtlasPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    """Token spend, cost estimate and p50/p95/p99 latency for this process."""
    return pipe.metrics()


@app.post("/retrieve")
def retrieve(req: AskRequest, pipe: AtlasPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    """Retrieval only, no generation. The endpoint the eval harness scores and
    the one to hit when debugging whether a bad answer is a retrieval problem or
    a generation problem."""
    result = pipe.retrieve(req.query, top_k=req.top_k, flt=req.filters or None)
    return {
        "query": req.query,
        "dense_hits": result.dense_hits,
        "lexical_hits": result.lexical_hits,
        "fused_hits": result.fused_hits,
        "reranked": result.reranked,
        "chunks": [sc.to_dict() for sc in result.chunks],
    }


@app.post("/ask")
def ask(req: AskRequest, pipe: AtlasPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    answer = pipe.ask(
        req.query,
        history=[t.model_dump() for t in req.history],
        top_k=req.top_k,
        flt=req.filters or None,
    )
    return answer.to_dict()


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/ask/stream")
async def ask_stream(
    req: AskRequest, pipe: AtlasPipeline = Depends(get_pipeline)
) -> StreamingResponse:
    def produce() -> Iterator[str]:
        try:
            for kind, payload in pipe.stream(
                req.query,
                history=[t.model_dump() for t in req.history],
                top_k=req.top_k,
                flt=req.filters or None,
            ):
                if kind == "sources":
                    yield _sse("sources", payload)
                elif kind == "token":
                    yield _sse("token", {"text": payload})
                else:
                    yield _sse("final", payload.to_dict())
        except Exception as exc:
            yield _sse("error", {"error": type(exc).__name__, "detail": str(exc)[:300]})
        yield _sse("done", {})

    return StreamingResponse(
        iterate_in_threadpool(produce()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this, nginx buffers the whole response and streaming
            # silently stops working the moment you deploy behind a proxy.
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/admin/reindex")
def reindex(pipe: AtlasPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    token = os.getenv("ADMIN_TOKEN", "")
    if not token:
        raise HTTPException(status_code=403, detail="ADMIN_TOKEN not configured")
    stats = pipe.ingest()
    return {"status": "reindexed", "detail": stats.summary()}
