"""Lightweight in-process telemetry: token spend, cost and latency percentiles.

Enough to answer the questions interviewers actually ask -- "what's your p95?",
"what does a query cost?" -- without standing up a tracing backend. The span
format is deliberately OpenTelemetry-shaped so it can be exported later.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass
class LLMCall:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    streamed: bool = False
    at: float = field(default_factory=time.time)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000


class Telemetry:
    """Thread-safe collector. One instance per process; `reset()` between evals."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[LLMCall] = []
        self.spans: list[Span] = []
        self.embeddings: list[dict[str, Any]] = []
        self._stack: threading.local = threading.local()

    # -- recording -------------------------------------------------------
    def record_llm_call(self, **kw: Any) -> None:
        with self._lock:
            self.calls.append(LLMCall(**kw))

    def record_embedding(self, provider: str, count: int, cost_usd: float) -> None:
        with self._lock:
            self.embeddings.append(
                {"provider": provider, "count": count, "cost_usd": cost_usd}
            )

    @property
    def _current(self) -> list[Span]:
        if not hasattr(self._stack, "spans"):
            self._stack.spans = []
        return self._stack.spans

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = self._current[-1] if self._current else None
        sp = Span(
            name=name,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            start=time.perf_counter(),
            attributes=dict(attributes),
        )
        self._current.append(sp)
        try:
            yield sp
        except Exception as exc:
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.end = time.perf_counter()
            self._current.pop()
            with self._lock:
                self.spans.append(sp)

    # -- reporting -------------------------------------------------------
    @staticmethod
    def _pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
        return ordered[idx]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            calls = list(self.calls)
            spans = list(self.spans)
            embeds = list(self.embeddings)
        latencies = [c.latency_s * 1000 for c in calls]
        by_stage: dict[str, list[float]] = {}
        for sp in spans:
            by_stage.setdefault(sp.name, []).append(sp.duration_ms)
        return {
            "llm_calls": len(calls),
            "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "total_cost_usd": round(
                sum(c.cost_usd for c in calls) + sum(e["cost_usd"] for e in embeds), 6
            ),
            "embedded_texts": sum(e["count"] for e in embeds),
            "latency_ms": {
                "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
                "p50": round(self._pct(latencies, 50), 1),
                "p95": round(self._pct(latencies, 95), 1),
                "p99": round(self._pct(latencies, 99), 1),
            },
            "stage_ms_p95": {
                stage: round(self._pct(vals, 95), 1) for stage, vals in sorted(by_stage.items())
            },
            "errors": [sp.error for sp in spans if sp.error],
        }

    def export_spans(self, path: str) -> None:
        with self._lock, open(path, "w", encoding="utf-8") as fh:
            for sp in self.spans:
                row = asdict(sp)
                row["duration_ms"] = round(sp.duration_ms, 2)
                fh.write(json.dumps(row) + "\n")

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()
            self.spans.clear()
            self.embeddings.clear()


TELEMETRY = Telemetry()

record_llm_call = TELEMETRY.record_llm_call
record_embedding = TELEMETRY.record_embedding
span = TELEMETRY.span
