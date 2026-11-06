"""Provider-agnostic LLM and embedding clients.

Every call site in the app depends on the two protocols at the bottom of this
module, never on a vendor SDK. Swapping Groq for OpenAI, Gemini or a local vLLM
server is a single environment variable, and the `offline` implementations let
the test suite and CI run with no API keys and no network.

Cost and latency for every call are recorded through `telemetry` so the eval
harness can report $/query and p95 without extra instrumentation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Protocol, Sequence

import httpx

from .config import Settings, get_settings
from .telemetry import record_embedding, record_llm_call

# Published free/low-tier prices per 1M tokens, used for cost estimates only.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gpt-4o-mini": (0.15, 0.60),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = _PRICE_PER_MTOK.get(model, (0.0, 0.0))
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Good enough for budget guards and
    avoids pulling a tokenizer into the request path."""
    return max(1, len(text) // 4)


class ProviderError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# protocols
# --------------------------------------------------------------------------


class LLM(Protocol):
    name: str

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str: ...

    def stream(self, messages: Sequence[dict[str, str]], **kw: Any) -> Iterator[str]: ...


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# --------------------------------------------------------------------------
# shared HTTP behaviour
# --------------------------------------------------------------------------


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int,
    **kw: Any,
) -> httpx.Response:
    """Retries only on 429 and 5xx, with exponential backoff plus jitter, and
    honours Retry-After when the provider sends it. Free tiers rate-limit
    aggressively, so this is what keeps a 500-question eval run alive."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.request(method, url, **kw)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code != 429 and resp.status_code < 500:
                raise ProviderError(
                    f"{method} {url} failed [{resp.status_code}]: {resp.text[:400]}"
                )
            retry_after = resp.headers.get("retry-after")
            if retry_after:
                try:
                    time.sleep(min(float(retry_after), 30.0))
                    continue
                except ValueError:
                    pass
            last_exc = ProviderError(f"{resp.status_code}: {resp.text[:200]}")

        if attempt < max_retries - 1:
            time.sleep(min(2**attempt + random.random(), 20.0))

    raise ProviderError(f"{method} {url} failed after {max_retries} attempts: {last_exc}")


# --------------------------------------------------------------------------
# OpenAI-compatible chat (Groq, OpenAI, Together, vLLM, Ollama, LM Studio)
# --------------------------------------------------------------------------


@dataclass
class OpenAICompatLLM:
    api_key: str
    model: str
    base_url: str
    timeout: float = 60.0
    max_retries: int = 3
    name: str = "openai-compatible"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProviderError(f"{self.name}: missing API key")
        self._client = httpx.Client(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def _payload(self, messages: Sequence[dict[str, str]], **kw: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": kw.get("temperature", 0.1),
            "max_tokens": kw.get("max_tokens", 900),
        }
        if kw.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}
        if kw.get("stop"):
            payload["stop"] = kw["stop"]
        return payload

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str:
        started = time.perf_counter()
        resp = _request_with_retry(
            self._client,
            "POST",
            f"{self.base_url}/chat/completions",
            max_retries=self.max_retries,
            json=self._payload(messages, **kw),
        )
        body = resp.json()
        text = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage") or {}
        pt = usage.get("prompt_tokens") or approx_tokens(json.dumps(list(messages)))
        ct = usage.get("completion_tokens") or approx_tokens(text)
        record_llm_call(
            provider=self.name,
            model=self.model,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_s=time.perf_counter() - started,
            cost_usd=estimate_cost(self.model, pt, ct),
        )
        return text

    def stream(self, messages: Sequence[dict[str, str]], **kw: Any) -> Iterator[str]:
        started = time.perf_counter()
        payload = self._payload(messages, **kw) | {"stream": True}
        emitted: list[str] = []
        with self._client.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise ProviderError(f"stream failed [{resp.status_code}]: {resp.text[:300]}")
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk in {"", "[DONE]"}:
                    continue
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content")
                if piece:
                    emitted.append(piece)
                    yield piece
        text = "".join(emitted)
        pt = approx_tokens(json.dumps(list(messages)))
        record_llm_call(
            provider=self.name,
            model=self.model,
            prompt_tokens=pt,
            completion_tokens=approx_tokens(text),
            latency_s=time.perf_counter() - started,
            cost_usd=estimate_cost(self.model, pt, approx_tokens(text)),
            streamed=True,
        )


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------


@dataclass
class GeminiLLM:
    api_key: str
    model: str
    base_url: str
    timeout: float = 60.0
    max_retries: int = 3
    name: str = "gemini"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProviderError("gemini: missing GEMINI_API_KEY")
        self._client = httpx.Client(timeout=self.timeout)

    @staticmethod
    def _convert(messages: Sequence[dict[str, str]]) -> dict[str, Any]:
        """Gemini keeps the system prompt out of the turn list and calls the
        assistant role 'model', so messages need remapping."""
        system_bits = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        body: dict[str, Any] = {"contents": contents}
        if system_bits:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_bits)}]}
        return body

    def _config(self, **kw: Any) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "temperature": kw.get("temperature", 0.1),
            "maxOutputTokens": kw.get("max_tokens", 900),
        }
        if kw.get("json_mode"):
            cfg["responseMimeType"] = "application/json"
        return cfg

    @staticmethod
    def _text_from(body: dict[str, Any]) -> str:
        out: list[str] = []
        for cand in body.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    out.append(part["text"])
        return "".join(out)

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str:
        started = time.perf_counter()
        payload = self._convert(messages) | {"generationConfig": self._config(**kw)}
        resp = _request_with_retry(
            self._client,
            "POST",
            f"{self.base_url}/models/{self.model}:generateContent",
            max_retries=self.max_retries,
            params={"key": self.api_key},
            json=payload,
        )
        body = resp.json()
        text = self._text_from(body)
        usage = body.get("usageMetadata") or {}
        pt = usage.get("promptTokenCount") or approx_tokens(json.dumps(list(messages)))
        ct = usage.get("candidatesTokenCount") or approx_tokens(text)
        record_llm_call(
            provider="gemini",
            model=self.model,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_s=time.perf_counter() - started,
            cost_usd=estimate_cost(self.model, pt, ct),
        )
        return text

    def stream(self, messages: Sequence[dict[str, str]], **kw: Any) -> Iterator[str]:
        started = time.perf_counter()
        payload = self._convert(messages) | {"generationConfig": self._config(**kw)}
        emitted: list[str] = []
        with self._client.stream(
            "POST",
            f"{self.base_url}/models/{self.model}:streamGenerateContent",
            params={"key": self.api_key, "alt": "sse"},
            json=payload,
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise ProviderError(f"gemini stream [{resp.status_code}]: {resp.text[:300]}")
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk:
                    continue
                try:
                    piece = self._text_from(json.loads(chunk))
                except json.JSONDecodeError:
                    continue
                if piece:
                    emitted.append(piece)
                    yield piece
        text = "".join(emitted)
        pt = approx_tokens(json.dumps(list(messages)))
        record_llm_call(
            provider="gemini",
            model=self.model,
            prompt_tokens=pt,
            completion_tokens=approx_tokens(text),
            latency_s=time.perf_counter() - started,
            cost_usd=estimate_cost(self.model, pt, approx_tokens(text)),
            streamed=True,
        )


# --------------------------------------------------------------------------
# offline stub -- keeps CI green and makes retrieval tests deterministic
# --------------------------------------------------------------------------


@dataclass
class OfflineLLM:
    name: str = "offline"
    model: str = "offline-stub"

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str:
        user = next(
            (m["content"] for m in reversed(list(messages)) if m["role"] == "user"), ""
        )
        if kw.get("json_mode"):
            return json.dumps({"answer": "offline stub", "citations": [], "grounded": True})
        cites = re.findall(r"\[(\d+)\]", user)
        tag = f" [{cites[0]}]" if cites else ""
        record_llm_call(
            provider="offline",
            model=self.model,
            prompt_tokens=approx_tokens(user),
            completion_tokens=12,
            latency_s=0.0,
            cost_usd=0.0,
        )
        return f"[offline stub] deterministic answer for evaluation runs.{tag}"

    def stream(self, messages: Sequence[dict[str, str]], **kw: Any) -> Iterator[str]:
        for word in self.complete(messages, **kw).split(" "):
            yield word + " "


@dataclass
class HashingEmbedder:
    """Deterministic, dependency-free embeddings via hashed character n-grams.

    Not competitive with a trained model, but it is stable across runs and
    preserves enough lexical similarity that retrieval tests assert real
    behaviour instead of mocks. This is what CI uses.
    """

    dim: int = 768
    name: str = "hashing-offline"

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        norm = re.sub(r"\s+", " ", text.lower()).strip()
        tokens = norm.split(" ")
        grams: list[str] = list(tokens)
        grams += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for i in range(len(norm) - 3):
            grams.append(norm[i : i + 4])
        for gram in grams:
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        mag = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / mag for v in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        record_embedding(provider=self.name, count=len(texts), cost_usd=0.0)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@dataclass
class GeminiEmbedder:
    api_key: str
    model: str
    base_url: str
    dim: int = 768
    timeout: float = 60.0
    max_retries: int = 3
    batch_size: int = 64
    name: str = "gemini-embeddings"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProviderError("gemini embeddings: missing GEMINI_API_KEY")
        self._client = httpx.Client(timeout=self.timeout)

    def _call(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        requests = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
            }
            for t in texts
        ]
        resp = _request_with_retry(
            self._client,
            "POST",
            f"{self.base_url}/models/{self.model}:batchEmbedContents",
            max_retries=self.max_retries,
            params={"key": self.api_key},
            json={"requests": requests},
        )
        return [e["values"] for e in resp.json().get("embeddings", [])]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Asymmetric embedding matters: documents and queries are embedded with
        different task types, which measurably lifts recall on Gemini."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._call(texts[i : i + self.batch_size], "RETRIEVAL_DOCUMENT"))
        record_embedding(provider=self.name, count=len(texts), cost_usd=0.0)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], "RETRIEVAL_QUERY")[0]


# --------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------


def build_llm(settings: Settings | None = None) -> LLM:
    s = settings or get_settings()
    provider = s.llm_provider.lower()

    if provider == "offline":
        return OfflineLLM()
    if provider == "groq":
        if not s.groq_api_key:
            return OfflineLLM()
        return OpenAICompatLLM(
            api_key=s.groq_api_key,
            model=s.groq_model,
            base_url=s.groq_base_url,
            timeout=s.request_timeout,
            max_retries=s.max_retries,
            name="groq",
        )
    if provider == "gemini":
        if not s.gemini_api_key:
            return OfflineLLM()
        return GeminiLLM(
            api_key=s.gemini_api_key,
            model=s.gemini_model,
            base_url=s.gemini_base_url,
            timeout=s.request_timeout,
            max_retries=s.max_retries,
        )
    if provider in {"openai", "openai-compatible", "vllm", "ollama"}:
        if not s.openai_api_key:
            return OfflineLLM()
        return OpenAICompatLLM(
            api_key=s.openai_api_key,
            model=s.openai_model,
            base_url=s.openai_base_url,
            timeout=s.request_timeout,
            max_retries=s.max_retries,
            name=provider,
        )
    raise ProviderError(f"unknown LLM_PROVIDER: {s.llm_provider}")


def build_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    provider = s.embedding_provider.lower()
    if provider in {"offline", "hashing"} or (provider == "gemini" and not s.gemini_api_key):
        return HashingEmbedder(dim=s.embedding_dim)
    if provider == "gemini":
        return GeminiEmbedder(
            api_key=s.gemini_api_key,
            model=s.gemini_embedding_model,
            base_url=s.gemini_base_url,
            dim=s.embedding_dim,
            timeout=s.request_timeout,
            max_retries=s.max_retries,
        )
    raise ProviderError(f"unknown EMBEDDING_PROVIDER: {s.embedding_provider}")
