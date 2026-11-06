"""LLM-as-judge faithfulness scoring.

Deterministic metrics cannot tell you whether an answer is *supported by* the
passages it cites -- only whether a citation exists. This judge reads the answer
alongside the retrieved context and scores support, which is the metric that
correlates with users trusting the system.

Two safeguards that matter and are usually skipped:

* The judge sees the context and the answer, never the expected answer, so it
  grades support rather than agreement with a reference.
* The judge is asked for a verdict per claim rather than one holistic score.
  Holistic scores collapse to "7/10" for everything.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from atlas.providers import LLM

JUDGE_PROMPT = """You are a strict evaluator of grounded question answering.

You receive CONTEXT passages and an ANSWER. Decide, for each factual claim in \
the ANSWER, whether the CONTEXT supports it.

Return JSON only:
{
  "claims": [
    {"claim": "<short paraphrase>", "supported": true|false, "passage": <int|null>}
  ],
  "unsupported_count": <int>,
  "faithfulness": <float 0.0-1.0>,
  "verdict": "faithful" | "partially_faithful" | "unfaithful"
}

Rules:
- A claim is supported only if the CONTEXT states it. Plausible-but-absent is \
NOT supported.
- Ignore hedging, pleasantries and restatements of the question; they are not claims.
- A refusal with no claims is "faithful" with faithfulness 1.0.
- faithfulness = supported claims / total claims.
"""


@dataclass
class Judgement:
    faithfulness: float
    verdict: str
    unsupported: int
    claims: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""
    ok: bool = True

    @classmethod
    def failed(cls, raw: str = "") -> "Judgement":
        return cls(faithfulness=0.0, verdict="unknown", unsupported=0, raw=raw, ok=False)


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_answer(llm: LLM, question: str, context: str, answer: str) -> Judgement:
    if not answer.strip():
        return Judgement(faithfulness=0.0, verdict="unfaithful", unsupported=0)

    raw = llm.complete(
        [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\nCONTEXT:\n{context[:9000]}\n\n"
                    f"ANSWER:\n{answer}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=800,
        json_mode=True,
    )
    data = _extract_json(raw)
    if not data:
        return Judgement.failed(raw)

    claims = data.get("claims") or []
    try:
        faith = float(data.get("faithfulness", 0.0))
    except (TypeError, ValueError):
        faith = 0.0
    if not 0.0 <= faith <= 1.0 and claims:
        supported = sum(1 for c in claims if c.get("supported"))
        faith = supported / len(claims)

    return Judgement(
        faithfulness=round(max(0.0, min(1.0, faith)), 4),
        verdict=str(data.get("verdict", "unknown")),
        unsupported=int(data.get("unsupported_count") or 0),
        claims=claims,
        raw=raw,
    )
