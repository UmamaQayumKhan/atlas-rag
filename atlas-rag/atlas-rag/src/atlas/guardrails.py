"""Input and output guardrails.

A RAG system over company documents has two specific exposures that a plain
chatbot does not:

1. **Indirect prompt injection.** The retrieved context is attacker-influenced
   whenever any ingested document can be edited by a non-admin (a wiki page, an
   uploaded PDF, a support ticket). Instructions inside a *document* must never
   be treated as instructions from the *user*. We detect, strip and log them.
2. **Ungrounded answers.** The model will happily answer from parametric memory
   when retrieval misses. We verify every claim-bearing sentence carries a
   citation that actually exists, and downgrade to a refusal when it does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# Patterns seen in real indirect-injection payloads. Matched against retrieved
# document text, never used to block the user's own question.
_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions", "override"),
    (r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)", "override"),
    (r"you\s+are\s+now\s+(?:a|an|in)\b", "persona-switch"),
    (r"\bnew\s+(?:system\s+)?(?:instructions?|prompt)\b", "prompt-injection"),
    (r"</?(?:system|assistant|instructions?)>", "tag-injection"),
    (r"reveal\s+(?:your\s+)?(?:system\s+prompt|instructions)", "exfiltration"),
    (r"print\s+(?:your\s+)?(?:system\s+prompt|configuration|api[_\s]?key)", "exfiltration"),
    (r"\b(?:send|post|email|exfiltrate)\s+.{0,40}\bto\s+https?://", "exfiltration"),
    (r"do\s+not\s+(?:cite|mention)\s+(?:any\s+)?sources?", "citation-suppression"),
)

_PII_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b", "EMAIL"),
    (r"\b(?:\d[ -]?){13,16}\b", "CARD"),
    # Allows internal separators so `sk-live-AbCdEf...` and `api_key_prod_...`
    # match, while the 12-char minimum on the final segment keeps ordinary
    # hyphenated words like "key-value" from being redacted.
    (r"\b(?:sk|pk|api|key)[-_](?:[A-Za-z0-9]+[-_])*[A-Za-z0-9]{12,}\b", "SECRET"),
    (r"\bBearer\s+[A-Za-z0-9._-]{20,}\b", "TOKEN"),
)


@dataclass
class GuardrailReport:
    injection_found: bool = False
    injection_kinds: list[str] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.injection_found and not self.redactions


def scan_for_injection(text: str) -> list[str]:
    found: list[str] = []
    for pattern, kind in _INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(kind)
    return sorted(set(found))


def neutralise_context(text: str) -> tuple[str, list[str]]:
    """Replace injection attempts inside retrieved context with a visible
    marker. Stripping silently would hide a live attack from the operator;
    leaving the text in would let it reach the model."""
    kinds: list[str] = []
    out = text
    for pattern, kind in _INJECTION_PATTERNS:
        out, n = re.subn(pattern, "[redacted: instruction-like text]", out, flags=re.IGNORECASE)
        if n:
            kinds.append(kind)
    return out, sorted(set(kinds))


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    out = text
    for pattern, label in _PII_PATTERNS:
        out, n = re.subn(pattern, f"[{label}_REDACTED]", out)
        if n:
            counts[label] = counts.get(label, 0) + n
    return out, counts


def guard_context(text: str, redact: bool = True) -> tuple[str, GuardrailReport]:
    report = GuardrailReport()
    out, kinds = neutralise_context(text)
    if kinds:
        report.injection_found = True
        report.injection_kinds = kinds
        report.notes.append(
            "Instruction-like text was found in retrieved documents and neutralised."
        )
    if redact:
        out, counts = redact_pii(out)
        if counts:
            report.redactions = counts
            report.notes.append(f"Redacted from context: {counts}")
    return out, report


_CITATION = re.compile(r"\[(\d+)\]")

# Sentences that make no factual claim need no citation.
_NON_CLAIM = re.compile(
    r"^\s*(?:"
    r"i (?:do not|don't|cannot|can't) (?:have|find|see)|"
    r"based on the (?:provided|available)|"
    r"(?:here|below) (?:is|are)|"
    r"in (?:short|summary)|"
    r"(?:to summarise|to summarize|overall)"
    r")",
    re.IGNORECASE,
)


@dataclass
class GroundingReport:
    cited_sentences: int
    claim_sentences: int
    unknown_citations: list[int]
    valid_citations: list[int]

    @property
    def coverage(self) -> float:
        if self.claim_sentences == 0:
            return 1.0
        return self.cited_sentences / self.claim_sentences

    @property
    def grounded(self) -> bool:
        return not self.unknown_citations and self.coverage >= 0.6


def check_grounding(answer: str, n_sources: int) -> GroundingReport:
    """Verify citations point at real sources and that claim sentences carry one.

    This is the output-side half of the refusal logic: retrieval can succeed and
    the model can still answer from memory. Catching a citation index that does
    not exist is the cheapest reliable hallucination signal available.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    claim_sentences = 0
    cited = 0
    all_refs: list[int] = []

    for sent in sentences:
        refs = [int(m) for m in _CITATION.findall(sent)]
        all_refs.extend(refs)
        if _NON_CLAIM.match(sent) or len(sent.split()) < 5:
            continue
        claim_sentences += 1
        if refs:
            cited += 1

    unknown = sorted({r for r in all_refs if r < 1 or r > n_sources})
    valid = sorted({r for r in all_refs if 1 <= r <= n_sources})
    return GroundingReport(
        cited_sentences=cited,
        claim_sentences=claim_sentences,
        unknown_citations=unknown,
        valid_citations=valid,
    )


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(
        marker in lowered
        for marker in (
            "i don't have enough information",
            "i do not have enough information",
            "not covered in the provided",
            "could not find",
            "no relevant information",
        )
    )
