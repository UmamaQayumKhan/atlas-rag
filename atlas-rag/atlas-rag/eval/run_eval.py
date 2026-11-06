#!/usr/bin/env python3
"""Evaluation harness and CI gate.

    # retrieval only -- no LLM, no key, ~2 seconds. This is what CI runs.
    python eval/run_eval.py --mode retrieval

    # full answer eval, needs a provider key
    python eval/run_eval.py --mode answer

    # add LLM-as-judge faithfulness on top
    python eval/run_eval.py --mode answer --judge

    # sweep chunking strategies and print the comparison table
    python eval/run_eval.py --sweep

    # fail the build if quality regressed
    python eval/run_eval.py --mode retrieval --gate-recall5 0.85 --gate-mrr 0.70

The gate is the point of the whole file. An eval you look at once is a number; an
eval wired to a threshold in CI is a regression test, and it is what stops a
"harmless" chunking tweak from quietly dropping recall 8 points.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from atlas.config import Settings, reset_settings_cache  # noqa: E402
from atlas.pipeline import AtlasPipeline  # noqa: E402
from atlas.telemetry import TELEMETRY  # noqa: E402
from judge import judge_answer  # noqa: E402
from metrics import (  # noqa: E402
    Aggregate,
    contains_any,
    load_golden,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def build_for_eval(settings: Settings) -> AtlasPipeline:
    """Always ingests into a fresh in-memory store so a sweep compares
    strategies on identical input and never reads a stale index."""
    pipe = AtlasPipeline(settings=settings)
    stats = pipe.ingest(settings.data_dir)
    print(
        f"  indexed: {stats.summary()} "
        f"[strategy={settings.chunking.strategy}, tokens={settings.chunking.chunk_tokens}]"
    )
    return pipe


def evaluate(
    pipe: AtlasPipeline,
    golden_path: Path,
    mode: str = "retrieval",
    use_judge: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    items = load_golden(golden_path)
    if limit:
        items = items[:limit]
    agg = Aggregate()
    started = time.perf_counter()

    for item in items:
        result = pipe.retrieve(item.question)
        # Relevance order, deduplicated to document level -- NOT the prompt
        # order, which is reading order and would make rank metrics meaningless.
        retrieved = result.ranked_doc_ids()

        row: dict[str, Any] = {
            "id": item.id,
            "question": item.question,
            "category": item.category,
            "answerable": item.answerable,
            "retrieved_docs": retrieved,
            "recall@1": recall_at_k(retrieved, item.expected_docs, 1),
            "recall@3": recall_at_k(retrieved, item.expected_docs, 3),
            "recall@5": recall_at_k(retrieved, item.expected_docs, 5),
            "precision@5": precision_at_k(retrieved, item.expected_docs, 5),
            "mrr": reciprocal_rank(retrieved, item.expected_docs),
            "ndcg@5": ndcg_at_k(retrieved, item.expected_docs, 5),
        }

        # Context recall: did the passage text actually handed to the generator
        # contain the expected answer? On a small corpus, document-level
        # recall@5 saturates at 1.0 and tells you nothing, while this metric
        # still discriminates -- it is the one that moves when chunking changes,
        # because it measures whether the answer survived the chunk boundary.
        if item.answerable and item.must_contain:
            context = result.context_block()
            row["answer_in_context"] = 1.0 if contains_any(context, item.must_contain) else 0.0
            if row["answer_in_context"] == 0.0:
                row["failure_reason"] = (
                    f"answer text {item.must_contain} absent from retrieved context"
                )

        if mode == "answer":
            answer = pipe.generator.answer(item.question, result)
            row["answer"] = answer.text
            row["refused"] = 1.0 if answer.refused else 0.0
            row["grounded"] = 1.0 if answer.grounded else 0.0
            row["citation_coverage"] = (
                answer.grounding.coverage if answer.grounding else 0.0
            )
            row["unknown_citations"] = (
                len(answer.grounding.unknown_citations) if answer.grounding else 0
            )

            if item.answerable:
                hit = contains_any(answer.text, item.must_contain) and not answer.refused
                row["answer_correct"] = 1.0 if hit else 0.0
                if not hit:
                    row["failure_reason"] = (
                        "refused an answerable question"
                        if answer.refused
                        else f"missing expected content: {item.must_contain}"
                    )
            else:
                row["answer_correct"] = 1.0 if answer.refused else 0.0
                if not answer.refused:
                    row["failure_reason"] = "answered an unanswerable question"

            if use_judge:
                judgement = judge_answer(
                    pipe.llm, item.question, result.context_block(), answer.text
                )
                row["faithfulness"] = judgement.faithfulness
                row["judge_verdict"] = judgement.verdict

        agg.add(**row)

    report = agg.report()
    report["meta"] = {
        "mode": mode,
        "judge": use_judge,
        "items": len(items),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "health": pipe.health(),
        "telemetry": TELEMETRY.summary(),
    }
    if any("faithfulness" in r for r in agg.rows):
        report["overall"]["faithfulness"] = agg._mean("faithfulness")
    report["rows"] = agg.rows
    return report


def print_report(report: dict[str, Any]) -> None:
    overall = report["overall"]
    print("\n" + "=" * 68)
    print("  RETRIEVAL")
    print("=" * 68)
    for key in ("recall@1", "recall@3", "recall@5", "precision@5", "mrr", "ndcg@5", "answer_in_context"):
        print(f"  {key:<22} {overall.get(key, 0):.3f}")

    if "answer_accuracy" in overall:
        print("\n" + "=" * 68)
        print("  ANSWER QUALITY")
        print("=" * 68)
        for key in (
            "answer_accuracy",
            "faithfulness",
            "citation_coverage",
            "grounded_rate",
            "refusal_accuracy",
            "false_answer_rate",
            "over_refusal_rate",
        ):
            if key in overall:
                print(f"  {key:<22} {overall[key]:.3f}")
        print(f"  {'hallucinated_cites':<22} {overall.get('hallucinated_citations', 0)}")

    print("\n" + "=" * 68)
    print("  BY CATEGORY")
    print("=" * 68)
    for cat, vals in report["by_category"].items():
        acc = vals.get("answer_accuracy")
        suffix = f"  answer={acc:.3f}" if acc is not None else ""
        print(f"  {cat:<16} n={vals['n']:<4} recall@5={vals['recall@5']:.3f}{suffix}")

    failures = report.get("failures", [])
    if failures:
        print("\n" + "=" * 68)
        print(f"  FAILURES ({len(failures)})")
        print("=" * 68)
        for f in failures[:15]:
            print(f"  [{f['id']}] {f['question'][:62]}")
            print(f"       -> {f['reason']}")

    tel = report["meta"]["telemetry"]
    print(
        f"\n  wall={report['meta']['wall_seconds']}s  llm_calls={tel['llm_calls']}  "
        f"cost=${tel['total_cost_usd']:.4f}  p95={tel['latency_ms']['p95']}ms"
    )


def sweep(golden_path: Path, mode: str) -> None:
    """Re-ingest and re-evaluate under each chunking strategy. This is the table
    in the README; it is regenerated, not written by hand."""
    results: dict[str, dict[str, Any]] = {}
    chunk_counts: dict[str, int] = {}
    for strategy in ("fixed", "recursive", "structural"):
        print(f"\n--- {strategy} ---")
        os.environ["CHUNK_STRATEGY"] = strategy
        os.environ["VECTOR_BACKEND"] = "memory"
        os.environ["INDEX_DIR"] = f".atlas_eval_{strategy}"
        reset_settings_cache()
        TELEMETRY.reset()
        settings = Settings()
        pipe = build_for_eval(settings)
        chunk_counts[strategy] = pipe.store.count()
        results[strategy] = evaluate(pipe, golden_path, mode=mode)["overall"]

    print("\n" + "=" * 78)
    print("  CHUNKING STRATEGY COMPARISON")
    print("=" * 78)
    header = (
        f"  {'strategy':<14}{'chunks':>8}{'recall@1':>10}{'mrr':>8}"
        f"{'ndcg@5':>9}{'ans_in_ctx':>12}"
    )
    if mode == "answer":
        header += f"{'answer_acc':>12}{'refusal_acc':>13}"
    print(header)
    print("  " + "-" * 55)
    for strategy, vals in results.items():
        line = (
            f"  {strategy:<14}{chunk_counts.get(strategy, 0):>8}"
            f"{vals['recall@1']:>10.3f}{vals['mrr']:>8.3f}"
            f"{vals['ndcg@5']:>9.3f}{vals['answer_in_context']:>12.3f}"
        )
        if mode == "answer":
            line += (
                f"{vals.get('answer_accuracy', 0):>12.3f}"
                f"{vals.get('refusal_accuracy', 0):>13.3f}"
            )
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="AtlasRAG evaluation harness")
    ap.add_argument("--mode", choices=["retrieval", "answer"], default="retrieval")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge faithfulness")
    ap.add_argument("--sweep", action="store_true", help="compare chunking strategies")
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "eval" / "report.json"))
    ap.add_argument("--gate-recall5", type=float, default=None)
    ap.add_argument("--gate-mrr", type=float, default=None)
    ap.add_argument("--gate-answer", type=float, default=None)
    ap.add_argument("--gate-refusal", type=float, default=None)
    args = ap.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"golden set not found: {golden_path}", file=sys.stderr)
        return 2

    if args.sweep:
        sweep(golden_path, args.mode)
        return 0

    os.environ.setdefault("VECTOR_BACKEND", "memory")
    os.environ.setdefault("INDEX_DIR", ".atlas_eval")
    reset_settings_cache()
    settings = Settings()
    print(f"provider={settings.llm_provider} offline={settings.offline}")
    if args.mode == "answer" and settings.offline:
        print(
            "  note: no API key found, answers come from the offline stub.\n"
            "        Set GROQ_API_KEY or GEMINI_API_KEY for a meaningful answer eval.",
            file=sys.stderr,
        )

    pipe = build_for_eval(settings)
    report = evaluate(pipe, golden_path, mode=args.mode, use_judge=args.judge, limit=args.limit)
    print_report(report)

    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  report written to {args.out}")

    overall = report["overall"]
    gates = [
        ("recall@5", args.gate_recall5, overall.get("recall@5", 0.0)),
        ("mrr", args.gate_mrr, overall.get("mrr", 0.0)),
        ("answer_accuracy", args.gate_answer, overall.get("answer_accuracy", 0.0)),
        ("refusal_accuracy", args.gate_refusal, overall.get("refusal_accuracy", 0.0)),
    ]
    failed = [(n, t, v) for n, t, v in gates if t is not None and v < t]
    if failed:
        print("\n  QUALITY GATE FAILED")
        for name, threshold, value in failed:
            print(f"    {name}: {value:.3f} < required {threshold:.3f}")
        return 1
    if any(t is not None for _, t, _ in gates):
        print("\n  quality gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
