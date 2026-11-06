"""Command line entry point: ingest, ask, retrieve, health, metrics.

`python -m atlas.cli ask "..."` is the fastest way to sanity-check a change
without booting the API or the UI.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings
from .pipeline import build_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atlas", description="AtlasRAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="index a folder of documents")
    p_ingest.add_argument("path", nargs="?", default=None)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("query")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--stream", action="store_true")
    p_ask.add_argument("--json", action="store_true")

    p_ret = sub.add_parser("retrieve", help="show retrieved chunks only")
    p_ret.add_argument("query")
    p_ret.add_argument("--top-k", type=int, default=None)

    sub.add_parser("health", help="print pipeline health")
    sub.add_parser("metrics", help="print telemetry summary")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "ingest":
        pipe = build_pipeline(settings, load=False)
        stats = pipe.ingest(args.path)
        print(stats.summary())
        for skip in stats.skipped:
            print(f"  skipped: {skip}", file=sys.stderr)
        return 0

    pipe = build_pipeline(settings, load=True)

    if args.cmd == "health":
        print(json.dumps(pipe.health(), indent=2))
        return 0

    if args.cmd == "metrics":
        print(json.dumps(pipe.metrics(), indent=2))
        return 0

    if pipe.store.count() == 0:
        print("index is empty -- run `atlas ingest` first", file=sys.stderr)
        return 2

    if args.cmd == "retrieve":
        result = pipe.retrieve(args.query, top_k=args.top_k)

        # Relevance order, not the prompt's reading order. `retrieve` exists to
        # debug the retriever, so showing chunks in document order made a 0.55
        # top hit appear below a 0.25 neighbour and made retrieval look broken
        # when it was not.
        print(f"\nRanked by relevance ({len(result.ranked)} hits):")
        for i, sc in enumerate(result.ranked, start=1):
            print(f"\n  [{i}] {sc.score:.4f} via {sc.retriever} -- {sc.chunk.citation_label}")
            print(f"      {sc.chunk.text[:240].replace(chr(10), ' ')}...")

        extras = [sc for sc in result.chunks if sc.retriever == "neighbour"]
        if extras:
            print(
                f"\n+ {len(extras)} neighbour chunk(s) added for context "
                "(not independently ranked):"
            )
            for sc in extras:
                print(f"    {sc.chunk.citation_label}")

        print(
            f"\ndense={result.dense_hits} lexical={result.lexical_hits} "
            f"fused={result.fused_hits} reranked={result.reranked} "
            f"in_prompt={len(result.chunks)}"
        )
        return 0

    if args.cmd == "ask":
        if args.stream:
            final = None
            for kind, payload in pipe.stream(args.query, top_k=args.top_k):
                if kind == "token":
                    sys.stdout.write(payload)
                    sys.stdout.flush()
                elif kind == "final":
                    final = payload
            print("\n")
            if final:
                for src in final.sources:
                    print(f"  [{src['n']}] {src['label']}")
            return 0

        answer = pipe.ask(args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps(answer.to_dict(), indent=2))
            return 0
        print(answer.text + "\n")
        for src in answer.sources:
            print(f"  [{src['n']}] {src['label']}  (score {src['score']})")
        print(
            f"\ngrounded={answer.grounded} refused={answer.refused} "
            f"coverage={answer.grounding.coverage:.2f}" if answer.grounding else ""
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
