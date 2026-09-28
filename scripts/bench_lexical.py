"""Latency of lexical retrieval on the loaded slice (PLAN 4.1: < 300 ms per query).

Runs the benchmark queries (``juris.retrieval.bench``) in every mode, after one warm-up pass,
and reports the median of ``--repeats`` timings per query plus p50/p95/max per mode. The first
hit is shown so the numbers can be read against what came back.

    uv run scripts/bench_lexical.py [--k 50] [--repeats 3] [--out docs/retrieval/lexical_latency.md]
"""

import argparse
import datetime as dt
import statistics
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine

from juris.config import Settings
from juris.retrieval.bench import QUERIES, chunk_count, header, label, pct
from juris.retrieval.filters import SearchFilters
from juris.retrieval.lexical import LexicalRetriever, Mode

MODES: list[Mode] = ["any", "all", "phrase"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, help="write the results as Markdown")
    args = ap.parse_args(argv)

    engine = create_engine(Settings().database_url())
    retriever = LexicalRetriever(engine)
    for mode in MODES:  # warm-up: caches, plans, the chunk count
        for _, q, f in QUERIES:
            retriever.search(q, f, k=args.k, mode=mode)

    rows: list[tuple[Mode, str, str, SearchFilters | None, float, int, str]] = []
    for mode in MODES:
        for kind, q, f in QUERIES:
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                hits = retriever.search(q, f, k=args.k, mode=mode)
                times.append((time.perf_counter() - t0) * 1000)
            top = header(engine, hits[0].chunk_id) if hits else ""
            rows.append((mode, kind, q, f, statistics.median(times), len(hits), top))

    lines = [
        "# Lexical retrieval latency (PLAN 4.1)",
        "",
        f"Made by `scripts/bench_lexical.py` on {dt.date.today()}. "
        f"Corpus: {chunk_count(engine):,} chunks. "
        f"k = {args.k}; median of {args.repeats} runs per query after a warm-up pass. "
        "Target: < 300 ms per query. Empty = no hits (phrase mode on a question, for example).",
        "",
        "| Mode | p50 ms | p95 ms | max ms | Queries | Empty |",
        "|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        ms = [r[4] for r in rows if r[0] == mode]
        empty = sum(1 for r in rows if r[0] == mode and r[5] == 0)
        lines.append(
            f"| {mode} | {pct(ms, 0.5):.0f} | {pct(ms, 0.95):.0f} | {max(ms):.0f} | {len(ms)} "
            f"| {empty} |"
        )
    lines += ["", "| Mode | Kind | Query | ms | Hits | First hit |", "|---|---|---|---|---|---|"]
    for mode, kind, q, f, ms_, n, top in rows:
        lines.append(f"| {mode} | {kind} | {label(q, f)} | {ms_:.0f} | {n} | {top[:70]} |")
    report = "\n".join(lines) + "\n"
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    worst = max(r[4] for r in rows)
    return 0 if worst < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
