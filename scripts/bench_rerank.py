"""Latency of hybrid retrieval with and without reranking, and what reranking changes (PLAN 4.4).

For every benchmark query (``juris.retrieval.bench``): hybrid search (lexical + dense, RRF)
with the reranker off and on, the median time of ``--repeats`` runs after a warm-up pass, and
how many of the top 10 changed. The first hit before and after is shown for reading. Quality
is measured in PLAN 5.3.

    JURIS_DATA_DIR=C:/juris-data uv run --group embed scripts/bench_rerank.py \
        [--k 10] [--top-n 50] [--repeats 3] [--out docs/retrieval/rerank_latency.md]
"""

import argparse
import datetime as dt
import statistics
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine

from juris.config import get_settings
from juris.db.embeddings import prewarm
from juris.ingest.tokens import REVISION, tokenizer_dir
from juris.retrieval.bench import QUERIES, header, label, pct
from juris.retrieval.dense import DenseRetriever
from juris.retrieval.embed import BgeM3Encoder
from juris.retrieval.filters import SearchFilters
from juris.retrieval.hybrid import HybridRetriever
from juris.retrieval.lexical import LexicalRetriever
from juris.retrieval.rerank import REVISION as RERANK_REVISION
from juris.retrieval.rerank import BgeReranker, Reranker, reranker_dir


def stats_row(name: str, ms: list[float]) -> str:
    return f"| {name} | {pct(ms, 0.5):.0f} | {pct(ms, 0.95):.0f} | {max(ms):.0f} |"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, help="write the results as Markdown")
    args = ap.parse_args(argv)

    engine = create_engine(get_settings().database_url())
    encoder = BgeM3Encoder(path=tokenizer_dir(), revision=REVISION)
    cross = BgeReranker(reranker_dir())
    hybrid = HybridRetriever(
        LexicalRetriever(engine),
        DenseRetriever(engine, encoder),
        reranker=Reranker(engine, cross, top_n=args.top_n),
    )
    with engine.begin() as conn:
        prewarm(conn, encoder.model)
    for _, q, f in QUERIES:  # warm-up
        hybrid.search(q, f, k=args.k)

    rows: list[tuple[str, str, SearchFilters | None, float, float, int, tuple[str, str]]] = []
    for kind, q, f in QUERIES:
        timings: dict[bool, list[float]] = {False: [], True: []}
        results = {}
        for on in (False, True):
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                results[on] = hybrid.search(q, f, k=args.k, rerank=on)
                timings[on].append((time.perf_counter() - t0) * 1000)
        off_ms, on_ms = statistics.median(timings[False]), statistics.median(timings[True])
        before = [h.chunk_id for h in results[False]]
        after = [h.chunk_id for h in results[True]]
        changed = len(set(after) - set(before))
        first = (
            header(engine, before[0]) if before else "",
            header(engine, after[0]) if after else "",
        )
        rows.append((kind, q, f, off_ms, on_ms, changed, first))

    off_ms_all = [r[3] for r in rows]
    on_ms_all = [r[4] for r in rows]
    added = [r[4] - r[3] for r in rows]
    lines = [
        "# Reranking latency (PLAN 4.4)",
        "",
        f"Made by `scripts/bench_rerank.py` on {dt.date.today()}. Hybrid search (lexical + "
        f"bge-m3, RRF, depth 100) with and without {cross.model}@{RERANK_REVISION[:8]} on "
        f"{cross.device}, reranking the fused top {args.top_n}; k = {args.k}; median of "
        f"{args.repeats} runs per query after a warm-up pass. 'New in top {args.k}' counts "
        "hits reranking brought in from below the fused top k.",
        "",
        "| | p50 ms | p95 ms | max ms |",
        "|---|---|---|---|",
        stats_row("hybrid", off_ms_all),
        stats_row("hybrid + rerank", on_ms_all),
        stats_row("rerank added", added),
        "",
        f"| Kind | Query | Hybrid ms | + rerank ms | New in top {args.k} | First hit before "
        "| First hit after |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, q, f, off_ms, on_ms, changed, (b, a) in rows:
        lines.append(
            f"| {kind} | {label(q, f)} | {off_ms:.0f} | {on_ms:.0f} | {changed} | {b[:55]} "
            f"| {a[:55]} |"
        )
    report = "\n".join(lines) + "\n"
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
