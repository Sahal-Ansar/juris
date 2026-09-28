"""Latency of dense retrieval on the loaded slice (PLAN 4.2: < 300 ms per query).

Loads bge-m3 (GPU if available), prewarms the model's HNSW index and vectors, runs the
benchmark queries (``juris.retrieval.bench``) once to warm up, then reports the median of
``--repeats`` timings per query: encoding the query, the search, and the total. The search
path (HNSW or exact) is shown per query.

Recall of the HNSW path against an exact scan is measured too (``--recall``), on the same
queries without filters.

    JURIS_DATA_DIR=C:/juris-data uv run --group embed scripts/bench_dense.py \
        [--k 50] [--repeats 3] [--out docs/retrieval/dense_latency.md]
"""

import argparse
import datetime as dt
import statistics
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from juris.config import get_settings
from juris.db.embeddings import prewarm
from juris.ingest.tokens import REVISION, tokenizer_dir
from juris.retrieval.bench import QUERIES, chunk_count, header, label, pct
from juris.retrieval.dense import DenseRetriever
from juris.retrieval.embed import BgeM3Encoder, _literal
from juris.retrieval.filters import SearchFilters


def exact_top(engine: Engine, model: str, dim: int, vector: list[float], k: int) -> set[str]:
    distance = f"(e.embedding::vector({dim})) <=> CAST(:q AS vector({dim}))"
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT e.chunk_id, 1 - ({distance}) AS s FROM chunk_embeddings e "
                "WHERE e.model = :m ORDER BY s DESC LIMIT :k"
            ),
            {"q": _literal(vector), "m": model, "k": k},
        )
        return {r[0] for r in rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--recall", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", type=Path, help="write the results as Markdown")
    args = ap.parse_args(argv)

    engine = create_engine(get_settings().database_url())
    encoder = BgeM3Encoder(path=tokenizer_dir(), revision=REVISION)
    retriever = DenseRetriever(engine, encoder)
    t0 = time.perf_counter()
    with engine.begin() as conn:
        blocks = prewarm(conn, encoder.model)
    warm_s = time.perf_counter() - t0
    for _, q, f in QUERIES:  # warm-up: CUDA kernels, plans, the per-document chunk counts
        retriever.search(q, f, k=args.k)

    rows: list[tuple[str, str, SearchFilters | None, str, float, float, float, int, str]] = []
    for kind, q, f in QUERIES:
        enc, srch = [], []
        path = ""
        hits = []
        for _ in range(args.repeats):
            t1 = time.perf_counter()
            vec = retriever.encode([q])[0]
            t2 = time.perf_counter()
            hits = retriever.search_vector(vec, f, k=args.k)
            t3 = time.perf_counter()
            enc.append((t2 - t1) * 1000)
            srch.append((t3 - t2) * 1000)
            path = retriever.last_path or ""
        e_ms, s_ms = statistics.median(enc), statistics.median(srch)
        top = header(engine, hits[0].chunk_id) if hits else ""
        rows.append((kind, q, f, path, e_ms, s_ms, e_ms + s_ms, len(hits), top))

    recalls = []
    if args.recall:
        for _, q, f in QUERIES:
            if f is None:
                vec = retriever.encode([q])[0]
                got = {h.chunk_id for h in retriever.search_vector(vec, None, k=args.k)}
                truth = exact_top(engine, encoder.model, encoder.dim, vec, args.k)
                recalls.append(len(got & truth) / len(truth))

    total = [r[6] for r in rows]
    lines = [
        "# Dense retrieval latency (PLAN 4.2)",
        "",
        f"Made by `scripts/bench_dense.py` on {dt.date.today()}. Corpus: "
        f"{chunk_count(engine):,} chunks; {encoder.model} on {encoder.device}; "
        f"ef_search {retriever.ef_search}; exact scan at or below {retriever.exact_below:,} "
        f"allowed chunks. Prewarm read {blocks:,} blocks in {warm_s:.1f} s. k = {args.k}; "
        f"median of {args.repeats} runs per query after a warm-up pass. Target: < 300 ms.",
        "",
        "| | p50 ms | p95 ms | max ms |",
        "|---|---|---|---|",
    ]
    columns = {
        "encode": [r[4] for r in rows],
        "search": [r[5] for r in rows],
        "total": [r[6] for r in rows],
    }
    for name, ms in columns.items():
        lines.append(f"| {name} | {pct(ms, 0.5):.0f} | {pct(ms, 0.95):.0f} | {max(ms):.0f} |")
    if recalls:
        lines += [
            "",
            f"HNSW recall@{args.k} against an exact scan, {len(recalls)} unfiltered queries: "
            f"mean {statistics.mean(recalls):.3f}, min {min(recalls):.2f}.",
        ]
    lines += [
        "",
        "| Kind | Query | Path | Encode ms | Search ms | Total ms | Hits | First hit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for kind, q, f, path, e_ms, s_ms, t_ms, n, top in rows:
        lines.append(
            f"| {kind} | {label(q, f)} | {path} | {e_ms:.0f} | {s_ms:.0f} | {t_ms:.0f} | {n} "
            f"| {top[:70]} |"
        )
    report = "\n".join(lines) + "\n"
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    return 0 if max(total) < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
