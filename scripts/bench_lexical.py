"""Latency of lexical retrieval on the loaded slice (PLAN 4.1: < 300 ms per query).

Runs a fixed set of queries (legal terms, section references, case names, questions, single
common words, and filtered searches) in every mode, after one warm-up pass, and reports the
median of ``--repeats`` timings per query plus p50/p95/max per mode. The first hit is shown
so the numbers can be read against what came back.

    uv run scripts/bench_lexical.py [--k 50] [--repeats 3] [--out docs/retrieval/lexical.md]
"""

import argparse
import datetime as dt
import statistics
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, text

from juris.config import Settings
from juris.models import CourtLevel, DocumentKind
from juris.retrieval.filters import SearchFilters
from juris.retrieval.lexical import LexicalRetriever, Mode

SC = SearchFilters(court_levels=(CourtLevel.SUPREME_COURT,))
QUERIES: list[tuple[str, str, SearchFilters | None]] = [
    ("term", "undue influence", None),
    ("term", "frustration of contract", None),
    ("term", "liquidated damages", None),
    ("term", "readiness and willingness", None),
    ("term", "quantum meruit", None),
    ("term", "privity of contract", None),
    ("term", "restraint of trade", None),
    ("section", "Section 74", None),
    ("section", "s. 16(2) undue influence", None),
    ("section", "Section 20 of the Specific Relief Act", None),
    ("section", "u/s 73 compensation for breach", None),
    ("section", "Section 56 impossibility of performance", None),
    ("section", "Section 19A", None),
    ("case", "Satyabrata Ghose v. Mugneeram Bangur", None),
    ("case", "Fateh Chand v. Balkishan Dass", None),
    ("case", "ONGC v. Saw Pipes", None),
    ("case", "Kailash Nath Associates", None),
    ("case", "Mohori Bibee", None),
    (
        "question",
        "Is a contract entered into under coercion by a threat to a third party "
        "voidable at the option of the coerced party?",
        None,
    ),
    ("question", "Can earnest money be forfeited without proof of actual loss?", None),
    (
        "question",
        "When is time of the essence in a contract for sale of immovable property?",
        None,
    ),
    (
        "question",
        "Does a minor's agreement bind the minor or can it be ratified on majority?",
        None,
    ),
    ("common", "contract", None),
    ("common", "court", None),
    ("common", "agreement party", None),
    ("filtered", "specific performance", SC),
    (
        "filtered",
        "liquidated damages",
        SearchFilters(date_from=dt.date(2000, 1, 1), date_to=dt.date(2020, 12, 31)),
    ),
    ("filtered", "specific performance discretion", SearchFilters(acts=("SRA",))),
    ("filtered", "compensation for loss", SearchFilters(doc_kinds=(DocumentKind.STATUTE,))),
    ("filtered", "contract", SearchFilters(court_levels=(CourtLevel.HIGH_COURT,), acts=("ICA",))),
]
MODES: list[Mode] = ["any", "all", "phrase"]


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


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
            top = ""
            if hits:
                with engine.connect() as conn:
                    top = conn.execute(
                        text(
                            "SELECT coalesce(context_header, chunk_id) FROM chunks "
                            "WHERE chunk_id = :c"
                        ),
                        {"c": hits[0].chunk_id},
                    ).scalar_one()
            rows.append((mode, kind, q, f, statistics.median(times), len(hits), top))

    with engine.connect() as conn:
        chunks: int = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    lines = [
        "# Lexical retrieval latency (PLAN 4.1)",
        "",
        f"Made by `scripts/bench_lexical.py` on {dt.date.today()}. Corpus: {chunks:,} chunks. "
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
        label = q if not f else f"{q} [{f.model_dump(exclude_defaults=True, mode='json')}]"
        lines.append(f"| {mode} | {kind} | {label} | {ms_:.0f} | {n} | {top[:70]} |")
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
