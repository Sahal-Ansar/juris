"""Retrieval evaluation runs and the results report (PLAN 5.3, D-037).

    uv run --group embed scripts/eval_retrieval.py run --datasets juris-dev aila-precedents \
        --systems lexical dense-bge hybrid-bge ...
    uv run scripts/eval_retrieval.py report     # docs/eval/retrieval_results.md from saved runs

Datasets (query plan in brackets):
    juris-dev[case]      Juris-Eval dev over our corpus; one query = facts + question
    juris-dev[issues]    the same items with the hand-written issue queries (fused)
    aila-precedents[head|win], aila-statutes[head|win], il-pcr-dev[head|win], il-pcr-test[head|win]
        benchmark pools (``scripts/index_pools.py``); head = the whole text as one query (dense
        models read its first 512 tokens), win = 300-word windows fused with RRF (at most 10)

Each run is saved to ``eval/results/raw/retrieval/<dataset>/<system>.json`` (per-query metrics).
Tune on juris-dev and il-pcr-dev only; AILA (no split) and il-pcr-test are reported, not tuned.
"""

import argparse
import datetime as dt
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path

import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from juris.config import REPO_ROOT, get_settings
from juris.eval.datasets import RetrievalDataset, load_aila2019, load_il_pcr
from juris.eval.juris_eval import EVAL_DIR, load_source
from juris.eval.pool import pool_engine
from juris.eval.retrieval_eval import (
    EvalQuery,
    Result,
    System,
    UnitOf,
    corpus_unit,
    doc_unit,
    evaluate,
    hybrid,
    single,
    table,
    windows,
)
from juris.retrieval.dense import DenseHit, DenseRetriever
from juris.retrieval.embed import Encoder
from juris.retrieval.hybrid import HybridRetriever
from juris.retrieval.lexical import LexicalHit, LexicalRetriever
from juris.retrieval.rerank import Reranker

RAW = REPO_ROOT / "eval" / "results" / "raw" / "retrieval"
REPORT = REPO_ROOT / "docs" / "eval" / "retrieval_results.md"
MAX_TERMS = 64  # lexical terms kept from a long query (its rarest)
MAX_WINDOWS = 10


# ---- encoders (loaded once) ----------------------------------------------------------------


@cache
def encoder(name: str) -> Encoder:
    if name == "bge":
        from juris.ingest.tokens import REVISION, tokenizer_dir
        from juris.retrieval.embed import BgeM3Encoder

        return BgeM3Encoder(path=tokenizer_dir(), revision=REVISION)
    from juris.retrieval.embed import E5_MODEL, E5_REVISION, E5Encoder
    from juris.retrieval.weights import model_dir

    return E5Encoder(path=model_dir(E5_MODEL), revision=E5_REVISION)


@cache
def cross_encoder():  # type: ignore[no-untyped-def]
    from juris.retrieval.rerank import BgeReranker, reranker_dir

    return BgeReranker(reranker_dir())


# ---- datasets ------------------------------------------------------------------------------


def juris_dev(plan: str) -> tuple[Engine, UnitOf, list[EvalQuery]]:
    engine = create_engine(get_settings().database_url())
    items = [i for i in load_source() if i.split == "dev"]
    hand = yaml.safe_load((EVAL_DIR / "search_queries.yaml").read_text(encoding="utf-8"))["queries"]
    queries = []
    for item in items:
        relevant = frozenset(
            [a.doc_id for a in item.gold_supporting_authorities]
            + [a.doc_id for a in item.gold_contrary_authorities]
            + list(item.gold_sections)
        )
        texts = (f"{item.facts}\n{item.question}",) if plan == "case" else tuple(hand[item.id])
        queries.append(EvalQuery(item.id, texts, relevant))
    return engine, corpus_unit(engine), queries


def pool(name: str, load: Callable[[], RetrievalDataset], plan: str):  # type: ignore[no-untyped-def]
    ds = load()
    queries = [
        EvalQuery(
            q.id,
            (q.text,) if plan == "head" else tuple(windows(q.text, 300, MAX_WINDOWS)),
            q.relevant,
        )
        for q in ds.queries
    ]
    return pool_engine(name), doc_unit, queries


DATASETS = {
    "juris-dev[case]": lambda: juris_dev("case"),
    "juris-dev[issues]": lambda: juris_dev("issues"),
}
POOLS: dict[str, tuple[str, Callable[[], RetrievalDataset]]] = {
    "aila-precedents": ("aila2019-precedents", lambda: load_aila2019("precedents")),
    "aila-statutes": ("aila2019-statutes", lambda: load_aila2019("statutes")),
    "il-pcr-dev": ("il-pcr-dev", lambda: load_il_pcr("dev")),
    "il-pcr-test": ("il-pcr-test", lambda: load_il_pcr("test")),
}
for _name, (_pool, _load) in POOLS.items():
    for _plan in ("head", "win"):
        DATASETS[f"{_name}[{_plan}]"] = partial(pool, _pool, _load, _plan)


# ---- systems -------------------------------------------------------------------------------


@dataclass
class LexicalSearch:
    retriever: LexicalRetriever

    def __call__(self, query: str, k: int) -> Sequence[LexicalHit]:
        return self.retriever.search(query, k=k)


@dataclass
class DenseSearch:
    retriever: DenseRetriever

    def __call__(self, query: str, k: int) -> Sequence[DenseHit]:
        return self.retriever.search(query, k=k)


def systems(engine: Engine, unit_of: UnitOf, names: list[str]) -> list[System]:
    lexical = LexicalRetriever(engine, max_terms=MAX_TERMS)
    out: list[System] = []
    for name in names:
        if name == "lexical":
            out.append(single(name, LexicalSearch(lexical), unit_of))
        elif name.startswith("dense-"):
            dense = DenseRetriever(engine, encoder(name.split("-")[1]))
            out.append(single(name, DenseSearch(dense), unit_of))
        elif name.startswith("hybrid-"):
            # hybrid-bge, hybrid-e5, hybrid-both (lexical + bge + e5), with -rerank / -rerank30
            parts = name.split("-")
            models = ["bge", "e5"] if parts[1] == "both" else [parts[1]]
            dense = DenseRetriever(engine, encoder(models[0]))
            extra = [DenseRetriever(engine, encoder(m)) for m in models[1:]]
            reranker = None
            if len(parts) > 2:
                top_n = int(parts[2].removeprefix("rerank") or 50)
                reranker = Reranker(engine, cross_encoder(), top_n=top_n)
            retriever = HybridRetriever(lexical, dense, reranker=reranker, extra_dense=extra)
            out.append(hybrid(name, retriever, unit_of, rerank=reranker is not None))
        else:
            raise SystemExit(f"unknown system {name!r}")
    return out


# ---- commands ------------------------------------------------------------------------------


def save(result: Result) -> Path:
    path = RAW / result.dataset / f"{result.system}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": result.system,
        "dataset": result.dataset,
        "seconds": result.seconds,
        "per_query": result.per_query,
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_results() -> dict[str, list[Result]]:
    out: dict[str, list[Result]] = {}
    for path in sorted(RAW.glob("*/*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        r = Result(d["system"], d["dataset"], d["per_query"], d["seconds"])
        out.setdefault(r.dataset, []).append(r)
    return out


ORDER = [
    "lexical", "dense-bge", "dense-e5", "hybrid-bge", "hybrid-e5", "hybrid-both",
    "hybrid-bge-rerank30", "hybrid-bge-rerank", "hybrid-e5-rerank", "hybrid-both-rerank30",
]  # fmt: skip


def report() -> str:
    lines = [
        "# Retrieval evaluation (PLAN 5.3)",
        "",
        f"Generated by `scripts/eval_retrieval.py report` on {dt.date.today()} from the runs in "
        "`eval/results/raw/retrieval/`. Units are authorities (judgments, and statute sections "
        "in our corpus); a system ranks them by their best chunk. Binary relevance; means over "
        "queries; bold = best in the table. Method and choices: D-037.",
        "",
        "Systems: `lexical` = Postgres FTS any-mode (4.1; at most 64 query terms, the rarest); "
        "`dense-bge` / `dense-e5` = bge-m3 / multilingual-e5-large (4.2); `hybrid-*` = both "
        "fused with RRF k=60 (4.3), `hybrid-both` = lexical + bge-m3 + e5; "
        "`-rerank` = bge-reranker-v2-m3 over the fused top 50 "
        "(`-rerank30`: top 30) (4.4).",
    ]
    for dataset, results in sorted(load_results().items()):
        results.sort(key=lambda r: ORDER.index(r.system) if r.system in ORDER else 99)
        n = len(results[0].per_query)
        lines += ["", f"## {dataset} ({n} queries)", "", table(results)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="+", required=True, choices=list(DATASETS))
    r.add_argument("--systems", nargs="+", required=True)
    sub.add_parser("report")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if args.command == "report":
        REPORT.write_text(report(), encoding="utf-8")
        print(f"wrote {REPORT}")
        return 0
    for dataset in args.datasets:
        engine, unit_of, queries = DATASETS[dataset]()
        for system in systems(engine, unit_of, args.systems):
            result = evaluate(system, dataset, queries)
            save(result)
            m = result.mean
            print(
                f"{dataset:24} {system.name:22} R@10 {m['R@10']:.3f} R@50 {m['R@50']:.3f} "
                f"MRR {m['MRR']:.3f} nDCG@10 {m['nDCG@10']:.3f} "
                f"({result.seconds_per_query:.2f} s/q)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
