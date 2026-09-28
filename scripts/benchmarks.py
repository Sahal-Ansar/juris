"""Fetch and summarise the external benchmarks (PLAN 5.2).

    uv run scripts/benchmarks.py fetch aila2019   # open, ~21 MB, MD5-checked
    uv run scripts/benchmarks.py fetch il-pcr     # needs approved Hugging Face access + token
    uv run scripts/benchmarks.py summary          # load and validate every benchmark on disk

Data goes to ``<data_dir>/raw/benchmarks/`` and is never committed. COLIEE is placed there by
hand after signing its memoranda (see ``juris.eval.datasets.coliee``).
"""

import argparse
import statistics
import sys
from collections.abc import Callable

from juris.eval.datasets import (
    BenchmarkMissing,
    EntailmentDataset,
    RetrievalDataset,
    fetch_aila2019,
    fetch_il_pcr,
    load_aila2019,
    load_il_pcr,
    load_task2,
    load_task4,
)
from juris.eval.datasets.coliee import coliee_dir

FETCHERS = {"aila2019": fetch_aila2019, "il-pcr": fetch_il_pcr}
Loader = Callable[[], RetrievalDataset | EntailmentDataset]


def loaders() -> dict[str, Loader]:
    root = coliee_dir()
    return {
        "aila2019 precedents": lambda: load_aila2019("precedents"),
        "aila2019 statutes": lambda: load_aila2019("statutes"),
        "il-pcr test": lambda: load_il_pcr("test"),
        "il-pcr dev": lambda: load_il_pcr("dev"),
        "coliee task2": lambda: load_task2(
            root / "task2" / "files", root / "task2" / "labels.json"
        ),
        "coliee task4": lambda: load_task4(root / "task4"),
    }


def describe(ds: RetrievalDataset | EntailmentDataset) -> str:
    if isinstance(ds, RetrievalDataset):
        rel = [len(q.relevant) for q in ds.queries]
        words = [len(q.text.split()) for q in ds.queries]
        return (
            f"{len(ds.queries)} queries (median {statistics.median(words):.0f} words), "
            f"{len(ds.documents)} documents, relevant per query {min(rel)}-{max(rel)} "
            f"(median {statistics.median(rel):g})"
        )
    positives = sum(e.label for e in ds.examples)
    return f"{len(ds.examples)} examples, {positives} entailed"


def summary() -> int:
    failed = 0
    for name, load in loaders().items():
        try:
            ds = load()
        except BenchmarkMissing as exc:
            print(f"- {name}: MISSING. {exc}")
            continue
        problems = ds.validate()
        failed += bool(problems)
        status = "valid" if not problems else f"{len(problems)} PROBLEMS: {problems[:3]}"
        print(f"- {name}: {describe(ds)}; {status}; licence: {ds.licence}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("benchmark", choices=sorted(FETCHERS))
    sub.add_parser("summary")
    args = ap.parse_args(argv)
    if args.command == "fetch":
        try:
            print(f"{args.benchmark}: {FETCHERS[args.benchmark]()}")
        except BenchmarkMissing as exc:
            print(f"{args.benchmark}: {exc}")
            return 1
        return 0
    return summary()


if __name__ == "__main__":
    sys.exit(main())
