"""Index and embed the benchmark pools for retrieval evaluation (PLAN 5.3).

Each pool goes into its own database (``juris_pool_<name>``) through the corpus pipeline:
segment, chunk, load, lexeme stats, then vectors for the chosen embedding model(s). Resumable.

    JURIS_DATA_DIR=C:/juris-data uv run --group embed scripts/index_pools.py \
        [--pools aila2019-precedents aila2019-statutes il-pcr-dev il-pcr-test] \
        [--models bge-m3 e5] [--no-embed]
"""

import argparse
import sys
import time
from collections.abc import Callable

from juris.eval.datasets import RetrievalDataset, load_aila2019, load_il_pcr
from juris.eval.pool import index_pool, pool_engine
from juris.retrieval.embed import Encoder, embed_missing

POOLS: dict[str, tuple[Callable[[], RetrievalDataset], str]] = {
    "aila2019-precedents": (lambda: load_aila2019("precedents"), "judgment"),
    "aila2019-statutes": (lambda: load_aila2019("statutes"), "statute"),
    "il-pcr-dev": (lambda: load_il_pcr("dev"), "judgment"),
    "il-pcr-test": (lambda: load_il_pcr("test"), "judgment"),
}


def encoder(name: str) -> Encoder:
    if name == "bge-m3":
        from juris.ingest.tokens import REVISION, tokenizer_dir
        from juris.retrieval.embed import BgeM3Encoder

        return BgeM3Encoder(path=tokenizer_dir(), revision=REVISION)
    if name == "e5":
        from juris.retrieval.embed import E5_MODEL, E5_REVISION, E5Encoder
        from juris.retrieval.weights import model_dir

        return E5Encoder(path=model_dir(E5_MODEL), revision=E5_REVISION)
    raise SystemExit(f"unknown model {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS))
    ap.add_argument("--models", nargs="+", default=["bge-m3"], choices=["bge-m3", "e5"])
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args(argv)
    encoders = [] if args.no_embed else [encoder(m) for m in args.models]
    for name in args.pools:
        started = time.monotonic()
        load, kind = POOLS[name]
        ds = load()
        engine = pool_engine(name)
        stats = index_pool(engine, ds.documents.values(), kind, ds.licence)
        print(f"{name}: {stats} ({time.monotonic() - started:.0f} s)", flush=True)
        for enc in encoders:
            report = embed_missing(engine, enc)
            print(
                f"  {enc.model}: embedded {report.embedded:,} in {report.seconds / 60:.1f} min",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
