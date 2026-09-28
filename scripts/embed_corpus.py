"""Embed every chunk with the primary model and index it (PLAN 3.8).

Resumable: only chunks without a vector for the model are embedded, committed per batch, so an
interrupted run continues where it stopped. For a bulk load the model's HNSW index is dropped
first and rebuilt at the end. The model name, weights revision and coverage are recorded in
``snapshots.embeddings``. Needs ``uv sync --group embed`` and the weights in
``<data_dir>/models/BAAI__bge-m3``.

Usage:
    uv run --group embed scripts/embed_corpus.py [--snapshot ID] [--batch-size 32]
    uv run --group embed scripts/embed_corpus.py --check CHUNK_ID [CHUNK_ID ...]
"""

import argparse
import sys
import time

from sqlalchemy import create_engine

from juris.config import get_settings
from juris.ingest.tokens import REVISION, tokenizer_dir
from juris.retrieval.embed import (
    BgeM3Encoder,
    count_missing,
    embed_missing,
    nearest,
    record_in_snapshot,
)

MODEL, DIM = "BAAI/bge-m3", 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed chunks with bge-m3 (PLAN 3.8)")
    parser.add_argument("--snapshot", default="mvp_contract-1f53c208a8")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--check", nargs="+", metavar="CHUNK_ID", help="print neighbours only")
    args = parser.parse_args(argv)

    engine = create_engine(get_settings().database_url())
    if args.check:
        with engine.connect() as conn:
            for chunk_id in args.check:
                print(f"== {chunk_id}")
                for cid, header, dist in nearest(conn, MODEL, DIM, chunk_id):
                    print(f"   {dist:.3f}  {cid}  {header}")
        return 0

    encoder = BgeM3Encoder(path=tokenizer_dir(), revision=REVISION, batch_size=args.batch_size)
    with engine.connect() as conn:
        total, missing = count_missing(conn, MODEL)
    print(f"{MODEL}@{REVISION[:8]} on {encoder.device}: {missing:,} of {total:,} chunks to embed")
    started = time.monotonic()

    def progress(done: int, todo: int) -> None:
        rate = done / max(1e-6, time.monotonic() - started)
        print(
            f"  {done:,}/{todo:,}  {rate:.0f}/s  eta {(todo - done) / rate / 60:.0f} min",
            flush=True,
        )

    report = embed_missing(engine, encoder, progress=progress)
    with engine.begin() as conn:
        entry = record_in_snapshot(conn, args.snapshot, encoder)
    print(
        f"embedded {report.embedded:,} in {report.seconds / 60:.1f} min "
        f"(index rebuilt: {report.index_rebuilt}, truncated inputs: {report.truncated}); "
        f"coverage {entry['chunks']:,}/{entry['of']:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
