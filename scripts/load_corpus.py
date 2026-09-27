"""Load a snapshot's processed data into Postgres, idempotently (PLAN 3.6).

Run ``uv run alembic upgrade head`` first. Loads the snapshot row, documents with their
normalised metadata (3.4), paragraphs (3.3), statute sections (3.5) and registers the candidate
embedding models (``configs/embeddings.yaml``). Chunks, vectors and citation edges are loaded
by 3.7-3.9. Loading twice leaves the same row counts.

Usage: uv run scripts/load_corpus.py [--snapshot ID] [--url SQLALCHEMY_URL]
"""

import argparse
import sys
import time

from sqlalchemy import create_engine

from juris.config import get_settings
from juris.db.load import CorpusLoader, row_counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load processed corpus data (PLAN 3.6)")
    parser.add_argument("--snapshot", default="mvp_contract-1f53c208a8")
    parser.add_argument("--url", help="SQLAlchemy URL (default: POSTGRES_* settings)")
    args = parser.parse_args(argv)

    settings = get_settings()
    engine = create_engine(args.url or settings.database_url())
    loader = CorpusLoader(settings.data_dir, args.snapshot)
    started = time.monotonic()
    with engine.begin() as conn:  # one transaction: a failed load leaves the DB unchanged
        stats = loader.load_all(conn)
        counts = row_counts(conn)
    print(f"loaded {args.snapshot} in {time.monotonic() - started:.0f} s")
    print("  written:", ", ".join(f"{k} {v}" for k, v in stats.counts.items()))
    print("  rows:   ", ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
