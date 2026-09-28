"""Download the pinned bge-reranker-v2-m3 files (about 2.3 GB, SHA-256 checked) (PLAN 4.4).

Files already present with the right hash are skipped, so the script can be re-run.

    JURIS_DATA_DIR=C:/juris-data uv run scripts/fetch_reranker.py
"""

import sys
import time

from juris.retrieval.rerank import MODEL, REVISION, fetch_reranker


def main() -> int:
    started = time.monotonic()
    path = fetch_reranker()
    print(f"{MODEL}@{REVISION[:8]} in {path} ({time.monotonic() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
