"""Download the pinned multilingual-e5-large files (about 2.3 GB, SHA-256 checked) (PLAN 5.3).

The second embedding candidate for the bake-off (configs/embeddings.yaml, D-026). Files
already present with the right hash are skipped.

    JURIS_DATA_DIR=C:/juris-data uv run scripts/fetch_e5.py
"""

import sys
import time

from juris.retrieval.embed import E5_FILES, E5_MODEL, E5_REVISION
from juris.retrieval.weights import fetch_pinned, model_dir


def main() -> int:
    started = time.monotonic()
    path = fetch_pinned(E5_MODEL, E5_REVISION, E5_FILES, model_dir(E5_MODEL))
    print(f"{E5_MODEL}@{E5_REVISION[:8]} in {path} ({time.monotonic() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
