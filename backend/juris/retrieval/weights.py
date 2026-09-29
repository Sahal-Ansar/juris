"""Pinned model downloads from the Hugging Face Hub (reranker 4.4, embedding bake-off 5.3).

Files are fetched at a fixed revision into ``<data_dir>/models/<org>__<name>``; large files
are checked against the SHA-256 the Hub lists for them. Files already present with the right
hash are skipped, so a fetch can be re-run after an interruption.
"""

import hashlib
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from juris.config import get_settings


def model_dir(model: str, data_dir: Path | None = None) -> Path:
    return (data_dir or get_settings().data_dir) / "models" / model.replace("/", "__")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pinned(model: str, revision: str, files: Mapping[str, str | None], target: Path) -> Path:
    """Download ``files`` (name -> SHA-256 or None) of ``model`` at ``revision`` into ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    for name, digest in files.items():
        path = target / name
        if path.exists() and (digest is None or sha256(path) == digest):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        url = f"https://huggingface.co/{model}/resolve/{revision}/{name}"
        urllib.request.urlretrieve(url, part)
        if digest is not None and sha256(part) != digest:
            part.unlink()
            raise ValueError(f"{name}: SHA-256 mismatch")
        part.replace(path)
    return target
