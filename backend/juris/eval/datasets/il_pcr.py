"""IL-PCR (IL-TUR ``pcr``): prior case retrieval over Indian Supreme Court judgments (D-011).

Gated on Hugging Face (``Exploration-Lab/IL-TUR``, manual approval, CC-BY-NC-SA-4.0,
research-only, no re-sharing). Each split is a parquet file ``pcr/<split>-00000-of-00001.parquet``
with ``id`` (Indian Kanoon ID), ``text`` (a list of sentences) and ``relevant_candidates``
(for queries). Queries are whole judgments; the pool is the candidates of the same split
(train 827 queries / 4,320 candidates, dev 118 / 1,023, test 237 / 1,727). Schema and counts
are from the dataset's public metadata at revision ``d16219ad``.

``fetch_il_pcr`` downloads the six files with the author's Hugging Face token once access is
granted; nothing from IL-PCR is committed or redistributed.
"""

import os
import urllib.request
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from juris.eval.datasets.common import (
    BenchmarkMissing,
    Document,
    RetrievalDataset,
    RetrievalQuery,
    benchmarks_dir,
)

REPO = "Exploration-Lab/IL-TUR"
REVISION = "d16219ad0423cc181ec8460d930fd10a907664b6"
LICENCE = "CC-BY-NC-SA-4.0, research use only, no re-sharing (IL-TUR)"
Split = Literal["train", "dev", "test"]
PARTS = [f"{s}_{kind}" for s in ("train", "dev", "test") for kind in ("queries", "candidates")]


def il_pcr_dir(data_dir: Path | None = None) -> Path:
    return benchmarks_dir(data_dir) / "il-tur" / "pcr"


def _file(root: Path, part: str) -> Path:
    return root / f"{part}-00000-of-00001.parquet"


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and path.exists():
        token = path.read_text(encoding="utf-8").strip()
    return token or None


def fetch_il_pcr(data_dir: Path | None = None, token: str | None = None) -> Path:
    """Download the pcr parquet files at the pinned revision (needs approved access)."""
    token = token or hf_token()
    if not token:
        raise BenchmarkMissing(
            "IL-PCR needs a Hugging Face token with approved access to Exploration-Lab/IL-TUR "
            "(set HF_TOKEN or run `huggingface-cli login`)"
        )
    root = il_pcr_dir(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        target = _file(root, part)
        if target.exists():
            continue
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/pcr/{target.name}"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        part_file = target.with_name(target.name + ".part")
        with urllib.request.urlopen(request) as response, part_file.open("wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        part_file.replace(target)
    return root


def _rows(path: Path) -> list[dict[str, object]]:
    return pq.read_table(path).to_pylist()


def _text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(s) for s in value)
    return str(value or "")


def load_il_pcr(split: Split = "test", root: Path | None = None) -> RetrievalDataset:
    root = root or il_pcr_dir()
    queries_file, candidates_file = (
        _file(root, f"{split}_queries"),
        _file(root, f"{split}_candidates"),
    )
    if not (queries_file.exists() and candidates_file.exists()):
        raise BenchmarkMissing(
            f"IL-PCR {split} not found in {root}: request access to {REPO} on Hugging Face, "
            "then run `uv run scripts/benchmarks.py fetch il-pcr`"
        )
    documents = {
        str(r["id"]): Document(str(r["id"]), _text(r["text"])) for r in _rows(candidates_file)
    }
    queries = [
        RetrievalQuery(
            str(r["id"]),
            _text(r["text"]),
            frozenset(str(c) for c in (r.get("relevant_candidates") or [])),  # type: ignore[attr-defined]
        )
        for r in _rows(queries_file)
    ]
    return RetrievalDataset(
        name="il-pcr",
        split=split,
        queries=queries,
        documents=documents,
        licence=LICENCE,
        notes="Queries are whole judgments; tune on dev, report on test (D-011).",
    )
