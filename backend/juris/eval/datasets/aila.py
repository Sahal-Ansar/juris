"""AILA 2019 (FIRE 2019): precedent and statute retrieval for Indian law (D-011).

Open data (CC-BY-4.0) on Zenodo, record 4063986: one zip with 50 queries (descriptions of
legal situations), 2,914 Supreme Court case documents (``Object_casedocs/C<n>.txt``), 197
statutes (``Object_statutes/S<n>.txt``: a "Title:" and a "Desc:" line) and TREC-style relevance
judgments (``<query> Q0 <doc> <0|1>``).

- Task 1, ``precedents``: query -> relevant prior cases (1-22 per query).
- Task 2, ``statutes``: query -> relevant statutes (2-5 per query). The statute pool is mostly
  constitutional, penal and procedural provisions, not contract law, so it measures the
  retriever rather than our domain (docs/eval/benchmarks.md).

All 50 queries form one split; 5.3 decides how to use them without tuning on the test data.
"""

import hashlib
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Literal

from juris.eval.datasets.common import (
    BenchmarkMissing,
    Document,
    RetrievalDataset,
    RetrievalQuery,
    benchmarks_dir,
)

URL = "https://zenodo.org/api/records/4063986/files/AILA_2019_dataset.zip/content"
MD5 = "07f9621e385ff0d4540ce8dfd76b0c21"
LICENCE = "CC-BY-4.0 (AILA 2019, Zenodo 4063986)"
Task = Literal["precedents", "statutes"]


def aila_dir(data_dir: Path | None = None) -> Path:
    return benchmarks_dir(data_dir) / "aila2019"


def fetch_aila2019(data_dir: Path | None = None) -> Path:
    """Download (about 21 MB), check the MD5 and unpack; files already there are kept."""
    root = aila_dir(data_dir)
    if (root / "Query_doc.txt").exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "AILA_2019_dataset.zip"
    if not archive.exists():
        part = archive.with_name(archive.name + ".part")
        urllib.request.urlretrieve(URL, part)
        part.replace(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        archive.unlink()
        raise ValueError("AILA 2019 archive: MD5 mismatch")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    return root


def _qrels(path: Path) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[3] == "1":
            relevant[parts[0]].add(parts[2])
    return relevant


def _case(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    title = text.split("\n", 1)[0].strip() or None
    return Document(path.stem, text, title)


def _statute(path: Path) -> Document:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = next((ln.removeprefix("Title:").strip() for ln in lines if ln.startswith("Title:")), "")
    desc = next((ln.removeprefix("Desc:").strip() for ln in lines if ln.startswith("Desc:")), "")
    return Document(path.stem, f"{title}\n{desc}".strip(), title or None)


def load_aila2019(task: Task = "precedents", root: Path | None = None) -> RetrievalDataset:
    root = root or aila_dir()
    queries_file = root / "Query_doc.txt"
    if not queries_file.exists():
        raise BenchmarkMissing(
            f"AILA 2019 not found in {root}: run `uv run scripts/benchmarks.py fetch aila2019`"
        )
    if task == "precedents":
        folder, qrels, make = "Object_casedocs", "relevance_judgments_priorcases.txt", _case
    elif task == "statutes":
        folder, qrels, make = "Object_statutes", "relevance_judgments_statutes.txt", _statute
    else:
        raise ValueError(f"unknown AILA task: {task!r}")
    relevant = _qrels(root / qrels)
    documents = {d.id: d for d in map(make, sorted((root / folder).glob("*.txt")))}
    # The released statute pool lacks S32, S58 and S162, yet S58 is judged relevant for four
    # queries. A document no system can retrieve is dropped from the judgments, and recorded.
    dropped = sorted(
        f"{qid}:{doc}" for qid, docs in relevant.items() for doc in docs if doc not in documents
    )
    queries = []
    for line in queries_file.read_text(encoding="utf-8").splitlines():
        if "||" in line:
            qid, text = (part.strip() for part in line.split("||", 1))
            gold = frozenset(d for d in relevant[qid] if d in documents)
            queries.append(RetrievalQuery(qid, text, gold))
    notes = "Evaluate over the AILA pool; its IDs do not map to our corpus (D-011)."
    if dropped:
        notes += f" Relevant documents missing from the released pool, dropped: {dropped}."
    return RetrievalDataset(
        name=f"aila2019-{task}",
        split="all",
        queries=queries,
        documents=documents,
        licence=LICENCE,
        notes=notes,
    )
