"""COLIEE entailment data for the citation verifier's sanity check (D-011, IDEA_final R11).

Needs signed memoranda (coliee.org/data-request), so nothing is downloaded automatically. Put
the unpacked data under ``<data_dir>/raw/benchmarks/coliee/`` and point the loaders at it.
Formats follow the COLIEE task descriptions and must be confirmed on the real files:

- Task 2 (case-law entailment, English): one directory per case holding
  ``entailed_fragment.txt`` and ``paragraphs/NNN.txt`` (``base_case.txt`` is ignored), plus a
  labels JSON ``{"<case>": ["NNN.txt", ...]}`` naming the paragraphs that entail the fragment.
  Each paragraph becomes one example: premise = paragraph, hypothesis = fragment, label = named.
- Task 4 (statute entailment, English 2025 archive): XML files of
  ``<pair id=".." label="Y|N"><t1>articles</t1><t2>question</t2></pair>``. Premise = articles,
  hypothesis = question, label = ``Y``.

Neither is Indian law: they check only that the verifier's entailment step behaves sensibly.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from juris.eval.datasets.common import (
    BenchmarkMissing,
    EntailmentDataset,
    EntailmentExample,
    benchmarks_dir,
)

LICENCE = "COLIEE memorandum terms (research use; not redistributed)"


def coliee_dir(data_dir: Path | None = None) -> Path:
    return benchmarks_dir(data_dir) / "coliee"


def _missing(what: str, where: Path) -> BenchmarkMissing:
    return BenchmarkMissing(
        f"COLIEE {what} not found at {where}: sign the memorandum at coliee.org/data-request "
        "and unpack the data there (docs/eval/benchmarks.md)"
    )


def load_task2(cases_dir: Path, labels_file: Path, split: str = "train") -> EntailmentDataset:
    if not cases_dir.is_dir() or not labels_file.exists():
        raise _missing("Task 2", cases_dir)
    labels: dict[str, list[str]] = json.loads(labels_file.read_text(encoding="utf-8"))
    examples: list[EntailmentExample] = []
    for case in sorted(p for p in cases_dir.iterdir() if p.is_dir()):
        fragment_file = case / "entailed_fragment.txt"
        if not fragment_file.exists():
            continue
        fragment = fragment_file.read_text(encoding="utf-8").strip()
        gold = set(labels.get(case.name, []))
        for para in sorted((case / "paragraphs").glob("*.txt")):
            examples.append(
                EntailmentExample(
                    id=f"{case.name}/{para.name}",
                    premise=para.read_text(encoding="utf-8").strip(),
                    hypothesis=fragment,
                    label=para.name in gold,
                    meta={"case": case.name, "paragraph": para.name},
                )
            )
    return EntailmentDataset("coliee-task2", split, examples, LICENCE)


def load_task4(xml_dir: Path, split: str = "all") -> EntailmentDataset:
    files = sorted(xml_dir.glob("*.xml")) if xml_dir.is_dir() else []
    if not files:
        raise _missing("Task 4", xml_dir)
    examples: list[EntailmentExample] = []
    for path in files:
        for pair in ET.parse(path).getroot().iter("pair"):
            t1, t2 = pair.find("t1"), pair.find("t2")
            if t1 is None or t2 is None:
                continue
            examples.append(
                EntailmentExample(
                    id=pair.get("id") or f"{path.stem}/{len(examples)}",
                    premise=(t1.text or "").strip(),
                    hypothesis=(t2.text or "").strip(),
                    label=(pair.get("label") or "").strip().upper() == "Y",
                    meta={"file": path.name},
                )
            )
    return EntailmentDataset("coliee-task4", split, examples, LICENCE)
