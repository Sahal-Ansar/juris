"""External benchmark loaders (PLAN 5.2): each format on a small synthetic sample.

Samples are written in the benchmarks' published layouts. The real AILA 2019 data is checked
too when it has been fetched (``scripts/benchmarks.py fetch aila2019``); IL-PCR and COLIEE
need the author's access (docs/eval/benchmarks.md).
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from juris.eval.datasets import (
    BenchmarkMissing,
    Document,
    EntailmentDataset,
    EntailmentExample,
    RetrievalDataset,
    RetrievalQuery,
    load_aila2019,
    load_il_pcr,
    load_task2,
    load_task4,
)
from juris.eval.datasets import aila as aila_module
from juris.eval.datasets import il_pcr as il_pcr_module
from juris.eval.datasets.aila import aila_dir
from juris.eval.datasets.coliee import find_task2, find_task4

# ---- AILA 2019 -----------------------------------------------------------------------------


def write_aila(root: Path) -> None:
    (root / "Object_casedocs").mkdir(parents=True)
    (root / "Object_statutes").mkdir()
    (root / "Query_doc.txt").write_text(
        "AILA_Q1||The appellant was dismissed without an inquiry.\n"
        "AILA_Q2||A contract was signed under a threat to the promisor's son.\n",
        encoding="utf-8",
    )
    for n, title in ((1, "A v State"), (2, "B v Union of India"), (3, "C v D")):
        (root / "Object_casedocs" / f"C{n}.txt").write_text(
            f"{title}\nSupreme Court of India\n\nThe judgment text of case {n}.", encoding="utf-8"
        )
    (root / "Object_statutes" / "S1.txt").write_text(
        "Title: Coercion defined\nDesc: Coercion is the committing of any act ...", encoding="utf-8"
    )
    (root / "Object_statutes" / "S2.txt").write_text(
        "Title: Equality before law\nDesc: The State shall not deny ...", encoding="utf-8"
    )
    (root / "relevance_judgments_priorcases.txt").write_text(
        "AILA_Q1 Q0 C1 1\nAILA_Q1 Q0 C2 0\nAILA_Q2 Q0 C3 1\nAILA_Q2 Q0 C2 1\n", encoding="utf-8"
    )
    # S9 is judged relevant but missing from the pool, as S58 is in the real release
    (root / "relevance_judgments_statutes.txt").write_text(
        "AILA_Q1 Q0 S2 1\nAILA_Q2 Q0 S1 1\nAILA_Q2 Q0 S9 1\nAILA_Q2 Q0 S2 0\n", encoding="utf-8"
    )


def test_aila_precedents_and_statutes(tmp_path: Path) -> None:
    write_aila(tmp_path)
    cases = load_aila2019("precedents", tmp_path)
    assert [q.id for q in cases.queries] == ["AILA_Q1", "AILA_Q2"]
    assert cases.queries[0].relevant == {"C1"}  # relevance 0 is not relevant
    assert cases.queries[1].relevant == {"C2", "C3"}
    assert cases.documents["C2"].title == "B v Union of India"
    assert cases.validate() == [] and cases.licence.startswith("CC-BY-4.0")

    statutes = load_aila2019("statutes", tmp_path)
    assert (
        statutes.documents["S1"].text
        == "Coercion defined\nCoercion is the committing of any act ..."
    )
    assert statutes.documents["S1"].title == "Coercion defined"
    assert statutes.queries[1].relevant == {"S1"}  # S9 dropped: not in the pool
    assert "AILA_Q2:S9" in statutes.notes
    assert statutes.validate() == []
    with pytest.raises(ValueError, match="unknown AILA task"):
        load_aila2019("summaries", tmp_path)  # type: ignore[arg-type]


def test_aila_missing_says_how_to_fetch(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkMissing, match="fetch aila2019"):
        load_aila2019("precedents", tmp_path)


def test_aila_fetch_checks_the_md5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "src"
    write_aila(source)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for path in source.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source).as_posix())
    payload = buffer.getvalue()
    monkeypatch.setattr(
        aila_module.urllib.request,
        "urlretrieve",
        lambda url, path: Path(path).write_bytes(payload),
    )
    data_dir = tmp_path / "data"
    monkeypatch.setattr(aila_module, "MD5", "0" * 32)
    with pytest.raises(ValueError, match="MD5"):
        aila_module.fetch_aila2019(data_dir)
    monkeypatch.setattr(aila_module, "MD5", hashlib.md5(payload).hexdigest())
    root = aila_module.fetch_aila2019(data_dir)
    assert root == aila_dir(data_dir)
    assert len(load_aila2019("precedents", root).queries) == 2


@pytest.mark.skipif(not (aila_dir() / "Query_doc.txt").exists(), reason="AILA 2019 not fetched")
def test_real_aila_2019_loads_and_validates() -> None:
    cases, statutes = load_aila2019("precedents"), load_aila2019("statutes")
    assert (len(cases.queries), len(cases.documents), len(statutes.documents)) == (50, 2914, 197)
    assert cases.validate() == [] and statutes.validate() == []
    assert sum(len(q.relevant) for q in cases.queries) == 195
    assert "S58" in statutes.notes  # judged relevant, missing from the released pool


# ---- IL-PCR ----------------------------------------------------------------------------


def write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_il_pcr_reads_the_hf_parquet_layout(tmp_path: Path) -> None:
    write_parquet(
        tmp_path / "test_queries-00000-of-00001.parquet",
        [
            {"id": "101", "text": ["First sentence.", "Second."], "relevant_candidates": ["7"]},
            {"id": "102", "text": ["Another query."], "relevant_candidates": ["7", "8"]},
            # as in the real data: a query whose only "relevant" ID is empty
            {"id": "103", "text": ["Nothing to find."], "relevant_candidates": [""]},
        ],
    )
    write_parquet(
        tmp_path / "test_candidates-00000-of-00001.parquet",
        [
            {"id": "7", "text": ["Candidate seven."], "relevant_candidates": []},
            {"id": "8", "text": ["Candidate eight."], "relevant_candidates": []},
        ],
    )
    ds = load_il_pcr("test", tmp_path)
    assert ds.queries[0].text == "First sentence.\nSecond."
    assert ds.queries[1].relevant == {"7", "8"}
    assert set(ds.documents) == {"7", "8"} and ds.validate() == []
    assert [q.id for q in ds.queries] == ["101", "102"] and "['103']" in ds.notes
    assert "NC" in ds.licence
    with pytest.raises(BenchmarkMissing, match="request access"):
        load_il_pcr("dev", tmp_path)


def test_il_pcr_fetch_needs_a_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(il_pcr_module.Path, "home", lambda: tmp_path)
    with pytest.raises(BenchmarkMissing, match="token"):
        il_pcr_module.fetch_il_pcr(tmp_path)


# ---- COLIEE ----------------------------------------------------------------------------


def test_coliee_task2_one_example_per_paragraph(tmp_path: Path) -> None:
    case = tmp_path / "files" / "001"
    (case / "paragraphs").mkdir(parents=True)
    (case / "base_case.txt").write_text("The decision under review.", encoding="utf-8")
    (case / "entailed_fragment.txt").write_text("The duty of fairness applies.", encoding="utf-8")
    (case / "paragraphs" / "001.txt").write_text("Background facts.", encoding="utf-8")
    (case / "paragraphs" / "002.txt").write_text("Fairness is owed here.", encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"001": ["002.txt"]}), encoding="utf-8")
    ds = load_task2(tmp_path / "files", labels)
    assert [(e.id, e.label) for e in ds.examples] == [("001/001.txt", False), ("001/002.txt", True)]
    assert ds.examples[1].hypothesis == "The duty of fairness applies."
    assert ds.validate() == []
    with pytest.raises(BenchmarkMissing, match="memorandum"):
        load_task2(tmp_path / "nowhere", labels)
    # 100 of the 925 cases in the 2026 training labels are a comma-separated string, and a
    # named paragraph missing from the files is recorded
    labels.write_text(json.dumps({"001": "001.txt, 002.txt, 009.txt"}), encoding="utf-8")
    both = load_task2(tmp_path / "files", labels)
    assert [e.label for e in both.examples] == [True, True]
    assert "001/009.txt" in both.notes


def test_coliee_release_layout_is_found(tmp_path: Path) -> None:
    release = tmp_path / "task2" / "task2_train_files_2026"
    (release / "cases").mkdir(parents=True)
    (release / "task2_train_labels_2026.json").write_text("{}", encoding="utf-8")
    statutes = tmp_path / "task34_en" / "COLIEE2025statute_data-English" / "train"
    statutes.mkdir(parents=True)
    (statutes / "riteval_H18_en.xml").write_text("<dataset/>", encoding="utf-8")
    assert find_task2(tmp_path) == (release / "cases", release / "task2_train_labels_2026.json")
    assert find_task4(tmp_path) == statutes


def test_coliee_task4_pairs(tmp_path: Path) -> None:
    (tmp_path / "task4").mkdir()
    (tmp_path / "task4" / "riteval_R05_en.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<dataset>
  <pair id="R05-01-A" label="Y"><t1>Article 90 A juristic act ... is void.</t1>
    <t2>A contract against public policy is void.</t2></pair>
  <pair id="R05-01-B" label="N"><t1>Article 5 A minor ...</t1>
    <t2>A minor may always rescind.</t2></pair>
</dataset>""",
        encoding="utf-8",
    )
    ds = load_task4(tmp_path / "task4")
    assert [(e.id, e.label) for e in ds.examples] == [("R05-01-A", True), ("R05-01-B", False)]
    assert ds.examples[0].premise.startswith("Article 90") and ds.validate() == []
    with pytest.raises(BenchmarkMissing):
        load_task4(tmp_path / "empty")


# ---- validation ------------------------------------------------------------------------


def test_validators_report_problems() -> None:
    ds = RetrievalDataset(
        "x",
        "all",
        [RetrievalQuery("q1", " ", frozenset()), RetrievalQuery("q2", "text", frozenset({"d9"}))],
        {"d1": Document("d1", "")},
        "licence",
    )
    problems = "\n".join(ds.validate())
    assert "d1: empty text" in problems and "q1: empty text" in problems
    assert "q1: no relevant" in problems and "q2: relevant not in pool: ['d9']" in problems
    assert RetrievalDataset("x", "all", [], {}, "l").validate() == ["no queries"]
    ent = EntailmentDataset(
        "y",
        "all",
        [EntailmentExample("e", "p", "h", True), EntailmentExample("e", "", "h", False)],
        "l",
    )
    assert "duplicate example IDs" in ent.validate() and len(ent.validate()) == 2


@pytest.mark.skipif(
    not any(find_task4().glob("riteval_*_en.xml")), reason="COLIEE Task 4 (English) not unpacked"
)
def test_real_coliee_task4_loads_and_validates() -> None:
    ds = load_task4(find_task4())
    assert len(ds.examples) == 1206 and sum(e.label for e in ds.examples) == 614
    assert ds.validate() == []


@pytest.mark.skipif(
    not (il_pcr_module.il_pcr_dir() / "test_queries-00000-of-00001.parquet").exists(),
    reason="IL-PCR not fetched",
)
def test_real_il_pcr_loads_and_validates() -> None:
    sizes = {}
    for split in ("train", "dev", "test"):
        ds = load_il_pcr(split)  # type: ignore[arg-type]
        assert ds.validate() == []
        sizes[split] = (len(ds.queries), len(ds.documents))
    # 827/118/237 queries at the pinned revision, less those with no relevant candidates
    assert sizes == {"train": (817, 4320), "dev": (118, 1023), "test": (234, 1727)}
