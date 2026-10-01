"""Verifier evaluation metrics and samplers (PLAN 6.1): pure functions, no model, no data."""

import json
from pathlib import Path

import pytest

from juris.eval.datasets import EntailmentExample, load_task2, load_task4
from juris.eval.verifier_eval import (
    cohens_kappa,
    coliee_metrics,
    coliee_pairs,
    confusion_matrix,
    juris_metrics,
    juris_pairs,
    precision_recall_f1,
    sample_balanced,
    sample_case_entailment,
    status_of,
)

S, P, D, C = "supports", "partially_supports", "does_not_support", "contradicts"


def test_status_mapping_merges_the_two_negative_labels() -> None:
    assert [status_of(x) for x in (S, P, D, C)] == [
        "verified",
        "weak",
        "unsupported",
        "unsupported",
    ]


def test_precision_recall_f1_and_zero_cases() -> None:
    p, r, f = precision_recall_f1([True, True, False, False], [True, False, True, False])
    assert (p, r, f) == (0.5, 0.5, 0.5)
    assert precision_recall_f1([False, False], [False, False]) == (0.0, 0.0, 0.0)


def test_cohens_kappa() -> None:
    a = ["x", "x", "y", "y"]
    assert cohens_kappa(a, a) == 1.0
    assert cohens_kappa(a, ["x", "y", "x", "y"]) == 0.0
    assert cohens_kappa(a, ["y", "y", "x", "x"]) == -1.0
    assert cohens_kappa(["x", "x"], ["x", "x"]) == 1.0
    assert cohens_kappa([], []) == 0.0


def test_confusion_matrix_rows_are_gold() -> None:
    m = confusion_matrix([S, S, D], [S, P, C])
    assert m[0] == [1, 1, 0, 0]  # gold supports: one right, one partial
    assert m[2] == [0, 0, 0, 1]  # gold does_not_support predicted contradicts
    assert sum(map(sum, m)) == 3


def test_juris_metrics() -> None:
    gold = [S, S, P, D, C, D]
    pred = [S, P, P, C, D, S]
    cons = ["gold", "gold", "overstated", "negated", "negated", "wrong_passage"]
    m = juris_metrics(gold, pred, cons)
    # status: S=S ok, S vs P no, P=P ok, D vs C ok (both unsupported), C vs D ok, D vs S no
    assert m["n"] == 6
    assert m["status_agreement"] == pytest.approx(4 / 6)
    assert m["label_accuracy"] == pytest.approx(2 / 6)
    assert m["binary_accuracy"] == pytest.approx(4 / 6)  # S/S, P/P, D/C, C/D agree on "not S"
    assert m["supports_precision"] == 0.5 and m["supports_recall"] == 0.5
    lo, hi = m["status_agreement_ci"]
    assert 0 <= lo <= m["status_agreement"] <= hi <= 1
    assert m["by_construction"]["negated"] == {"n": 2, "status_agreement": 1.0}
    assert m["by_construction"]["gold"]["status_agreement"] == 0.5
    assert m["confusion"][0][1] == 1
    assert -1 <= m["status_kappa"] <= 1


def test_coliee_metrics_strict_and_lenient() -> None:
    gold = [True, True, False, False]
    pred = [S, P, P, D]
    m = coliee_metrics(gold, pred)
    assert m["strict"]["accuracy"] == 0.75
    assert m["strict"]["recall"] == 0.5 and m["strict"]["precision"] == 1.0
    assert m["lenient"]["accuracy"] == 0.75
    assert m["lenient"]["precision"] == pytest.approx(2 / 3) and m["lenient"]["recall"] == 1.0
    assert m["label_distribution"] == {S: 1, P: 2, D: 1, C: 0}
    assert m["positives"] == 2


def _examples(cases: int, pos_per: int, neg_per: int) -> list[EntailmentExample]:
    out: list[EntailmentExample] = []
    for c in range(cases):
        for k in range(pos_per + neg_per):
            out.append(
                EntailmentExample(
                    id=f"{c:03d}/{k:03d}.txt",
                    premise=f"para {c}.{k}",
                    hypothesis=f"fragment {c}",
                    label=k < pos_per,
                    meta={"case": f"{c:03d}"},
                )
            )
    return out


def test_sample_balanced_is_seeded_balanced_and_order_free() -> None:
    ex = _examples(10, 1, 4)
    a = sample_balanced(ex, 12, seed=1)
    assert len(a) == 12 and sum(e.label for e in a) == 6
    assert a == sample_balanced(list(reversed(ex)), 12, seed=1)
    assert a != sample_balanced(ex, 12, seed=2)
    assert [e.id for e in a] == sorted(e.id for e in a)
    assert sum(e.label for e in sample_balanced(ex, 5)) == 3  # odd: the extra one is positive
    assert len(sample_balanced(ex, 1000)) == len(ex)  # fewer when the data runs out


def test_sample_case_entailment_takes_negatives_from_the_positives_cases() -> None:
    ex = _examples(20, 1, 3)
    s = sample_case_entailment(ex, 10, seed=3)
    pos = [e for e in s if e.label]
    neg = [e for e in s if not e.label]
    assert len(pos) == 5 and len(neg) == 5
    assert {e.meta["case"] for e in neg} == {e.meta["case"] for e in pos}
    assert s == sample_case_entailment(ex, 10, seed=3)


def test_sample_case_entailment_tops_up_from_other_cases() -> None:
    ex = _examples(6, 1, 1)
    ex += [
        EntailmentExample("zzz/001.txt", "p", "f", False, {"case": "zzz"}),
        EntailmentExample("zzz/002.txt", "p", "f", False, {"case": "zzz"}),
    ]
    # negatives of the chosen positives' cases run short of 6 only if cases repeat; ask for
    # more than the cases can give so the top-up path runs
    s = sample_case_entailment([e for e in ex if e.meta["case"] != "000"] + ex[:1], 8, seed=0)
    assert sum(not e.label for e in s) == min(4, sum(not e.label for e in ex[2:]))


def test_pairs_from_juris_rows_and_coliee_examples(tmp_path: Path) -> None:
    rows = [
        {
            "id": "VP-001",
            "claim": "c",
            "chunk_id": "D#c1",
            "quote": "q",
            "passage": "pq",
            "label": S,
            "construction": "gold",
        }
    ]
    [j] = juris_pairs(rows)
    assert (j.pair.claim, j.pair.quote, j.pair.passage, j.gold) == ("c", "q", "pq", S)
    assert j.pair.evidence_id == "D#c1" and j.construction == "gold"

    (tmp_path / "t4").mkdir()
    (tmp_path / "t4" / "riteval_R05_en.xml").write_text(
        '<dataset><pair id="A" label="Y"><t1>Article 1 text.</t1><t2>Question?</t2></pair>'
        '<pair id="B" label="N"><t1>Article 2 text.</t1><t2>Other?</t2></pair></dataset>',
        encoding="utf-8",
    )
    pairs = coliee_pairs(sample_balanced(load_task4(tmp_path / "t4").examples, 2))
    assert [(p.pair.claim, p.pair.quote == p.pair.passage, p.gold) for p in pairs] == [
        ("Question?", True, True),
        ("Other?", True, False),
    ]

    case = tmp_path / "cases" / "001"
    (case / "paragraphs").mkdir(parents=True)
    (case / "entailed_fragment.txt").write_text("Fragment.", encoding="utf-8")
    for n in ("001", "002"):
        (case / "paragraphs" / f"{n}.txt").write_text(f"Paragraph {n}.", encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"001": ["002.txt"]}), encoding="utf-8")
    ex = load_task2(tmp_path / "cases", labels).examples
    got = coliee_pairs(sample_case_entailment(ex, 2))
    assert [(p.pair.claim_id, p.gold) for p in got] == [
        ("001/001.txt", False),
        ("001/002.txt", True),
    ]
