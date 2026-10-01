"""Metrics and samplers for scoring the verifier's entailment check (PLAN 6.1).

Pure functions, no model or database: ``scripts/eval_verifier.py`` runs the judge and hands
the gold and predicted labels here. The 4-label Juris set is scored on what the verifier
reports to the user, the claim status (``supports`` is verified, ``partially_supports`` is
weak, the other two are unsupported), so a swap between ``does_not_support`` and
``contradicts`` is not an error. COLIEE has binary gold, scored strictly (only ``supports``
counts as entailed) and leniently (``partially_supports`` too).
"""

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from juris.eval.datasets.common import EntailmentExample
from juris.eval.stats import bootstrap_ci
from juris.models.common import EntailmentLabel
from juris.verify.entailment import ClaimEvidencePair
from juris.verify.verifier import LABEL_STATUS

LABELS = tuple(label.value for label in EntailmentLabel)
STATUSES = ("verified", "weak", "unsupported")


def status_of(label: str) -> str:
    return LABEL_STATUS[EntailmentLabel(label)].value


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def precision_recall_f1(
    gold: Sequence[bool], predicted: Sequence[bool]
) -> tuple[float, float, float]:
    """Precision, recall and F1 of the positive class; 0 where undefined."""
    tp = sum(g and p for g, p in zip(gold, predicted, strict=True))
    fp = sum(p and not g for g, p in zip(gold, predicted, strict=True))
    fn = sum(g and not p for g, p in zip(gold, predicted, strict=True))
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return precision, recall, _ratio(2 * precision * recall, precision + recall)


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Chance-corrected agreement of two labellings; 1.0 when both are one constant label."""
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def confusion_matrix(gold: Sequence[str], predicted: Sequence[str]) -> list[list[int]]:
    """Counts over ``LABELS``: gold in rows, predicted in columns."""
    index = {label: i for i, label in enumerate(LABELS)}
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for g, p in zip(gold, predicted, strict=True):
        matrix[index[g]][index[p]] += 1
    return matrix


def juris_metrics(
    gold: Sequence[str], predicted: Sequence[str], constructions: Sequence[str]
) -> dict[str, Any]:
    """Scores of predicted labels against the hand labels of the Juris pairs."""
    n = len(gold)
    gold_status = [status_of(g) for g in gold]
    pred_status = [status_of(p) for p in predicted]
    agree = [float(g == p) for g, p in zip(gold_status, pred_status, strict=True)]
    gold_pos = [g == EntailmentLabel.SUPPORTS for g in gold]
    pred_pos = [p == EntailmentLabel.SUPPORTS for p in predicted]
    precision, recall, f1 = precision_recall_f1(gold_pos, pred_pos)
    by_construction: dict[str, list[float]] = {}
    for c, a in zip(constructions, agree, strict=True):
        by_construction.setdefault(c, []).append(a)
    lo, hi = bootstrap_ci(agree) if agree else (0.0, 0.0)
    return {
        "n": n,
        "status_agreement": _ratio(sum(agree), n),
        "status_agreement_ci": [lo, hi],
        "label_accuracy": _ratio(sum(g == p for g, p in zip(gold, predicted, strict=True)), n),
        "binary_accuracy": _ratio(sum(g == p for g, p in zip(gold_pos, pred_pos, strict=True)), n),
        "supports_precision": precision,
        "supports_recall": recall,
        "supports_f1": f1,
        "status_kappa": cohens_kappa(gold_status, pred_status),
        "confusion_labels": list(LABELS),
        "confusion": confusion_matrix(gold, predicted),
        "by_construction": {
            c: {"n": len(v), "status_agreement": sum(v) / len(v)}
            for c, v in sorted(by_construction.items())
        },
    }


def _binary(gold: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float]:
    precision, recall, f1 = precision_recall_f1(gold, predicted)
    accuracy = _ratio(sum(g == p for g, p in zip(gold, predicted, strict=True)), len(gold))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def coliee_metrics(gold: Sequence[bool], predicted: Sequence[str]) -> dict[str, Any]:
    """Binary scores: strict (``supports`` is entailed) and lenient (``partially_supports`` too)."""
    strict = [p == EntailmentLabel.SUPPORTS for p in predicted]
    lenient = [
        p in (EntailmentLabel.SUPPORTS, EntailmentLabel.PARTIALLY_SUPPORTS) for p in predicted
    ]
    counts = Counter(predicted)
    return {
        "n": len(gold),
        "positives": sum(gold),
        "strict": _binary(gold, strict),
        "lenient": _binary(gold, lenient),
        "label_distribution": {label: counts.get(label, 0) for label in LABELS},
    }


@dataclass(frozen=True)
class EvalPair:
    """A pair to judge with its gold: a label (Juris) or entailed yes/no (COLIEE)."""

    pair: ClaimEvidencePair
    gold: str | bool
    construction: str = ""


def juris_pairs(rows: Sequence[Mapping[str, Any]]) -> list[EvalPair]:
    """The rows of ``eval/verifier/juris_pairs.yaml`` as given: claim, quote, passage."""
    return [
        EvalPair(
            ClaimEvidencePair(
                str(r["id"]),
                str(r["claim"]),
                str(r["chunk_id"]),
                str(r["quote"]),
                str(r["passage"]),
            ),
            str(r["label"]),
            str(r["construction"]),
        )
        for r in rows
    ]


def coliee_pairs(examples: Sequence[EntailmentExample]) -> list[EvalPair]:
    """Claim = hypothesis; passage and quote = the whole premise (there is no sub-quote)."""
    return [
        EvalPair(ClaimEvidencePair(e.id, e.hypothesis, e.id, e.premise, e.premise), e.label)
        for e in examples
    ]


def sample_balanced(
    examples: Sequence[EntailmentExample], n: int, seed: int = 0
) -> list[EntailmentExample]:
    """About ``n`` examples, half positive (the extra one for odd ``n``), by seeded sample.

    Fewer if a class has too few; the result is sorted by ID so it does not depend on order.
    """
    rng = random.Random(seed)
    pos = sorted((e for e in examples if e.label), key=lambda e: e.id)
    neg = sorted((e for e in examples if not e.label), key=lambda e: e.id)
    n_pos = n - n // 2
    picked = rng.sample(pos, min(n_pos, len(pos))) + rng.sample(neg, min(n // 2, len(neg)))
    return sorted(picked, key=lambda e: e.id)


def sample_case_entailment(
    examples: Sequence[EntailmentExample], n: int, seed: int = 0
) -> list[EntailmentExample]:
    """Half positives, and negatives from the same cases as those positives where possible.

    A negative paragraph of the case whose fragment it is judged against is the hard kind:
    same facts and vocabulary, but it does not entail. Cases with no spare negative are
    made up from negatives of other cases.
    """
    rng = random.Random(seed)
    pos = sorted((e for e in examples if e.label), key=lambda e: e.id)
    neg = sorted((e for e in examples if not e.label), key=lambda e: e.id)
    chosen_pos = rng.sample(pos, min(n - n // 2, len(pos)))
    wanted = min(n // 2, len(neg))
    by_case: dict[str, list[EntailmentExample]] = {}
    for e in neg:
        by_case.setdefault(str(e.meta.get("case", "")), []).append(e)
    chosen_neg: list[EntailmentExample] = []
    for e in chosen_pos:
        pool = by_case.get(str(e.meta.get("case", "")), [])
        if pool and len(chosen_neg) < wanted:
            chosen_neg.append(pool.pop(rng.randrange(len(pool))))
    if len(chosen_neg) < wanted:
        rest = sorted((e for pool in by_case.values() for e in pool), key=lambda e: e.id)
        chosen_neg += rng.sample(rest, wanted - len(chosen_neg))
    return sorted(chosen_pos + chosen_neg, key=lambda e: e.id)
