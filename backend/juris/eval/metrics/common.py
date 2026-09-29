"""What every answer metric shares: the result type and readers over a folded run.

A scored answer is a folded run (``CaseView``): its ``analysis`` is the ``CaseAnalysis``, and
the record around it (evidence, chunks, claims, arguments with their citations, verification
results) is what the analysis cites. Baselines produce the same view (PLAN 6.2-6.4).
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from juris.events.fold import CaseView
from juris.ingest.statutes import Act, load_acts, resolve_act
from juris.models import CaseAnalysis, Citation, StatuteMeta, VerificationResult
from juris.models.common import VerificationStatus

EVIDENCE_REF = re.compile(r"\bE-\d{3,}\b")

# Best first: an evidence item's support is the best status any of its citations got.
_STATUS_ORDER = (
    VerificationStatus.VERIFIED,
    VerificationStatus.WEAK,
    VerificationStatus.UNSUPPORTED,
    VerificationStatus.INVALID,
)
SUPPORTED = frozenset({VerificationStatus.VERIFIED, VerificationStatus.WEAK})


@dataclass
class MetricResult:
    """One metric on one item. ``values`` holds the metric's own value under its name plus
    any sub-values (``"<name>.<part>"``); ``None`` means not applicable to this item.
    ``details`` is JSON-serialisable and explains the numbers."""

    name: str
    values: dict[str, float | None]
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float | None:
        return self.values.get(self.name)


def ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def argument_citations(view: CaseView) -> list[Citation]:
    """Every citation in the record, in argument order (counterarguments reuse arguments)."""
    return [c for a in view.arguments.values() for c in a.citations]


def pair_results(view: CaseView) -> dict[tuple[str, str], VerificationResult]:
    """The latest verification of each (claim, evidence) pair: later gates (S7, S10) override
    earlier ones. A result stored on the citation itself counts when no event recorded one."""
    out: dict[tuple[str, str], VerificationResult] = {}
    for c in argument_citations(view):
        if c.verification is not None:
            out[(c.claim_id, c.evidence_id)] = c.verification
    for record in sorted(view.verifications, key=lambda r: r.seq):
        out[(record.claim_id, record.evidence_id)] = record.result
    return out


def evidence_support(view: CaseView) -> dict[str, VerificationStatus]:
    """Best verification status per evidence ID over all the claims citing it."""
    best: dict[str, VerificationStatus] = {}
    for (_, evidence_id), result in pair_results(view).items():
        current = best.get(evidence_id)
        if current is None or _STATUS_ORDER.index(result.status) < _STATUS_ORDER.index(current):
            best[evidence_id] = result.status
    return best


def chunk_text(view: CaseView, evidence_id: str) -> str | None:
    evidence = view.evidence.get(evidence_id)
    if evidence is None:
        return None
    chunk = view.chunks.get(evidence.chunk_id)
    return chunk.text if chunk is not None else None


def analysis_refs(analysis: CaseAnalysis) -> list[str]:
    """Evidence IDs the analysis cites (key evidence, and inline ``[E-###]`` in the summaries),
    in order of first appearance."""
    seen: dict[str, None] = {}
    for issue in analysis.issues:
        for position in issue.positions:
            for ref in position.key_evidence:
                seen.setdefault(ref.evidence_id, None)
            for ref_id in EVIDENCE_REF.findall(position.summary):
                seen.setdefault(ref_id, None)
    for ref_id in EVIDENCE_REF.findall(analysis.overall_summary):
        seen.setdefault(ref_id, None)
    return list(seen)


def analysis_documents(view: CaseView) -> set[str]:
    """Documents the analysis relies on: named authorities, both sides of key conflicts,
    sources, and the documents of the evidence it cites."""
    analysis = view.analysis
    if analysis is None:
        return set()
    docs: set[str] = {s.document_id for s in analysis.sources}
    for issue in analysis.issues:
        for position in issue.positions:
            docs.update(position.supporting_authorities)
        for conflict in issue.key_conflicts:
            docs.update((conflict.authority_a, conflict.authority_b))
    for ref_id in analysis_refs(analysis):
        evidence = view.evidence.get(ref_id)
        if evidence is not None:
            docs.add(evidence.document_id)
    return docs


@lru_cache
def _acts() -> dict[str, Act]:
    return load_acts()


def section_id(statute: StatuteMeta) -> str:
    """``contract_act:15``: the chunk carries the Act's name ("Indian Contract Act"), which
    resolves to its ID through the alias table; an unknown name is kept as it is."""
    act = resolve_act(statute.act, _acts()) or statute.act
    return f"{act}:{statute.section}"


def analysis_sections(view: CaseView) -> set[str]:
    """Statute sections (``contract_act:15``) whose text the analysis cites as evidence."""
    if view.analysis is None:
        return set()
    sections: set[str] = set()
    for ref_id in analysis_refs(view.analysis):
        evidence = view.evidence.get(ref_id)
        chunk = view.chunks.get(evidence.chunk_id) if evidence is not None else None
        if chunk is not None and chunk.statute is not None:
            sections.add(section_id(chunk.statute))
    return sections


# Tokens after which a full stop does not end a sentence in legal prose.
_ABBREVIATIONS = frozenset(
    {
        *("v", "vs", "s", "ss", "sec", "secs", "art", "arts", "no", "nos", "para", "paras"),
        *("cl", "co", "ltd", "pvt", "j", "jj", "cj", "hon", "ors", "anr", "mr", "mrs", "ms"),
        *("dr", "st", "i.e", "e.g", "cf", "viz", "etc", "ibid", "supra"),
    }
)
_SENTENCE_END = re.compile(r"(?<=[.!?])([\"')\]]*)\s+(?=[\"'(\[]?[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Sentences of a summary. Doesn't split after "v.", "s.", "Ltd.", initials and similar."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        words = text[start : match.start()].rstrip(".!?").split()
        last = words[-1].lstrip("([\"'").lower() if words else ""
        if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
            continue
        sentences.append(text[start : match.end(1)].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]
