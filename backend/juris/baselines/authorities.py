"""Resolve free-text authorities against the corpus (PLAN 6.2: B0's hallucination check).

A model answering without retrieval names its authorities in free text: a case name and a
reporter citation, or an Act and section. Each is looked up with the structured lookups
(4.5) and given a status:

- ``corpus``: a judgment or section in the corpus (by citation, else by a close title match);
- ``outside_corpus``: a real judgment known from the citation graph but not in the slice;
- ``mismatch``: the citation names a corpus judgment whose parties don't match the name given
  (a wrong citation, one kind of fabrication);
- ``unverified``: nothing in the corpus or the citation graph matches; possibly fabricated,
  possibly just outside the slice, so the rate is an upper bound on hallucination;
- ``unknown_act``: a statute outside the three Acts in the corpus, which cannot be checked.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from juris.retrieval.lookup import CaseMatch, Section, TitleMatch, parse_provision

Status = Literal["corpus", "outside_corpus", "mismatch", "unverified", "unknown_act"]

# A title match on its own must be close; citations are checked before titles.
TITLE_MIN_SCORE = 0.75

# Words that say nothing about which case is meant.
_COMMON = frozenset(
    {
        *("versus", "the", "and", "others", "ors", "anr", "another", "ltd", "limited", "pvt"),
        *("private", "company", "co", "corporation", "state", "union", "india", "of", "for"),
        *("bros", "brothers", "sons", "mrs", "smt", "shri", "sri", "dead", "lrs", "through"),
    }
)


class Lookups(Protocol):
    """The parts of ``juris.retrieval.lookup.Lookup`` the resolver uses."""

    def get_case_by_citation(self, citation: str) -> CaseMatch | None: ...
    def get_case_by_title(
        self, title: str, limit: int = 5, min_score: float = 0.4
    ) -> list[TitleMatch]: ...
    def get_section(self, act: str, section: str) -> Section | None: ...


@dataclass(frozen=True)
class Resolution:
    kind: Literal["judgment", "statute"]
    name: str
    citation: str | None
    status: Status
    doc_id: str | None = None  # the corpus document (judgment, or the Act's document)
    section_id: str | None = None
    matched_title: str | None = None
    how: str = ""  # citation | title | provision

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _COMMON}


def same_parties(name: str, title: str) -> bool:
    """Whether a case name and a corpus title share a distinctive party word (lenient on
    purpose: OCR-damaged titles and abbreviated names should not count as fabrication)."""
    return bool(_words(name) & _words(title))


class AuthorityResolver:
    def __init__(self, lookups: Lookups) -> None:
        self.lookups = lookups

    def judgment(self, name: str, citation: str | None) -> Resolution:
        if citation:
            match = self.lookups.get_case_by_citation(citation)
            if match is not None and match.doc_id is not None:
                status: Status = (
                    "corpus"
                    if match.title is None or same_parties(name, match.title)
                    else "mismatch"
                )
                return Resolution(
                    "judgment", name, citation, status, match.doc_id, None, match.title, "citation"
                )
            if match is not None:
                return Resolution(
                    "judgment", name, citation, "outside_corpus", None, None, None, "citation"
                )
        titles = self.lookups.get_case_by_title(name, limit=1, min_score=TITLE_MIN_SCORE)
        if titles:
            top = titles[0]
            return Resolution(
                "judgment", name, citation, "corpus", top.doc_id, None, top.title, "title"
            )
        return Resolution("judgment", name, citation, "unverified")

    def statute(self, name: str, citation: str | None) -> Resolution:
        reference = f"{name} {citation or ''}".strip()
        provision = parse_provision(reference)
        if provision is None:
            return Resolution("statute", name, citation, "unknown_act")
        section = self.lookups.get_section(provision.act_id, provision.section)
        if section is None:
            return Resolution("statute", name, citation, "unverified", how="provision")
        return Resolution(
            "statute",
            name,
            citation,
            "corpus",
            f"ACT-{section.act_id}",
            section.section_id,
            section.title,
            "provision",
        )


def summarise(resolutions: list[Resolution], prefix: str) -> dict[str, float | None]:
    """Counts by kind and status, and the unverified-judgment rate (mismatch + unverified over
    judgments cited): the hallucinated-authority rate, as an upper bound."""
    judgments = [r for r in resolutions if r.kind == "judgment"]
    statutes = [r for r in resolutions if r.kind == "statute"]
    bad = sum(1 for r in judgments if r.status in ("unverified", "mismatch"))
    return {
        f"{prefix}.authorities_cited": float(len(resolutions)),
        f"{prefix}.judgments_cited": float(len(judgments)),
        f"{prefix}.judgments_in_corpus": float(sum(r.status == "corpus" for r in judgments)),
        f"{prefix}.judgments_outside_corpus": float(
            sum(r.status == "outside_corpus" for r in judgments)
        ),
        f"{prefix}.judgments_mismatched": float(sum(r.status == "mismatch" for r in judgments)),
        f"{prefix}.judgments_unverified": float(sum(r.status == "unverified" for r in judgments)),
        f"{prefix}.unverified_judgment_rate": bad / len(judgments) if judgments else None,
        f"{prefix}.statutes_cited": float(len(statutes)),
        f"{prefix}.statutes_unverified": float(sum(r.status == "unverified" for r in statutes)),
    }
