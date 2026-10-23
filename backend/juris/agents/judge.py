"""The Judge (S9; IDEA_final §6.1, §10; PLAN 7.11): the final ``CaseAnalysis``, a leaning per
issue and never a verdict.

One call over the whole frozen record: per issue, the positions, every claim with its status,
arguments, quotes and their verification and fit, the precedent cards, the objections with
replies, and the jury's aggregate; then the **citable evidence** (best verification verified or
weak, each with a quote that verified) and the unresolved questions, numbered U1, U2, ...

The model writes, per issue: a short summary per position citing evidence inline (``[E-012]``),
the key evidence IDs, key conflicts between two judgments, the leaning and confidence with a
reason, and an answer to each of the issue's unresolved questions; then an overall summary
(at most 250 words, inline citations) and its own limitations.

Code builds the ``CaseAnalysis`` (``build_analysis``): inline references become one ``[E-nnn]``
per evidence (a claim or argument cited inline is replaced by the evidence it cites, for S10 to
check; ``normalise_refs``); positions get pinpoints from the chunks
and their supporting authorities from the judgments they cite; every judgment relied on gets a
limitation saying its treatment is not verified (D-013); the assumptions are listed as
limitations, and so are the evidence graph's checks (PLAN 8.2: positions with no verified
support, contested issues, standing claims with no or only weak support), which the Judge also
reads per issue as "Record checks"; the sources list every document cited, with court, date,
citation and pinpoints.

Checks (``validate``): every S0 issue once with each of its positions once (IDEA_final §8
invariant 5); key evidence and conflicts only from citable evidence (invariant 3), conflicts
between two different judgments; every unresolved question answered; the overall summary
within 250 words. Inline references are checked at S10.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from juris.agents.base import Agent, Problem
from juris.agents.cross_examiner import describe_claims, describe_precedents
from juris.agents.juror import describe_objections
from juris.config import Role
from juris.events import CaseView
from juris.graph.checks import hints
from juris.models import (
    CaseAnalysis,
    CreatedBy,
    EvidenceRef,
    IssueAnalysis,
    KeyConflict,
    PositionAnalysis,
    SourceRef,
)
from juris.models.analysis import SUMMARY_MAX_WORDS
from juris.models.common import Confidence, DocumentKind, Leaning, Stage, VerificationStatus
from juris.pipeline.case_record import CITABLE, CaseRecord
from juris.pipeline.citing import clean_evidence_id, clean_record_id, pinpoint

AGENT = "judge"
INLINE = re.compile(r"\[\s*(E\s*-?\s*\d+)\s*\]", re.IGNORECASE)
CORPUS_LIMITATION = (
    "The research covers the system's corpus only (Supreme Court judgments and central Acts on "
    "contract law); later or unreported decisions may change the analysis."
)


class JudgePosition(BaseModel):
    position_id: str = Field(description='e.g. "I-1.A"')
    summary: str = Field(description="2-4 sentences; cite evidence inline, e.g. [E-012]")
    key_evidence: list[str] = Field(description="The citable evidence IDs it rests on")


class JudgeConflict(BaseModel):
    evidence_a: str = Field(description="An evidence ID from one judgment")
    evidence_b: str = Field(description="An evidence ID from another judgment")
    distinction: str = Field(description="What separates the two")
    weightier: Literal["a", "b", "neither"]
    reason: str = Field(description="Why, by the rules of weight")


class JudgeAnswer(BaseModel):
    question: str = Field(description='The unresolved question\'s label, e.g. "U2"')
    answer: str = Field(description="One or two sentences")


class JudgeIssue(BaseModel):
    issue_id: str = Field(description='e.g. "I-1"')
    positions: list[JudgePosition]
    key_conflicts: list[JudgeConflict] = Field(description="May be empty")
    leaning: Leaning
    confidence: Confidence
    confidence_reason: str = Field(description="One sentence")
    unresolved: list[JudgeAnswer] = Field(description="An answer to each of the issue's questions")


class JudgeOutput(BaseModel):
    issues: list[JudgeIssue]
    overall_summary: str = Field(description="At most 250 words; cite evidence inline")
    limitations: list[str] = Field(description="What the analysis could not settle; may be empty")


def citable(view: CaseView) -> dict[str, str]:
    """Evidence the Judge may cite (best verification verified or weak), each with the quote of
    its best verified citation."""
    quotes = {
        (c.claim_id, c.evidence_id): c.quote for a in view.arguments.values() for c in a.citations
    }
    best: dict[str, tuple[int, str]] = {}  # evidence -> (0 verified / 1 weak, quote)
    for v in view.verifications:
        if v.result.status not in CITABLE:
            continue
        score = 0 if v.result.status is VerificationStatus.VERIFIED else 1
        if v.evidence_id not in best or score < best[v.evidence_id][0]:
            best[v.evidence_id] = (score, quotes.get((v.claim_id, v.evidence_id), ""))
    return {e: q for e, (_, q) in sorted(best.items())}


def unresolved(view: CaseView) -> dict[str, tuple[str, str]]:
    """The unresolved questions, labelled U1, U2, ...: (issue, question)."""
    return {
        f"U{n}": (q.issue_id or "", q.question) for n, q in enumerate(view.unresolved_questions, 1)
    }


def record_checks(view: CaseView) -> dict[str, list[str]]:
    """The evidence graph's integrity checks as sentences, by issue ("" for the whole case)."""
    return hints(view.graph)


def describe_jury(view: CaseView, issue_id: str) -> str:
    aggregate = view.jury_aggregate
    if aggregate is None:
        return "(no jury)"
    lines = []
    for position in view.issues[issue_id].positions:
        stats = aggregate.per_position.get(position.id, {})
        parts = [f"{name} {s.mean:.1f} (spread {s.spread:.1f})" for name, s in stats.items()]
        lines.append(f"{position.id}: " + ", ".join(parts))
    if issue_id in aggregate.contested_issue_ids:
        lines.append("The jurors disagree on this issue (contested).")
    return "\n".join(lines)


def describe_citable(view: CaseView, issue_id: str, quotes: dict[str, str]) -> str:
    lines = []
    for evidence_id, quote in quotes.items():
        evidence = view.evidence[evidence_id]
        if issue_id not in evidence.issue_ids:
            continue
        title = view.documents[evidence.document_id].title
        where = pinpoint(view.chunks[evidence.chunk_id], None)
        lines.append(f'[{evidence_id}] {title}, {where}: "{quote}"')
    return "\n".join(lines) or "(none: no evidence on this issue was verified)"


class Judge(Agent[JudgeOutput]):
    role = Role.JUDGE
    prompt = "judge"
    duty = (
        "The Judge writes the final analysis of every issue from the record: a leaning and a "
        "confidence, never a verdict, citing only verified or weak evidence."
    )
    output_model = JudgeOutput
    max_tokens = 4000

    def __init__(self) -> None:
        super().__init__(AGENT)

    def input_view(self, view: CaseView) -> dict[str, str]:
        quotes = citable(view)
        questions = unresolved(view)
        checks = record_checks(view)
        blocks = []
        for issue in view.issues.values():
            found = [*checks.get(issue.id, []), *checks.get("", [])]
            checked = "\n".join(f"- {c}" for c in found) or "(none)"
            positions = "\n".join(f"{p.id}: {p.statement}" for p in issue.positions)
            asked = [f"{label}: {q}" for label, (i, q) in questions.items() if i == issue.id]
            blocks.append(
                f'<issue id="{issue.id}">\n{issue.question}\n\nPositions:\n{positions}\n\n'
                f"Claims and arguments:\n{describe_claims(view, issue.id)}\n\n"
                f"Precedents cited:\n{describe_precedents(view, issue.id)}\n\n"
                f"Objections:\n{describe_objections(view, issue.id)}\n\n"
                f"Jury (mean score 1-5, higher is better for the position):\n"
                f"{describe_jury(view, issue.id)}\n\n"
                f"Citable evidence (cite only these):\n"
                f"{describe_citable(view, issue.id, quotes)}\n\n"
                f"Record checks:\n{checked}\n\n"
                f"Unresolved questions to answer:\n" + ("\n".join(asked) or "(none)")
                + "\n</issue>"
            )  # fmt: skip
        facts = [*view.facts, *(f"(assumed) {a}" for a in view.assumptions)]
        return {
            "question": view.question,
            "facts": "\n".join(f"- {f}" for f in facts) or "(none stated)",
            "record": "\n\n".join(blocks),
            "max_words": str(SUMMARY_MAX_WORDS),
        }

    def validate(self, output: JudgeOutput, record: CaseRecord) -> list[Problem]:
        view = record.view
        quotes = citable(view)
        questions = unresolved(view)
        problems: list[Problem] = []
        given = [i.issue_id.strip() for i in output.issues]
        missing = [i for i in view.issues if i not in given]
        if missing:
            problems.append(Problem(f"the analysis does not cover {', '.join(missing)}", soft=True))
        seen: set[str] = set()
        for n, issue in enumerate(output.issues):
            at = ("issues", n)
            issue_id = issue.issue_id.strip()
            if issue_id not in view.issues:
                problems.append(Problem(f"{issue.issue_id!r} is not an issue of this case", at))
                continue
            if issue_id in seen:
                problems.append(Problem(f"{issue_id} is analysed twice", at))
                continue
            seen.add(issue_id)
            expected = [p.id for p in view.issues[issue_id].positions]
            covered: set[str] = set()
            for k, position in enumerate(issue.positions):
                here = (*at, "positions", k)
                position_id = position.position_id.strip()
                if position_id not in expected:
                    problems.append(
                        Problem(f"{position.position_id!r} is not a position of {issue_id}", here)
                    )
                    continue
                if position_id in covered:
                    problems.append(Problem(f"{position_id} is analysed twice", here))
                    continue
                covered.add(position_id)
                if not position.summary.strip():
                    problems.append(Problem("the summary is empty", here, soft=True))
                for j, raw in enumerate(position.key_evidence):
                    if not resolve(view, raw, quotes):
                        where = (*here, "key_evidence", j)
                        problems.append(Problem(f"{raw!r} is not in the citable evidence", where))
            left = [p for p in expected if p not in covered]
            if left:
                problems.append(
                    Problem(f"{issue_id} does not analyse {', '.join(left)}", at, soft=True)
                )
            if not issue.confidence_reason.strip():
                problems.append(Problem("the confidence has no reason", at, soft=True))
            for k, conflict in enumerate(issue.key_conflicts):
                problem = _conflict_problem(view, quotes, conflict)
                if problem:
                    problems.append(Problem(problem, (*at, "key_conflicts", k)))
        answered = answers(output)
        for label in questions:
            if label not in answered:
                problems.append(Problem(f"unresolved question {label} has no answer", soft=True))
        words = len(output.overall_summary.split())
        if not words or words > SUMMARY_MAX_WORDS:
            message = f"the overall summary has {words} words (1 to {SUMMARY_MAX_WORDS})"
            problems.append(Problem(message, soft=True))
        return problems

    def apply(self, output: JudgeOutput, record: CaseRecord, *, stage: Stage) -> None:
        record.set_analysis(build_analysis(output, record.view, stage=stage), stage=stage,
                            agent=self.name)  # fmt: skip


def _conflict_problem(view: CaseView, quotes: dict[str, str], conflict: JudgeConflict) -> str:
    a, b = clean_evidence_id(conflict.evidence_a), clean_evidence_id(conflict.evidence_b)
    for raw, evidence_id in ((conflict.evidence_a, a), (conflict.evidence_b, b)):
        if evidence_id not in quotes:
            return f"{raw!r} is not in the citable evidence"
        document = view.documents[view.evidence[evidence_id].document_id]
        if document.kind is not DocumentKind.JUDGMENT:
            return f"{raw!r} is not from a judgment"
    if view.evidence[a].document_id == view.evidence[b].document_id:
        return "a conflict needs two different judgments"
    if not conflict.distinction.strip() or not conflict.reason.strip():
        return "the conflict needs a distinction and a reason"
    return ""


def inline_refs(text: str) -> list[str]:
    """The evidence IDs cited inline, in order, once each."""
    return list(dict.fromkeys(clean_evidence_id(m) for m in INLINE.findall(text)))


def resolve(view: CaseView, raw: str, quotes: dict[str, str]) -> list[str]:
    """The citable evidence an ID stands for: an evidence ID itself, or the citable evidence
    that a claim's or an argument's citations name (a small model often gives those)."""
    text = raw.strip().strip("[]()").strip()
    prefix = text[:1].upper()
    if prefix == "E":
        evidence_id = clean_evidence_id(text)
        return [evidence_id] if evidence_id in quotes else []
    record_id = clean_record_id(text, prefix) if prefix in {"C", "A"} else text
    arguments = [
        a for a in view.arguments.values() if record_id == a.id or record_id in a.claim_ids
    ]
    return list(
        dict.fromkeys(
            c.evidence_id for a in arguments for c in a.citations if c.evidence_id in quotes
        )
    )


_GROUP = re.compile(r"\[([^\[\]]+)\]")
_TOKEN = re.compile(r"^\s*([ECA])\s*-?\s*(\d+)\s*$", re.IGNORECASE)


def normalise_refs(text: str, view: CaseView) -> str:
    """Inline references as one ``[E-nnn]`` per evidence: ``[E-1, E-3]`` is split, and a claim
    or argument (``[C-004]``, which a small model often cites) becomes the evidence its
    citations name, citable or not, for S10 to check. A bracket that holds anything else is
    left alone."""

    def expand(match: re.Match[str]) -> str:
        ids: list[str] = []
        for part in match.group(1).split(","):
            token = _TOKEN.match(part)
            if token is None:
                return match.group(0)
            prefix, number = token.group(1).upper(), int(token.group(2))
            record_id = f"{prefix}-{number:03d}"
            if prefix == "E":
                ids.append(record_id)
            else:
                ids += [
                    c.evidence_id
                    for a in view.arguments.values()
                    if record_id == a.id or record_id in a.claim_ids
                    for c in a.citations
                ]
        return " ".join(f"[{e}]" for e in dict.fromkeys(ids))

    return re.sub(r"\s+([.,;])", r"\1", _GROUP.sub(expand, text)).strip()


def answers(output: JudgeOutput) -> dict[str, str]:
    """The answers by question label, from any issue (a small model files them loosely)."""
    out: dict[str, str] = {}
    for issue in output.issues:
        for a in issue.unresolved:
            label = a.question.strip().upper()
            if a.answer.strip() and label not in out:
                out[label] = a.answer.strip()
    return out


NOT_ANALYSED = "The Judge gave no analysis of this position."


def build_analysis(output: JudgeOutput, view: CaseView, *, stage: Stage) -> CaseAnalysis:
    """The ``CaseAnalysis`` from a checked output (see the module docstring). A gap the Judge
    left (a soft problem) is filled explicitly and listed in ``limitations``: a position or
    issue it did not analyse, a question it did not answer, a summary missing or too long."""
    by_issue: dict[str, JudgeIssue] = {}
    for i in output.issues:
        by_issue.setdefault(i.issue_id.strip(), i)
    questions = unresolved(view)
    answered = answers(output)
    quotes = citable(view)
    gaps: list[str] = []
    cited: dict[str, list[str]] = {}  # document -> pinpoints used

    def ref(evidence_id: str) -> EvidenceRef:
        evidence = view.evidence[evidence_id]
        where = pinpoint(view.chunks[evidence.chunk_id], None)
        cited.setdefault(evidence.document_id, [])
        if where not in cited[evidence.document_id]:
            cited[evidence.document_id].append(where)
        return EvidenceRef(evidence_id=evidence_id, pinpoint=where)

    def judgments(evidence_ids: list[str]) -> list[str]:
        documents = [view.evidence[e].document_id for e in evidence_ids]
        return list(
            dict.fromkeys(d for d in documents if view.documents[d].kind is DocumentKind.JUDGMENT)
        )

    issues = []
    for issue in view.issues.values():
        out = by_issue.get(issue.id)
        if out is None:
            gaps.append(f"The Judge gave no analysis of {issue.id}; it is marked insufficient.")
        given = {p.position_id.strip(): p for p in out.positions} if out else {}
        positions = []
        for position in issue.positions:
            p = given.get(position.id)
            if p is None or not p.summary.strip():
                if out is not None:
                    gaps.append(f"The Judge gave no analysis of {position.id}.")
                positions.append(PositionAnalysis(position_id=position.id, summary=NOT_ANALYSED))
                continue
            keys = list(
                dict.fromkeys(e for raw in p.key_evidence for e in resolve(view, raw, quotes))
            )
            text = normalise_refs(p.summary, view)
            inline = [e for e in inline_refs(text) if e in quotes]
            for e in inline:
                ref(e)
            positions.append(
                PositionAnalysis(
                    position_id=position.id,
                    summary=text,
                    key_evidence=[ref(e) for e in keys],
                    supporting_authorities=judgments(keys + inline),
                )
            )
        conflicts = []
        for c in out.key_conflicts if out else []:
            a = view.evidence[clean_evidence_id(c.evidence_a)].document_id
            b = view.evidence[clean_evidence_id(c.evidence_b)].document_id
            conflicts.append(
                KeyConflict(
                    authority_a=a,
                    authority_b=b,
                    distinction=c.distinction.strip(),
                    weightier={"a": a, "b": b, "neither": None}[c.weightier],
                    reason=c.reason.strip(),
                )
            )
        asked = [(label, q) for label, (owner, q) in questions.items() if owner == issue.id]
        gaps += [
            f"The Judge did not answer unresolved question {label}."
            for label, _ in asked
            if label not in answered
        ]
        reason = out.confidence_reason.strip() if out else ""
        issues.append(
            IssueAnalysis(
                issue_id=issue.id,
                issue_text=issue.question,
                positions=positions,
                key_conflicts=conflicts,
                leaning=out.leaning if out else Leaning.INSUFFICIENT_EVIDENCE,
                confidence=out.confidence if out else Confidence.LOW,
                confidence_reason=reason or "No reason was given.",
                unresolved_questions=[
                    f"{q} Answer: {answered.get(label, 'not addressed by the Judge.')}"
                    for label, q in asked
                ],
            )
        )
    summary = normalise_refs(output.overall_summary, view)
    words = summary.split()
    if not words:
        summary = "The Judge gave no overall summary."
        gaps.append(summary)
    elif len(words) > SUMMARY_MAX_WORDS:
        summary = " ".join(words[: SUMMARY_MAX_WORDS - 1]) + " [...]"
        gaps.append(f"The overall summary was cut to {SUMMARY_MAX_WORDS} words.")
    for e in inline_refs(summary):
        if e in quotes:
            ref(e)
    relied = list(
        dict.fromkeys(a for i in issues for p in i.positions for a in p.supporting_authorities)
    )
    limitations = [x.strip() for x in output.limitations if x.strip()] + gaps
    limitations += [f"Assumed: {a}" for a in view.assumptions]
    checked = dict.fromkeys(c for found in record_checks(view).values() for c in found)
    limitations += [f"Record check: {c}" for c in checked]
    limitations += [
        f"Treatment of {view.documents[d].title} not verified: whether later judgments followed, "
        "doubted or overruled it was not checked."
        for d in relied
        if not (card := view.precedent_cards.get(d)) or not card.treatment_verified
    ]
    limitations.append(CORPUS_LIMITATION)
    sources = [
        SourceRef(
            document_id=d,
            court=view.documents[d].court,
            date=view.documents[d].decision_date,
            citation=view.documents[d].citations[0] if view.documents[d].citations else None,
            pinpoints_used=pins,
            link=view.documents[d].source_url,
        )
        for d, pins in cited.items()
    ]
    return CaseAnalysis(
        question=view.question,
        facts=list(view.facts),
        assumptions=list(view.assumptions),
        issues=issues,
        overall_summary=summary,
        limitations=limitations,
        sources=sources,
        created_by=CreatedBy(stage=stage, agent=AGENT),
    )
