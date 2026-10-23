"""S10, final verification (IDEA_final §6, PLAN 7.11): the last verification gate, then the
Judge's own citations.

1. **The gate** (``VerificationGate(S10)``): argument citations still without a result.
2. **The Judge's citations.** Every sentence of the position summaries and the overall summary
   that cites evidence inline (``[E-012]``) is checked against each evidence it cites:
   - evidence that is not citable (not verified or weak; invariant 3) fails without a call;
   - otherwise the entailment judge reads the sentence as the claim, with the evidence's
     verified quote and its chunk as the passage (the 6.1 verifier's third check).
     ``supports`` or ``partially_supports`` keeps the citation.
3. **Rewrite.** A failing citation is removed from its sentence. A sentence left with no
   citation is listed in ``limitations``. Supporting authorities and sources are recomputed
   from what is still cited (key evidence included), and the analysis is written again; S10's
   is the final one.

Recorded (``s10.*``, besides the gate's): the Judge's citations checked, failing as not citable
or as unsupported, and the sentences flagged.

Passthrough: the gate's passthrough, and the Judge's citations checked in code only (the
non-citable ones removed).
"""

import re
from dataclasses import dataclass

from juris.agents.judge import INLINE, citable, inline_refs
from juris.config import Role
from juris.events import CaseView
from juris.models import CaseAnalysis, IssueAnalysis, PositionAnalysis
from juris.models.common import DocumentKind, EntailmentLabel, Stage
from juris.pipeline.budget import Spend
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.verify import VerificationGate
from juris.verify import LLMEntailmentJudge
from juris.verify.entailment import ClaimEvidencePair
from juris.verify.verifier import chunk_text

AGENT = "final_verifier"
KEEP = frozenset({EntailmentLabel.SUPPORTS, EntailmentLabel.PARTIALLY_SUPPORTS})
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Ref:
    """One inline citation: sentence ``sentence`` of text ``text`` cites ``evidence_id``."""

    text: str  # "I-1.A" for a position summary, "overall" for the overall summary
    sentence: int
    evidence_id: str
    keep: bool = True
    reason: str = ""


def sentences(text: str) -> list[str]:
    """The sentences; a piece holding only citations ("... restraint. [E-012]") belongs to
    the sentence before it."""
    out: list[str] = []
    for piece in _SENTENCE.split(text.strip()):
        if out and not INLINE.sub("", piece).strip(" .,;"):
            out[-1] += " " + piece
        elif piece:
            out.append(piece)
    return out


def drop_ref(sentence: str, evidence_id: str) -> str:
    """The sentence without its inline citation of ``evidence_id``."""
    number = int(evidence_id.split("-")[1])
    pattern = rf"\s*\[\s*E\s*-?\s*0*{number}\s*\]"
    return re.sub(pattern, "", sentence, flags=re.IGNORECASE)


class FinalVerificationStage:
    stage = Stage.S10
    optional = False

    def __init__(self, *, batch_size: int = 1, max_passage_chars: int | None = None) -> None:
        self.gate = VerificationGate(
            Stage.S10, batch_size=batch_size, max_passage_chars=max_passage_chars
        )
        self.batch_size = batch_size
        self.max_passage_chars = max_passage_chars

    def estimate(self, ctx: StageContext) -> Spend | None:
        return None  # required: never skipped

    async def run(self, ctx: StageContext) -> None:
        await self.gate.run(ctx)
        await self._judge_citations(ctx, entailment=True)

    async def passthrough(self, ctx: StageContext) -> None:
        await self.gate.passthrough(ctx)
        await self._judge_citations(ctx, entailment=False)

    async def _judge_citations(self, ctx: StageContext, *, entailment: bool) -> None:
        view = ctx.record.view
        analysis = view.analysis
        values = ctx.run.values
        if analysis is None:
            values["s10.judge_citations"] = 0.0
            return
        texts = {p.position_id: p.summary for i in analysis.issues for p in i.positions}
        texts["overall"] = analysis.overall_summary
        split = {name: sentences(text) for name, text in texts.items()}
        refs = [
            Ref(name, n, e)
            for name, parts in split.items()
            for n, sentence in enumerate(parts)
            for e in inline_refs(sentence)
        ]
        quotes = citable(view)
        for ref in refs:
            if ref.evidence_id not in quotes:
                ref.keep, ref.reason = False, "not citable"
        to_judge = [r for r in refs if r.keep]
        if entailment and to_judge:
            model = ctx.run.config.model_for(Role.VERIFIER)
            judge = LLMEntailmentJudge(
                ctx.run.gateway,
                model,
                batch_size=self.batch_size,
                seed=ctx.run.seed,
                max_passage_chars=self.max_passage_chars,
            )
            pairs = [
                ClaimEvidencePair(
                    claim_id=f"{r.text}:{r.sentence}",
                    claim=split[r.text][r.sentence],
                    evidence_id=r.evidence_id,
                    quote=quotes[r.evidence_id],
                    passage=chunk_text(view, r.evidence_id) or "",
                )
                for r in to_judge
            ]
            for ref, judgement in zip(to_judge, await judge.judge(pairs), strict=True):
                if judgement.label not in KEEP:
                    ref.keep, ref.reason = False, judgement.label.value
        flagged: list[str] = []
        for ref in refs:
            if not ref.keep:
                split[ref.text][ref.sentence] = drop_ref(
                    split[ref.text][ref.sentence], ref.evidence_id
                )
        for name, parts in split.items():
            for n, sentence in enumerate(parts):
                mine = [r for r in refs if r.text == name and r.sentence == n]
                if mine and not any(r.keep for r in mine):
                    flagged.append(sentence)
        final = rebuild(analysis, view, {name: " ".join(parts) for name, parts in split.items()},
                        flagged)  # fmt: skip
        ctx.record.set_analysis(final, stage=self.stage, agent=AGENT)
        values["s10.judge_citations"] = float(len(refs))
        values["s10.judge_not_citable"] = float(sum(r.reason == "not citable" for r in refs))
        values["s10.judge_unsupported"] = float(
            sum(not r.keep and r.reason != "not citable" for r in refs)
        )
        values["s10.sentences_flagged"] = float(len(flagged))
        ctx.run.details["s10.removed"] = [
            {"text": r.text, "sentence": r.sentence, "evidence": r.evidence_id, "why": r.reason}
            for r in refs
            if not r.keep
        ]


def rebuild(
    analysis: CaseAnalysis, view: CaseView, texts: dict[str, str], flagged: list[str]
) -> CaseAnalysis:
    """The analysis with the checked texts; authorities and sources from what is still cited."""
    quotes = citable(view)

    def judgments(evidence_ids: list[str]) -> list[str]:
        return list(dict.fromkeys(
            view.evidence[e].document_id for e in evidence_ids
            if view.documents[view.evidence[e].document_id].kind is DocumentKind.JUDGMENT
        ))  # fmt: skip

    issues: list[IssueAnalysis] = []
    still: list[str] = []  # evidence still cited anywhere
    for issue in analysis.issues:
        positions: list[PositionAnalysis] = []
        for p in issue.positions:
            summary = texts[p.position_id]
            cited = [r.evidence_id for r in p.key_evidence]
            cited += [e for e in inline_refs(summary) if e in quotes]
            still += cited
            positions.append(
                p.model_copy(
                    update={"summary": summary, "supporting_authorities": judgments(cited)}
                )
            )
        issues.append(issue.model_copy(update={"positions": positions}))
    still += [e for e in inline_refs(texts["overall"]) if e in quotes]
    documents = {view.evidence[e].document_id for e in still}
    limitations = list(analysis.limitations) + [
        f'Stated without support: its citations did not check out at the final verification: "{s}"'
        for s in flagged
    ]
    return analysis.model_copy(
        update={
            "issues": issues,
            "overall_summary": texts["overall"],
            "limitations": limitations,
            "sources": [s for s in analysis.sources if s.document_id in documents],
            "created_by": analysis.created_by.model_copy(update={"stage": Stage.S10}),
        }
    )
