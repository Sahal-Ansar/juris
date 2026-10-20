"""The Precedent Analyst (S3; IDEA_final §6.1, §9; PLAN 7.6): one call per judgment cited in
S2, giving its ``PrecedentCard`` and a fit label for every argument that relies on it.

Code, not the model, supplies the facts it can know (IDEA_final §6.1): court, level, bench
strength, date and citations come from the document. Treatment comes from the citation graph
(3.9): each later judgment citing this one with a cue such as "followed" or "overruled". Cues
are weak hints, so ``treatment_verified`` is always False (D-013). The model gives the ratio,
the material facts, how they differ from the dispute's, and a fit per argument.

The model sees the dispute (question, facts, the issues the citing arguments are about), the
judgment's metadata and cues, passages (its headnote, and each cited chunk with its
neighbours: ``passages``), and the arguments with their claims and quotes.

Checks: a principle and material facts are required, or the whole output is dropped. A fit
must name an argument that cites this judgment, once, or that fit is dropped. If nothing
usable is left, S3 writes ``fallback_card`` (code facts only), so every cited judgment gets a
card.
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from juris.agents.base import Agent, Problem
from juris.config import Role
from juris.events import CaseView
from juris.models import CreatedBy, Document, PrecedentCard
from juris.models.common import CourtLevel, PrecedentFit, Stage, Treatment
from juris.models.record import TreatmentSignal
from juris.pipeline.case_record import CaseRecord
from juris.retrieval.lookup import LinkedCase

AGENT = "precedent_analyst"
NOT_EXTRACTED = "Not extracted: {why}"

# graph cue label (3.9) -> treatment; "referred" says nothing about weight
CUE_TREATMENT = {
    "followed": Treatment.FOLLOWED,
    "relied_on": Treatment.FOLLOWED,
    "approved": Treatment.FOLLOWED,
    "affirmed": Treatment.AFFIRMED,
    "distinguished": Treatment.DISTINGUISHED,
    "doubted": Treatment.DOUBTED,
    "overruled": Treatment.OVERRULED,
    "reversed": Treatment.OVERRULED,
    "per_incuriam": Treatment.PER_INCURIAM,
}
LEVEL_WEIGHT = {
    CourtLevel.SUPREME_COURT: "binds every court in India (Article 141)",
    CourtLevel.HIGH_COURT: "binds courts below it in its State; persuasive elsewhere",
    CourtLevel.TRIBUNAL: "little weight",
    CourtLevel.SUBORDINATE: "little weight",
    CourtLevel.OTHER: "historical or persuasive weight only",
}


class FitJudgement(BaseModel):
    argument_id: str = Field(description='An argument ID from the list, e.g. "A-003"')
    fit: PrecedentFit
    reason: str = Field(description="One sentence")


class PrecedentOutput(BaseModel):
    principle: str = Field(description="The ratio as it bears on the issues, 1-2 sentences")
    material_facts: str = Field(description="The facts the decision turned on, 1-2 sentences")
    distinguishing_factors: list[str] = Field(description="Material differences from the dispute")
    fits: list[FitJudgement] = Field(description="One per argument listed")


def treatment_signals(citing: Sequence[LinkedCase]) -> list[TreatmentSignal]:
    """One signal per (later judgment, cue) from the citation graph; unknown cues are left out."""
    signals: list[TreatmentSignal] = []
    for case in citing:
        for label in case.treatments:
            kind = CUE_TREATMENT.get(label)
            if kind is None:
                continue
            where = f", para {', '.join(case.paragraphs[:3])}" if case.paragraphs else ""
            name = case.title or case.ref
            signals.append(
                TreatmentSignal(
                    kind=kind,
                    by_document_id=case.ref,
                    cue_text=f'"{label}" near the citation in {name}{where}',
                )
            )
    return signals


def metadata(document: Document) -> str:
    """The judgment's facts that code knows, one per line."""
    lines = [f"Title: {document.title}", f"Document ID: {document.id}"]
    if document.court:
        lines.append(f"Court: {document.court}")
    if document.court_level:
        weight = LEVEL_WEIGHT[document.court_level]
        lines.append(f"Level: {document.court_level.value.replace('_', ' ')} ({weight})")
    if document.bench_strength:
        lines.append(f"Bench: {document.bench_strength} judge(s)")
    if document.decision_date:
        lines.append(f"Decided: {document.decision_date.isoformat()}")
    if document.citations:
        lines.append(f"Citations: {'; '.join(document.citations)}")
    return "\n".join(lines)


def fallback_card(
    document_id: str, signals: Sequence[TreatmentSignal], why: str, *, stage: Stage
) -> PrecedentCard:
    """A card from code facts alone, when the model gave nothing usable (or S3 is a
    passthrough): every cited judgment still has a card, honestly marked."""
    return PrecedentCard(
        document_id=document_id,
        principle=NOT_EXTRACTED.format(why=why),
        material_facts=NOT_EXTRACTED.format(why=why),
        treatment=list(signals),
        treatment_verified=False,
        created_by=CreatedBy(stage=stage, agent=AGENT),
    )


class PrecedentAnalyst(Agent[PrecedentOutput]):
    role = Role.PRECEDENT_ANALYST
    prompt = "precedent_analyst"
    duty = (
        "the Precedent Analyst says what a cited judgment actually decides, how much weight it "
        "carries, and how well it fits each argument that relies on it."
    )
    output_model = PrecedentOutput
    max_tokens = 2000

    def __init__(
        self,
        document_id: str,
        *,
        passages: str,
        signals: Sequence[TreatmentSignal],
        facts: Sequence[str] = (),
        name: str = AGENT,
    ) -> None:
        super().__init__(name)
        self.document_id = document_id
        self.passages = passages
        self.signals = list(signals)
        self.facts = list(facts)

    @property
    def call_tags(self) -> dict[str, str]:
        return {"document": self.document_id}

    def citing_arguments(self, view: CaseView) -> list[str]:
        """The arguments that cite this judgment, in ID order."""
        return [
            a.id
            for a in view.arguments.values()
            if any(
                view.evidence[c.evidence_id].document_id == self.document_id
                for c in a.citations
                if c.evidence_id in view.evidence
            )
        ]

    def input_view(self, view: CaseView) -> dict[str, str]:
        arguments = self.citing_arguments(view)
        issue_ids: list[str] = []
        blocks: list[str] = []
        for argument_id in arguments:
            argument = view.arguments[argument_id]
            claims = [view.claims[c] for c in argument.claim_ids]
            position = claims[0].position_id
            issue_ids.append(position.rsplit(".", 1)[0])
            quotes = [
                f'  {c.evidence_id}: "{c.quote}"'
                for c in argument.citations
                if view.evidence[c.evidence_id].document_id == self.document_id
            ]
            blocks.append(
                f"{argument_id} (for {position}): "
                + " ".join(c.text for c in claims)
                + "\n  Quoted from this judgment:\n"
                + "\n".join(quotes)
            )
        issues = [view.issues[i] for i in dict.fromkeys(issue_ids)]
        facts = [*view.facts, *(f"(assumed) {a}" for a in view.assumptions)] or self.facts
        cues = "\n".join(f"- {s.kind.value}: {s.cue_text}" for s in self.signals)
        return {
            "question": view.question,
            "facts": "\n".join(f"- {f}" for f in facts) or "(none stated)",
            "issues": "\n".join(
                f"{i.id}: {i.question} ("
                + "; ".join(f"{p.id}: {p.statement}" for p in i.positions)
                + ")"
                for i in issues
            ),
            "metadata": metadata(view.documents[self.document_id]),
            "treatment": cues or "(no later judgment in the corpus is known to treat it)",
            "passages": self.passages,
            "arguments": "\n\n".join(blocks),
        }

    def validate(self, output: PrecedentOutput, record: CaseRecord) -> list[Problem]:
        problems: list[Problem] = []
        if not output.principle.strip():
            problems.append(Problem("the principle is empty"))
        if not output.material_facts.strip():
            problems.append(Problem("the material facts are empty"))
        citing = set(self.citing_arguments(record.view))
        seen: set[str] = set()
        for k, fit in enumerate(output.fits):
            argument_id = fit.argument_id.strip()
            if argument_id not in citing:
                message = f"{fit.argument_id!r} is not an argument that cites this judgment"
                problems.append(Problem(message, ("fits", k)))
            elif argument_id in seen:
                problems.append(Problem(f"{argument_id} already has a fit", ("fits", k)))
            seen.add(argument_id)
        return problems

    def card(self, output: PrecedentOutput, *, stage: Stage) -> PrecedentCard:
        return PrecedentCard(
            document_id=self.document_id,
            principle=output.principle.strip(),
            material_facts=output.material_facts.strip(),
            distinguishing_factors=[d.strip() for d in output.distinguishing_factors if d.strip()],
            treatment=self.signals,
            treatment_verified=False,  # cues are hints, never verified treatment (D-013)
            fit_by_argument={f.argument_id.strip(): f.fit for f in output.fits},
            created_by=CreatedBy(stage=stage, agent=self.name),
        )

    def apply(self, output: PrecedentOutput, record: CaseRecord, *, stage: Stage) -> None:
        record.add_precedent_card(self.card(output, stage=stage), stage=stage, agent=self.name)
