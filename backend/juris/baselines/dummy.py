"""The dummy system: an offline stand-in for testing the runner (``configs/pipeline/dummy.yaml``).

It frames the item's gold issues, registers each gold supporting authority as one piece of
evidence (its proposition as the chunk text), argues position A of the first issue from
them, marks every citation verified, makes one call to the ``dummy`` provider, and writes a
valid ``CaseAnalysis``. Nothing touches the corpus or a real model, and its scores mean
nothing: it exists so the runner, events, manifests and metrics can run end to end.
"""

from collections.abc import Collection

from juris.baselines.base import RunContext
from juris.config import Role
from juris.events.catalog import (
    AgentCompletedPayload,
    AgentStartedPayload,
    AnalysisReadyPayload,
    ArgumentCreatedPayload,
    ClaimCreatedPayload,
    EvidenceRegisteredPayload,
    IssueFramedPayload,
    VerificationResultPayload,
)
from juris.llm.types import ChatMessage, ProviderRequest, ProviderResponse
from juris.models import (
    Argument,
    CaseAnalysis,
    Chunk,
    Citation,
    Claim,
    CreatedBy,
    Document,
    Evidence,
    EvidenceRef,
    Issue,
    IssueAnalysis,
    Position,
    PositionAnalysis,
    SourceRef,
    VerificationResult,
)
from juris.models.common import (
    Confidence,
    DocumentKind,
    EntailmentLabel,
    Leaning,
    Stage,
    VerificationCheck,
    VerificationStatus,
)

AGENT = "dummy"


class EchoProvider:
    """Offline provider for the dummy profile: answers with the last message's first line."""

    name = "dummy"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        prompt = request.messages[-1].content
        text = prompt.splitlines()[0] if prompt else ""
        return ProviderResponse(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            stop_reason="end_turn",
        )


class DummySystem:
    corpus_snapshot_id = "dummy"
    embedding_model = "none"

    def __init__(self, fail_items: Collection[str] = ()) -> None:
        self.fail_items = frozenset(fail_items)  # these items raise, to test run_failed

    async def run(self, ctx: RunContext) -> None:
        item, emit = ctx.item, ctx.emitter.emit
        model = ctx.config.model_for(Role.BASELINE)
        emit(AgentStartedPayload(role=Role.BASELINE, model=model.name), agent=AGENT)

        framed = CreatedBy(stage=Stage.S0, agent=AGENT)
        issues = [
            Issue(
                id=f"I-{n}",
                question=question,
                positions=[
                    Position(id=f"I-{n}.A", statement="Yes."),
                    Position(id=f"I-{n}.B", statement="No."),
                ],
                created_by=framed,
            )
            for n, question in enumerate(item.gold_issues[:4], 1)
        ]
        for issue in issues:
            emit(IssueFramedPayload(issue=issue, facts=[item.facts]), stage=Stage.S0, agent=AGENT)

        found = CreatedBy(stage=Stage.S1, agent=AGENT)
        evidence: list[Evidence] = []
        for n, a in enumerate(item.gold_supporting_authorities, 1):
            para = int(a.pinpoints[0]) if a.pinpoints[0].isdigit() else 0  # "h-2": front matter
            chunk = Chunk(
                id=f"{a.doc_id}#p{a.pinpoints[0]}",
                document_id=a.doc_id,
                text=a.proposition,
                para_start=para,
                para_end=para,
                char_start=0,
                char_end=len(a.proposition),
            )
            document = Document(
                id=a.doc_id,
                kind=DocumentKind.JUDGMENT,
                title=a.title,
                citations=[a.citation],
                licence="dummy",
                corpus_snapshot=self.corpus_snapshot_id,
            )
            ev = Evidence(
                id=f"E-{n:03d}",
                chunk_id=chunk.id,
                document_id=a.doc_id,
                issue_ids=["I-1"],
                retrieved_by_query=item.question,
                registered_at_stage=Stage.S1,
                created_by=found,
            )
            emit(
                EvidenceRegisteredPayload(evidence=ev, document=document, chunk=chunk),
                stage=Stage.S1,
                agent=AGENT,
            )
            evidence.append(ev)

        argued = CreatedBy(stage=Stage.S2, agent=AGENT)
        for n, (ev, a) in enumerate(
            zip(evidence, item.gold_supporting_authorities, strict=True), 1
        ):
            claim = Claim(
                id=f"C-{n:03d}",
                text=a.proposition,
                position_id="I-1.A",
                author_agent=AGENT,
                created_by=argued,
            )
            quote = " ".join(a.proposition.split()[:8])
            emit(ClaimCreatedPayload(claim=claim), stage=Stage.S2, agent=AGENT)
            emit(
                ArgumentCreatedPayload(
                    argument=Argument(
                        id=f"A-{n:03d}",
                        claim_ids=[claim.id],
                        reasoning="The authority states the rule.",
                        citations=[
                            Citation(
                                claim_id=claim.id,
                                evidence_id=ev.id,
                                pinpoint=f"para {a.pinpoints[0]}",
                                quote=quote,
                            )
                        ],
                        created_by=argued,
                    )
                ),
                stage=Stage.S2,
                agent=AGENT,
            )
            emit(
                VerificationResultPayload(
                    claim_id=claim.id,
                    evidence_id=ev.id,
                    result=VerificationResult(
                        status=VerificationStatus.VERIFIED,
                        check=VerificationCheck.ENTAILMENT,
                        label=EntailmentLabel.SUPPORTS,
                        justification="Dummy: the chunk is the claim.",
                        created_by=CreatedBy(stage=Stage.S4, agent=AGENT),
                    ),
                ),
                stage=Stage.S4,
                agent=AGENT,
            )

        if item.id in self.fail_items:
            raise RuntimeError(f"dummy failure on {item.id}")

        reply = await ctx.gateway.complete(
            [ChatMessage(role="user", content=f"{item.question}\n\n{item.facts}")],
            model=model,
            seed=ctx.seed,
            tags={"agent": AGENT, "item": item.id},
        )
        emit(
            AgentCompletedPayload(
                role=Role.BASELINE,
                input_tokens=reply.usage.input_tokens,
                output_tokens=reply.usage.output_tokens,
                cost_usd=reply.usage.cost_usd,
                cached=reply.cached,
            ),
            agent=AGENT,
        )

        judged = CreatedBy(stage=Stage.S9, agent=AGENT)
        cited = " ".join(
            f"{a.proposition.rstrip('.')} [{ev.id}]."
            for ev, a in zip(evidence, item.gold_supporting_authorities, strict=True)
        )
        analysis = CaseAnalysis(
            question=item.question,
            facts=[item.facts],
            issues=[
                IssueAnalysis(
                    issue_id=issue.id,
                    issue_text=issue.question,
                    positions=[
                        PositionAnalysis(
                            position_id=f"{issue.id}.A",
                            summary=reply.text or "Yes.",
                            supporting_authorities=[e.document_id for e in evidence]
                            if n == 1
                            else [],
                            key_evidence=[
                                EvidenceRef(evidence_id=e.id, pinpoint="para") for e in evidence
                            ]
                            if n == 1
                            else [],
                        ),
                        PositionAnalysis(position_id=f"{issue.id}.B", summary="No."),
                    ],
                    leaning=Leaning.LEANING_A if n == 1 else Leaning.INSUFFICIENT_EVIDENCE,
                    confidence=Confidence.LOW,
                    confidence_reason="Dummy system.",
                )
                for n, issue in enumerate(issues, 1)
            ],
            overall_summary=" ".join(cited.split()[:240]) or "No authority was found.",
            sources=[SourceRef(document_id=e.document_id) for e in evidence],
            created_by=judged,
        )
        emit(AnalysisReadyPayload(analysis=analysis), stage=Stage.S9, agent=AGENT)
