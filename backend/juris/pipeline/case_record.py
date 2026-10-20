"""The Case Record, the Scribe (IDEA_final §6.1, §8; PLAN 7.1): deterministic code that every
stage writes through. It never calls a model.

Each typed method checks its input against the record so far, then appends the event; nothing
is written when a check fails. A failed check raises ``InvariantViolation`` with every error
found, so an agent can be sent all of them in one repair attempt (PLAN 7.2). The invariants of
IDEA_final §8:

1. a citation references registered evidence (``add_argument``);
2. its quote is an exact normalised substring of the chunk (``add_argument``; the verifier's
   rule, ``verify.check_citation``);
3. the analysis cites only evidence whose best verification is verified or weak
   (``set_analysis``);
4. an objection targets an existing ID (``raise_objection``);
5. the analysis's issues are the S0 issues (``set_analysis``: each one, with its positions).

Every other reference (a claim's position, a counterargument's target, a ballot's positions,
...) must exist too, and claim statuses follow the lifecycle of §8. IDs come from one
``IdAllocator``, so a replayed run gets the same IDs.

``view`` is the folded ``CaseView``. It includes events that other code emitted through the
same emitter (the evidence registry, the Citation Verifier), so the checks see them too.
"""

from collections.abc import Container, Iterable, Sequence

from juris.config import DeliberationParams
from juris.events import CaseView, EventEmitter, fold
from juris.events.catalog import (
    AnalysisReadyPayload,
    ArgumentCreatedPayload,
    ClaimCreatedPayload,
    ClaimStatusChangedPayload,
    CounterargumentCreatedPayload,
    IssueFramedPayload,
    JuryAggregatedPayload,
    JuryBallotCastPayload,
    ObjectionAnsweredPayload,
    ObjectionRaisedPayload,
    PrecedentCardCreatedPayload,
    QueryIssuedPayload,
    QueryKind,
    UnresolvedQuestionAddedPayload,
    ValidationFailedPayload,
    VerificationResultPayload,
)
from juris.models import (
    Argument,
    CaseAnalysis,
    Citation,
    Claim,
    Counterargument,
    CreatedBy,
    Evidence,
    IdAllocator,
    Issue,
    JuryBallot,
    Objection,
    Position,
    PrecedentCard,
    VerificationResult,
)
from juris.models.common import (
    ClaimStatus,
    DocumentKind,
    ObjectionType,
    Stage,
    VerificationStatus,
)
from juris.pipeline.evidence_registry import CorpusRecords, EvidenceRegistry, Registrable
from juris.verify import check_citation

SCRIBE = "scribe"

# IDEA_final §8: proposed -> verified | weak | unsupported -> contested -> survives | falls.
# A verification gate may move a claim between the three verdicts (S7 checks new citations).
_VERDICTS = frozenset({ClaimStatus.VERIFIED, ClaimStatus.WEAK, ClaimStatus.UNSUPPORTED})
TRANSITIONS: dict[ClaimStatus, frozenset[ClaimStatus]] = {
    ClaimStatus.PROPOSED: _VERDICTS,
    ClaimStatus.VERIFIED: _VERDICTS | {ClaimStatus.CONTESTED},
    ClaimStatus.WEAK: _VERDICTS | {ClaimStatus.CONTESTED},
    ClaimStatus.UNSUPPORTED: _VERDICTS | {ClaimStatus.CONTESTED},
    ClaimStatus.CONTESTED: frozenset({ClaimStatus.SURVIVES, ClaimStatus.FALLS}),
    ClaimStatus.SURVIVES: frozenset(),
    ClaimStatus.FALLS: frozenset(),
}
_RANK = {
    VerificationStatus.VERIFIED: 3,
    VerificationStatus.WEAK: 2,
    VerificationStatus.UNSUPPORTED: 1,
    VerificationStatus.INVALID: 0,
}
CITABLE = frozenset({VerificationStatus.VERIFIED, VerificationStatus.WEAK})


class InvariantViolation(ValueError):
    """The input breaks the record's rules. ``errors`` lists every problem found."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class CaseRecord:
    def __init__(
        self,
        emitter: EventEmitter,
        corpus: CorpusRecords,
        *,
        params: DeliberationParams | None = None,
        ids: IdAllocator | None = None,
    ) -> None:
        self.emitter = emitter
        self.params = params or DeliberationParams()
        self.ids = ids or IdAllocator()
        self.registry = EvidenceRegistry(corpus, emitter, self.ids)
        self._view: CaseView | None = None
        self._seq = 0

    @property
    def view(self) -> CaseView:
        """The record as folded so far (a snapshot: later writes give a new object)."""
        new = self.emitter.events(self._seq)
        if new:
            self._view = fold(new, self._view)
            self._seq = new[-1].seq
        if self._view is None:
            raise RuntimeError("the run has no case_created event yet")
        return self._view

    # ---- S0-S1 ------------------------------------------------------------------------------

    def frame_issue(
        self,
        question: str,
        positions: Sequence[str],
        *,
        stage: Stage,
        agent: str,
        facts: Sequence[str] = (),
        assumptions: Sequence[str] = (),
    ) -> Issue:
        """A new issue ``I-n`` with positions ``I-n.A``, ``I-n.B`` (and ``.C``)."""
        errors: list[str] = []
        if len(self.view.issues) >= self.params.max_issues:
            errors.append(f"at most {self.params.max_issues} issues per case")
        most = max(2, self.params.positions_per_issue)
        if not 2 <= len(positions) <= most:
            errors.append(f"an issue needs 2 to {most} positions, got {len(positions)}")
        if not question.strip():
            errors.append("the issue's question is empty")
        if any(not p.strip() for p in positions):
            errors.append("a position statement is empty")
        _raise(errors)
        issue_id = self.ids.issue()
        issue = Issue(
            id=issue_id,
            question=question.strip(),
            positions=[
                Position(id=IdAllocator.position(issue_id, n), statement=p.strip())
                for n, p in enumerate(positions)
            ],
            created_by=CreatedBy(stage=stage, agent=agent),
        )
        self.emitter.emit(
            IssueFramedPayload(issue=issue, facts=list(facts), assumptions=list(assumptions)),
            stage=stage,
            agent=agent,
        )
        return issue

    def add_query(
        self,
        query: str,
        issue_ids: Sequence[str],
        *,
        stage: Stage,
        agent: str,
        kind: QueryKind = "search",
    ) -> None:
        """A search query, or (``provision``) a statute section to fetch by name."""
        errors = self._unknown(issue_ids, self.view.issues, "issue")
        if not query.strip():
            errors.append("the query is empty")
        _raise(errors)
        self.emitter.emit(
            QueryIssuedPayload(query=query.strip(), issue_ids=list(issue_ids), kind=kind),
            stage=stage,
            agent=agent,
        )

    def register_evidence(
        self, hit: Registrable, issue_ids: Sequence[str], *, stage: Stage, agent: str
    ) -> Evidence:
        """Register a retrieved chunk (or extend an earlier registration's issues)."""
        errors = self._unknown(issue_ids, self.view.issues, "issue")
        if not issue_ids:
            errors.append("evidence must be registered for at least one issue")
        _raise(errors)
        return self.registry.register(hit, issue_ids, stage, agent=agent)

    # ---- S2, S5: claims and arguments -------------------------------------------------------

    def add_claim(self, text: str, position_id: str, *, stage: Stage, agent: str) -> Claim:
        errors = [] if text.strip() else ["the claim's text is empty"]
        if position_id not in self._positions():
            errors.append(f"unknown position {position_id!r}")
        _raise(errors)
        claim = Claim(
            id=self.ids.next("C"),
            text=text.strip(),
            position_id=position_id,
            author_agent=agent,
            created_by=CreatedBy(stage=stage, agent=agent),
        )
        self.emitter.emit(ClaimCreatedPayload(claim=claim), stage=stage, agent=agent)
        return claim

    def check_citation(self, citation: Citation) -> str | None:
        """Why the citation breaks invariant 1 or 2, or None if it holds."""
        failed = check_citation(self.registry, citation)
        if failed is None:
            return None
        check, reason = failed
        return f"{citation.evidence_id}: {reason} ({check.value} check)"

    def add_argument(
        self,
        claim_ids: Sequence[str],
        reasoning: str,
        citations: Sequence[Citation],
        *,
        stage: Stage,
        agent: str,
    ) -> Argument:
        """An argument for ``claim_ids``. Every citation must belong to one of them, cite
        registered evidence and quote it exactly (invariants 1-2)."""
        view = self.view
        errors = self._unknown(claim_ids, view.claims, "claim")
        if not claim_ids:
            errors.append("an argument needs at least one claim")
        if not reasoning.strip():
            errors.append("the argument's reasoning is empty")
        for n, citation in enumerate(citations, 1):
            if citation.claim_id not in claim_ids:
                errors.append(f"citation {n} is for {citation.claim_id}, not this argument")
            problem = self.check_citation(citation)
            if problem is not None:
                errors.append(f"citation {n}: {problem}")
        _raise(errors)
        argument = Argument(
            id=self.ids.next("A"),
            claim_ids=list(claim_ids),
            reasoning=reasoning.strip(),
            citations=list(citations),
            created_by=CreatedBy(stage=stage, agent=agent),
        )
        self.emitter.emit(ArgumentCreatedPayload(argument=argument), stage=stage, agent=agent)
        return argument

    def add_counterargument(
        self, target_claim_id: str, argument_id: str, *, stage: Stage, agent: str
    ) -> Counterargument:
        """A rebuttal: ``argument_id`` answers an opposing claim of the same issue."""
        view = self.view
        errors: list[str] = []
        target = view.claims.get(target_claim_id)
        argument = view.arguments.get(argument_id)
        if target is None:
            errors.append(f"unknown claim {target_claim_id!r}")
        if argument is None:
            errors.append(f"unknown argument {argument_id!r}")
        if target is not None and argument is not None:
            own = {view.claims[c].position_id for c in argument.claim_ids}
            issue = _issue_of(target.position_id)
            if target.position_id in own:
                errors.append(f"{argument_id} argues for {target_claim_id}'s own position")
            elif any(_issue_of(p) != issue for p in own):
                errors.append(f"{argument_id} is not about {target_claim_id}'s issue ({issue})")
        _raise(errors)
        counter = Counterargument(
            id=self.ids.next("R"),
            targets=target_claim_id,
            argument_id=argument_id,
            created_by=CreatedBy(stage=stage, agent=agent),
        )
        self.emitter.emit(
            CounterargumentCreatedPayload(counterargument=counter), stage=stage, agent=agent
        )
        return counter

    def set_claim_status(
        self, claim_id: str, status: ClaimStatus, reason: str, *, stage: Stage, agent: str
    ) -> bool:
        """Move a claim along the lifecycle. Returns False (and writes nothing) if the claim
        already has ``status``."""
        claim = self.view.claims.get(claim_id)
        if claim is None:
            raise InvariantViolation([f"unknown claim {claim_id!r}"])
        if claim.status is status:
            return False
        if status not in TRANSITIONS[claim.status]:
            raise InvariantViolation(
                [f"{claim_id} cannot go from {claim.status.value} to {status.value}"]
            )
        self.emitter.emit(
            ClaimStatusChangedPayload(claim_id=claim_id, status=status, reason=reason),
            stage=stage,
            agent=agent,
        )
        return True

    # ---- S3, S4/S7/S10 ----------------------------------------------------------------------

    def add_precedent_card(self, card: PrecedentCard, *, stage: Stage, agent: str) -> None:
        """One card per cited judgment; its fit labels only for arguments citing it."""
        view = self.view
        errors: list[str] = []
        document = view.documents.get(card.document_id)
        if document is None:
            errors.append(f"document {card.document_id!r} is not in the record")
        elif document.kind is not DocumentKind.JUDGMENT:
            errors.append(f"{card.document_id} is a {document.kind.value}, not a judgment")
        if card.document_id in view.precedent_cards:
            errors.append(f"{card.document_id} already has a precedent card")
        for argument_id in card.fit_by_argument:
            argument = view.arguments.get(argument_id)
            if argument is None:
                errors.append(f"unknown argument {argument_id!r}")
            elif not any(
                view.evidence[c.evidence_id].document_id == card.document_id
                for c in argument.citations
                if c.evidence_id in view.evidence
            ):
                errors.append(f"{argument_id} does not cite {card.document_id}")
        _raise(errors)
        self.emitter.emit(PrecedentCardCreatedPayload(card=card), stage=stage, agent=agent)

    def add_verification(
        self,
        claim_id: str,
        evidence_id: str,
        result: VerificationResult,
        *,
        stage: Stage,
        agent: str,
    ) -> None:
        """A verifier result for a citation that exists in the record."""
        view = self.view
        errors = self._unknown([claim_id], view.claims, "claim")
        errors += self._unknown([evidence_id], view.evidence, "evidence")
        cited = any(
            c.claim_id == claim_id and c.evidence_id == evidence_id
            for a in view.arguments.values()
            for c in a.citations
        )
        if not errors and not cited:
            errors.append(f"no argument cites {evidence_id} for {claim_id}")
        _raise(errors)
        self.emitter.emit(
            VerificationResultPayload(claim_id=claim_id, evidence_id=evidence_id, result=result),
            stage=stage,
            agent=agent,
        )

    def best_verification(self, evidence_id: str) -> VerificationStatus | None:
        """The best verdict any citation of this evidence got, or None if never checked."""
        statuses = [
            v.result.status for v in self.view.verifications if v.evidence_id == evidence_id
        ]
        return max(statuses, key=_RANK.__getitem__) if statuses else None

    # ---- S6 ---------------------------------------------------------------------------------

    def raise_objection(
        self,
        kind: ObjectionType,
        target_id: str,
        text: str,
        *,
        stage: Stage,
        agent: str,
    ) -> Objection:
        """Invariant 4: the target must be an ID in the record."""
        if not self._exists(target_id):
            raise InvariantViolation([f"objection target {target_id!r} is not in the record"])
        objection = Objection(
            id=self.ids.next("O"),
            type=kind,
            target_id=target_id,
            raised_by=agent,
            text=text,
            created_by=CreatedBy(stage=stage, agent=agent),
        )
        self.emitter.emit(ObjectionRaisedPayload(objection=objection), stage=stage, agent=agent)
        return objection

    def answer_objection(
        self, objection_id: str, response: str, *, resolved: bool, stage: Stage, agent: str
    ) -> None:
        """The targeted counsel's one reply (IDEA_final §6.1)."""
        objection = self.view.objections.get(objection_id)
        if objection is None:
            raise InvariantViolation([f"unknown objection {objection_id!r}"])
        if objection.response is not None:
            raise InvariantViolation([f"{objection_id} has already been answered"])
        self.emitter.emit(
            ObjectionAnsweredPayload(
                objection_id=objection_id, response=response, resolved=resolved
            ),
            stage=stage,
            agent=agent,
        )

    def add_unresolved_question(
        self,
        question: str,
        *,
        stage: Stage,
        agent: str,
        issue_id: str | None = None,
        from_objection_id: str | None = None,
    ) -> None:
        view = self.view
        errors: list[str] = []
        if issue_id is not None:
            errors += self._unknown([issue_id], view.issues, "issue")
        if from_objection_id is not None:
            errors += self._unknown([from_objection_id], view.objections, "objection")
        _raise(errors)
        self.emitter.emit(
            UnresolvedQuestionAddedPayload(
                question=question, issue_id=issue_id, from_objection_id=from_objection_id
            ),
            stage=stage,
            agent=agent,
        )

    # ---- S8-S9 ------------------------------------------------------------------------------

    def cast_ballot(self, ballot: JuryBallot, *, stage: Stage, agent: str) -> None:
        """One ballot per juror; it scores existing positions and cites existing IDs."""
        view = self.view
        errors: list[str] = []
        if ballot.juror_id in view.jury_ballots:
            errors.append(f"{ballot.juror_id} has already voted")
        errors += self._unknown(ballot.per_position_scores, self._positions(), "position")
        for n, justification in enumerate(ballot.justifications, 1):
            missing = [i for i in justification.cited_ids if not self._exists(i)]
            if missing:
                errors.append(f"justification {n} cites unknown IDs {missing}")
        _raise(errors)
        self.emitter.emit(JuryBallotCastPayload(ballot=ballot), stage=stage, agent=agent)

    def set_jury_aggregate(
        self, aggregate: JuryAggregatedPayload, *, stage: Stage, agent: str
    ) -> None:
        view = self.view
        errors = self._unknown(aggregate.per_position, self._positions(), "position")
        errors += self._unknown(aggregate.contested_issue_ids, view.issues, "issue")
        if not view.jury_ballots:
            errors.append("no ballots have been cast")
        _raise(errors)
        self.emitter.emit(aggregate, stage=stage, agent=agent)

    def set_analysis(self, analysis: CaseAnalysis, *, stage: Stage, agent: str) -> None:
        """Invariants 3 and 5: the S0 issues with their positions, each once; evidence and
        authorities only from verified or weak citations; sources from the record."""
        view = self.view
        errors: list[str] = []
        covered = [i.issue_id for i in analysis.issues]
        for missing in [i for i in view.issues if i not in covered]:
            errors.append(f"the analysis does not cover {missing}")
        supported_documents: set[str] = set()
        for evidence_id, evidence in view.evidence.items():
            if self.best_verification(evidence_id) in CITABLE:
                supported_documents.add(evidence.document_id)
        for issue in analysis.issues:
            framed = view.issues.get(issue.issue_id)
            if framed is None:
                errors.append(f"{issue.issue_id} is not an S0 issue")
                continue
            expected = [p.id for p in framed.positions]
            if sorted(p.position_id for p in issue.positions) != expected:
                errors.append(f"{issue.issue_id} must analyse positions {expected}")
            for position in issue.positions:
                for ref in position.key_evidence:
                    best = self.best_verification(ref.evidence_id)
                    if ref.evidence_id not in view.evidence:
                        errors.append(f"{position.position_id} cites unknown {ref.evidence_id}")
                    elif best not in CITABLE:
                        state = best.value if best else "never verified"
                        errors.append(f"{position.position_id} cites {ref.evidence_id} ({state})")
                for document_id in position.supporting_authorities:
                    if document_id not in supported_documents:
                        errors.append(
                            f"{position.position_id} relies on {document_id}, which has no "
                            "verified or weak citation"
                        )
        for source in analysis.sources:
            if source.document_id not in view.documents:
                errors.append(f"source {source.document_id} is not in the record")
        _raise(errors)
        self.emitter.emit(AnalysisReadyPayload(analysis=analysis), stage=stage, agent=agent)

    # ---- failures ---------------------------------------------------------------------------

    def reject(
        self, errors: Sequence[str], *, stage: Stage, agent: str, dropped: str | None = None
    ) -> None:
        """Log output that failed validation (after its repair attempt, PLAN 7.2)."""
        self.emitter.emit(
            ValidationFailedPayload(errors=list(errors) or ["(no detail)"], dropped=dropped),
            stage=stage,
            agent=agent,
        )

    # ---- lookups ----------------------------------------------------------------------------

    def _positions(self) -> dict[str, Position]:
        return {p.id: p for i in self.view.issues.values() for p in i.positions}

    def _exists(self, record_id: str) -> bool:
        view = self.view
        return any(
            record_id in group
            for group in (
                view.issues,
                self._positions(),
                view.evidence,
                view.claims,
                view.arguments,
                view.counterarguments,
                view.objections,
            )
        )

    @staticmethod
    def _unknown(ids: Iterable[str], known: Container[str], what: str) -> list[str]:
        return [f"unknown {what} {i!r}" for i in ids if i not in known]


def _issue_of(position_id: str) -> str:
    return position_id.rsplit(".", 1)[0]


def _raise(errors: Sequence[str]) -> None:
    if errors:
        raise InvariantViolation(errors)
