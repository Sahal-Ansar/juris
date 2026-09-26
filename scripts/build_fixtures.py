"""Build the mock run fixtures for the UI track (PLAN 2.4).

Everything here is FICTITIOUS: party names, judgments, quotes and paragraph numbers are
invented, every document is titled "[MOCK] ..." with a mock:// URL, and statute sections
are paraphrased, not quoted. Nothing may be cited as a real authority.

Writes fixtures/runs/{coercion_full,failure_case}.jsonl and their folded .caseview.json.
Usage: uv run scripts/build_fixtures.py [--check]
"""

import datetime as dt
import sys
from collections.abc import Iterator
from itertools import count
from pathlib import Path

from juris.config import DEFAULT_PRICES
from juris.events import CaseView, Event, EventEmitter, InMemoryEventStore, fold
from juris.events import catalog as c
from juris.models import (
    DISCLAIMER,
    Argument,
    CaseAnalysis,
    Chunk,
    Citation,
    Claim,
    Counterargument,
    CreatedBy,
    Document,
    Evidence,
    EvidenceRef,
    Issue,
    IssueAnalysis,
    JurisModel,
    JuryBallot,
    Justification,
    KeyConflict,
    Objection,
    Position,
    PositionAnalysis,
    PrecedentCard,
    RubricScores,
    RunTotals,
    SourceRef,
    StatuteMeta,
    TreatmentSignal,
    VerificationResult,
)
from juris.models.common import (
    ClaimStatus,
    CourtLevel,
    DocumentKind,
    EntailmentLabel,
    ObjectionType,
    PrecedentFit,
    Stage,
    Treatment,
    VerificationCheck,
    VerificationStatus,
)

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "runs"
MODEL = "claude-opus-5-5"
PRICE = DEFAULT_PRICES[MODEL]
SNAPSHOT = "mock-snapshot-2026-09"
LICENCE = "MOCK (fictitious, for UI development only)"
T0 = dt.datetime(2026, 9, 27, 10, 0, tzinfo=dt.UTC)


def by(stage: Stage, agent: str) -> CreatedBy:
    return CreatedBy(stage=stage, agent=agent)


# ---- Mock corpus ----------------------------------------------------------------------


def judgment(
    doc_id: str, title: str, court: str, level: CourtLevel, bench: int, date: str, cite: str
) -> Document:
    return Document(
        id=doc_id,
        kind=DocumentKind.JUDGMENT,
        title=f"[MOCK] {title}",
        court=f"[MOCK] {court}",
        court_level=level,
        bench_strength=bench,
        judges=["[MOCK] Justice A", "[MOCK] Justice B"][:bench] or ["[MOCK] Justice A"],
        decision_date=dt.date.fromisoformat(date),
        citations=[f"[MOCK] {cite}"],
        source_url=f"mock://judgments/{doc_id}",
        licence=LICENCE,
        corpus_snapshot=SNAPSHOT,
    )


def statute(doc_id: str, section: str, title: str) -> Document:
    return Document(
        id=doc_id,
        kind=DocumentKind.STATUTE,
        title=f"[MOCK] Indian Contract Act, 1872, s. {section} ({title}) - paraphrase",
        licence=LICENCE,
        corpus_snapshot=SNAPSHOT,
        source_url=f"mock://statutes/ica-1872/s{section}",
    )


SC, HC = CourtLevel.SUPREME_COURT, CourtLevel.HIGH_COURT
DOCS = {
    d.id: d
    for d in [
        judgment(
            "MOCK-SC-1962-01",
            "Bharat Cotton Mills v. Lakshmi Traders",
            "Supreme Court of India",
            SC,
            3,
            "1962-04-11",
            "[1962] 9 S.C.R. 101",
        ),
        judgment(
            "MOCK-SC-1978-02",
            "Union Transport Co. v. Meera Devi",
            "Supreme Court of India",
            SC,
            2,
            "1978-08-02",
            "[1978] 3 S.C.R. 455",
        ),
        judgment(
            "MOCK-SC-1994-03",
            "Deccan Infrastructure Ltd. v. State Supply Corp.",
            "Supreme Court of India",
            SC,
            3,
            "1994-02-17",
            "[1994] 1 S.C.R. 812",
        ),
        judgment(
            "MOCK-SC-2003-04",
            "Anand Engineering v. Rao Constructions",
            "Supreme Court of India",
            SC,
            2,
            "2003-11-05",
            "[2003] 5 S.C.R. 230",
        ),
        judgment(
            "MOCK-SC-2011-05",
            "Sunrise Housing v. Kavita Sharma",
            "Supreme Court of India",
            SC,
            2,
            "2011-07-21",
            "[2011] 8 S.C.R. 640",
        ),
        judgment(
            "MOCK-SC-2016-06",
            "Horizon Power Ltd. v. Western Coal Suppliers",
            "Supreme Court of India",
            SC,
            2,
            "2016-03-09",
            "2016 INSC 9001",
        ),
        judgment(
            "MOCK-SC-2019-07",
            "Metro Rail Corp. v. Prakash Builders",
            "Supreme Court of India",
            SC,
            3,
            "2019-10-14",
            "2019 INSC 9002",
        ),
        judgment(
            "MOCK-DHC-2020-08",
            "Northstar Logistics v. Gupta Warehousing",
            "High Court of Delhi",
            HC,
            1,
            "2020-12-03",
            "2020 DHC 9003",
        ),
        judgment(
            "MOCK-BHC-2021-09",
            "Konkan Shipping v. Patil Exports",
            "Bombay High Court",
            HC,
            2,
            "2021-06-25",
            "2021 BHC 9004",
        ),
        statute("MOCK-ICA-S14", "14", "free consent"),
        statute("MOCK-ICA-S15", "15", "coercion"),
        statute("MOCK-ICA-S19", "19", "voidability of agreements without free consent"),
    ]
}

# (doc_id, first paragraph, text) per evidence item, E-001 ... E-020.
PASSAGES: list[tuple[str, int | None, str, list[str]]] = [
    (
        "MOCK-ICA-S15",
        None,
        "Paraphrase: coercion is committing, or threatening to commit, any act forbidden by the Indian Penal Code, or unlawfully detaining or threatening to detain property, to the prejudice of any person, with the intention of causing any person to enter into an agreement.",
        ["I-1"],
    ),
    (
        "MOCK-ICA-S14",
        None,
        "Paraphrase: consent is free when it is not caused by coercion, undue influence, fraud, misrepresentation or mistake, and consent is so caused when it would not have been given but for that factor.",
        ["I-1"],
    ),
    (
        "MOCK-ICA-S19",
        None,
        "Paraphrase: when consent to an agreement is caused by coercion, fraud or misrepresentation, the agreement is a contract voidable at the option of the party whose consent was so caused.",
        ["I-2"],
    ),
    (
        "MOCK-SC-1962-01",
        9,
        "9. The word coercion in section 15 is not a term of art borrowed from English law. The Act defines it exhaustively. A threat to do something the party is legally entitled to do, such as declining to renew a supply arrangement, is not an act forbidden by the Penal Code and does not by itself amount to coercion.",
        ["I-1"],
    ),
    (
        "MOCK-SC-1962-01",
        12,
        "12. What must be shown is a nexus between the unlawful act or threat and the consent. Consent that would have been given in any event is not caused by coercion within the meaning of section 14.",
        ["I-1"],
    ),
    (
        "MOCK-SC-1978-02",
        7,
        "7. A threat to set the criminal law in motion on a false accusation, made to extract a promise to pay, is a threat to commit an act forbidden by the Penal Code. The resulting promise is not freely given.",
        ["I-1"],
    ),
    (
        "MOCK-SC-1994-03",
        21,
        "21. Commercial pressure is a normal incident of bargaining. Hard bargaining, or insistence on strict contractual rights, does not become coercion merely because one party is in a weaker position.",
        ["I-1"],
    ),
    (
        "MOCK-SC-1994-03",
        24,
        "24. The line is crossed where a party uses an unlawful means, such as detaining goods it has no right to detain, to compel the other side to accept terms it would otherwise have rejected.",
        ["I-1"],
    ),
    (
        "MOCK-SC-2003-04",
        15,
        "15. A party who, after the pressure has ceased, accepts the benefit of the agreement and allows a considerable time to pass without protest may be held to have affirmed it. The right to avoid the contract is then lost.",
        ["I-2"],
    ),
    (
        "MOCK-SC-2003-04",
        18,
        "18. Affirmation is a question of fact. Acceptance of a payment while the financial distress that caused the agreement continues will not ordinarily amount to affirmation.",
        ["I-2"],
    ),
    (
        "MOCK-SC-2011-05",
        31,
        "31. A no-dues certificate signed by a contractor who has no real choice, because the release of admitted amounts is made conditional on signing it, does not by itself bar a later claim.",
        ["I-1", "I-2"],
    ),
    (
        "MOCK-SC-2011-05",
        34,
        "34. The contractor must still show the circumstances in which the certificate was signed. A bare assertion of pressure, made long after the event, will not suffice.",
        ["I-2"],
    ),
    (
        "MOCK-SC-2016-06",
        40,
        "40. The party alleging coercion must plead its particulars. The date of the threat, the person who made it, and the manner in which consent was extracted must appear in the pleading, and the burden of proving them rests on that party.",
        ["I-1"],
    ),
    (
        "MOCK-SC-2019-07",
        27,
        "27. The decision in Deccan Infrastructure turned on an unlawful detention of goods. Where the payer withholds money in a genuine dispute about quantum, the withholding is not unlawful and the settlement is not vitiated.",
        ["I-1"],
    ),
    (
        "MOCK-SC-2019-07",
        29,
        "29. We follow the principle that commercial pressure is not coercion, while making clear that withholding amounts admitted to be due stands on a different footing.",
        ["I-1"],
    ),
    (
        "MOCK-DHC-2020-08",
        18,
        "18. Withholding amounts that the payer itself admits are due, in order to force the payee to accept a lower settlement, is an unlawful detention of the payee's property. Consent to such a settlement is caused by coercion.",
        ["I-1"],
    ),
    (
        "MOCK-DHC-2020-08",
        22,
        "22. The settlement was repudiated within three weeks of the payment being released. Acceptance of the payment in these circumstances cannot be treated as affirmation.",
        ["I-2"],
    ),
    (
        "MOCK-BHC-2021-09",
        11,
        "11. The appellant waited eleven months after receiving the settlement amount before seeking to avoid the agreement. Such delay, with full knowledge of the facts, is strong evidence of affirmation.",
        ["I-2"],
    ),
    (
        "MOCK-BHC-2021-09",
        14,
        "14. We do not lay down that delay by itself defeats the right to rescind. Delay is one circumstance among others, to be weighed with the conduct of the parties.",
        ["I-2"],
    ),
    (
        "MOCK-SC-1978-02",
        9,
        "9. The court will look at the substance of the transaction. A promise obtained while the promisor was in fear of an unlawful act is voidable at his option.",
        ["I-2"],
    ),
]

QUESTION = "Can a contract be enforced when consent was obtained through coercion?"
FACTS = [
    "A contractor (the claimant) completed works for a developer (the respondent).",
    "The respondent admitted that Rs 48 lakh was due but withheld payment for five months.",
    "The claimant signed a settlement accepting Rs 30 lakh 'in full and final settlement'.",
    "The Rs 30 lakh was paid; the claimant repudiated the settlement four months later.",
]
ASSUMPTIONS = [
    "The claimant's pleadings describe how and when the pressure was applied.",
    "No arbitration clause governs the dispute.",
]


class Run:
    """Emits one run with a deterministic clock and accumulated agent costs."""

    def __init__(self, run_id: str, case_id: str) -> None:
        self.store = InMemoryEventStore()
        ticks = count()
        self.emitter = EventEmitter(
            self.store,
            run_id=run_id,
            case_id=case_id,
            clock=lambda: T0 + dt.timedelta(seconds=3 * next(ticks)),
        )
        self.totals = RunTotals()

    def e(self, payload: JurisModel, stage: Stage | None = None, agent: str | None = None) -> None:
        self.emitter.emit(payload, stage=stage, agent=agent)

    def agent(
        self, stage: Stage, agent: str, role: str, tokens_in: int, tokens_out: int
    ) -> Iterator[None]:
        self.e(c.AgentStartedPayload(role=role, model=MODEL), stage, agent)  # type: ignore[arg-type]
        yield
        usd = (tokens_in * PRICE.input + tokens_out * PRICE.output) / 1_000_000
        self.totals = RunTotals(
            llm_calls=self.totals.llm_calls + 1,
            input_tokens=self.totals.input_tokens + tokens_in,
            output_tokens=self.totals.output_tokens + tokens_out,
            usd=round(self.totals.usd + usd, 6),
        )
        self.e(
            c.AgentCompletedPayload(
                role=role, input_tokens=tokens_in, output_tokens=tokens_out, cost_usd=round(usd, 6)
            ),  # type: ignore[arg-type]
            stage,
            agent,
        )

    def stage(self, stage: Stage) -> None:
        self.e(c.StageStartedPayload(), stage)

    def done(self, stage: Stage, outcome: str = "completed", reason: str | None = None) -> None:
        self.e(c.StageCompletedPayload(outcome=outcome, reason=reason), stage)  # type: ignore[arg-type]

    def events(self) -> list[Event]:
        return self.store.read(self.emitter.run_id)


def _run_agent(
    run: Run, stage: Stage, agent: str, role: str, tin: int, tout: int
) -> Iterator[None]:
    return run.agent(stage, agent, role, tin, tout)


def evidence_items() -> tuple[list[c.EvidenceRegisteredPayload], dict[str, str]]:
    """E-001..E-020 with document, chunk and evidence. Returns payloads and chunk text by E-ID."""
    payloads, text = [], {}
    for n, (doc_id, para, body, issues) in enumerate(PASSAGES, start=1):
        eid = f"E-{n:03d}"
        doc = DOCS[doc_id]
        chunk = Chunk(
            id=f"{doc_id}#p{para}" if para is not None else f"{doc_id}#s",
            document_id=doc_id,
            text=body,
            para_start=para,
            para_end=para,
            char_start=1000 * (para or 0),
            char_end=1000 * (para or 0) + len(body),
            statute=None
            if para is not None
            else StatuteMeta(
                act="contract_act",
                section=doc_id.rsplit("S", 1)[1],
                in_force_from=dt.date(1872, 9, 1),
            ),
        )
        ev = Evidence(
            id=eid,
            chunk_id=chunk.id,
            document_id=doc_id,
            issue_ids=issues,
            retrieval_score=round(0.82 - n * 0.013, 3),
            rerank_score=round(0.95 - n * 0.02, 3),
            retrieved_by_query="coercion consent section 15 settlement"
            if "I-1" in issues
            else "voidable affirmation delay section 19",
            registered_at_stage=Stage.S1,
            created_by=by(Stage.S1, "researcher"),
        )
        payloads.append(c.EvidenceRegisteredPayload(evidence=ev, document=doc, chunk=chunk))
        text[eid] = body
    return payloads, text


def build_full() -> list[Event]:
    run = Run("run_mock_coercion_full", "CASE-MOCK-001")
    S = Stage
    run.e(
        c.CaseCreatedPayload(question=QUESTION, profile="juris_full", corpus_snapshot_id=SNAPSHOT)
    )

    # S0: issue framing
    issues = [
        Issue(
            id="I-1",
            question="Was the claimant's consent to the settlement caused by coercion within s. 15 of the Indian Contract Act?",
            positions=[
                Position(
                    id="I-1.A",
                    statement="Yes: withholding admitted dues to force a lower settlement is an unlawful detention of property, so consent was caused by coercion.",
                ),
                Position(
                    id="I-1.B",
                    statement="No: this was commercial pressure in a dispute about money, not an act forbidden by law, so consent was free.",
                ),
            ],
            created_by=by(S.S0, "issue_framer"),
        ),
        Issue(
            id="I-2",
            question="If consent was caused by coercion, can the claimant still avoid the settlement after accepting payment and waiting four months?",
            positions=[
                Position(
                    id="I-2.A",
                    statement="Yes: the settlement is voidable under s. 19, and accepting payment while in financial distress is not affirmation.",
                ),
                Position(
                    id="I-2.B",
                    statement="No: by accepting the payment and delaying, the claimant affirmed the settlement and lost the right to avoid it.",
                ),
            ],
            created_by=by(S.S0, "issue_framer"),
        ),
    ]
    run.stage(S.S0)
    for _ in run.agent(S.S0, "issue_framer", "issue_framer", 1_800, 900):
        run.e(
            c.IssueFramedPayload(issue=issues[0], facts=FACTS, assumptions=ASSUMPTIONS),
            S.S0,
            "issue_framer",
        )
        run.e(c.IssueFramedPayload(issue=issues[1]), S.S0, "issue_framer")
    run.done(S.S0)

    # S1: research
    run.stage(S.S1)
    payloads, text = evidence_items()
    for _ in run.agent(S.S1, "researcher", "researcher", 14_000, 2_200):
        for q, iss in [
            ("section 15 coercion unlawful detention of property", ["I-1"]),
            ("withholding admitted dues settlement coercion", ["I-1"]),
            ("commercial pressure not coercion", ["I-1"]),
            ("voidable contract affirmation delay section 19", ["I-2"]),
        ]:
            run.e(c.QueryIssuedPayload(query=q, issue_ids=iss), S.S1, "researcher")
        for p in payloads:
            run.e(p, S.S1, "researcher")
    run.done(S.S1)

    def cite(
        claim: str, eid: str, quote: str, *, pinpoint: str | None = None, check: bool = True
    ) -> Citation:
        if check:
            assert quote in text[eid], f"{eid}: quote is not in the chunk"
        para = PASSAGES[int(eid[2:]) - 1][1]
        return Citation(
            claim_id=claim,
            evidence_id=eid,
            pinpoint=pinpoint or (f"para {para}" if para else "section text"),
            quote=quote,
        )

    def claim(cid: str, pos: str, author: str, stage: Stage, body: str) -> Claim:
        return Claim(
            id=cid, text=body, position_id=pos, author_agent=author, created_by=by(stage, author)
        )

    # S2: opening arguments
    claims_a = [
        claim(
            "C-001",
            "I-1.A",
            "counsel_a",
            S.S2,
            "Withholding amounts the respondent admitted were due, to force a lower settlement, is an unlawful detention of property and so coercion under s. 15.",
        ),
        claim(
            "C-002",
            "I-1.A",
            "counsel_a",
            S.S2,
            "The respondent's threat to blacklist the claimant was a threat to commit an act forbidden by the Penal Code.",
        ),
        claim(
            "C-005",
            "I-2.A",
            "counsel_a",
            S.S2,
            "Consent caused by coercion makes the settlement voidable at the claimant's option under s. 19.",
        ),
        claim(
            "C-006",
            "I-2.A",
            "counsel_a",
            S.S2,
            "Signing a 'full and final settlement' under pressure does not by itself bar the claim.",
        ),
    ]
    claims_b = [
        claim(
            "C-003",
            "I-1.B",
            "counsel_b",
            S.S2,
            "Commercial pressure in a money dispute is not coercion unless the act is forbidden by law.",
        ),
        claim(
            "C-004",
            "I-1.B",
            "counsel_b",
            S.S2,
            "The claimant has not pleaded the particulars of the alleged coercion.",
        ),
        claim(
            "C-007",
            "I-2.B",
            "counsel_b",
            S.S2,
            "By accepting Rs 30 lakh and waiting four months, the claimant affirmed the settlement.",
        ),
        claim(
            "C-008",
            "I-2.B",
            "counsel_b",
            S.S2,
            "Delay alone defeats the right to rescind a voidable contract.",
        ),
    ]
    args_a = [
        Argument(
            id="A-001",
            claim_ids=["C-001"],
            reasoning="S. 15 covers unlawful detention of property; admitted dues belong to the claimant, and the only authority on admitted dues treats withholding them as coercion.",
            citations=[
                cite("C-001", "E-001", "unlawfully detaining or threatening to detain property"),
                cite("C-001", "E-016", "is an unlawful detention of the payee's property"),
                cite(
                    "C-001",
                    "E-011",
                    "the release of admitted amounts is made conditional on signing it",
                ),
            ],
            created_by=by(S.S2, "counsel_a"),
        ),
        Argument(
            id="A-002",
            claim_ids=["C-002"],
            reasoning="A threat of adverse action to extract a promise is coercion.",
            citations=[
                cite(
                    "C-002",
                    "E-006",
                    "A threat to set the criminal law in motion on a false accusation",
                )
            ],
            created_by=by(S.S2, "counsel_a"),
        ),
        Argument(
            id="A-003",
            claim_ids=["C-005", "C-006"],
            reasoning="S. 19 makes the settlement voidable; a no-dues or settlement document signed without real choice is not a bar.",
            citations=[
                cite(
                    "C-005",
                    "E-003",
                    "the agreement is a contract voidable at the option of the party whose consent was so caused",
                ),
                cite("C-006", "E-011", "does not by itself bar a later claim"),
            ],
            created_by=by(S.S2, "counsel_a"),
        ),
    ]
    args_b = [
        Argument(
            id="A-004",
            claim_ids=["C-003"],
            reasoning="The definition is exhaustive; hard bargaining is not an act forbidden by law.",
            citations=[
                cite("C-003", "E-004", "does not by itself amount to coercion"),
                cite("C-003", "E-007", "Commercial pressure is a normal incident of bargaining"),
            ],
            created_by=by(S.S2, "counsel_b"),
        ),
        Argument(
            id="A-005",
            claim_ids=["C-004"],
            reasoning="Particulars of coercion must be pleaded and proved by the party alleging it.",
            citations=[
                cite("C-004", "E-013", "The party alleging coercion must plead its particulars")
            ],
            created_by=by(S.S2, "counsel_b"),
        ),
        Argument(
            id="A-006",
            claim_ids=["C-007", "C-008"],
            reasoning="Acceptance of the benefit plus delay is affirmation; the delay here is long.",
            citations=[
                cite("C-007", "E-009", "may be held to have affirmed it"),
                cite("C-007", "E-018", "is strong evidence of affirmation"),
                cite("C-008", "E-018", "delay by itself defeats the right to rescind", check=False),
            ],
            created_by=by(S.S2, "counsel_b"),
        ),
    ]
    run.stage(S.S2)
    for agent_id, claims, args in (
        ("counsel_a", claims_a, args_a),
        ("counsel_b", claims_b, args_b),
    ):
        for _ in run.agent(S.S2, agent_id, "counsel", 22_000, 3_100):
            for cl in claims:
                run.e(c.ClaimCreatedPayload(claim=cl), S.S2, agent_id)
            for ar in args:
                run.e(c.ArgumentCreatedPayload(argument=ar), S.S2, agent_id)
    run.done(S.S2)

    # S3: precedent analysis
    cards = [
        PrecedentCard(
            document_id="MOCK-SC-1962-01",
            principle="Coercion under s. 15 is defined exhaustively; a threat to do what one is legally entitled to do is not coercion.",
            material_facts="Supplier declined to renew a supply arrangement unless prices were revised.",
            distinguishing_factors=["No admitted debt was withheld"],
            treatment=[
                TreatmentSignal(
                    kind=Treatment.FOLLOWED,
                    by_document_id="MOCK-SC-1994-03",
                    cue_text="the exhaustive definition ... was followed",
                )
            ],
            treatment_verified=True,
            fit_by_argument={
                "A-004": PrecedentFit.DIRECTLY_SUPPORTS,
                "A-001": PrecedentFit.DISTINGUISHABLE,
            },
            created_by=by(S.S3, "precedent_analyst"),
        ),
        PrecedentCard(
            document_id="MOCK-SC-1978-02",
            principle="A threat to prosecute on a false accusation to extract a promise is coercion.",
            material_facts="Promise to pay obtained under threat of a false criminal complaint.",
            distinguishing_factors=[
                "Threat of criminal proceedings, not of commercial blacklisting"
            ],
            treatment=[],
            treatment_verified=True,
            fit_by_argument={"A-002": PrecedentFit.DOES_NOT_SUPPORT},
            created_by=by(S.S3, "precedent_analyst"),
        ),
        PrecedentCard(
            document_id="MOCK-SC-1994-03",
            principle="Commercial pressure is not coercion; the line is crossed only by unlawful means such as unlawful detention of goods.",
            material_facts="State corporation detained goods to force acceptance of revised terms.",
            distinguishing_factors=["Goods detained, not money withheld"],
            treatment=[
                TreatmentSignal(
                    kind=Treatment.DISTINGUISHED,
                    by_document_id="MOCK-SC-2019-07",
                    cue_text="The decision in Deccan Infrastructure turned on an unlawful detention of goods",
                )
            ],
            treatment_verified=True,
            fit_by_argument={
                "A-004": PrecedentFit.DIRECTLY_SUPPORTS,
                "A-001": PrecedentFit.SUPPORTS_BY_ANALOGY,
            },
            created_by=by(S.S3, "precedent_analyst"),
        ),
        PrecedentCard(
            document_id="MOCK-SC-2003-04",
            principle="Accepting the benefit and delaying after the pressure has ceased may amount to affirmation; acceptance during continuing distress ordinarily does not.",
            material_facts="Contractor accepted a reduced final bill and sued fourteen months later.",
            distinguishing_factors=["Pressure had ended long before suit"],
            treatment=[],
            treatment_verified=False,
            fit_by_argument={"A-006": PrecedentFit.SUPPORTS_BY_ANALOGY},
            created_by=by(S.S3, "precedent_analyst"),
        ),
        PrecedentCard(
            document_id="MOCK-SC-2011-05",
            principle="A no-dues certificate signed without real choice does not bar a later claim, but the circumstances must be proved.",
            material_facts="Release of admitted dues conditional on signing a no-dues certificate.",
            distinguishing_factors=[],
            treatment=[
                TreatmentSignal(
                    kind=Treatment.FOLLOWED,
                    by_document_id="MOCK-DHC-2020-08",
                    cue_text="following Sunrise Housing",
                )
            ],
            treatment_verified=True,
            fit_by_argument={
                "A-001": PrecedentFit.DIRECTLY_SUPPORTS,
                "A-003": PrecedentFit.SUPPORTS_BY_ANALOGY,
            },
            created_by=by(S.S3, "precedent_analyst"),
        ),
        PrecedentCard(
            document_id="MOCK-SC-2016-06",
            principle="Particulars of coercion must be pleaded and proved by the party alleging it.",
            material_facts="Supplier alleged coercion for the first time in evidence.",
            distinguishing_factors=[],
            treatment=[],
            treatment_verified=True,
            fit_by_argument={"A-005": PrecedentFit.DIRECTLY_SUPPORTS},
            created_by=by(S.S3, "precedent_analyst"),
        ),
        PrecedentCard(
            document_id="MOCK-DHC-2020-08",
            principle="Withholding admitted dues to force a lower settlement is unlawful detention of property, so consent is caused by coercion.",
            material_facts="Payer admitted the amount due and released it only against a reduced settlement.",
            distinguishing_factors=[
                "High Court decision: persuasive, not binding on the Supreme Court"
            ],
            treatment=[],
            treatment_verified=True,
            fit_by_argument={"A-001": PrecedentFit.DIRECTLY_SUPPORTS},
            created_by=by(S.S3, "precedent_analyst"),
        ),
        PrecedentCard(
            document_id="MOCK-BHC-2021-09",
            principle="Delay with full knowledge is strong evidence of affirmation, but delay alone does not defeat the right to rescind.",
            material_facts="Appellant waited eleven months after receiving the settlement amount.",
            distinguishing_factors=["Eleven months' delay versus four months here"],
            treatment=[],
            treatment_verified=True,
            fit_by_argument={"A-006": PrecedentFit.SUPPORTS_BY_ANALOGY},
            created_by=by(S.S3, "precedent_analyst"),
        ),
    ]
    run.stage(S.S3)
    for _ in run.agent(S.S3, "precedent_analyst", "precedent_analyst", 30_000, 4_800):
        for card in cards:
            run.e(c.PrecedentCardCreatedPayload(card=card), S.S3, "precedent_analyst")
    run.done(S.S3)

    # S4: verification gate #1
    def verdict(
        status: VerificationStatus,
        label: EntailmentLabel | None,
        why: str,
        stage: Stage = S.S4,
        check: VerificationCheck = VerificationCheck.ENTAILMENT,
    ) -> VerificationResult:
        return VerificationResult(
            status=status,
            check=check,
            label=label,
            justification=why,
            verifier_model=MODEL if check is VerificationCheck.ENTAILMENT else None,
            created_by=by(stage, "verifier"),
        )

    V, W, U, INV = (
        VerificationStatus.VERIFIED,
        VerificationStatus.WEAK,
        VerificationStatus.UNSUPPORTED,
        VerificationStatus.INVALID,
    )
    SUP, PART, NOT = (
        EntailmentLabel.SUPPORTS,
        EntailmentLabel.PARTIALLY_SUPPORTS,
        EntailmentLabel.DOES_NOT_SUPPORT,
    )
    gate1 = [
        ("C-001", "E-001", verdict(V, SUP, "The section covers unlawful detention of property.")),
        ("C-001", "E-016", verdict(V, SUP, "States the proposition directly.")),
        (
            "C-001",
            "E-011",
            verdict(
                W, PART, "Supports that conditional release is pressure, not that it is coercion."
            ),
        ),
        (
            "C-002",
            "E-006",
            verdict(U, NOT, "Concerns a threat of criminal prosecution, not blacklisting."),
        ),
        ("C-005", "E-003", verdict(V, SUP, "Paraphrase of s. 19 states voidability directly.")),
        (
            "C-006",
            "E-011",
            verdict(W, PART, "Concerns a no-dues certificate, not a settlement agreement."),
        ),
        ("C-003", "E-004", verdict(V, SUP, "States that lawful threats are not coercion.")),
        ("C-003", "E-007", verdict(V, SUP, "States that commercial pressure is not coercion.")),
        ("C-004", "E-013", verdict(V, SUP, "States the pleading requirement directly.")),
        (
            "C-007",
            "E-009",
            verdict(V, SUP, "States that acceptance plus delay may be affirmation."),
        ),
        ("C-007", "E-018", verdict(W, PART, "Eleven months' delay; the facts here involve four.")),
        (
            "C-008",
            "E-018",
            verdict(
                INV,
                None,
                "The quoted words do not appear in the chunk.",
                check=VerificationCheck.QUOTE,
            ),
        ),
    ]
    status_after_gate1 = {
        "C-001": ClaimStatus.VERIFIED, "C-002": ClaimStatus.UNSUPPORTED, "C-003": ClaimStatus.VERIFIED,
        "C-004": ClaimStatus.VERIFIED, "C-005": ClaimStatus.VERIFIED, "C-006": ClaimStatus.WEAK,
        "C-007": ClaimStatus.VERIFIED, "C-008": ClaimStatus.UNSUPPORTED,
    }  # fmt: skip
    run.stage(S.S4)
    for _ in run.agent(S.S4, "verifier", "verifier", 18_000, 2_400):
        for cid, eid, result in gate1:
            run.e(
                c.VerificationResultPayload(claim_id=cid, evidence_id=eid, result=result),
                S.S4,
                "verifier",
            )
    for cid, st in status_after_gate1.items():
        run.e(
            c.ClaimStatusChangedPayload(claim_id=cid, status=st, reason="Verification gate #1"),
            S.S4,
            "case_record",
        )
    run.done(S.S4)

    # S5: rebuttal round
    rebuttals = [
        (
            "counsel_b",
            claim(
                "C-009",
                "I-1.B",
                "counsel_b",
                S.S5,
                "The withholding arose from a genuine dispute about quantum, so it was not unlawful.",
            ),
            Argument(
                id="A-007",
                claim_ids=["C-009"],
                reasoning="The later Supreme Court decision confines the unlawful-detention line to goods and to amounts admitted due; the respondent disputed quantum.",
                citations=[
                    cite(
                        "C-009",
                        "E-014",
                        "Where the payer withholds money in a genuine dispute about quantum, the withholding is not unlawful",
                    )
                ],
                created_by=by(S.S5, "counsel_b"),
            ),
            Counterargument(
                id="R-001", targets="C-001", argument_id="A-007", created_by=by(S.S5, "counsel_b")
            ),
        ),
        (
            "counsel_a",
            claim(
                "C-010",
                "I-2.A",
                "counsel_a",
                S.S5,
                "Accepting payment while the financial distress continued is not affirmation.",
            ),
            Argument(
                id="A-008",
                claim_ids=["C-010"],
                reasoning="The claimant was still in distress when paid, and the authority on affirmation says acceptance in continuing distress is not affirmation.",
                citations=[
                    cite("C-010", "E-010", "will not ordinarily amount to affirmation"),
                    cite("C-010", "E-017", "cannot be treated as affirmation"),
                ],
                created_by=by(S.S5, "counsel_a"),
            ),
            Counterargument(
                id="R-002", targets="C-007", argument_id="A-008", created_by=by(S.S5, "counsel_a")
            ),
        ),
    ]
    run.stage(S.S5)
    for agent_id, cl, ar, rb in rebuttals:
        for _ in run.agent(S.S5, agent_id, "counsel", 26_000, 2_000):
            run.e(c.ClaimCreatedPayload(claim=cl), S.S5, agent_id)
            run.e(c.ArgumentCreatedPayload(argument=ar), S.S5, agent_id)
            run.e(c.CounterargumentCreatedPayload(counterargument=rb), S.S5, agent_id)
    run.done(S.S5)

    # S6: cross-examination
    objections = [
        Objection(
            id="O-001",
            type=ObjectionType.DISTINGUISHABLE_PRECEDENT,
            target_id="C-001",
            raised_by="cross_examiner",
            text="MOCK-DHC-2020-08 involved dues the payer admitted; the respondent says quantum was disputed. Is the admission established?",
            created_by=by(S.S6, "cross_examiner"),
        ),
        Objection(
            id="O-002",
            type=ObjectionType.UNSUPPORTED_CLAIM,
            target_id="C-002",
            raised_by="cross_examiner",
            text="C-002 cites a judgment about threats of prosecution; nothing in the record shows a threat of an act forbidden by law.",
            created_by=by(S.S6, "cross_examiner"),
        ),
        Objection(
            id="O-003",
            type=ObjectionType.OVERRULED_OR_DOUBTED,
            target_id="C-007",
            raised_by="cross_examiner",
            text="C-007 relies on MOCK-SC-2003-04, whose later treatment is not verified.",
            created_by=by(S.S6, "cross_examiner"),
        ),
        Objection(
            id="O-004",
            type=ObjectionType.FACT_ASSUMPTION,
            target_id="C-004",
            raised_by="cross_examiner",
            text="C-004 turns on the pleadings, which are not in the facts; the framing assumed they describe the pressure.",
            created_by=by(S.S6, "cross_examiner"),
        ),
    ]
    answers = [
        (
            "O-001",
            "counsel_a",
            "The respondent's own ledger entry records Rs 48 lakh as payable; that is an admission.",
            False,
        ),
        ("O-002", "counsel_a", "Conceded: C-002 is withdrawn.", True),
        ("O-003", "counsel_b", "No later decision in the corpus doubts MOCK-SC-2003-04.", False),
        (
            "O-004",
            "counsel_b",
            "The objection is noted; the pleadings are outside the record.",
            False,
        ),
    ]
    run.stage(S.S6)
    for _ in run.agent(S.S6, "cross_examiner", "cross_examiner", 40_000, 3_300):
        for ob in objections:
            run.e(c.ObjectionRaisedPayload(objection=ob), S.S6, "cross_examiner")
    for cid in ("C-001", "C-004", "C-007"):
        run.e(
            c.ClaimStatusChangedPayload(
                claim_id=cid, status=ClaimStatus.CONTESTED, reason="Open objection"
            ),
            S.S6,
            "case_record",
        )
    for oid, agent_id, response, resolved in answers:
        for _ in run.agent(S.S6, f"{agent_id}_reply_{oid}", "counsel", 6_000, 400):
            run.e(
                c.ObjectionAnsweredPayload(objection_id=oid, response=response, resolved=resolved),
                S.S6,
                agent_id,
            )
    for q, issue_id, oid in [
        (
            "Did the respondent admit that Rs 48 lakh was due, or was quantum genuinely disputed?",
            "I-1",
            "O-001",
        ),
        ("Has MOCK-SC-2003-04 been doubted or overruled by a later decision?", "I-2", "O-003"),
        ("Do the claimant's pleadings give particulars of the coercion?", "I-1", "O-004"),
    ]:
        run.e(
            c.UnresolvedQuestionAddedPayload(question=q, issue_id=issue_id, from_objection_id=oid),
            S.S6,
            "case_record",
        )
    run.e(
        c.ClaimStatusChangedPayload(
            claim_id="C-002", status=ClaimStatus.FALLS, reason="Withdrawn in reply to O-002"
        ),
        S.S6,
        "case_record",
    )
    run.done(S.S6)

    # S7: verification gate #2 (new material from S5)
    run.stage(S.S7)
    gate2 = [
        (
            "C-009",
            "E-014",
            verdict(V, SUP, "States the quantum-dispute distinction directly.", S.S7),
        ),
        ("C-010", "E-010", verdict(V, SUP, "States the continuing-distress rule directly.", S.S7)),
        ("C-010", "E-017", verdict(V, SUP, "Applies the same rule to a settlement.", S.S7)),
    ]
    for _ in run.agent(S.S7, "verifier", "verifier", 9_000, 900):
        for cid, eid, result in gate2:
            run.e(
                c.VerificationResultPayload(claim_id=cid, evidence_id=eid, result=result),
                S.S7,
                "verifier",
            )
    for cid in ("C-009", "C-010"):
        run.e(
            c.ClaimStatusChangedPayload(
                claim_id=cid, status=ClaimStatus.VERIFIED, reason="Verification gate #2"
            ),
            S.S7,
            "case_record",
        )
    run.e(
        c.ClaimStatusChangedPayload(
            claim_id="C-008", status=ClaimStatus.FALLS, reason="Its only citation is invalid"
        ),
        S.S7,
        "case_record",
    )
    run.done(S.S7)

    # S8: jury
    def rubric(e: int, p: int, ch: int, r: int) -> RubricScores:
        return RubricScores(
            evidence_strength=e, precedent_fit=p, counterargument_handling=ch, unresolved_risk=r
        )

    ballots = [
        JuryBallot(
            juror_id="J-1",
            per_position_scores={
                "I-1.A": rubric(4, 3, 3, 3),
                "I-1.B": rubric(3, 4, 3, 3),
                "I-2.A": rubric(4, 4, 4, 2),
                "I-2.B": rubric(3, 3, 2, 3),
            },
            justifications=[
                Justification(
                    text="I-1 turns on whether the dues were admitted (O-001); I-2.A is better supported after R-002.",
                    cited_ids=["O-001", "R-002", "E-010"],
                )
            ],
            created_by=by(S.S8, "juror_1"),
        ),
        JuryBallot(
            juror_id="J-2",
            per_position_scores={
                "I-1.A": rubric(3, 3, 3, 3),
                "I-1.B": rubric(4, 4, 3, 2),
                "I-2.A": rubric(4, 3, 4, 2),
                "I-2.B": rubric(3, 3, 2, 3),
            },
            justifications=[
                Justification(
                    text="C-009 narrows the admitted-dues authority; I-2 favours A.",
                    cited_ids=["C-009", "E-014", "C-010"],
                )
            ],
            created_by=by(S.S8, "juror_2"),
        ),
        JuryBallot(
            juror_id="J-3",
            per_position_scores={
                "I-1.A": rubric(5, 4, 3, 2),
                "I-1.B": rubric(2, 3, 3, 4),
                "I-2.A": rubric(4, 4, 3, 2),
                "I-2.B": rubric(3, 2, 2, 3),
            },
            justifications=[
                Justification(
                    text="E-016 is squarely on admitted dues; C-008 fell.",
                    cited_ids=["E-016", "C-008"],
                )
            ],
            created_by=by(S.S8, "juror_3"),
        ),
    ]
    run.stage(S.S8)
    for ballot in ballots:
        juror = ballot.created_by.agent
        for _ in run.agent(S.S8, juror, "juror", 35_000, 1_500):
            run.e(c.JuryBallotCastPayload(ballot=ballot), S.S8, juror)
    run.e(aggregate(ballots), S.S8, "case_record")
    run.done(S.S8)

    # S9 + S10: judge and final verification
    analysis = CaseAnalysis(
        question=QUESTION,
        facts=FACTS,
        assumptions=ASSUMPTIONS,
        issues=[
            IssueAnalysis(
                issue_id="I-1",
                issue_text=issues[0].question,
                positions=[
                    PositionAnalysis(
                        position_id="I-1.A",
                        summary="Withholding dues admitted to be payable, to force a lower settlement, can be unlawful detention of property and so coercion.",
                        supporting_authorities=["MOCK-DHC-2020-08", "MOCK-SC-2011-05"],
                        key_evidence=[
                            EvidenceRef(evidence_id="E-001", pinpoint="section text"),
                            EvidenceRef(evidence_id="E-016", pinpoint="para 18"),
                        ],
                    ),
                    PositionAnalysis(
                        position_id="I-1.B",
                        summary="Commercial pressure in a genuine quantum dispute is not coercion; the definition in s. 15 is exhaustive.",
                        supporting_authorities=[
                            "MOCK-SC-1962-01",
                            "MOCK-SC-1994-03",
                            "MOCK-SC-2019-07",
                        ],
                        key_evidence=[
                            EvidenceRef(evidence_id="E-004", pinpoint="para 9"),
                            EvidenceRef(evidence_id="E-007", pinpoint="para 21"),
                            EvidenceRef(evidence_id="E-014", pinpoint="para 27"),
                        ],
                    ),
                ],
                key_conflicts=[
                    KeyConflict(
                        authority_a="MOCK-DHC-2020-08",
                        authority_b="MOCK-SC-2019-07",
                        distinction="Both accept that withholding admitted dues differs from withholding disputed sums; they point different ways only on whether the amount was admitted.",
                        weightier="MOCK-SC-2019-07",
                        reason="Supreme Court, three-judge bench; the High Court decision is persuasive only.",
                    )
                ],
                leaning="balanced",  # type: ignore[arg-type]
                confidence="low",  # type: ignore[arg-type]
                confidence_reason="The answer depends on an unresolved fact: whether Rs 48 lakh was admitted as due (O-001).",
                unresolved_questions=[
                    "Did the respondent admit that Rs 48 lakh was due, or was quantum genuinely disputed?",
                    "Do the claimant's pleadings give particulars of the coercion?",
                ],
            ),
            IssueAnalysis(
                issue_id="I-2",
                issue_text=issues[1].question,
                positions=[
                    PositionAnalysis(
                        position_id="I-2.A",
                        summary="If coercion is shown, the settlement is voidable under s. 19, and accepting payment during continuing distress is not affirmation.",
                        supporting_authorities=["MOCK-SC-2003-04", "MOCK-DHC-2020-08"],
                        key_evidence=[
                            EvidenceRef(evidence_id="E-003", pinpoint="section text"),
                            EvidenceRef(evidence_id="E-010", pinpoint="para 18"),
                            EvidenceRef(evidence_id="E-017", pinpoint="para 22"),
                        ],
                    ),
                    PositionAnalysis(
                        position_id="I-2.B",
                        summary="Acceptance of the benefit and delay may amount to affirmation, though delay alone does not defeat rescission.",
                        supporting_authorities=["MOCK-SC-2003-04", "MOCK-BHC-2021-09"],
                        key_evidence=[
                            EvidenceRef(evidence_id="E-009", pinpoint="para 15"),
                            EvidenceRef(evidence_id="E-018", pinpoint="para 11"),
                        ],
                    ),
                ],
                key_conflicts=[],
                leaning="leaning_A",  # type: ignore[arg-type]
                confidence="moderate",  # type: ignore[arg-type]
                confidence_reason="Four months' delay during continuing distress fits the continuing-distress rule, but that rule rests on a precedent whose treatment is not verified.",
                unresolved_questions=[
                    "Has MOCK-SC-2003-04 been doubted or overruled by a later decision?"
                ],
            ),
        ],
        overall_summary=(
            "On I-1 the record is balanced. Section 15 covers unlawful detention of property [E-001], "
            "and withholding admitted dues to force a lower settlement has been treated as coercion [E-016], "
            "but commercial pressure in a genuine dispute about quantum is not coercion [E-007, E-014]. "
            "The outcome turns on whether the Rs 48 lakh was admitted as due, which the record does not settle. "
            "On I-2, if coercion is shown the settlement is voidable at the claimant's option [E-003], "
            "and accepting payment while the distress continued is ordinarily not affirmation [E-010, E-017]; "
            "four months' delay is weaker evidence of affirmation than the eleven months in the contrary authority [E-018]. "
            "The claim that a blacklisting threat was a criminal act fell for want of support."
        ),
        limitations=[
            "MOCK DATA: every authority in this analysis is fictitious and exists only for UI development.",
            "Treatment of MOCK-SC-2003-04 is not verified; later decisions may have doubted it.",
            "The pleadings are not in the record; the analysis assumes they describe the pressure.",
        ],
        sources=[
            SourceRef(
                document_id=d,
                court=DOCS[d].court,
                date=DOCS[d].decision_date,
                citation=DOCS[d].citations[0] if DOCS[d].citations else None,
                pinpoints_used=p,
                link=DOCS[d].source_url,
            )
            for d, p in [
                ("MOCK-ICA-S15", ["section text"]),
                ("MOCK-ICA-S19", ["section text"]),
                ("MOCK-SC-1962-01", ["para 9"]),
                ("MOCK-SC-1994-03", ["para 21"]),
                ("MOCK-SC-2003-04", ["para 15", "para 18"]),
                ("MOCK-SC-2019-07", ["para 27"]),
                ("MOCK-DHC-2020-08", ["para 18", "para 22"]),
                ("MOCK-BHC-2021-09", ["para 11"]),
            ]
        ],
        disclaimer=DISCLAIMER,
        created_by=by(S.S9, "judge"),
    )
    run.stage(S.S9)
    for _ in run.agent(S.S9, "judge", "judge", 60_000, 5_500):
        run.e(c.AnalysisReadyPayload(analysis=analysis), S.S9, "judge")
    run.done(S.S9)
    run.stage(S.S10)
    for _ in run.agent(S.S10, "verifier", "verifier", 8_000, 700):
        pass
    for cid, st in [
        ("C-003", ClaimStatus.SURVIVES),
        ("C-005", ClaimStatus.SURVIVES),
        ("C-009", ClaimStatus.SURVIVES),
        ("C-010", ClaimStatus.SURVIVES),
    ]:
        run.e(
            c.ClaimStatusChangedPayload(
                claim_id=cid, status=st, reason="Relied on in the final analysis"
            ),
            S.S10,
            "case_record",
        )
    run.done(S.S10)
    run.e(c.RunCompletedPayload(totals=run.totals))
    return run.events()


def aggregate(ballots: list[JuryBallot]) -> c.JuryAggregatedPayload:
    """Deterministic jury aggregate: mean and spread per position and criterion."""
    per_position: dict[str, dict[str, c.ScoreStats]] = {}
    contested: set[str] = set()
    for pos in ballots[0].per_position_scores:
        stats = {}
        for crit in RubricScores.model_fields:
            scores = [getattr(b.per_position_scores[pos], crit) for b in ballots]
            spread = max(scores) - min(scores)
            stats[crit] = c.ScoreStats(mean=round(sum(scores) / len(scores), 3), spread=spread)
            if spread >= 2:
                contested.add(pos.split(".")[0])
        per_position[pos] = stats
    return c.JuryAggregatedPayload(per_position=per_position, contested_issue_ids=sorted(contested))


def build_failure() -> list[Event]:
    """A run that hits validation failures and runs out of budget before the Judge."""
    run = Run("run_mock_failure", "CASE-MOCK-002")
    S = Stage
    run.e(
        c.CaseCreatedPayload(question=QUESTION, profile="juris_full", corpus_snapshot_id=SNAPSHOT)
    )
    run.stage(S.S0)
    issue = Issue(
        id="I-1",
        question="Was consent to the settlement caused by coercion within s. 15?",
        positions=[
            Position(id="I-1.A", statement="Yes: coercion."),
            Position(id="I-1.B", statement="No: commercial pressure."),
        ],
        created_by=by(S.S0, "issue_framer"),
    )
    for _ in run.agent(S.S0, "issue_framer", "issue_framer", 1_700, 700):
        run.e(
            c.IssueFramedPayload(issue=issue, facts=FACTS[:3], assumptions=ASSUMPTIONS[:1]),
            S.S0,
            "issue_framer",
        )
    run.done(S.S0)
    run.stage(S.S1)
    payloads, _ = evidence_items()
    for _ in run.agent(S.S1, "researcher", "researcher", 9_000, 1_200):
        run.e(
            c.QueryIssuedPayload(query="section 15 coercion settlement", issue_ids=["I-1"]),
            S.S1,
            "researcher",
        )
        for p in (payloads[0], payloads[6], payloads[15]):
            run.e(p, S.S1, "researcher")
    run.done(S.S1)
    run.stage(S.S2)
    for _ in run.agent(S.S2, "counsel_a", "counsel", 20_000, 2_600):
        run.e(
            c.ValidationFailedPayload(
                errors=["Argument A-001 cites evidence E-099, which is not registered"]
            ),
            S.S2,
            "counsel_a",
        )
        run.e(
            c.ValidationFailedPayload(
                errors=["Repair attempt still cites E-099"],
                dropped="Argument A-001 and claim C-001",
            ),
            S.S2,
            "counsel_a",
        )
    claim_b = Claim(
        id="C-002",
        text="Commercial pressure is not coercion.",
        position_id="I-1.B",
        author_agent="counsel_b",
        created_by=by(S.S2, "counsel_b"),
    )
    for _ in run.agent(S.S2, "counsel_b", "counsel", 19_000, 2_300):
        run.e(c.ClaimCreatedPayload(claim=claim_b), S.S2, "counsel_b")
    run.done(S.S2)
    run.e(
        c.BudgetWarningPayload(
            tokens_used=76_500,
            tokens_budget=80_000,
            usd_used=round(run.totals.usd, 6),
            usd_budget=0.5,
            action="Skipping S3 (precedent analysis) and S5 (rebuttal)",
        ),
        S.S3,
        "orchestrator",
    )
    run.done(S.S3, "skipped", "budget")
    run.done(S.S5, "skipped", "budget")
    run.e(
        c.RunFailedPayload(
            reason="Budget exhausted before S9 (Judge); the partial record is preserved."
        )
    )
    return run.events()


def write(name: str, events: list[Event], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = "".join(e.model_dump_json() + "\n" for e in events)
    (out_dir / f"{name}.jsonl").write_text(lines, encoding="utf-8", newline="\n")
    view: CaseView = fold(events)
    (out_dir / f"{name}.caseview.json").write_text(
        view.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


FIXTURES = {"coercion_full": build_full, "failure_case": build_failure}


def main(argv: list[str]) -> int:
    if "--check" in argv:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            stale = []
            for name, builder in FIXTURES.items():
                write(name, builder(), Path(tmp))
                for suffix in (".jsonl", ".caseview.json"):
                    fresh = (Path(tmp) / f"{name}{suffix}").read_text("utf-8")
                    committed = OUT / f"{name}{suffix}"
                    if (
                        not committed.exists()
                        or committed.read_text("utf-8").replace("\r\n", "\n") != fresh
                    ):
                        stale.append(f"{name}{suffix}")
        if stale:
            print("stale fixtures (run: uv run scripts/build_fixtures.py):", *stale, sep="\n  ")
            return 1
        print("fixtures are up to date")
        return 0
    for name, builder in FIXTURES.items():
        events = builder()
        write(name, events, OUT)
        print(f"{name}: {len(events)} events")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
