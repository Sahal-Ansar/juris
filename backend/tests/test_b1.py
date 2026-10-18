"""Baseline B1 (PLAN 6.3): single-agent RAG through the runner, on a scripted provider and a
fake retrieval service (no database, no model)."""

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from juris.baselines.b1 import B1Answer, B1System, clean_evidence_id, evidence_block, pinpoint
from juris.config import (
    BaselineParams,
    ModelsConfig,
    ModelSpec,
    PipelineProfile,
    Settings,
    load_profile,
    resolve_run_config,
)
from juris.eval.juris_eval import JurisEvalItem, load_split
from juris.eval.runner import SYSTEMS, ExperimentSpec, GatewayFactory, run_experiment
from juris.events.fold import CaseView
from juris.llm.types import ProviderRequest, ProviderResponse
from juris.models import Chunk, Document, StatuteMeta
from juris.models.common import ClaimStatus, DocumentKind
from juris.retrieval.service import Hit

STATUTE_TEXT = (
    "When a contract has been broken, the party who complains is entitled to receive "
    "reasonable compensation not exceeding the amount so named in the contract."
)
FATEH_TEXT = (
    "Section 74 applies to stipulations by way of penalty. "
    "The party must prove actual loss in the case of forfeiture of earnest money."
)
SATYABRATA_TEXT = (
    "Forfeiture of a reasonable sum paid as earnest money is permissible. It is not a penalty."
)
ACT = "ACT-contract_act"
FATEH = "SC-1964_1_515_528"
SATYABRATA = "SC-1954_1_310_329"


def hit(chunk_id: str, doc_id: str, text: str, header: str, rank: int, query: str) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        rank=rank,
        text=text,
        context_header=header,
        title=header.split(" - ")[0],
        query=query,
        retrieval_score=1.0 / rank,
    )


HITS = [
    hit("c-74", ACT, STATUTE_TEXT, "Indian Contract Act - s. 74 - Compensation", 1, "q one"),
    hit("c-fateh-12", FATEH, FATEH_TEXT, "Fateh Chand - SC - 1963 - para 12", 2, "q one"),
    hit("c-sat-3", SATYABRATA, SATYABRATA_TEXT, "Satyabrata Ghose - SC - 1953", 3, "q two"),
]


def _chunks() -> dict[str, tuple[Document, Chunk]]:
    act = Document(
        id=ACT,
        kind=DocumentKind.STATUTE,
        title="Indian Contract Act, 1872",
        licence="test",
        corpus_snapshot="snap",
    )
    fateh = Document(
        id=FATEH,
        kind=DocumentKind.JUDGMENT,
        title="Fateh Chand v. Balkishan Dass",
        court="Supreme Court of India",
        citations=["[1964] 1 SCR 515"],
        licence="test",
        corpus_snapshot="snap",
    )
    sat = Document(
        id=SATYABRATA,
        kind=DocumentKind.JUDGMENT,
        title="Satyabrata Ghose v. Mugneeram Bangur",
        licence="test",
        corpus_snapshot="snap",
    )
    return {
        "c-74": (
            act,
            Chunk(
                id="c-74",
                document_id=ACT,
                text=STATUTE_TEXT,
                char_start=0,
                char_end=len(STATUTE_TEXT),
                statute=StatuteMeta(act="Indian Contract Act", section="74"),
            ),
        ),
        "c-fateh-12": (
            fateh,
            Chunk(
                id="c-fateh-12",
                document_id=FATEH,
                text=FATEH_TEXT,
                para_start=12,
                para_end=12,
                char_start=0,
                char_end=len(FATEH_TEXT),
            ),
        ),
        "c-sat-3": (
            sat,
            Chunk(
                id="c-sat-3",
                document_id=SATYABRATA,
                text=SATYABRATA_TEXT,
                para_start=3,
                para_end=5,
                char_start=0,
                char_end=len(SATYABRATA_TEXT),
            ),
        ),
    }


@dataclass
class FakeService:
    """A retrieval service with canned hits that records how it was searched."""

    hits: list[Hit] = field(default_factory=lambda: list(HITS))
    searches: list[dict[str, Any]] = field(default_factory=list)

    def search(self, queries: Sequence[str], *, k: int = 10, rerank: bool = True) -> list[Hit]:
        self.searches.append({"queries": list(queries), "k": k, "rerank": rerank})
        return self.hits[:k]

    def records(self, chunk_id: str) -> tuple[Document, Chunk]:
        return _chunks()[chunk_id]


class Scripted:
    """A provider that answers by prompt: queued replies per prompt, and for the verifier a
    label chosen by a phrase in the claim (default ``supports``)."""

    name = "fake"

    def __init__(self, replies: dict[str, list[str]], verdicts: dict[str, str] | None = None):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.verdicts = verdicts or {}
        self.calls: list[ProviderRequest] = []

    def prompts(self, prompt: str) -> list[ProviderRequest]:
        return [c for c in self.calls if c.tags["prompt"].startswith(prompt + "@")]

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.calls.append(request)
        prompt = request.tags["prompt"].split("@")[0]
        if prompt == "verifier_entailment":
            body = request.messages[0].content
            label = next((v for k, v in self.verdicts.items() if k in body), "supports")
            text = json.dumps({"label": label, "justification": "Scripted."})
        else:
            text = self.replies[prompt].pop(0)
        return ProviderResponse(
            text=text, input_tokens=10, output_tokens=len(text), stop_reason="end_turn"
        )


QUERIES = {
    "queries": [
        "liquidated damages s. 74",
        "  ",
        "forfeiture of earnest money",
        "liquidated damages s. 74",
        "penalty clause",
        "burden of proving loss",
    ]
}

STATUTE_QUOTE = "reasonable compensation not exceeding the amount so named in the contract"
SATYABRATA_QUOTE = "Forfeiture of a reasonable sum paid as earnest money is permissible."


def claim(text: str, *cites: tuple[str, str]) -> dict[str, Any]:
    return {"text": text, "citations": [{"evidence_id": e, "quote": q} for e, q in cites]}


ANSWER: dict[str, Any] = {
    "facts": ["The buyer paid earnest money."],
    "assumptions": ["No arbitration clause."],
    "issues": [
        {
            "question": "Can the seller forfeit the whole earnest money?",
            "position_a": {
                "statement": "No.",
                "summary": "Only reasonable compensation is recoverable.",
                "claims": [
                    claim(
                        "Compensation under s. 74 is capped at the stipulated sum.",
                        ("[E-001]", STATUTE_QUOTE),  # brackets are tolerated
                        ("E-002", "the party need not prove anything at all"),  # not in the chunk
                    )
                ],
            },
            "position_b": {
                "statement": "Yes.",
                "summary": "A reasonable earnest sum may be kept.",
                "claims": [
                    claim("A reasonable earnest money may be forfeited (partial).",
                          ("E-003", SATYABRATA_QUOTE)),
                    claim("Forfeiture is never a penalty (unrelated).",
                          ("E-003", "It is not a penalty.")),
                ],
            },
            "leaning": "leaning_A",
            "confidence": "moderate",
            "confidence_reason": "Two sources, one thin.",
            "unresolved_questions": ["Was the sum reasonable?"],
        },
        {
            "question": "Is notice required?",
            "position_a": {
                "statement": "Yes.",
                "summary": "Notice is needed.",
                "claims": [
                    claim(
                        "Notice is required.",
                        ("Evidence 9", "malformed id"),
                        ("E-099", "unknown id"),
                    )
                ],
            },
            "position_b": {"statement": "No.", "summary": "Not needed.", "claims": []},
            "leaning": "insufficient_evidence",
            "confidence": "low",
            "confidence_reason": "Nothing retrieved.",
            "unresolved_questions": [],
        },
    ],
    "overall_summary": "Compensation is capped by s. 74 [E-001]. Earnest money may be kept [E-003]",
    "limitations": ["Only three passages were retrieved."],
}  # fmt: skip

VERDICTS = {"(partial)": "partially_supports", "(unrelated)": "does_not_support"}


def profile(**baseline: Any) -> PipelineProfile:
    base = load_profile("b1")
    return base.model_copy(update={"baseline": BaselineParams(**baseline)})


@dataclass
class Result:
    view: CaseView
    scores: dict[str, Any]
    provider: Scripted
    service: FakeService
    types: list[str]


async def run(
    tmp_path: Path,
    *,
    answer: dict[str, Any] | None = None,
    profile_: PipelineProfile | None = None,
    extra: dict[str, list[str]] | None = None,
    verdicts: dict[str, str] | None = None,
    service: FakeService | None = None,
) -> Result:
    item: JurisEvalItem = load_split("dev")[0]
    model = ModelSpec(provider="fake", name="local-model", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    config = resolve_run_config(settings, profile_ or profile())
    provider = Scripted(
        {
            "baseline_b1_queries": [json.dumps(QUERIES)],
            "baseline_b1": [json.dumps(ANSWER if answer is None else answer)],
            **(extra or {}),
        },
        VERDICTS if verdicts is None else verdicts,
    )
    fake = service or FakeService()
    spec = ExperimentSpec("b1-test", "b1", "b1", "dev", (item.id,), (0,))
    result = await run_experiment(
        spec,
        [item],
        config,
        system=B1System(service=lambda rerank: fake),
        gateways=GatewayFactory({"fake": provider}, prices={}),
        results_dir=tmp_path,
    )
    assert [o.status for o in result.outcomes] == ["completed"]
    run_dir = tmp_path / "b1-test" / "runs" / f"{item.id}-s0"
    view = CaseView.model_validate_json((run_dir / "caseview.json").read_text("utf-8"))
    scores = json.loads((run_dir / "scores.json").read_text("utf-8"))
    types = [
        json.loads(line)["type"]
        for line in (run_dir / "events.jsonl").read_text("utf-8").splitlines()
    ]
    return Result(view, scores, provider, fake, types)


def test_the_answer_schema_accepts_the_fixture() -> None:
    B1Answer.model_validate(ANSWER)


def test_the_profile_and_registry() -> None:
    p = load_profile("b1")
    assert p.baseline == BaselineParams(repair_quotes=True)  # the yaml sets repair on
    assert (p.baseline.queries, p.baseline.evidence_k) == ("llm", 20)
    assert not p.baseline.rerank and p.baseline.repair_quotes
    assert SYSTEMS["b1"] is B1System


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("E-003", "E-003"),
        ("E-3", "E-003"),
        ("E3", "E-003"),
        ("e-03", "E-003"),
        ("[E-003]", "E-003"),
        (" E - 3 ", "E-003"),
        ("E-0003", "E-003"),
        ("E-1234", "E-1234"),
        ("Evidence 3", "Evidence 3"),  # stays as it is, and is dropped later
        ("E-", "E-"),
        ("C-001", "C-001"),
    ],
)
def test_evidence_ids_are_normalised(raw: str, expected: str) -> None:
    assert clean_evidence_id(raw) == expected


async def test_loosely_written_ids_still_cite(tmp_path: Path) -> None:
    answer = copy.deepcopy(ANSWER)
    cites = answer["issues"][0]["position_b"]["claims"][0]["citations"]
    cites[0]["evidence_id"] = "e-3"
    answer["issues"][0]["position_b"]["claims"][1]["citations"][0]["evidence_id"] = "E3"
    r = await run(tmp_path, answer=answer)
    values = r.scores["values"]
    assert values["b1.ids_normalised"] == 3  # "[E-001]", "e-3", "E3"
    assert values["b1.malformed_citations_dropped"] == 1
    statuses = {(v.claim_id, v.evidence_id): v.result.status for v in r.view.verifications}
    assert statuses[("C-002", "E-003")] == "weak"


def test_evidence_block_and_pinpoints() -> None:
    block = evidence_block(HITS)
    assert block.startswith("[E-001] Indian Contract Act - s. 74 - Compensation\nWhen a contract")
    assert "\n\n[E-003] Satyabrata Ghose - SC - 1953\nForfeiture" in block
    chunks = _chunks()
    assert pinpoint(chunks["c-74"][1], None) == "s. 74"
    assert pinpoint(chunks["c-fateh-12"][1], None) == "para 12"
    assert pinpoint(chunks["c-sat-3"][1], None) == "paras 3-5"
    pages = Chunk(id="x", document_id="d", text="t", page_start=7, page_end=7, char_start=0,
                  char_end=1)  # fmt: skip
    assert pinpoint(pages, None) == "p. 7"
    assert pinpoint(pages.model_copy(update={"page_end": 9}), None) == "pp. 7-9"
    assert pinpoint(pages.model_copy(update={"page_start": None, "page_end": None}), "Hdr") == "Hdr"


async def test_queries_from_the_model(tmp_path: Path) -> None:
    r = await run(tmp_path)
    queries = ["liquidated damages s. 74", "forfeiture of earnest money", "penalty clause",
               "burden of proving loss"]  # fmt: skip
    # blanks and duplicates go; at most max_queries (5) are kept
    assert r.service.searches == [{"queries": queries, "k": 20, "rerank": False}]
    assert [q.query for q in r.view.queries] == queries
    assert all(q.stage == "S0" and q.agent == "b1" for q in r.view.queries)
    asked = r.provider.prompts("baseline_b1_queries")
    assert len(asked) == 1 and "$" not in asked[0].messages[0].content
    assert asked[0].seed == 0 and asked[0].json_schema is not None
    assert r.scores["details"]["system"]["queries"] == queries
    assert r.scores["values"]["b1.queries"] == 4


async def test_queries_are_capped(tmp_path: Path) -> None:
    r = await run(tmp_path, profile_=profile(max_queries=2, evidence_k=2))
    assert r.service.searches == [
        {
            "queries": ["liquidated damages s. 74", "forfeiture of earnest money"],
            "k": 2,
            "rerank": False,
        }
    ]
    assert len(r.view.evidence) == 2


async def test_question_mode_makes_no_query_call(tmp_path: Path) -> None:
    r = await run(tmp_path, profile_=profile(queries="question", rerank=True))
    item = load_split("dev")[0]
    assert r.service.searches == [{"queries": [item.question], "k": 20, "rerank": True}]
    assert r.provider.prompts("baseline_b1_queries") == []
    assert r.scores["values"]["b1.queries"] == 1


async def test_evidence_ids_and_registry(tmp_path: Path) -> None:
    r = await run(tmp_path)
    sent = r.provider.prompts("baseline_b1")[0]
    assert sent.max_tokens == 6000 and sent.tags["prompt"] == "baseline_b1@v2"
    text = sent.messages[0].content
    assert "[E-001] Indian Contract Act - s. 74 - Compensation\n" + STATUTE_TEXT in text
    assert "[E-002] Fateh Chand - SC - 1963 - para 12\n" + FATEH_TEXT in text
    view = r.view
    assert list(view.evidence) == ["E-001", "E-002", "E-003"]  # the prompt's IDs
    assert [e.chunk_id for e in view.evidence.values()] == ["c-74", "c-fateh-12", "c-sat-3"]
    # every item is cited, by issue 1's claims only
    assert view.evidence["E-001"].issue_ids == ["I-1"]
    assert view.evidence["E-002"].issue_ids == ["I-1"]
    assert view.evidence["E-003"].issue_ids == ["I-1"]
    assert view.evidence["E-001"].retrieved_by_query == "q one"
    assert view.evidence["E-001"].registered_at_stage == "S1"


async def test_uncited_evidence_belongs_to_every_issue(tmp_path: Path) -> None:
    answer = copy.deepcopy(ANSWER)
    answer["issues"][0]["position_b"]["claims"] = []  # nothing cites E-003 now
    r = await run(tmp_path, answer=answer)
    assert r.view.evidence["E-003"].issue_ids == ["I-1", "I-2"]


async def test_events_in_order_and_the_folded_record(tmp_path: Path) -> None:
    r = await run(tmp_path)
    t = r.types
    first = {name: t.index(name) for name in dict.fromkeys(t)}
    assert t[0] == "case_created" and t[-1] == "run_completed"
    order = [
        "agent_started",
        "query_issued",
        "issue_framed",
        "evidence_registered",
        "claim_created",
        "argument_created",
        "verification_result",
        "analysis_ready",
    ]
    assert [first[n] for n in order] == sorted(first[n] for n in order)
    assert t.index("analysis_ready") > max(
        i for i, n in enumerate(t) if n in {"verification_result", "claim_status_changed"}
    )
    view = r.view
    assert list(view.issues) == ["I-1", "I-2"]
    assert view.facts == ["The buyer paid earnest money."]
    assert list(view.claims) == ["C-001", "C-002", "C-003", "C-004"]
    assert {c.position_id for c in view.claims.values()} == {"I-1.A", "I-1.B", "I-2.A"}
    assert all(c.author_agent == "b1" for c in view.claims.values())
    assert list(view.arguments) == ["A-001", "A-002", "A-003", "A-004"]
    assert [a.claim_ids for a in view.arguments.values()] == [[c] for c in view.claims]
    assert view.arguments["A-001"].reasoning == "Only reasonable compensation is recoverable."
    assert view.agents["b1"].input_tokens == 20  # queries + answer
    assert view.status == "completed"


async def test_pinpoints_come_from_the_chunks(tmp_path: Path) -> None:
    view = (await run(tmp_path)).view
    cites = {
        (c.claim_id, c.evidence_id): c.pinpoint
        for a in view.arguments.values()
        for c in a.citations
    }
    assert cites[("C-001", "E-001")] == "s. 74"
    assert cites[("C-001", "E-002")] == "para 12"
    assert cites[("C-002", "E-003")] == "paras 3-5"
    assert cites[("C-004", "E-099")] == "unknown"  # never presented: fails existence later


async def test_verification_results_and_claim_statuses(tmp_path: Path) -> None:
    r = await run(tmp_path)
    view = r.view
    by_pair = {(v.claim_id, v.evidence_id): v.result for v in view.verifications}
    assert len(by_pair) == 6 - 1  # the malformed ID was dropped before it was a citation
    assert by_pair[("C-001", "E-001")].status == "verified"
    bad_quote = by_pair[("C-001", "E-002")]
    assert (bad_quote.status, bad_quote.check) == ("invalid", "quote")
    assert by_pair[("C-002", "E-003")].status == "weak"
    assert by_pair[("C-003", "E-003")].status == "unsupported"
    unknown = by_pair[("C-004", "E-099")]
    assert (unknown.status, unknown.check) == ("invalid", "existence")
    assert {c: x.status for c, x in view.claims.items()} == {
        "C-001": ClaimStatus.VERIFIED,
        "C-002": ClaimStatus.WEAK,
        "C-003": ClaimStatus.UNSUPPORTED,
        "C-004": ClaimStatus.UNSUPPORTED,
    }
    # only citations that pass the code checks reach the model, one pair per call
    verifier_calls = r.provider.prompts("verifier_entailment")
    assert len(verifier_calls) == 3
    assert all(c.tags["pairs"] == "1" for c in verifier_calls)


async def test_the_analysis_keeps_only_verified_and_weak_evidence(tmp_path: Path) -> None:
    analysis = (await run(tmp_path)).view.analysis
    assert analysis is not None
    first, second = analysis.issues
    a, b = first.positions
    assert [(e.evidence_id, e.pinpoint) for e in a.key_evidence] == [("E-001", "s. 74")]
    assert a.supporting_authorities == []  # a statute is not a judgment
    assert [(e.evidence_id, e.pinpoint) for e in b.key_evidence] == [("E-003", "paras 3-5")]
    assert b.supporting_authorities == [SATYABRATA]
    assert a.summary == "Only reasonable compensation is recoverable."
    assert all(not p.key_evidence for p in second.positions)
    assert (first.leaning, first.confidence) == ("leaning_A", "moderate")
    assert first.unresolved_questions == ["Was the sum reasonable?"]
    assert {s.document_id: s.pinpoints_used for s in analysis.sources} == {
        ACT: ["s. 74"],
        SATYABRATA: ["paras 3-5"],
    }
    assert analysis.limitations == [
        "Only three passages were retrieved.",
        "I-1: 2 of 4 citations failed verification and were left out of the key evidence.",
        "I-2: 1 of 1 citations failed verification and were left out of the key evidence.",
    ]
    # the overall summary keeps its inline references as written
    assert "[E-001]" in analysis.overall_summary and "[E-003]" in analysis.overall_summary


async def test_long_summaries_are_cut(tmp_path: Path) -> None:
    answer = copy.deepcopy(ANSWER)
    answer["overall_summary"] = "word " * 300
    analysis = (await run(tmp_path, answer=answer)).view.analysis
    assert analysis is not None and len(analysis.overall_summary.split()) == 250
    assert any("cut to 250 words" in lim for lim in analysis.limitations)


async def test_malformed_ids_are_dropped_and_counted(tmp_path: Path) -> None:
    r = await run(tmp_path)
    values = r.scores["values"]
    assert values["b1.malformed_citations_dropped"] == 1  # "Evidence 9"
    assert values["b1.ids_normalised"] == 1  # "[E-001]"
    cited = {c.evidence_id for a in r.view.arguments.values() for c in a.citations}
    assert cited == {"E-001", "E-002", "E-003", "E-099"}
    assert values["b1.claims"] == 4 and values["b1.citations"] == 5
    assert values["b1.evidence"] == 3


async def test_values_details_and_citation_validity(tmp_path: Path) -> None:
    r = await run(tmp_path)
    values = r.scores["values"]
    assert values["b1.share_verified"] == pytest.approx(1 / 5)
    assert values["b1.share_weak"] == pytest.approx(1 / 5)
    assert values["b1.share_unsupported"] == pytest.approx(1 / 5)
    assert values["b1.share_invalid"] == pytest.approx(2 / 5)
    assert values["b1.invalid_citations"] == 2
    assert "b1.citation_failures_before_repair" not in values  # repair is off
    outcomes = r.scores["details"]["system"]["citations"]
    assert [(o["claim_id"], o["evidence_id"], o["status"]) for o in outcomes] == [
        ("C-001", "E-001", "verified"),
        ("C-001", "E-002", "invalid"),
        ("C-002", "E-003", "weak"),
        ("C-003", "E-003", "unsupported"),
        ("C-004", "E-099", "invalid"),
    ]
    # the metrics read B1 like Juris: 3 of 5 quotes valid, 4 of 5 IDs exist
    assert values["citation_validity"] == pytest.approx(3 / 5)
    assert values["citation_validity.existence"] == pytest.approx(4 / 5)
    assert values["citation_faithfulness"] == pytest.approx(1 / 5)  # 1 supports of 5 judged
    assert 0 < (values["unsupported_claim_rate"] or 0) < 1
    assert r.scores["details"]["system"]["queries"]


async def test_no_citations_gives_no_shares(tmp_path: Path) -> None:
    answer = copy.deepcopy(ANSWER)
    for issue in answer["issues"]:
        for side in ("position_a", "position_b"):
            issue[side]["claims"] = []
    r = await run(tmp_path, answer=answer)
    assert r.scores["values"]["b1.share_verified"] is None
    assert r.scores["values"]["citation_validity"] is None
    assert r.provider.prompts("verifier_entailment") == []


FIXED_QUOTE = "The party must prove actual loss in the case of forfeiture of earnest money."


async def test_repair_fixes_a_quote(tmp_path: Path) -> None:
    reply = json.dumps({"fixes": [{"number": 1, "quote": FIXED_QUOTE}]})
    r = await run(
        tmp_path,
        profile_=profile(repair_quotes=True),
        extra={"baseline_b1_repair": [reply]},
    )
    asked = r.provider.prompts("baseline_b1_repair")
    assert len(asked) == 1
    body = asked[0].messages[0].content
    assert "the party need not prove anything at all" in body  # the bad quote
    assert FATEH_TEXT in body and "Compensation under s. 74" in body  # evidence and claim
    assert "E-099" not in body  # an unknown ID has no evidence text to repair from
    values = r.scores["values"]
    assert values["b1.citation_failures_before_repair"] == 2  # bad quote + unknown E-099
    assert values["b1.citation_failures_after_repair"] == 1
    assert values["b1.citations_dropped_by_repair"] == 0
    quotes = {c.evidence_id: c.quote for a in r.view.arguments.values() for c in a.citations}
    assert quotes["E-002"] == FIXED_QUOTE
    statuses = {(v.claim_id, v.evidence_id): v.result.status for v in r.view.verifications}
    assert statuses[("C-001", "E-002")] == "verified"  # now checked by the model too


async def test_repair_can_drop_a_citation(tmp_path: Path) -> None:
    reply = json.dumps({"fixes": [{"number": 1, "quote": None}, {"number": 7, "quote": "x"}]})
    r = await run(
        tmp_path, profile_=profile(repair_quotes=True), extra={"baseline_b1_repair": [reply]}
    )
    values = r.scores["values"]
    assert values["b1.citations_dropped_by_repair"] == 1
    assert values["b1.citation_failures_after_repair"] == 1
    assert values["b1.citations"] == 4
    assert ("C-001", "E-002") not in {(v.claim_id, v.evidence_id) for v in r.view.verifications}


async def test_a_wrong_repair_stays_in(tmp_path: Path) -> None:
    reply = json.dumps({"fixes": [{"number": 1, "quote": "still not in the chunk"}]})
    r = await run(
        tmp_path, profile_=profile(repair_quotes=True), extra={"baseline_b1_repair": [reply]}
    )
    assert r.scores["values"]["b1.citation_failures_after_repair"] == 2
    statuses = {(v.claim_id, v.evidence_id): v.result.status for v in r.view.verifications}
    assert statuses[("C-001", "E-002")] == "invalid"


async def test_repair_is_skipped_when_every_quote_holds(tmp_path: Path) -> None:
    answer = copy.deepcopy(ANSWER)
    answer["issues"][0]["position_a"]["claims"][0]["citations"][1]["quote"] = FIXED_QUOTE
    answer["issues"][1]["position_a"]["claims"] = []
    r = await run(tmp_path, answer=answer, profile_=profile(repair_quotes=True))
    assert r.provider.prompts("baseline_b1_repair") == []
    assert r.scores["values"]["b1.citation_failures_after_repair"] == 0


async def test_repair_off_makes_no_repair_call(tmp_path: Path) -> None:
    r = await run(tmp_path)
    assert r.provider.prompts("baseline_b1_repair") == []


async def test_the_service_is_created_once(tmp_path: Path) -> None:
    made: list[bool] = []
    fake = FakeService()

    def factory(rerank: bool) -> FakeService:
        made.append(rerank)
        return fake

    system = B1System(service=factory)
    assert made == []  # lazy
    assert system._search_service(True) is fake and system._search_service(False) is fake
    assert made == [True]


def _bare(answer: dict[str, Any]) -> dict[str, Any]:
    """The answer with one more claim that has no citation."""
    out = copy.deepcopy(answer)
    out["issues"][1]["position_b"]["claims"] = [claim("Uncited claim.")]
    return out


async def test_a_claim_without_citations_is_fixed_by_the_gateway_retry(tmp_path: Path) -> None:
    r = await run(
        tmp_path,
        answer=_bare(ANSWER),
        extra={"baseline_b1": [json.dumps(_bare(ANSWER)), json.dumps(ANSWER)]},  # answer, retry
    )
    assert r.scores["values"]["b1.uncited_claims_dropped"] == 0
    assert r.scores["values"]["b1.claims"] == 4


async def test_claims_still_uncited_after_the_retry_are_dropped(tmp_path: Path) -> None:
    bare = json.dumps(_bare(ANSWER))
    r = await run(tmp_path, answer=_bare(ANSWER), extra={"baseline_b1": [bare, bare]})
    assert r.scores["values"]["b1.uncited_claims_dropped"] == 1
    assert r.scores["values"]["b1.claims"] == 4  # the four cited claims remain


async def test_claims_left_without_citations_are_dropped(tmp_path: Path) -> None:
    answer = copy.deepcopy(ANSWER)
    answer["issues"][0]["position_a"]["claims"][0]["citations"] = [
        {"evidence_id": "nothing", "quote": "x"}  # malformed: dropped, leaving the claim bare
    ]
    r = await run(tmp_path, answer=answer)
    assert r.scores["values"]["b1.malformed_citations_dropped"] == 2
    assert r.scores["values"]["b1.uncited_claims_dropped"] == 1
    assert "C-001" not in r.view.claims or r.view.claims["C-001"].text != (
        "Compensation under s. 74 is capped at the stipulated sum."
    )
