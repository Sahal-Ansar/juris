"""The Precedent Analyst and S3 (PLAN 7.6) on canned outputs: a card for every cited judgment,
code facts and graph cues, what the analyst reads, fits, and the fallback. No model, no
database."""

import json
from dataclasses import dataclass
from typing import Any

from juris.agents.precedent_analyst import metadata, treatment_signals
from juris.baselines.base import RunContext
from juris.config import ModelsConfig, ModelSpec, PipelineProfile, Settings, resolve_run_config
from juris.eval.juris_eval import load_split
from juris.events import EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.llm.gateway import LLMGateway
from juris.llm.providers.fake import FakeProvider
from juris.models import Chunk, Citation, Document
from juris.models.common import CourtLevel, DocumentKind, Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.precedent import PrecedentStage, cited_judgments
from juris.retrieval.lookup import LinkedCase
from tests.test_b1 import (
    FATEH,
    FATEH_TEXT,
    HITS,
    SATYABRATA,
    SATYABRATA_QUOTE,
    STATUTE_QUOTE,
    FakeService,
    _chunks,
)

FATEH_QUOTE = "The party must prove actual loss in the case of forfeiture of earnest money."
HEADNOTE = "HELD: a stipulation for forfeiture of earnest money is subject to s. 74."
AFTER = "The plaintiff was entitled only to reasonable compensation."


def linked(ref: str, treatments: list[str], paras: list[str] | None = None) -> LinkedCase:
    return LinkedCase(ref=ref, doc_id=ref, title=f"Later case {ref}", decision_date=None,
                      mentions=1, paragraphs=paras or [], citations=[],
                      treatments=treatments)  # fmt: skip


@dataclass
class FakePrecedents(FakeService):
    """B1's fake corpus plus Fateh Chand's headnote and the paragraph after the cited one."""

    def records(self, chunk_id: str) -> tuple[Document, Chunk]:
        document, chunk = _chunks()["c-fateh-12"]
        extra = {"c-fateh-h1": (HEADNOTE, 1), "c-fateh-13": (AFTER, 13)}
        if chunk_id in extra:
            text, para = extra[chunk_id]
            return document, chunk.model_copy(
                update={
                    "id": chunk_id,
                    "text": text,
                    "para_start": para,
                    "para_end": para,
                    "char_end": len(text),
                }
            )
        return super().records(chunk_id)

    def chunk_ids(self, doc_id: str, section: str | None = None) -> list[str]:
        if doc_id == FATEH:
            return ["c-fateh-h1"] if section == "headnote" else [
                "c-fateh-h1", "c-fateh-12", "c-fateh-13"]  # fmt: skip
        return ["c-sat-3"] if section is None else []

    def citing_cases(self, doc_id: str) -> list[LinkedCase]:
        if doc_id == FATEH:
            return [
                linked("SC-later-1", ["followed", "referred"], ["14", "15"]),
                linked("SC-later-2", ["reversed"]),
            ]
        return []


def cite(claim_id: str, evidence_id: str, quote: str) -> Citation:
    return Citation(claim_id=claim_id, evidence_id=evidence_id, pinpoint="p", quote=quote)


def context(responses: dict[str, list[Any]]) -> tuple[StageContext, FakeProvider]:
    provider = FakeProvider(
        {k: [r if isinstance(r, str) else json.dumps(r) for r in v] for k, v in responses.items()},
        key_tag="document",
    )
    model = ModelSpec(provider="fake", name="m", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    config = resolve_run_config(settings, PipelineProfile(name="t", kind="juris"))
    item = load_split("dev")[0]
    emitter = EventEmitter(InMemoryEventStore(), run_id="r", case_id=item.id)
    emitter.emit(CaseCreatedPayload(question=item.question, profile="t", corpus_snapshot_id="s"))
    record = CaseRecord(emitter, FakePrecedents())
    record.frame_issue("Can the seller forfeit?", ["No.", "Yes."], stage=Stage.S0, agent="f",
                       facts=["Earnest money was paid."])  # fmt: skip
    for hit in HITS:  # E-001 statute, E-002 Fateh Chand, E-003 Satyabrata Ghose
        record.register_evidence(hit, ["I-1"], stage=Stage.S1, agent="researcher")
    S2 = Stage.S2
    for text, position, cites in [
        ("Capped.", "I-1.A", [("E-001", STATUTE_QUOTE), ("E-002", FATEH_QUOTE)]),
        ("Earnest money may be kept.", "I-1.B", [("E-003", SATYABRATA_QUOTE)]),
        ("Loss must be proved.", "I-1.A", [("E-002", FATEH_QUOTE)]),
    ]:
        claim = record.add_claim(text, position, stage=S2, agent="counsel")
        record.add_argument([claim.id], "r", [cite(claim.id, e, q) for e, q in cites], stage=S2,
                            agent="counsel")  # fmt: skip
    run = RunContext(item, config, LLMGateway({"fake": provider}, prices={}), emitter, 0)
    budget = BudgetManager(config.budget, Spend)
    return StageContext(run, record, budget, Stage.S3, "on"), provider


def output(
    principle: str, *fits: tuple[str, str], factors: list[str] | None = None
) -> dict[str, Any]:
    return {
        "principle": principle,
        "material_facts": "A buyer defaulted after paying earnest money.",
        "distinguishing_factors": factors or [],
        "fits": [{"argument_id": a, "fit": f, "reason": "Because."} for a, f in fits],
    }


FATEH_OUT = output(
    "Forfeiture of earnest money is limited to reasonable compensation under s. 74.",
    ("A-001", "directly_supports"),
    ("A-003", "supports_by_analogy"),
    factors=["", "The sum here was a part payment, not earnest money."],
)
SAT_OUT = output("A reasonable earnest sum may be forfeited.", ("A-002", "distinguishable"))


def stage() -> PrecedentStage:
    return PrecedentStage(lambda: FakePrecedents())


async def test_every_cited_judgment_gets_a_card() -> None:
    ctx, _ = context({FATEH: [FATEH_OUT], SATYABRATA: [SAT_OUT]})
    await stage().run(ctx)
    cards = ctx.record.view.precedent_cards
    assert list(cards) == [FATEH, SATYABRATA]  # first-citation order; the statute has none
    fateh = cards[FATEH]
    assert fateh.principle.startswith("Forfeiture of earnest money is limited")
    assert fateh.distinguishing_factors == ["The sum here was a part payment, not earnest money."]
    assert fateh.fit_by_argument == {"A-001": "directly_supports", "A-003": "supports_by_analogy"}
    assert fateh.treatment_verified is False
    assert [(s.kind.value, s.by_document_id) for s in fateh.treatment] == [
        ("followed", "SC-later-1"),
        ("overruled", "SC-later-2"),  # "reversed" is a negative cue
    ]
    assert (
        fateh.treatment[0].cue_text
        == '"followed" near the citation in Later case SC-later-1, para 14, 15'
    )
    assert cards[SATYABRATA].fit_by_argument == {"A-002": "distinguishable"}
    values = ctx.run.values
    assert values == {
        "s3.judgments": 2.0,
        "s3.cards": 2.0,
        "s3.fallback_cards": 0.0,
        "s3.fits": 3.0,
        "s3.fits_missing": 0.0,
        "s3.treatment_signals": 2.0,
    }
    assert ctx.run.details["s3.fits"][0] == {
        "document": FATEH, "argument": "A-001", "fit": "directly_supports", "reason": "Because."
    }  # fmt: skip


async def test_what_the_analyst_reads() -> None:
    ctx, provider = context({FATEH: [FATEH_OUT], SATYABRATA: [SAT_OUT]})
    await stage().run(ctx)
    body = next(c for c in provider.calls if c.tags["document"] == FATEH).messages[0].content
    assert (
        "Title: Fateh Chand v. Balkishan Dass" in body and "Court: Supreme Court of India" in body
    )
    assert "Citations: [1964] 1 SCR 515" in body
    assert '- followed: "followed" near the citation in Later case SC-later-1' in body
    passages = body[body.index("<passages>") : body.index("</passages>")]
    assert passages.index("[headnote; para 1]") < passages.index("[cited as E-002; para 12]")
    assert passages.index("[cited as E-002; para 12]") < passages.index("[context; para 13]")
    assert HEADNOTE in passages and FATEH_TEXT in passages and AFTER in passages
    arguments = body[body.index("<arguments_relying_on_it>") :]
    assert "A-001 (for I-1.A): Capped." in arguments and "A-003 (for I-1.A)" in arguments
    assert f'E-002: "{FATEH_QUOTE}"' in arguments
    assert STATUTE_QUOTE not in arguments  # only the quotes from this judgment
    assert "A-002" not in arguments
    assert "I-1: Can the seller forfeit? (I-1.A: No.; I-1.B: Yes.)" in body
    assert "- Earnest money was paid." in body


async def test_the_passage_budget_keeps_the_cited_chunk() -> None:
    ctx, provider = context({FATEH: [FATEH_OUT], SATYABRATA: [SAT_OUT]})
    tight = PrecedentStage(lambda: FakePrecedents(), max_passage_chars=len(FATEH_TEXT))
    await tight.run(ctx)
    body = next(c for c in provider.calls if c.tags["document"] == FATEH).messages[0].content
    assert FATEH_TEXT in body and HEADNOTE not in body and AFTER not in body


async def test_bad_fits_are_dropped_and_missing_ones_counted() -> None:
    bad = output("A principle.", ("A-002", "contrary"), ("A-001", "contrary"),
                 ("A-001", "distinguishable"))  # fmt: skip
    ctx, provider = context({FATEH: [bad, bad], SATYABRATA: [SAT_OUT]})
    await stage().run(ctx)
    feedback = next(c for c in provider.calls if c.tags.get("repair")).messages[2].content
    assert "fits[0]: 'A-002' is not an argument that cites this judgment" in feedback
    assert "fits[2]: A-001 already has a fit" in feedback
    assert ctx.record.view.precedent_cards[FATEH].fit_by_argument == {"A-001": "contrary"}
    assert ctx.run.values["s3.fits_missing"] == 1.0  # A-003 got no fit


async def test_no_usable_output_gives_a_fallback_card() -> None:
    empty = output(" ", ("A-001", "contrary"))
    ctx, _ = context({FATEH: [empty, empty], SATYABRATA: [SAT_OUT]})
    await stage().run(ctx)
    card = ctx.record.view.precedent_cards[FATEH]
    assert card.principle == "Not extracted: the Precedent Analyst gave no valid output"
    assert card.fit_by_argument == {} and len(card.treatment) == 2  # code facts kept
    assert ctx.run.values["s3.fallback_cards"] == 1.0 and ctx.run.values["s3.cards"] == 2.0


async def test_passthrough_writes_code_only_cards() -> None:
    ctx, provider = context({})
    await stage().passthrough(ctx)
    assert provider.calls == []
    cards = ctx.record.view.precedent_cards
    assert list(cards) == [FATEH, SATYABRATA]
    assert cards[FATEH].principle == "Not extracted: S3 ran as a passthrough"


def test_cited_judgments_and_code_facts() -> None:
    ctx, _ = context({})
    assert cited_judgments(ctx.record.view) == {
        FATEH: {"c-fateh-12": ["E-002"]},
        SATYABRATA: {"c-sat-3": ["E-003"]},
    }
    assert treatment_signals([linked("x", ["referred", "approved", "per_incuriam"])]) == [
        s for s in treatment_signals([linked("x", ["approved", "per_incuriam"])])
    ]
    document = Document(id="SC-x", kind=DocumentKind.JUDGMENT, title="X v. Y",
                        court_level=CourtLevel.SUPREME_COURT, bench_strength=5, licence="t",
                        corpus_snapshot="s")  # fmt: skip
    text = metadata(document)
    assert "Level: supreme court (binds every court in India (Article 141))" in text
    assert "Bench: 5 judge(s)" in text
