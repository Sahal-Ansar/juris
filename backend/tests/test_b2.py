"""Baseline B2 (PLAN 6.4): B1's draft, then critique -> revise rounds with a second search,
through the runner on a scripted provider and a fake retrieval service (no database, no model)."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from juris.baselines.b1 import B1Answer
from juris.baselines.b2 import B2System, _quote_errors, _shown
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
from juris.models import Chunk, Document
from tests.test_b1 import (
    ANSWER,
    FATEH,
    HITS,
    QUERIES,
    SATYABRATA_QUOTE,
    STATUTE_QUOTE,
    VERDICTS,
    FakeService,
    Scripted,
    _chunks,
    claim,
    hit,
)

EXTRA_TEXT = "Earnest money is part of the purchase price when the transaction goes forward."
EXTRA = hit("c-fateh-13", FATEH, EXTRA_TEXT, "Fateh Chand - SC - 1963 - para 13", 4, "follow")


@dataclass
class MoreService(FakeService):
    """B1's fake service with one more chunk, which only a search for more than 3 reaches."""

    def records(self, chunk_id: str) -> tuple[Document, Chunk]:
        if chunk_id == EXTRA.chunk_id:
            document, chunk = _chunks()["c-fateh-12"]
            return document, chunk.model_copy(
                update={
                    "id": EXTRA.chunk_id,
                    "text": EXTRA_TEXT,
                    "para_start": 13,
                    "para_end": 13,
                    "char_end": len(EXTRA_TEXT),
                }
            )
        return super().records(chunk_id)


def service() -> MoreService:
    return MoreService(hits=[*HITS, EXTRA])


CRITIQUE = {
    "problems": ["  ", "Position B of issue 1 lacks the earnest-money rule.", "Fix E-002's quote."],
    "queries": ["earnest money part of price", "earnest money part of price", "", "a", "b"],
}
NO_PROBLEMS: dict[str, list[str]] = {"problems": [], "queries": []}

REVISED: dict[str, Any] = copy.deepcopy(ANSWER)
REVISED["issues"][0]["position_b"]["claims"].append(
    claim("Earnest money is part of the price.", ("E-004", EXTRA_TEXT))
)
REVISED["issues"] = REVISED["issues"][:1]


def profile(**baseline: Any) -> PipelineProfile:
    base = load_profile("b2")
    params = base.baseline.model_dump() | {"evidence_k": 3, "repair_quotes": False} | baseline
    return base.model_copy(update={"baseline": BaselineParams(**params)})


@dataclass
class Result:
    view: CaseView
    scores: dict[str, Any]
    provider: Scripted
    service: MoreService
    events: list[dict[str, Any]]


async def run(
    tmp_path: Path,
    *,
    replies: dict[str, list[Any]],
    profile_: PipelineProfile | None = None,
) -> Result:
    item: JurisEvalItem = load_split("dev")[0]
    model = ModelSpec(provider="fake", name="local-model", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    config = resolve_run_config(settings, profile_ or profile())
    provider = Scripted(
        {
            "baseline_b1_queries": [json.dumps(QUERIES)],
            **{
                k: [r if isinstance(r, str) else json.dumps(r) for r in v]
                for k, v in replies.items()
            },
        },
        VERDICTS,
    )
    fake = service()
    spec = ExperimentSpec("b2-test", "b2", "b2", "dev", (item.id,), (0,))
    result = await run_experiment(
        spec,
        [item],
        config,
        system=B2System(service=lambda rerank: fake),
        gateways=GatewayFactory({"fake": provider}, prices={}),
        results_dir=tmp_path,
    )
    assert [o.status for o in result.outcomes] == ["completed"]
    run_dir = tmp_path / "b2-test" / "runs" / f"{item.id}-s0"
    view = CaseView.model_validate_json((run_dir / "caseview.json").read_text("utf-8"))
    scores = json.loads((run_dir / "scores.json").read_text("utf-8"))
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text("utf-8").splitlines()
    ]
    return Result(view, scores, provider, fake, events)


def test_the_profile_and_registry() -> None:
    assert SYSTEMS["b2"] is B2System
    params = load_profile("b2").baseline
    assert params.max_rounds == 8 and params.target_tokens == 170000
    assert (params.follow_up_queries, params.follow_up_k, params.revise_evidence_k) == (3, 8, 12)
    b1 = load_profile("b1").baseline
    same = ("queries", "max_queries", "evidence_k", "rerank", "repair_quotes")
    assert all(getattr(params, f) == getattr(b1, f) for f in same)  # B2's draft is B1's


async def test_critique_search_and_revise(tmp_path: Path) -> None:
    r = await run(
        tmp_path,
        replies={
            "baseline_b1": [ANSWER],
            "baseline_b2_critique": [CRITIQUE, NO_PROBLEMS],
            "baseline_b2_revise": [REVISED],
        },
    )
    values = r.scores["values"]
    assert (values["b2.critiques"], values["b2.revisions"]) == (2, 1)  # 2nd found nothing
    assert values["b2.follow_up_queries"] == 3
    assert values["b2.follow_up_evidence"] == 1
    assert values["b2.evidence"] == 4

    # the second search: the critique's queries, deduplicated and capped, after the first's
    assert r.service.searches[1] == {
        "queries": ["earnest money part of price", "a", "b"],
        "k": 8 + 3,
        "rerank": False,
    }
    follow = [e for e in r.events if e["type"] == "query_issued" and e["stage"] == "S1"]
    assert [e["payload"]["query"] for e in follow] == ["earnest money part of price", "a", "b"]

    # critique 1 sees the code's quote errors; the revise sees the new chunk and the review
    first = r.provider.prompts("baseline_b2_critique")[0].messages[0].content
    assert '"the party need not prove anything at all" is not word for word in E-002' in first
    assert '"E-099" is not an evidence ID' in first and "[E-004]" not in first
    revise = r.provider.prompts("baseline_b2_revise")[0].messages[0].content
    assert f"[E-004] Fateh Chand - SC - 1963 - para 13\n{EXTRA_TEXT}" in revise
    assert "- Position B of issue 1 lacks the earnest-money rule.\n- Fix E-002's quote." in revise
    assert '"overall_summary"' in revise  # the draft, as JSON
    second = r.provider.prompts("baseline_b2_critique")[1].messages[0].content
    assert "Earnest money is part of the price." in second  # critique 2 reads the revision

    # the record is the revision, citing the new chunk under its registry ID
    assert len(r.view.evidence) == 4 and r.view.evidence["E-004"].chunk_id == EXTRA.chunk_id
    cites = {(c.evidence_id, c.quote) for a in r.view.arguments.values() for c in a.citations}
    assert ("E-004", EXTRA_TEXT) in cites
    assert len(r.view.issues) == 1
    assert r.scores["details"]["system"]["critiques"] == [CRITIQUE["problems"][1:], []]


async def test_values_and_events_are_named_b2(tmp_path: Path) -> None:
    r = await run(
        tmp_path,
        replies={"baseline_b1": [ANSWER], "baseline_b2_critique": [NO_PROBLEMS]},
        profile_=profile(follow_up_queries=0),
    )
    values = r.scores["values"]
    assert not [k for k in values if k.startswith("b1.")]
    assert values["b2.claims"] == 4 and values["b2.tokens_estimate"] > 0
    agents = {e["agent"] for e in r.events if e.get("agent")}
    assert agents <= {"b2", "citation_verifier"} and "b2" in agents
    tags = {c.tags["agent"] for c in r.provider.calls if c.tags["prompt"].startswith("baseline")}
    assert tags == {"b2"}


async def test_no_problems_and_no_new_evidence_means_no_revision(tmp_path: Path) -> None:
    r = await run(
        tmp_path,
        replies={"baseline_b1": [ANSWER], "baseline_b2_critique": [NO_PROBLEMS]},
        profile_=profile(follow_up_queries=0),
    )
    assert r.provider.prompts("baseline_b2_revise") == []
    assert len(r.service.searches) == 1
    values = r.scores["values"]
    assert (values["b2.critiques"], values["b2.revisions"]) == (1, 0)
    assert values["b2.claims"] == 4  # the draft stands


async def test_the_budget_stops_rounds(tmp_path: Path) -> None:
    r = await run(tmp_path, replies={"baseline_b1": [ANSWER]}, profile_=profile(target_tokens=1))
    assert r.provider.prompts("baseline_b2_critique") == []
    assert r.scores["values"]["b2.critiques"] == 0


async def test_rounds_stop_at_max_rounds(tmp_path: Path) -> None:
    r = await run(
        tmp_path,
        replies={
            "baseline_b1": [ANSWER],
            "baseline_b2_critique": [CRITIQUE, CRITIQUE],
            "baseline_b2_revise": [REVISED, REVISED],
        },
        profile_=profile(max_rounds=2, target_tokens=None),
    )
    assert len(r.provider.prompts("baseline_b2_critique")) == 2
    assert len(r.service.searches) == 2  # the second search runs in round 1 only
    assert r.scores["values"]["b2.revisions"] == 2


async def test_zero_rounds_is_b1_with_a_b2_name(tmp_path: Path) -> None:
    r = await run(tmp_path, replies={"baseline_b1": [ANSWER]}, profile_=profile(max_rounds=0))
    assert r.provider.prompts("baseline_b2_critique") == []
    assert r.scores["values"]["b2.claims"] == 4


async def test_a_failed_revision_keeps_the_last_answer(tmp_path: Path) -> None:
    r = await run(
        tmp_path,
        replies={
            "baseline_b1": [ANSWER],
            "baseline_b2_critique": [CRITIQUE],
            "baseline_b2_revise": ["not json", "still not json"],  # the call and its retry
        },
    )
    values = r.scores["values"]
    assert values["b2.revise_failures"] == 1 and values["b2.revisions"] == 0
    assert values["b2.claims"] == 4 and len(r.view.issues) == 2  # the draft
    assert values["b2.evidence"] == 4  # the second search's chunk is still registered


def test_shown_puts_cited_then_new_then_rank_first() -> None:
    answer = B1Answer.model_validate(ANSWER)  # cites E-001, E-002, E-003, E-099
    assert _shown(answer, total=10, first=6, cap=5) == [0, 1, 2, 6, 7]
    assert _shown(answer, total=10, first=6, cap=2) == [0, 1]
    assert _shown(answer, total=4, first=4, cap=12) == [0, 1, 2, 3]


def test_quote_errors_follow_the_verifier_rule() -> None:
    good = copy.deepcopy(ANSWER)
    good["issues"] = good["issues"][:1]
    good["issues"][0]["position_a"]["claims"] = [claim("x", ("E-001", STATUTE_QUOTE))]
    good["issues"][0]["position_b"]["claims"] = [claim("y", ("e-3", SATYABRATA_QUOTE))]
    assert _quote_errors(B1Answer.model_validate(good), HITS) == ""
