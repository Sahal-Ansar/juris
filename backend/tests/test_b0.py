"""Baseline B0 (PLAN 6.2): authority resolution, and B0 through the runner (FakeProvider)."""

import datetime as dt
import json
from pathlib import Path
from typing import ClassVar

import pytest
from sqlalchemy import create_engine

from juris.baselines.authorities import AuthorityResolver, same_parties, summarise
from juris.baselines.b0 import B0Answer, B0System
from juris.config import ModelsConfig, ModelSpec, Settings, load_profile, resolve_run_config
from juris.eval.juris_eval import load_split
from juris.eval.runner import ExperimentSpec, GatewayFactory, run_experiment
from juris.events.fold import CaseView
from juris.llm.providers.fake import FakeProvider
from juris.retrieval.lookup import CaseMatch, Lookup, Section, TitleMatch

FATEH = "SC-1964_1_515_528"


class StubLookups:
    """Citations, titles and sections a test corpus knows."""

    cases: ClassVar[dict[str, CaseMatch]] = {
        "[1964] 1 SCR 515": CaseMatch(
            "[1964] 1 SCR 515", FATEH, FATEH, "FATEH CHAND versus BALKISHAN DAS", "document"
        ),
        "(1872) 1 IA 1": CaseMatch("(1872) 1 IA 1", "PC-1872", None, None, "citation_graph"),
    }
    titles: ClassVar[dict[str, str]] = {"Satyabrata Ghose v. Mugneeram Bangur": "SC-1954_1_310_329"}
    sections: ClassVar[dict[tuple[str, str], str]] = {("contract_act", "74"): "contract_act:74"}

    def get_case_by_citation(self, citation: str) -> CaseMatch | None:
        return self.cases.get(citation)

    def get_case_by_title(
        self, title: str, limit: int = 5, min_score: float = 0.4
    ) -> list[TitleMatch]:
        doc = self.titles.get(title)
        return [TitleMatch(doc, title.upper(), dt.date(1953, 11, 16), 0.9)] if doc else []

    def get_section(self, act: str, section: str) -> Section | None:
        sid = self.sections.get((act, section))
        if sid is None:
            return None
        return Section(sid, act, "Indian Contract Act", 1872, section, "Compensation", "t",
                       False, None, None)  # fmt: skip


def resolver() -> AuthorityResolver:
    return AuthorityResolver(StubLookups())


def test_judgments_resolve_by_citation_then_title() -> None:
    r = resolver()
    found = r.judgment("Fateh Chand v. Balkishan Das", "[1964] 1 SCR 515")
    assert (found.status, found.doc_id, found.how) == ("corpus", FATEH, "citation")
    wrong = r.judgment("Maula Bux v. Union of India", "[1964] 1 SCR 515")
    assert wrong.status == "mismatch" and wrong.doc_id == FATEH
    outside = r.judgment("Mohori Bibee v. Dharmodas Ghose", "(1872) 1 IA 1")
    assert outside.status == "outside_corpus" and outside.doc_id is None
    by_title = r.judgment("Satyabrata Ghose v. Mugneeram Bangur", "[1954] SCR 999")
    assert (by_title.status, by_title.how) == ("corpus", "title")
    made_up = r.judgment("Ramesh Kumar v. Suresh Traders", "(2099) 99 SCC 999")
    assert made_up.status == "unverified" and made_up.doc_id is None


def test_statutes_resolve_by_provision() -> None:
    r = resolver()
    found = r.statute("Indian Contract Act, 1872, s. 74", None)
    assert (found.status, found.doc_id, found.section_id) == (
        "corpus",
        "ACT-contract_act",
        "contract_act:74",
    )
    assert r.statute("Indian Contract Act, 1872, s. 999", None).status == "unverified"
    assert r.statute("Transfer of Property Act, 1882, s. 108", None).status == "unknown_act"


def test_same_parties_is_lenient() -> None:
    assert same_parties("Satyabrata Ghose v. Mugneeram", "SATYABRATA GROSE versus MUGNEERAM BANGUR")
    assert not same_parties("Maula Bux v. Union of India", "FATEH CHAND versus BALKISHAN DAS")
    # only generic words in common ("state", "union") is not a match
    assert not same_parties("Union of India v. State of Kerala", "STATE OF BIHAR versus UNION")


def test_summarise_counts_and_rate() -> None:
    r = resolver()
    values = summarise(
        [
            r.judgment("Fateh Chand v. Balkishan Das", "[1964] 1 SCR 515"),
            r.judgment("Maula Bux v. Union of India", "[1964] 1 SCR 515"),
            r.judgment("Ramesh Kumar v. Suresh Traders", "(2099) 99 SCC 999"),
            r.judgment("Mohori Bibee v. Dharmodas Ghose", "(1872) 1 IA 1"),
            r.statute("Indian Contract Act, 1872, s. 74", None),
        ],
        "b0",
    )
    assert values["b0.authorities_cited"] == 5 and values["b0.judgments_cited"] == 4
    assert values["b0.judgments_in_corpus"] == 1 and values["b0.judgments_outside_corpus"] == 1
    assert values["b0.unverified_judgment_rate"] == pytest.approx(2 / 4)  # mismatch + made up
    assert values["b0.statutes_cited"] == 1 and values["b0.statutes_unverified"] == 0
    assert summarise([], "b0")["b0.unverified_judgment_rate"] is None


ANSWER = {
    "facts": ["The buyer paid earnest money."],
    "assumptions": ["No arbitration clause."],
    "issues": [
        {
            "question": "Can the seller forfeit the whole earnest money?",
            "position_a": {
                "statement": "Yes.",
                "summary": "Earnest money may be forfeited under the contract.",
                "authorities": [1, 3],
            },
            "position_b": {
                "statement": "No.",
                "summary": "Only reasonable compensation under s. 74.",
                "authorities": [2, 4],
            },
            "leaning": "leaning_B",
            "confidence": "moderate",
            "confidence_reason": "The cases go both ways.",
            "unresolved_questions": [],
        }
    ],
    "authorities": [
        {"number": 1, "kind": "judgment", "name": "Fateh Chand v. Balkishan Das",
         "citation": "[1964] 1 SCR 515", "proposition": "Section 74 applies."},
        {"number": 2, "kind": "judgment", "name": "Ramesh Kumar v. Suresh Traders",
         "citation": "(2099) 99 SCC 999", "proposition": "Made up."},
        {"number": 3, "kind": "statute", "name": "Indian Contract Act, 1872, s. 74",
         "citation": None, "proposition": "Reasonable compensation."},
        {"number": 4, "kind": "judgment", "name": "Mohori Bibee v. Dharmodas Ghose",
         "citation": "(1872) 1 IA 1", "proposition": "Not in the slice."},
    ],
    "overall_summary": "The seller may keep only reasonable compensation. " * 3,
    "limitations": ["Answered without documents."],
}  # fmt: skip


def test_the_answer_schema_accepts_the_fixture() -> None:
    B0Answer.model_validate(ANSWER)


async def test_b0_through_the_runner(tmp_path: Path) -> None:
    item = load_split("dev")[0]
    model = ModelSpec(provider="fake", name="local-model", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    config = resolve_run_config(settings, load_profile("b0"))
    fake = FakeProvider({"b0": [json.dumps(ANSWER)]})
    spec = ExperimentSpec("b0-test", "b0", "b0", "dev", (item.id,), (0,))
    result = await run_experiment(
        spec,
        [item],
        config,
        system=B0System(resolver=resolver),
        gateways=GatewayFactory({"fake": fake}, prices={}),
        results_dir=tmp_path,
    )
    assert [o.status for o in result.outcomes] == ["completed"]
    request = fake.calls[0]
    assert request.model == model and request.tags["prompt"] == "baseline_b0@v1"
    assert item.question in request.messages[0].content
    assert request.json_schema is not None  # structured output

    run = tmp_path / "b0-test" / "runs" / f"{item.id}-s0"
    view = CaseView.model_validate_json((run / "caseview.json").read_text("utf-8"))
    analysis = view.analysis
    assert analysis is not None and len(analysis.issues) == 1
    positions = analysis.issues[0].positions
    assert positions[0].supporting_authorities == [FATEH]  # only corpus judgments
    assert positions[1].supporting_authorities == []
    assert {(s.document_id, tuple(s.pinpoints_used)) for s in analysis.sources} == {
        (FATEH, ()),
        ("ACT-contract_act", ("contract_act:74",)),
    }
    assert any("Ramesh Kumar" in lim and "not found" in lim for lim in analysis.limitations)
    assert any(
        "Mohori Bibee" in lim and "outside the corpus" in lim for lim in analysis.limitations
    )

    scores = json.loads((run / "scores.json").read_text("utf-8"))
    values = scores["values"]
    assert values["b0.judgments_cited"] == 3
    assert values["b0.unverified_judgment_rate"] == pytest.approx(1 / 3)
    assert values["citation_validity"] is None  # B0 registers no evidence
    assert len(scores["details"]["system"]["authorities"]) == 4
    assert "b0.unverified_judgment_rate" in result.summary["aggregate"]


@pytest.mark.db
def test_resolver_on_the_corpus() -> None:
    from juris.config import get_settings

    r = AuthorityResolver(Lookup(create_engine(get_settings().database_url())))
    saw_pipes = r.judgment("Oil & Natural Gas Corporation v. Saw Pipes", "(2003) 5 SCC 705")
    assert saw_pipes.status == "corpus" and saw_pipes.doc_id == "SC-2003_3_691_741"
    by_title = r.judgment("Satyabrata Ghose v. Mugneeram Bangur", None)
    assert by_title.status == "corpus" and by_title.how == "title"
    assert r.judgment("Ramesh Kumar v. Suresh Traders", "(2099) 99 SCC 999").status == "unverified"
    assert r.statute("Indian Contract Act, 1872, s. 56", None).section_id == "contract_act:56"
