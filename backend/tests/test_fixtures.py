"""UI fixtures (PLAN 2.4): valid against the exported schemas, fold to their CaseView,
obey the Case Record rules, and are unmistakably mock."""

import importlib.util
import json
from types import ModuleType
from typing import Any

import jsonschema
import pytest

from juris.config import REPO_ROOT
from juris.events import CaseView, Event, fold, parse_event
from juris.models.common import VerificationStatus

RUNS = REPO_ROOT / "fixtures" / "runs"
SCHEMAS = REPO_ROOT / "schemas"
NAMES = ["coercion_full", "failure_case"]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_lines(name: str) -> list[dict[str, Any]]:
    text = (RUNS / f"{name}.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def events(name: str) -> list[Event]:
    return [parse_event(line) for line in raw_lines(name)]


def validator(schema_file: str) -> jsonschema.protocols.Validator:
    schema = json.loads((SCHEMAS / schema_file).read_text(encoding="utf-8"))
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


@pytest.mark.parametrize("name", NAMES)
def test_events_validate_against_the_exported_schema(name: str) -> None:
    check = validator("event.schema.json")
    for line in raw_lines(name):
        errors = list(check.iter_errors(line))
        assert not errors, f"{name} seq {line.get('seq')}: {errors[0].message}"


@pytest.mark.parametrize("name", NAMES)
def test_fold_matches_committed_caseview(name: str) -> None:
    committed = (RUNS / f"{name}.caseview.json").read_text(encoding="utf-8")
    assert fold(events(name)) == CaseView.model_validate_json(committed)
    errors = list(validator("case_view.schema.json").iter_errors(json.loads(committed)))
    assert not errors, errors[0].message


def test_fixtures_are_fresh() -> None:
    """The committed fixtures are exactly what scripts/build_fixtures.py produces."""
    assert _script("build_fixtures").main(["--check"]) == 0


@pytest.mark.parametrize("name", NAMES)
def test_every_document_is_marked_mock(name: str) -> None:
    view = fold(events(name))
    for doc in view.documents.values():
        assert doc.title.startswith("[MOCK] "), doc.id
        assert doc.source_url and doc.source_url.startswith("mock://"), doc.id


def test_full_run_has_the_required_shape() -> None:
    view = fold(events("coercion_full"))
    assert view.status == "completed"
    assert len(view.issues) == 2 and all(len(i.positions) == 2 for i in view.issues.values())
    assert len(view.evidence) == 20
    assert len(view.jury_ballots) == 3 and view.jury_aggregate is not None
    assert view.analysis is not None
    statuses = {v.result.status for v in view.verifications}
    assert {VerificationStatus.UNSUPPORTED, VerificationStatus.INVALID} <= statuses
    assert view.precedent_cards and view.objections and view.unresolved_questions
    assert view.counterarguments


def test_full_run_obeys_the_case_record_rules() -> None:
    """The fixture must model a *correct* run (IDEA_final §8 invariants, D-013)."""
    view = fold(events("coercion_full"))
    assert view.analysis is not None
    # 1-2: citations reference registered evidence and quote the chunk verbatim, except the
    # one deliberately invalid quote, which the verifier flags as `invalid`.
    invalid = {
        (v.claim_id, v.evidence_id)
        for v in view.verifications
        if v.result.status is VerificationStatus.INVALID
    }
    for argument in view.arguments.values():
        for cite in argument.citations:
            assert cite.evidence_id in view.evidence
            chunk = view.chunks[view.evidence[cite.evidence_id].chunk_id]
            assert (cite.quote in chunk.text) != ((cite.claim_id, cite.evidence_id) in invalid)
    # 3: the Judge cites only evidence whose best verification is verified or weak.
    best: dict[str, VerificationStatus] = {}
    rank = [
        VerificationStatus.INVALID,
        VerificationStatus.UNSUPPORTED,
        VerificationStatus.WEAK,
        VerificationStatus.VERIFIED,
    ]
    for v in view.verifications:
        if rank.index(v.result.status) > rank.index(best.get(v.evidence_id, rank[0])):
            best[v.evidence_id] = v.result.status
    for issue in view.analysis.issues:
        for position in issue.positions:
            for ref in position.key_evidence:
                assert best.get(ref.evidence_id) in (
                    VerificationStatus.VERIFIED,
                    VerificationStatus.WEAK,
                ), ref.evidence_id
    # 4: objections target existing IDs.
    known = set(view.claims) | set(view.evidence) | set(view.arguments) | set(view.issues)
    assert all(o.target_id in known for o in view.objections.values())
    # 5: the analysis covers exactly the framed issues.
    assert [i.issue_id for i in view.analysis.issues] == list(view.issues)
    # D-013: every relied-on precedent with unverified treatment is in the limitations.
    relied = {
        a for i in view.analysis.issues for p in i.positions for a in p.supporting_authorities
    }
    for doc_id in relied:
        card = view.precedent_cards.get(doc_id)
        if card and not card.treatment_verified:
            assert any(doc_id in line for line in view.analysis.limitations), doc_id


def test_failure_run_shows_budget_and_validation_problems() -> None:
    view = fold(events("failure_case"))
    assert view.status == "failed" and view.error
    assert len(view.validation_failures) == 2 and view.budget_warnings
    assert {s.status for s in view.stages.values()} >= {"completed", "skipped"}
    assert view.analysis is None


def test_replay_emits_sse_frames_with_capped_delays() -> None:
    replay = _script("replay_fixture")
    sleeps: list[float] = []
    frames = list(
        replay.replay(
            replay.read_events(RUNS / "failure_case.jsonl"),
            speed=2.0,
            max_delay=1.0,
            sleep=sleeps.append,
        )
    )
    assert len(frames) == len(raw_lines("failure_case"))
    assert frames[0].startswith("id: 1\nevent: case_created\ndata: {")
    # Events are 3 s apart: 3 / 2 = 1.5 s, capped at 1.0 s.
    assert sleeps and all(s == 1.0 for s in sleeps)
    uncapped: list[float] = []
    list(
        replay.replay(
            replay.read_events(RUNS / "failure_case.jsonl"),
            speed=2.0,
            max_delay=5.0,
            sleep=uncapped.append,
        )
    )
    assert set(uncapped) == {1.5}
