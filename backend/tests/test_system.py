"""The whole Juris pipeline (PLAN 7.12) through the experiment runner, on a scripted model and
a fake retrieval service: juris_full and every ablation finish with an analysis, and each
ablation changes what it should. No model, no database."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from juris.config import ModelsConfig, ModelSpec, Settings, load_profile, resolve_run_config
from juris.eval.juris_eval import load_split
from juris.eval.runner import ExperimentSpec, GatewayFactory, run_experiment
from juris.llm.types import ProviderRequest, ProviderResponse
from juris.pipeline.system import ABLATIONS, JurisSystem
from juris.retrieval.lookup import LinkedCase
from tests.test_b1 import HITS, STATUTE_QUOTE
from tests.test_rebuttal import Service
from tests.test_verify_gate import FATEH_QUOTE

ID = re.compile(r"\b([ACO]-\d{3})\b")


class FullService(Service):
    """The rebuttal tests' fake retrieval, with S3's lookups (nothing beyond the cited chunks)."""

    def chunk_ids(self, doc_id: str, section: str | None = None) -> list[str]:
        return []

    def citing_cases(self, doc_id: str) -> list[LinkedCase]:
        return []


def reply(prompt: str, agent: str, body: str) -> Any:
    """A plausible output for each prompt, built from what the prompt shows."""
    if prompt == "issue_framer":
        return {
            "facts": ["Earnest money was paid."],
            "assumptions": [],
            "issues": [
                {
                    "question": "Can the seller forfeit the earnest money?",
                    "positions": ["No.", "Yes."],
                    "queries": ["forfeiture of earnest money", "section 74 penalty"],
                    "statute_refs": [],
                }
            ],
        }
    if prompt == "counsel":
        evidence, quote = (
            ("E-001", STATUTE_QUOTE) if agent == "counsel_a" else ("E-002", FATEH_QUOTE)
        )
        claim = {"text": f"Claim of {agent}.", "reasoning": "Because.",
                 "citations": [{"evidence_id": evidence, "quote": quote}]}  # fmt: skip
        return {"claims": [claim]}
    if prompt == "precedent_analyst":
        arguments = sorted(set(re.findall(r"\b(A-\d{3})\b", body)))
        return {
            "principle": "p",
            "material_facts": "f",
            "distinguishing_factors": [],
            "fits": [{"argument_id": a, "fit": "supports_by_analogy", "reason": "r"}
                     for a in arguments],
        }  # fmt: skip
    if prompt == "verifier_entailment":
        return {"label": "supports", "justification": "ok"}
    if prompt == "counsel_search":
        return {"queries": []}
    if prompt == "counsel_rebuttal":
        return {"counterarguments": []}
    if prompt == "cross_examiner":
        return {"objections": [{"type": "logical_gap", "target_id": "C-001", "text": "A gap.",
                                "severity": "medium"}]}  # fmt: skip
    if prompt == "counsel_reply":
        ids = sorted({i for i in ID.findall(body) if i.startswith("O-")})
        return {"replies": [{"objection_id": i, "stance": "maintain", "response": "It holds."}
                            for i in ids]}  # fmt: skip
    if prompt == "juror":
        positions = re.search(r"exactly once: (.+)\.", body)
        assert positions is not None
        return {"scores": [
            {"position_id": p.strip(), "evidence_strength": 3, "precedent_fit": 3,
             "counterargument_handling": 3, "unresolved_risk": 3, "justification": "j",
             "cited_ids": ["C-001"]}
            for p in positions.group(1).split(",")
        ]}  # fmt: skip
    if prompt == "judge":
        labels = sorted(set(re.findall(r"\b(U\d+):", body)))
        return {
            "issues": [
                {
                    "issue_id": "I-1",
                    "positions": [
                        {"position_id": "I-1.A", "summary": "Capped [E-001].",
                         "key_evidence": ["E-001"]},
                        {"position_id": "I-1.B", "summary": "Kept.", "key_evidence": []},
                    ],
                    "key_conflicts": [],
                    "leaning": "leaning_A",
                    "confidence": "low",
                    "confidence_reason": "Thin.",
                    "unresolved": [{"question": u, "answer": "Open."} for u in labels],
                }
            ],
            "overall_summary": "Compensation is capped [E-001].",
            "limitations": [],
        }  # fmt: skip
    raise AssertionError(f"no reply for {prompt}")


class Model:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[ProviderRequest] = []

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.calls.append(request)
        prompt = request.tags["prompt"].split("@")[0]
        out = reply(prompt, request.tags.get("agent", ""), request.messages[0].content)
        text = json.dumps(out)
        return ProviderResponse(text=text, input_tokens=100, output_tokens=len(text),
                                stop_reason="end_turn")  # fmt: skip


async def run(profile_name: str, tmp_path: Path) -> tuple[dict[str, Any], Model]:
    model = Model()
    spec_model = ModelSpec(provider="fake", name="m", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=spec_model))  # type: ignore[call-arg]
    profile = load_profile(profile_name)
    config = resolve_run_config(settings, profile)
    items = load_split("dev")[:1]
    spec = ExperimentSpec(profile_name, profile_name, "juris", "dev", (items[0].id,), (0,))
    system = JurisSystem(service=lambda: FullService(hits=list(HITS)))
    result = await run_experiment(
        spec,
        items,
        config,
        system=system,
        gateways=GatewayFactory({"fake": model}, prices={}),
        results_dir=tmp_path,
    )
    [outcome] = result.outcomes
    run_dir = result.directory / "runs" / f"{items[0].id}-s0"
    view = json.loads((run_dir / "caseview.json").read_text(encoding="utf-8"))
    assert outcome.status == "completed", view.get("failure")
    return view, model


def prompts(model: Model) -> set[str]:
    return {c.tags["prompt"].split("@")[0] for c in model.calls}


async def test_juris_full_runs_end_to_end(tmp_path: Path) -> None:
    view, model = await run("juris_full", tmp_path)
    assert view["analysis"] is not None and view["status"] == "completed"
    assert prompts(model) == {
        "issue_framer", "counsel", "precedent_analyst", "verifier_entailment", "counsel_search",
        "counsel_rebuttal", "cross_examiner", "counsel_reply", "juror", "judge",
    }  # fmt: skip
    assert len(view["jury_ballots"]) == 3 and view["objections"]
    assert view["analysis"]["created_by"]["stage"] == "S10"


@pytest.mark.parametrize("name", ABLATIONS)
async def test_every_ablation_runs_and_changes_what_it_should(name: str, tmp_path: Path) -> None:
    view, model = await run(name, tmp_path)
    assert view["analysis"] is not None and view["status"] == "completed"
    used = prompts(model)
    claims = view["claims"].values()
    if name == "no_opponent":
        assert {c["position_id"] for c in claims} == {"I-1.A"}
        assert "counsel_rebuttal" not in used and "counsel_search" not in used
    if name == "no_precedent_analyst":
        assert "precedent_analyst" not in used and view["precedent_cards"]
    if name == "no_rebuttal":
        assert "counsel_rebuttal" not in used
    if name == "no_cross_exam":
        assert "cross_examiner" not in used and not view["objections"]
    if name == "no_jury":
        assert "juror" not in used and view["jury_aggregate"] is None
    if name == "no_verifier_gates":
        # counsel and the cross-examiner worked without verdicts; the citations were checked
        # once, before the Judge (events at S10)
        stages = {v["result"]["created_by"]["stage"] for v in view["verifications"]}
        assert stages == {"S10"}
