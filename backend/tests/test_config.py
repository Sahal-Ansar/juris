import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from juris.config import (
    PIPELINE_CONFIGS_DIR,
    ModelSpec,
    Role,
    RunConfig,
    Settings,
    load_profile,
    resolve_run_config,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real environment out of these tests."""
    for key in list(os.environ):
        if key.startswith(("JURIS_", "POSTGRES_")) or key.endswith("_API_KEY"):
            monkeypatch.delenv(key)


def write_env(tmp_path: Path) -> Path:
    env = tmp_path / ".env"
    env.write_text(
        "POSTGRES_PORT=6543\n"
        "POSTGRES_PASSWORD=p@ss word\n"
        "ANTHROPIC_API_KEY=sk-test-secret\n"
        "JURIS_LLM_CACHE=off\n"
        "JURIS_MODELS__DEFAULT__NAME=test-model\n"
        "JURIS_BUDGET__MAX_TOKENS_PER_CASE=200000\n",
        encoding="utf-8",
    )
    return env


def test_settings_load_from_env_file(tmp_path: Path) -> None:
    settings = Settings(_env_file=write_env(tmp_path))

    assert settings.postgres_port == 6543
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-test-secret"
    assert settings.llm_cache == "off"
    assert settings.models.default.name == "test-model"
    assert settings.budget.max_tokens_per_case == 200000
    assert settings.database_url() == (
        "postgresql+psycopg://juris:p%40ss%20word@127.0.0.1:6543/juris"
    )
    assert settings.database_url(driver=None).startswith("postgresql://")


def test_run_config_round_trips_to_json(tmp_path: Path) -> None:
    settings = Settings(_env_file=write_env(tmp_path))
    profile_path = tmp_path / "custom.yaml"
    profile_path.write_text(
        "name: custom\n"
        "kind: juris\n"
        "seed: 7\n"
        "models:\n"
        "  default: {provider: anthropic, name: main-model}\n"
        "  roles:\n"
        "    juror: {provider: openai, name: cheap-model}\n"
        "stages:\n"
        "  s5_rebuttal: 'off'\n"
        "  s6_cross_exam: passthrough\n"
        "params:\n"
        "  rebuttal_rounds: 2\n",
        encoding="utf-8",
    )

    run = resolve_run_config(settings, load_profile(profile_path))
    as_json = run.model_dump_json()

    assert RunConfig.model_validate_json(as_json) == run
    assert "sk-test-secret" not in as_json
    assert "p@ss" not in as_json
    assert run.model_for(Role.JUROR) == ModelSpec(provider="openai", name="cheap-model")
    assert run.model_for(Role.JUDGE).name == "main-model"
    # The profile didn't set a budget, so the settings default applies.
    assert run.budget.max_tokens_per_case == 200000
    assert run.llm_cache == "off"
    assert run.profile.stages.s5_rebuttal == "off"
    assert run.profile.stages.s6_cross_exam == "passthrough"


@pytest.mark.parametrize("path", sorted(PIPELINE_CONFIGS_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_repo_profiles_are_valid(path: Path) -> None:
    profile = load_profile(path.stem)
    assert profile.name == path.stem


def test_juris_full_uses_spec_defaults() -> None:
    profile = load_profile("juris_full")
    assert profile.kind == "juris"
    assert profile.params.jurors == 3
    assert profile.params.rebuttal_rounds == 1
    assert all(mode == "on" for mode in profile.stages.model_dump().values())


@pytest.mark.parametrize(
    "body",
    [
        "name: x\nkind: juris\nunknown_key: 1\n",
        "name: x\nkind: juris\nparams: {rebuttal_rounds: 3}\n",
        "name: x\nkind: juris\nparams: {evidence_per_issue_min: 30}\n",
        "name: x\nkind: juris\nstages: {s8_jury: maybe}\n",
        "name: x\nkind: b9\n",
    ],
)
def test_invalid_profiles_are_rejected(tmp_path: Path, body: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_profile(path)
