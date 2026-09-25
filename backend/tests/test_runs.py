import json
from pathlib import Path

import psycopg
import pytest

from juris.config import Role, Settings, load_profile, resolve_run_config
from juris.llm import ChatMessage, LLMGateway
from juris.llm.cache import ResponseCache
from juris.llm.providers import FakeProvider
from juris.prompts import PromptFormatError, load_prompt, parse_prompt, prompt_hashes
from juris.runs import RunManifest, RunRecorder, save_manifest_to_db

PROMPT = "---\nid: judge\nversion: 1\n---\nWrite the analysis for {question}.\n"


def write_prompts(folder: Path) -> Path:
    folder.mkdir()
    (folder / "judge.md").write_text(PROMPT, encoding="utf-8")
    (folder / "juror.md").write_text("---\nid: juror\nversion: 2\n---\nScore it.\n", "utf-8")
    (folder / "README.md").write_text("# not a template\n", encoding="utf-8")
    return folder


def test_prompt_loads_with_header(tmp_path: Path) -> None:
    prompt = load_prompt("judge", write_prompts(tmp_path / "prompts"))
    assert (prompt.id, prompt.version) == ("judge", 1)
    assert prompt.body == "Write the analysis for {question}.\n"


def test_changing_a_prompt_changes_its_hash(tmp_path: Path) -> None:
    folder = write_prompts(tmp_path / "prompts")
    before = prompt_hashes(folder)
    (folder / "judge.md").write_text(PROMPT.replace("analysis", "summary"), encoding="utf-8")
    after = prompt_hashes(folder)

    assert set(before) == {"judge@v1", "juror@v2"}
    assert before["judge@v1"] != after["judge@v1"]
    assert before["juror@v2"] == after["juror@v2"]


def test_hash_ignores_line_endings() -> None:
    assert parse_prompt(PROMPT).sha256 == parse_prompt(PROMPT.replace("\n", "\r\n")).sha256


@pytest.mark.parametrize("text", ["no header", "---\nid: x\n---\nbody", "---\nid: x\nversion: 1\n"])
def test_bad_prompt_headers_are_rejected(text: str) -> None:
    with pytest.raises(PromptFormatError):
        parse_prompt(text)


def test_repo_prompts_are_valid() -> None:
    prompt_hashes()  # raises if any committed template has a bad header


async def test_dummy_run_produces_complete_manifest(tmp_path: Path) -> None:
    settings = Settings(_env_file=None)
    config = resolve_run_config(settings, load_profile("juris_full"))
    fake = FakeProvider({"judge": ['{"ok": true}'], "juror": ["fine"]})
    gateway = LLMGateway(
        {"anthropic": fake}, prices=settings.llm_prices, cache=ResponseCache(tmp_path / "c.db")
    )
    question = [ChatMessage(role="user", content="q")]

    with RunRecorder(
        config,
        corpus_snapshot_id="snap-test",
        embedding_model="bge-m3",
        ledger=gateway.ledger,
        runs_dir=tmp_path / "runs",
        prompts_dir=write_prompts(tmp_path / "prompts"),
        seeds={"juror_1": 11},
    ) as run:
        assert json.loads(run.manifest_path.read_text("utf-8"))["status"] == "running"
        model = config.model_for(Role.JUDGE)
        await gateway.complete(question, model=model, tags={"agent": "judge"})
        await gateway.complete(question, model=model, tags={"agent": "judge"})  # cache hit
        await gateway.complete(question, model=model, seed=11, tags={"agent": "juror"})

    manifest = RunManifest.model_validate_json(run.manifest_path.read_text("utf-8"))
    assert manifest == run.manifest
    # Every field is filled on a successful run; only `error` stays empty.
    empty = [k for k, v in manifest.model_dump().items() if v in (None, "", {}, [])]
    assert empty == ["error"]
    assert manifest.status == "completed"
    assert manifest.ended_at is not None and manifest.ended_at >= manifest.started_at
    assert len(manifest.git.commit) == 40
    assert set(manifest.models) == set(Role)
    assert manifest.models[Role.JUROR].name == "claude-opus-5-5"
    assert set(manifest.prompt_hashes) == {"judge@v1", "juror@v2"}
    assert manifest.seeds == {"profile": 0, "juror_1": 11}
    assert manifest.totals.llm_calls == 3
    assert manifest.totals.cached_calls == 1
    assert manifest.totals.input_tokens == 20
    # Opus 5.5 at $4 / $20 per MTok: 20 in, len('{"ok": true}') + len("fine") out.
    assert manifest.totals.usd == pytest.approx((20 * 4 + 16 * 20) / 1_000_000)


def test_failed_run_is_recorded(tmp_path: Path) -> None:
    settings = Settings(_env_file=None)
    config = resolve_run_config(settings, load_profile("b1"))
    recorder = RunRecorder(
        config,
        corpus_snapshot_id="snap-test",
        embedding_model="bge-m3",
        ledger=LLMGateway({}, prices={}).ledger,
        runs_dir=tmp_path / "runs",
        prompts_dir=write_prompts(tmp_path / "prompts"),
    )
    with pytest.raises(RuntimeError), recorder:
        raise RuntimeError("retriever down")

    manifest = RunManifest.model_validate_json(recorder.manifest_path.read_text("utf-8"))
    assert manifest.status == "failed"
    assert manifest.error == "RuntimeError: retriever down"


@pytest.mark.db
def test_manifest_is_saved_to_postgres(tmp_path: Path) -> None:
    db_url = Settings().database_url(driver=None)
    try:
        psycopg.connect(db_url, connect_timeout=2).close()
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable (docker compose up -d)")

    config = resolve_run_config(Settings(_env_file=None), load_profile("b0"))
    with RunRecorder(
        config,
        corpus_snapshot_id="snap-test",
        embedding_model="bge-m3",
        ledger=LLMGateway({}, prices={}).ledger,
        runs_dir=tmp_path / "runs",
        prompts_dir=write_prompts(tmp_path / "prompts"),
        db_url=db_url,
    ) as run:
        pass

    with psycopg.connect(db_url, connect_timeout=5) as conn:
        row = conn.execute(
            "SELECT status, manifest->>'profile' FROM runs WHERE run_id = %s",
            (run.manifest.run_id,),
        ).fetchone()
        conn.execute("DELETE FROM runs WHERE run_id = %s", (run.manifest.run_id,))
    assert row == ("completed", "b0")

    # Saving again updates rather than duplicating.
    save_manifest_to_db(run.manifest, db_url)
    with psycopg.connect(db_url, connect_timeout=5) as conn:
        count = conn.execute(
            "SELECT count(*) FROM runs WHERE run_id = %s", (run.manifest.run_id,)
        ).fetchone()
        conn.execute("DELETE FROM runs WHERE run_id = %s", (run.manifest.run_id,))
    assert count == (1,)
