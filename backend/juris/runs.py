"""Run manifests: everything needed to reproduce and audit a run (IDEA_final §15).

    with RunRecorder(run_config, corpus_snapshot_id=..., embedding_model=...,
                     ledger=gateway.ledger) as run:
        ...  # model calls through the gateway
    run.manifest  # also in data/runs/<run_id>/manifest.json (and the DB if given)

The manifest is written when the run starts (status ``running``) and again when it
ends (``completed`` or ``failed``), so a crash still leaves a record.
"""

import secrets
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import psycopg

from juris.config import REPO_ROOT, Role, RunConfig
from juris.llm.cost import CostLedger
from juris.models.run import GitState, Run, RunStatus, RunTotals
from juris.prompts import PROMPTS_DIR, prompt_hashes

RunManifest = Run  # the manifest is the domain model's Run (juris.models.run)

__all__ = ["GitState", "RunManifest", "RunRecorder", "RunStatus", "RunTotals", "git_state"]


def new_run_id(profile: str) -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{profile}-{secrets.token_hex(3)}"


def git_state(repo: Path = REPO_ROOT) -> GitState:
    """HEAD commit and whether tracked files have uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return GitState(commit="unknown", dirty=True)
    return GitState(commit=commit, dirty=bool(status.strip()))


class RunRecorder:
    """Context manager that records one run's manifest."""

    def __init__(
        self,
        config: RunConfig,
        *,
        corpus_snapshot_id: str,
        embedding_model: str,
        ledger: CostLedger,
        runs_dir: Path,
        seeds: dict[str, int] | None = None,
        prompts_dir: Path = PROMPTS_DIR,
        db_url: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._ledger_start = len(ledger.records)
        self._db_url = db_url
        run_id = run_id or new_run_id(config.profile.name)
        self.run_dir = runs_dir / run_id
        self.manifest = RunManifest(
            run_id=run_id,
            status="running",
            profile=config.profile.name,
            config=config,
            git=git_state(),
            models={role: config.model_for(role) for role in Role},
            prompt_hashes=prompt_hashes(prompts_dir),
            corpus_snapshot_id=corpus_snapshot_id,
            embedding_model=embedding_model,
            seeds={"profile": config.profile.seed, **(seeds or {})},
            started_at=datetime.now(UTC),
        )

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def __enter__(self) -> Self:
        self._write()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.manifest = self.manifest.model_copy(
            update={
                "status": "failed" if exc is not None else "completed",
                "error": f"{exc_type.__name__}: {exc}" if exc_type is not None else None,
                "ended_at": datetime.now(UTC),
                "totals": self._totals(),
            }
        )
        self._write()

    def _totals(self) -> RunTotals:
        records = self._ledger.records[self._ledger_start :]
        live = [r for r in records if not r.cached]
        return RunTotals(
            llm_calls=len(records),
            cached_calls=len(records) - len(live),
            input_tokens=sum(r.input_tokens for r in live),
            output_tokens=sum(r.output_tokens for r in live),
            usd=sum(r.cost_usd or 0.0 for r in live),
            unpriced_calls=sum(1 for r in live if r.cost_usd is None),
        )

    def _write(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(self.manifest.model_dump_json(indent=2), encoding="utf-8")
        if self._db_url is not None:
            save_manifest_to_db(self.manifest, self._db_url)


RUNS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      text PRIMARY KEY,
    profile     text NOT NULL,
    status      text NOT NULL,
    started_at  timestamptz NOT NULL,
    ended_at    timestamptz,
    manifest    jsonb NOT NULL
)
"""


def save_manifest_to_db(manifest: RunManifest, db_url: str) -> None:
    """Upsert into ``runs``. The table moves under Alembic in PLAN 3.6."""
    with psycopg.connect(db_url, connect_timeout=5) as conn:
        conn.execute(RUNS_TABLE_DDL)
        conn.execute(
            """
            INSERT INTO runs (run_id, profile, status, started_at, ended_at, manifest)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id) DO UPDATE SET
                status = EXCLUDED.status,
                ended_at = EXCLUDED.ended_at,
                manifest = EXCLUDED.manifest
            """,
            (
                manifest.run_id,
                manifest.profile,
                manifest.status,
                manifest.started_at,
                manifest.ended_at,
                manifest.model_dump_json(),
            ),
        )
