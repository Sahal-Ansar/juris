"""Shared fixtures: a fresh, empty Postgres database per ``db`` test (dropped afterwards)."""

import secrets
from collections.abc import Iterator

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from juris.config import REPO_ROOT, Settings


@pytest.fixture
def engine() -> Iterator[Engine]:
    """An engine on a throwaway database; the test is skipped if Postgres isn't running."""
    settings = Settings()
    admin = settings.database_url(driver=None)
    try:
        conn = psycopg.connect(admin, connect_timeout=2, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable (docker compose up -d)")
    name = f"juris_test_{secrets.token_hex(4)}"
    conn.execute(f"CREATE DATABASE {name}")
    url = settings.database_url().rsplit("/", 1)[0] + f"/{name}"
    eng = create_engine(url)
    try:
        yield eng
    finally:
        eng.dispose()
        conn.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        conn.close()


def migrate(conn: Connection, revision: str = "head", down: bool = False) -> None:
    """Run Alembic on an open connection (the env passes it through)."""
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.attributes["connection"] = conn
    (command.downgrade if down else command.upgrade)(cfg, revision)
