"""Check that Postgres is reachable and pgvector is installed.

Usage: docker compose up -d && uv run scripts/db_check.py
"""

import sys

import psycopg

from juris.config import get_settings


def main() -> int:
    try:
        with psycopg.connect(get_settings().database_url(driver=None), connect_timeout=5) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            server = conn.info.server_version
    except psycopg.OperationalError as exc:
        print(f"FAIL: cannot connect to Postgres: {exc}", file=sys.stderr)
        return 1
    if row is None:
        print("FAIL: pgvector extension not installed", file=sys.stderr)
        return 1
    print(f"OK: Postgres {server // 10000}.{server % 10000}, pgvector {row[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
