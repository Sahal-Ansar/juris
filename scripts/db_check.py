"""Check that Postgres is reachable and pgvector is installed.

Usage: docker compose up -d && uv run scripts/db_check.py
"""

import os
import sys

import psycopg
from dotenv import load_dotenv


def conninfo() -> str:
    load_dotenv()
    return psycopg.conninfo.make_conninfo(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "juris"),
        password=os.getenv("POSTGRES_PASSWORD", "juris"),
        dbname=os.getenv("POSTGRES_DB", "juris"),
        connect_timeout=5,
    )


def main() -> int:
    try:
        with psycopg.connect(conninfo()) as conn:
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
