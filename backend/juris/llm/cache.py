"""SQLite response cache keyed on everything that determines a model's output."""

import hashlib
import json
import sqlite3
from pathlib import Path

from juris.llm.types import ProviderRequest, ProviderResponse


def cache_key(request: ProviderRequest) -> str:
    """Hash of provider, model, messages, params and schema. Tags are excluded."""
    payload = request.model_dump(mode="json", exclude={"tags"})
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            " key TEXT PRIMARY KEY,"
            " response TEXT NOT NULL,"
            " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        self._conn.commit()

    def get(self, key: str) -> ProviderResponse | None:
        row = self._conn.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return ProviderResponse.model_validate_json(row[0]) if row else None

    def put(self, key: str, response: ProviderResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, response) VALUES (?, ?)",
            (key, response.model_dump_json()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
