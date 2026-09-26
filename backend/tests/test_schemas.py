"""The committed JSON Schemas must match the models (PLAN 2.3). TypeScript is checked in CI."""

import importlib.util
from pathlib import Path
from types import ModuleType

from juris.config import REPO_ROOT


def _export_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "export_schemas", REPO_ROOT / "scripts" / "export_schemas.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_json_schemas_are_fresh() -> None:
    stale = _export_module().stale_files(with_ts=False)
    assert stale == [], f"re-run `uv run scripts/export_schemas.py`; stale: {stale}"


def test_export_is_deterministic() -> None:
    module = _export_module()
    assert module.json_schemas() == module.json_schemas()


def test_every_event_type_is_in_the_contract() -> None:
    import json

    from juris.events import EVENT_TYPES

    bundle = json.loads(Path(REPO_ROOT / "schemas" / "juris.schema.json").read_text("utf-8"))
    union = bundle["$defs"]["Event"]
    mapped = set(union["discriminator"]["mapping"])
    assert mapped == set(EVENT_TYPES)
