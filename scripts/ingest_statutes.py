"""Split the snapshot's statutes into sections with temporal validity (PLAN 3.5).

Reads the cleaned Act texts from 3.2 (``<data_dir>/interim/parsed/<snapshot>/ACT-<act>.json.gz``),
splits them (``juris.ingest.statutes``), applies ``configs/statutes/amendments.yaml`` and writes
``<data_dir>/processed/statutes/<act_id>.jsonl`` (one section per line). The statute text isn't
redistributable (docs/DATA_NOTICE.md), so the output stays in the data directory.

Usage: uv run scripts/ingest_statutes.py [--snapshot ID]
"""

import argparse
import gzip
import json
import sys

from juris.config import get_settings
from juris.ingest.statutes import (
    apply_amendments,
    arrangement,
    load_acts,
    load_amendments,
    split_act,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Section-level statute ingestion (PLAN 3.5)")
    parser.add_argument("--snapshot", default="mvp_contract-1f53c208a8")
    args = parser.parse_args(argv)

    data_dir = get_settings().data_dir
    acts = load_acts()
    amendments = load_amendments()
    out_dir = data_dir / "processed" / "statutes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for act_id, act in acts.items():
        path = data_dir / "interim" / "parsed" / args.snapshot / f"ACT-{act_id}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            text = json.load(fh)["clean_text"]
        sections = split_act(text, act)
        curated = apply_amendments(sections, amendments)
        listed = {no for no, _ in arrangement(text)[0]}
        with (out_dir / f"{act_id}.jsonl").open("w", encoding="utf-8") as fh:
            for s in sections:
                fh.write(json.dumps(s.to_json(), ensure_ascii=False) + "\n")
        found = {s.section for s in sections}
        print(
            f"{act_id}: {len(sections)} sections ({sum(s.repealed for s in sections)} repealed, "
            f"{curated} curated); arrangement lists {len(listed)}, not in body: "
            f"{sorted(listed - found) or 'none'}; body only: {sorted(found - listed) or 'none'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
