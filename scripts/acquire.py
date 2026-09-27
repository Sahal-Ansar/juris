"""Acquire the corpus slice (PLAN 3.1). Every step is resumable; re-running skips finished work.

Usage:
    uv run scripts/acquire.py all [--slice mvp_contract] [--workers 12]
    uv run scripts/acquire.py sc-metadata
    uv run scripts/acquire.py sc-scan [--years 2019-2023] [--workers 12]
    uv run scripts/acquire.py select
    uv run scripts/acquire.py sc-fetch
    uv run scripts/acquire.py hc
    uv run scripts/acquire.py statutes
    uv run scripts/acquire.py manifest
"""

import argparse
import json
import os
import sys
import time

from juris.config import get_settings
from juris.ingest import acquire as a
from juris.ingest.slice import CORPUS_CONFIGS_DIR, load_slice


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def years_arg(value: str | None, default: tuple[int, int]) -> list[int]:
    if not value:
        return list(range(default[0], default[1] + 1))
    lo, _, hi = value.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire the corpus slice (PLAN 3.1)")
    parser.add_argument(
        "step",
        choices=[
            "all",
            "sc-metadata",
            "sc-scan",
            "select",
            "sc-fetch",
            "hc",
            "statutes",
            "manifest",
        ],
    )
    parser.add_argument("--slice", default="mvp_contract")
    parser.add_argument("--years", help="e.g. 2019-2023 (sc-scan only; default: slice years)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)

    config = load_slice(args.slice)
    config_path = CORPUS_CONFIGS_DIR / f"{args.slice}.yaml"
    paths = a.Paths(get_settings().data_dir)
    fetcher = a.Fetcher()
    steps = (
        [args.step]
        if args.step != "all"
        else ["sc-metadata", "sc-scan", "select", "sc-fetch", "hc", "statutes", "manifest"]
    )
    log(f"data dir: {paths.data_dir}")

    for step in steps:
        if step == "sc-metadata":
            files = a.fetch_sc_metadata(fetcher, paths)
            log(f"SC metadata: {len(files)} files")
        elif step == "sc-scan":
            for year in years_arg(args.years, config.supreme_court.years):
                started = time.monotonic()
                out = a.scan_sc_year(year, fetcher, paths, workers=args.workers)
                log(f"SC scan {year}: {out.name} ({time.monotonic() - started:.0f}s)")
        elif step == "select":
            records = a.read_scan(paths)
            selection = a.select_sc(
                config.supreme_court, records, a.read_sc_metadata(paths), config.seed
            )
            paths.selection.write_text(selection.model_dump_json(indent=1) + "\n", encoding="utf-8")
            log(
                f"SC selection: core {len(selection.core)}, one-hop {len(selection.one_hop)}, "
                f"distractors {len(selection.distractors)} = {selection.total} "
                f"(scanned {selection.scanned}, errors {selection.scan_errors}); "
                f"adjustments: {selection.adjustments or 'none'}"
            )
        elif step == "sc-fetch":
            selection = a.ScSelection.model_validate_json(paths.selection.read_text("utf-8"))
            results = a.fetch_sc(selection, a.load_scan_index(paths), fetcher, paths)
            new = sum(r.downloaded for r in results)
            log(
                f"SC PDFs: {len(results)} selected, {new} downloaded, "
                f"{len(results) - new} already present"
            )
        elif step == "hc":
            hc = config.high_courts
            files = a.fetch_hc_metadata([c.code for c in hc.courts], hc.years, fetcher, paths)
            candidates = a.hc_candidates(config, paths)
            log(f"HC metadata: {len(files)} files, {len(candidates)} candidates by case type")
            kept = a.screen_hc(config, candidates, fetcher, paths)
            log(f"HC kept: {len(kept)} (cap {hc.max_judgments})")
        elif step == "statutes":
            for record in a.fetch_statutes(config, fetcher, paths):
                log(
                    f"statute {record['key']}: {record['source_title']} "
                    f"({len(record['text'])} chars, {record['candidates']} candidates)"
                )
        elif step == "manifest":
            selection = a.ScSelection.model_validate_json(paths.selection.read_text("utf-8"))
            notes = [*selection.adjustments]
            screened = (
                [
                    json.loads(line)
                    for line in paths.hc_progress.read_text("utf-8").splitlines()
                    if line.strip()
                ]
                if paths.hc_progress.exists()
                else []
            )
            notes.append(
                f"HC screened {len(screened)} candidates, kept {sum(e['kept'] for e in screened)}"
            )
            notes.append(
                f"SC scanned {selection.scanned} judgments ({selection.scan_errors} unreadable); "
                f"core rule {selection.core_rule.model_dump()}; one-hop cap {selection.one_hop_cap}"
            )
            manifest = a.build_manifest(config, config_path, selection, paths, notes)
            out = a.write_manifest(manifest, paths)
            log(f"snapshot {manifest.snapshot_id}: {manifest.counts} -> {out}")
    log(f"done ({fetcher.requests} HTTP requests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
