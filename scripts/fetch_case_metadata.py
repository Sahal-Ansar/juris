"""Fetch KanoonGPT case-law metadata rows for the snapshot's judgments (PLAN 3.4).

KanoonGPT/indian-case-laws (Apache-2.0) republishes the AWS SC/HC metadata with parsed parties,
coram, docket numbers and SC headnotes: 17M rows in one parquet file per year (10.9 GB). Only the
row groups whose CNR range can hold one of our judgments are read, over HTTP range requests
(about 0.4 GB in total), and only our rows are kept:
``<data_dir>/raw/hf_case_laws/structured_v1_subset.parquet`` plus a JSON manifest.

Usage: uv run scripts/fetch_case_metadata.py [--snapshot ID] [--workers 6]
"""

import argparse
import concurrent.futures as cf
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from juris.config import get_settings
from juris.ingest.acquire import USER_AGENT

REPO = "KanoonGPT/indian-case-laws"
REVISION = "42dd6a97345e9811b3d6219f15c250f1ce2bc5af"  # pinned for reproducibility
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/structured/v1"
YEARS = range(1950, 2027)
COLUMNS = [
    "case_title", "party_petitioner", "party_respondent", "party_caption", "docket_number",
    "cnr_number", "neutral_citation", "law_report_citation", "court_name", "court_code",
    "bench_name", "presiding_judge", "coram_members", "decision_date", "registration_date",
    "disposition_text", "source_relative_path", "source_filename", "headnote_text",
    "quality_json", "dataset_source",
]  # fmt: skip


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (for pyarrow's parquet reader)."""

    def __init__(self, url: str, retries: int = 5) -> None:
        self.url, self.retries, self.pos, self.fetched = url, retries, 0, 0
        head = self._request({}, method="HEAD")
        self.size = int(head.headers["Content-Length"])

    def _request(self, headers: dict[str, str], method: str = "GET") -> Any:
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.url, headers={"User-Agent": USER_AGENT, **headers}, method=method
                )
                return urllib.request.urlopen(req, timeout=120)
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        with self._request({"Range": f"bytes={self.pos}-{end - 1}"}) as response:
            data: bytes = response.read()
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def our_cnrs(data_dir: Path, snapshot_id: str) -> set[str]:
    snapshot = json.loads((data_dir / "snapshots" / f"{snapshot_id}.json").read_text("utf-8"))
    sc_paths: dict[str, str] = {}
    for f in (data_dir / "raw" / "aws_sc" / "metadata").glob("*.parquet"):
        for row in pq.read_table(f, columns=["path", "cnr"]).to_pylist():
            if row["cnr"]:
                sc_paths.setdefault(row["path"], row["cnr"])
    cnrs = set()
    for doc in snapshot["documents"]:
        if doc["source"] == "aws_sc":
            cnrs.add(sc_paths[doc["doc_id"].removeprefix("SC-")])
        elif doc["source"] == "aws_hc" and (m := re.match(r"HC-([A-Z]{4}\d+)_", doc["doc_id"])):
            cnrs.add(m.group(1))
    return cnrs


def fetch_year(year: int, cnrs: list[str]) -> tuple[int, pa.Table | None, dict[str, Any]]:
    f = HttpRangeFile(f"{BASE}/year={year}/indian_case_laws_structured_v1_year={year}.parquet")
    pf = pq.ParquetFile(f)
    md = pf.metadata
    cnr_col = pf.schema_arrow.names.index("cnr_number")
    groups = []
    for g in range(md.num_row_groups):
        stats = md.row_group(g).column(cnr_col).statistics
        lo, hi = (stats.min, stats.max) if stats and stats.has_min_max else ("", "￿")
        if any(lo <= c <= hi for c in cnrs):
            groups.append(g)
    wanted = pa.array(cnrs)
    tables = []
    for g in groups:
        table = pf.read_row_group(g, columns=COLUMNS)
        mask = pc.is_in(table["cnr_number"], value_set=wanted)
        if pc.any(mask).as_py():
            tables.append(table.filter(mask))
    result = pa.concat_tables(tables) if tables else None
    stats = {
        "year": year,
        "file_bytes": f.size,
        "row_groups_read": len(groups),
        "row_groups_total": md.num_row_groups,
        "bytes_fetched": f.fetched,
        "rows_kept": result.num_rows if result is not None else 0,
    }
    return year, result, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch KanoonGPT metadata rows (PLAN 3.4)")
    parser.add_argument("--snapshot", default="mvp_contract-1f53c208a8")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    data_dir = get_settings().data_dir
    out_dir = data_dir / "raw" / "hf_case_laws"
    out_dir.mkdir(parents=True, exist_ok=True)
    cnrs = sorted(our_cnrs(data_dir, args.snapshot))
    print(f"{len(cnrs)} CNRs to look up in {REPO}@{REVISION[:8]}", flush=True)
    tables, stats = [], []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for year, table, s in pool.map(lambda y: fetch_year(y, cnrs), YEARS):
            stats.append(s)
            if table is not None:
                tables.append(table)
            print(
                f"  {year}: {s['row_groups_read']}/{s['row_groups_total']} row groups, "
                f"{s['bytes_fetched'] / 1e6:.1f} MB, {s['rows_kept']} rows",
                flush=True,
            )
    subset = pa.concat_tables(tables, promote_options="permissive")
    pq.write_table(subset, out_dir / "structured_v1_subset.parquet")
    found = set(subset["cnr_number"].to_pylist())
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "license": "apache-2.0",
        "snapshot_id": args.snapshot,
        "cnrs_requested": len(cnrs),
        "cnrs_found": len(found & set(cnrs)),
        "rows": subset.num_rows,
        "bytes_fetched": sum(s["bytes_fetched"] for s in stats),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "years": stats,
    }
    (out_dir / "structured_v1_subset.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        f"kept {subset.num_rows} rows for {manifest['cnrs_found']}/{len(cnrs)} CNRs; "
        f"fetched {manifest['bytes_fetched'] / 1e9:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
