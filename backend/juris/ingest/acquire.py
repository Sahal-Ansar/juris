"""Acquire the corpus slice (PLAN 3.1), driven by ``configs/corpus/<slice>.yaml``.

The SC core rule needs judgment text, but the SC metadata has no Act fields (D-010), so:

1. ``sc_scan``: stream each year's English tar once, extract text with pdfium, and keep only
   a small signal record per judgment (Act mentions, section references, citations) in
   ``interim/sc_scan/``. The tar is deleted after each year.
2. ``select_sc``: apply the slice rules (core, one-hop, distractors, size guard) to the records.
3. ``fetch_sc``: download the selected judgments' PDFs individually into ``raw/``.

High Courts are screened candidate by candidate until the cap is reached; statutes come from
the KanoonGPT legal-documents shards. Everything is resumable and checksummed, and the result
is recorded in a snapshot manifest (``snapshots/<id>.json``).
"""

import concurrent.futures as cf
import gzip
import hashlib
import json
import random
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import pypdfium2
from pydantic import BaseModel, ConfigDict, Field

from juris.ingest.slice import (
    ActSignal,
    Adjustment,
    CoreRule,
    SliceConfig,
    SupremeCourtSlice,
    act_signal,
    cited_sc_authorities,
    is_core,
)

SC_BUCKET = "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com"
HC_BUCKET = "https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com"
HF_LEGAL_DOCS = "https://huggingface.co/datasets/KanoonGPT/indian-legal-documents/resolve/main"
USER_AGENT = "juris-research/0.1 (corpus acquisition; github.com/Sahal-Ansar/juris)"
_NUMBERED_PARA = re.compile(r"^\s*(\d{1,3})\.\s+\S", re.M)


# ---- Paths and fetching ---------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    data_dir: Path

    @property
    def sc_metadata(self) -> Path:
        return self.data_dir / "raw" / "aws_sc" / "metadata"

    @property
    def sc_pdf(self) -> Path:
        return self.data_dir / "raw" / "aws_sc" / "pdf"

    @property
    def sc_scan(self) -> Path:
        return self.data_dir / "interim" / "sc_scan"

    @property
    def hc_metadata(self) -> Path:
        return self.data_dir / "raw" / "aws_hc" / "metadata"

    @property
    def hc_pdf(self) -> Path:
        return self.data_dir / "raw" / "aws_hc" / "pdf"

    @property
    def hc_progress(self) -> Path:
        return self.data_dir / "interim" / "hc_screening.jsonl"

    @property
    def statutes(self) -> Path:
        return self.data_dir / "raw" / "statutes"

    @property
    def selection(self) -> Path:
        return self.data_dir / "interim" / "sc_selection.json"

    @property
    def snapshots(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def tmp(self) -> Path:
        return self.data_dir / "tmp"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class FetchResult:
    path: Path
    sha256: str
    bytes: int
    downloaded: bool


class Fetcher:
    """HTTP(S)/file downloads with retries, a minimum interval between requests, atomic
    writes (``.part`` then rename) and skip-if-present."""

    def __init__(self, min_interval: float = 0.05, retries: int = 5, timeout: float = 300) -> None:
        self.min_interval = min_interval
        self.retries = retries
        self.timeout = timeout
        self._last = 0.0
        self.requests = 0

    def _open(self, url: str) -> Any:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(self.retries + 1):
            try:
                self._last = time.monotonic()
                self.requests += 1
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                return urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise
                if attempt == self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries:
                    raise
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    def get(self, url: str) -> bytes:
        with self._open(url) as response:
            data: bytes = response.read()
            return data

    def download(self, url: str, dest: Path, expect_sha256: str | None = None) -> FetchResult:
        if dest.exists():
            sha = sha256_file(dest)
            if expect_sha256 and sha != expect_sha256:
                raise ValueError(f"{dest} exists with sha256 {sha}, expected {expect_sha256}")
            return FetchResult(dest, sha, dest.stat().st_size, downloaded=False)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        digest = hashlib.sha256()
        with self._open(url) as response, part.open("wb") as fh:
            for block in iter(lambda: response.read(1 << 20), b""):
                digest.update(block)
                fh.write(block)
        sha = digest.hexdigest()
        if expect_sha256 and sha != expect_sha256:
            part.unlink()
            raise ValueError(f"{url}: sha256 {sha} does not match expected {expect_sha256}")
        part.replace(dest)
        return FetchResult(dest, sha, dest.stat().st_size, downloaded=True)

    def s3_list(self, bucket: str, prefix: str) -> list[tuple[str, int]]:
        keys: list[tuple[str, int]] = []
        token = ""
        while True:
            query = {"list-type": "2", "prefix": prefix}
            if token:
                query["continuation-token"] = token
            xml = self.get(f"{bucket}/?{urllib.parse.urlencode(query)}").decode()
            for block in re.findall(r"<Contents>(.*?)</Contents>", xml, re.S):
                key = re.search(r"<Key>(.*?)</Key>", block)
                size = re.search(r"<Size>(\d+)</Size>", block)
                if key and size:
                    keys.append((key.group(1), int(size.group(1))))
            match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
            if not match:
                return keys
            token = match.group(1)


# ---- PDF text and signals --------------------------------------------------------------


def pdf_text(data: bytes) -> tuple[str, int]:
    """Text layer of a PDF (pdfium) and its page count. No OCR here (PLAN 3.2)."""
    pdf = pypdfium2.PdfDocument(data)
    try:
        pages = len(pdf)
        return "\n".join(page.get_textpage().get_text_range() for page in pdf), pages
    finally:
        pdf.close()


def numbered_paragraphs(text: str) -> int:
    """Length of the longest run of sequentially numbered paragraphs (1., 2., 3., ...)."""
    best = run = 0
    previous = None
    for number in (int(n) for n in _NUMBERED_PARA.findall(text)):
        run = run + 1 if previous is not None and number == previous + 1 else 1
        best = max(best, run)
        previous = number
    return best


def scan_record(name: str, data: bytes) -> dict[str, Any]:
    """Selection signals for one SC judgment PDF (no text is kept)."""
    record: dict[str, Any] = {
        "path": name.rsplit("/", 1)[-1].removesuffix(".pdf").removesuffix("_EN"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    try:
        text, pages = pdf_text(data)
    except Exception as exc:  # corrupt PDFs are recorded, not fatal
        return {**record, "error": f"{type(exc).__name__}: {exc}"}
    signal = act_signal(text)
    cites = cited_sc_authorities(text)
    return {
        **record,
        "pages": pages,
        "chars": len(text),
        "mentions": signal.mentions,
        "section_refs": signal.section_refs,
        "scr": sorted([int(y), int(v), int(p)] for y, v, p in cites["scr"]),
        "insc": sorted([int(y), int(n)] for y, n in cites["insc"]),
    }


def _scan_member(item: tuple[str, bytes]) -> dict[str, Any]:
    return scan_record(*item)


def _tar_pdfs(tar_path: Path) -> Iterator[tuple[str, bytes]]:
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if member.isfile() and member.name.lower().endswith(".pdf"):
                fh = tar.extractfile(member)
                if fh is not None:
                    yield member.name, fh.read()


def scan_sc_year(year: int, fetcher: Fetcher, paths: Paths, workers: int = 8) -> Path:
    """Stream one year's English tar into ``sc_scan/year=YYYY.jsonl.gz``. Skips done years."""
    out = paths.sc_scan / f"year={year}.jsonl.gz"
    if out.exists():
        return out
    tar_path = paths.tmp / f"sc_english_{year}.tar"
    fetcher.download(f"{SC_BUCKET}/data/tar/year={year}/english/english.tar", tar_path)
    records: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(_scan_member, _tar_pdfs(tar_path), chunksize=4):
            records.append({"year": year, **record})
    records.sort(key=lambda r: r["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with gzip.open(part, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    part.replace(out)
    tar_path.unlink()
    return out


def path_year(path: str) -> int:
    """Year of a judgment path: '2023_16_872_887' and 'S_1959_1_979_1008' (Supp. SCR)."""
    match = re.search(r"(\d{4})", path)
    if not match:
        raise ValueError(f"no year in judgment path {path!r}")
    return int(match.group(1))


def read_scan(paths: Paths) -> list[dict[str, Any]]:
    """Scan records, one per judgment. ~5k judgments appear byte-identical in two
    consecutive years' tars; the copy whose tar year matches the path's year wins."""
    by_path: dict[str, dict[str, Any]] = {}
    for file in sorted(paths.sc_scan.glob("year=*.jsonl.gz")):
        with gzip.open(file, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                kept = by_path.get(record["path"])
                if kept is None or (
                    kept["year"] != path_year(record["path"])
                    and record["year"] == path_year(record["path"])
                ):
                    by_path[record["path"]] = record
    return [by_path[p] for p in sorted(by_path)]


# ---- SC metadata and citation resolution ---------------------------------------------


def fetch_sc_metadata(fetcher: Fetcher, paths: Paths) -> list[Path]:
    files = []
    for key, _ in fetcher.s3_list(SC_BUCKET, "metadata/parquet/"):
        if key.endswith(".parquet"):
            year = re.search(r"year=(\d{4})", key)
            if year:
                dest = paths.sc_metadata / f"year={year.group(1)}.parquet"
                files.append(fetcher.download(f"{SC_BUCKET}/{key}", dest).path)
    return files


def read_sc_metadata(paths: Paths) -> dict[str, dict[str, Any]]:
    columns = ["path", "title", "citation", "case_id", "decision_date", "year", "cnr"]
    rows: dict[str, dict[str, Any]] = {}
    for file in sorted(paths.sc_metadata.glob("year=*.parquet")):
        table = pq.read_table(file)
        present = [c for c in columns if c in table.column_names]
        for row in table.select(present).to_pylist():
            if row.get("path"):
                rows[row["path"]] = row
    return rows


_PATH = re.compile(r"^(\d{4})_(\d+)_(\d+)_(\d+)$")


@dataclass
class CitationIndex:
    """Resolves SCR citations (any page inside a judgment's SCR page range, taken from its
    ``path`` = year_volume_firstpage_lastpage) and neutral citations to SC judgment paths."""

    scr: dict[tuple[int, int], list[tuple[int, int, str]]] = field(default_factory=dict)
    insc: dict[tuple[int, int], str] = field(default_factory=dict)

    @classmethod
    def build(cls, paths: Iterable[str], metadata: dict[str, dict[str, Any]]) -> "CitationIndex":
        index = cls()
        for path in paths:
            match = _PATH.match(path)
            if match:
                year, volume, first, last = (int(g) for g in match.groups())
                index.scr.setdefault((year, volume), []).append((first, max(first, last), path))
            case_id = (metadata.get(path) or {}).get("case_id") or ""
            neutral = re.match(r"^(\d{4})\s*INSC\s*(\d+)$", str(case_id).strip())
            if neutral:
                index.insc[(int(neutral.group(1)), int(neutral.group(2)))] = path
        return index

    def resolve(self, record: dict[str, Any], via: Iterable[str]) -> set[str]:
        targets: set[str] = set()
        if "scr" in via:
            for year, volume, page in record.get("scr", []):
                for first, last, path in self.scr.get((year, volume), []):
                    if first <= page <= last:
                        targets.add(path)
        if "insc" in via:
            for year, number in record.get("insc", []):
                found = self.insc.get((year, number))
                if found:
                    targets.add(found)
        targets.discard(record["path"])  # SCR page headers cite the judgment itself
        return targets


# ---- SC selection ---------------------------------------------------------------------


class ScSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    core: list[str]
    one_hop: list[str]
    distractors: list[str]
    core_rule: CoreRule
    one_hop_cap: int
    adjustments: list[str] = Field(default_factory=list)
    scanned: int
    in_years: int
    scan_errors: int
    cited_by: dict[str, int] = Field(default_factory=dict)
    source_year: dict[str, int] = Field(
        default_factory=dict, description="Tar/S3 year each selected judgment was found under"
    )

    @property
    def total(self) -> int:
        return len(self.core) + len(self.one_hop) + len(self.distractors)


def _signal(record: dict[str, Any]) -> ActSignal:
    return ActSignal(
        mentions=record.get("mentions", {}), section_refs=record.get("section_refs", {})
    )


def _apply(rule: CoreRule, cap: int, adj: Adjustment) -> tuple[CoreRule, int]:
    update = {
        k: v for k, v in adj.model_dump().items() if v is not None and k != "one_hop_max_judgments"
    }
    new_rule = rule.model_copy(update=update)
    return new_rule, adj.one_hop_max_judgments if adj.one_hop_max_judgments is not None else cap


def select_sc(
    config: SupremeCourtSlice,
    records: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    seed: int,
) -> ScSelection:
    lo_year, hi_year = config.years
    usable = [
        r
        for r in records
        if lo_year <= int(r["year"]) <= hi_year and "error" not in r and r.get("chars", 0) > 0
    ]
    index = CitationIndex.build((r["path"] for r in usable), metadata)
    usable_paths = {r["path"] for r in usable}

    def run(rule: CoreRule, cap: int) -> tuple[list[str], list[str], Counter[str]]:
        core = sorted(r["path"] for r in usable if is_core(_signal(r), rule))
        core_set = set(core)
        cited: Counter[str] = Counter()
        for r in usable:
            if r["path"] in core_set:
                for target in index.resolve(r, config.one_hop.resolve_via):
                    if target not in core_set and target in usable_paths:
                        cited[target] += 1
        eligible = [p for p, n in cited.items() if n >= config.one_hop.min_cited_by]
        eligible.sort(key=lambda p: (-cited[p], p))
        return core, eligible[:cap], cited

    def distractor_count(selected: int) -> int:
        d = config.distractors
        return max(d.min, int(d.fraction_of_selected * selected + 0.5))  # round half up

    rule, cap, notes = config.core, config.one_hop.max_judgments, []
    core, hop, cited = run(rule, cap)
    low, high = config.size_bounds

    def total() -> int:
        return len(core) + len(hop) + distractor_count(len(core) + len(hop))

    for direction, steps in (
        ("too large", config.if_too_large),
        ("too small", config.if_too_small),
    ):
        for step in steps:
            size = total()
            if (direction == "too large" and size <= high) or (
                direction == "too small" and size >= low
            ):
                break
            rule, cap = _apply(rule, cap, step)
            notes.append(f"{direction} ({size}): applied {step.model_dump(exclude_none=True)}")
            core, hop, cited = run(rule, cap)

    selected = set(core) | set(hop)
    pool = sorted(p for p in usable_paths if p not in selected)
    k = min(len(pool), distractor_count(len(selected)))
    distractors = sorted(random.Random(seed).sample(pool, k))
    return ScSelection(
        core=core,
        one_hop=hop,
        distractors=distractors,
        core_rule=rule,
        one_hop_cap=cap,
        adjustments=notes,
        scanned=len(records),
        in_years=sum(1 for r in records if lo_year <= int(r["year"]) <= hi_year),
        scan_errors=sum(1 for r in records if "error" in r),
        cited_by={p: cited[p] for p in hop},
        source_year={
            r["path"]: int(r["year"])
            for r in usable
            if r["path"] in selected or r["path"] in distractors
        },
    )


def sc_pdf_url(path: str, year: int) -> str:
    return f"{SC_BUCKET}/data/pdf/year={year}/english/{path}_EN.pdf"


def sc_local_pdf(paths: Paths, path: str) -> Path:
    return paths.sc_pdf / f"year={path_year(path)}" / f"{path}_EN.pdf"


def fetch_sc(
    selection: ScSelection, scan: dict[str, dict[str, Any]], fetcher: Fetcher, paths: Paths
) -> list[FetchResult]:
    results = []
    for path in [*selection.core, *selection.one_hop, *selection.distractors]:
        expected = scan.get(path, {}).get("sha256")
        year = selection.source_year.get(path, path_year(path))
        url = sc_pdf_url(path, year)
        results.append(fetcher.download(url, sc_local_pdf(paths, path), expect_sha256=expected))
    return results


# ---- High Courts ----------------------------------------------------------------------


def case_type(title: str) -> str:
    """Case type prefix of an HC metadata title: 'CS(COMM)/123/2020 of X Vs Y' -> 'CS(COMM)'."""
    return title.split("/", 1)[0].strip().upper() if "/" in title else ""


def fetch_hc_metadata(
    courts: Iterable[str], years: tuple[int, int], fetcher: Fetcher, paths: Paths
) -> list[Path]:
    files = []
    for court in courts:
        for year in range(years[0], years[1] + 1):
            for key, _ in fetcher.s3_list(
                HC_BUCKET, f"metadata/parquet/year={year}/court={court}/"
            ):
                if key.endswith(".parquet"):
                    rel = key.removeprefix("metadata/parquet/")
                    files.append(
                        fetcher.download(f"{HC_BUCKET}/{key}", paths.hc_metadata / rel).path
                    )
    return files


def hc_candidates(config: SliceConfig, paths: Paths) -> list[dict[str, Any]]:
    """Metadata rows whose case type is a candidate type, in a seeded random order."""
    hc = config.high_courts
    wanted = {c.code: {t.upper() for t in c.case_types} for c in hc.courts}
    rows: list[dict[str, Any]] = []
    for file in sorted(paths.hc_metadata.rglob("metadata.parquet")):
        parts = dict(
            p.split("=", 1) for p in file.relative_to(paths.hc_metadata).parts[:-1] if "=" in p
        )
        if parts.get("court") not in wanted or not (
            hc.years[0] <= int(parts.get("year", 0)) <= hc.years[1]
        ):
            continue
        for row in (
            pq.read_table(file).select(["title", "pdf_link", "cnr", "decision_date"]).to_pylist()
        ):
            if case_type(row.get("title") or "") in wanted[parts["court"]] and row.get("pdf_link"):
                rows.append(
                    {
                        **row,
                        "year": parts["year"],
                        "court": parts["court"],
                        "bench": parts.get("bench", ""),
                    }
                )
    rows.sort(key=lambda r: (r["court"], r["pdf_link"]))
    random.Random(config.seed).shuffle(rows)
    return rows


def hc_pdf_key(row: dict[str, Any]) -> str:
    name = str(row["pdf_link"]).rsplit("/", 1)[-1]
    return f"data/pdf/year={row['year']}/court={row['court']}/bench={row['bench']}/{name}"


def screen_hc(
    config: SliceConfig,
    candidates: list[dict[str, Any]],
    fetcher: Fetcher,
    paths: Paths,
    max_examined: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch candidates in order and keep reasoned judgments that pass the core rule, until the
    cap. Progress is appended to ``hc_screening.jsonl``, so an interrupted run resumes."""
    hc = config.high_courts
    limit = max_examined if max_examined is not None else hc.max_examined
    done: dict[str, dict[str, Any]] = {}
    if paths.hc_progress.exists():
        for line in paths.hc_progress.read_text(encoding="utf-8").splitlines():
            if line.strip():
                previous = json.loads(line)
                done[previous["key"]] = previous
    kept = [e for e in done.values() if e["kept"]]
    paths.hc_progress.parent.mkdir(parents=True, exist_ok=True)
    with paths.hc_progress.open("a", encoding="utf-8") as log:
        for row in candidates:
            if len(kept) >= hc.max_judgments or len(done) >= limit:
                break
            key = hc_pdf_key(row)
            if key in done:
                continue
            entry: dict[str, Any] = {"key": key, "cnr": row.get("cnr"), "kept": False}
            try:
                data = fetcher.get(f"{HC_BUCKET}/{key}")
                text, pages = pdf_text(data)
                paras = numbered_paragraphs(text)
                reasoned = (
                    pages >= hc.reasoned_only.min_pages
                    and paras >= hc.reasoned_only.min_numbered_paragraphs
                )
                core = is_core(act_signal(text), hc.core)
                entry.update(pages=pages, paragraphs=paras, reasoned=reasoned, core=core)
                if reasoned and core:
                    dest = paths.hc_pdf / key.removeprefix("data/pdf/")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    entry.update(
                        kept=True, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
                    )
                    kept.append(entry)
            except urllib.error.HTTPError as exc:
                entry["error"] = f"HTTP {exc.code}"
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            done[key] = entry
            log.write(json.dumps(entry, sort_keys=True) + "\n")
            log.flush()
    return kept


# ---- Statutes -------------------------------------------------------------------------


def _norm_title(title: str) -> str:
    title = re.sub(r"\(.*?\)", " ", title.lower())
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def fetch_statutes(
    config: SliceConfig, fetcher: Fetcher, paths: Paths, shards: Iterable[int] = (0, 1)
) -> list[dict[str, Any]]:
    """Download the shards and save the best English principal text of each Act as JSON."""
    rows: list[dict[str, Any]] = []
    for n in shards:
        dest = paths.statutes / "hf_legal_documents" / f"train-{n:05d}.parquet"
        result = fetcher.download(f"{HF_LEGAL_DOCS}/data/train-{n:05d}.parquet", dest)
        cols = [
            "doc_id",
            "document_title",
            "document_type",
            "document_jurisdiction",
            "issue_date",
            "text",
        ]
        for row in pq.read_table(result.path, columns=cols).to_pylist():
            rows.append({**row, "shard": dest.name, "shard_sha256": result.sha256})
    saved = []
    for spec in config.statutes.acts:
        wanted = _norm_title(spec.title)
        matches = [
            r
            for r in rows
            if r["document_jurisdiction"] == "Central"
            and _norm_title(r["document_title"] or "") == wanted
            and (r["document_title"] or "").isascii()
            and (r["text"] or "").strip()
        ]
        if not matches:
            if spec.optional:
                continue
            raise LookupError(f"no English principal text found for {spec.title}")
        best = max(matches, key=lambda r: (len(r["text"]), r["doc_id"]))
        record = {
            "key": spec.key,
            "title": spec.title,
            "act_no": spec.act_no,
            "source_doc_id": best["doc_id"],
            "source_title": best["document_title"],
            "source_shard": best["shard"],
            "source_shard_sha256": best["shard_sha256"],
            "candidates": len(matches),
            "text": best["text"],
            "text_sha256": hashlib.sha256(best["text"].encode("utf-8")).hexdigest(),
        }
        out = paths.statutes / f"{spec.key}.json"
        out.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        saved.append(record)
    return saved


# ---- Snapshot manifest ------------------------------------------------------------------

Reason = Literal["core", "one_hop", "distractor", "high_court", "statute"]


class SnapshotEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    kind: Literal["judgment", "statute"]
    reason: Reason
    source: str
    source_url: str
    local_path: str = Field(description="Relative to the data directory")
    sha256: str
    bytes: int


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    slice_name: str
    slice_config_sha256: str
    created_at: datetime
    counts: dict[str, int]
    notes: list[str]
    documents: list[SnapshotEntry]


def build_manifest(
    config: SliceConfig,
    config_path: Path,
    selection: ScSelection,
    paths: Paths,
    notes: list[str],
) -> SnapshotManifest:
    entries: list[SnapshotEntry] = []
    reasons: dict[str, Reason] = {
        **{p: "core" for p in selection.core},
        **{p: "one_hop" for p in selection.one_hop},
        **{p: "distractor" for p in selection.distractors},
    }
    for path, reason in sorted(reasons.items()):
        local = sc_local_pdf(paths, path)
        if not local.exists():
            raise FileNotFoundError(f"selected SC judgment not downloaded: {local}")
        entries.append(
            SnapshotEntry(
                doc_id=f"SC-{path}",
                kind="judgment",
                reason=reason,
                source="aws_sc",
                source_url=sc_pdf_url(path, selection.source_year.get(path, path_year(path))),
                local_path=local.relative_to(paths.data_dir).as_posix(),
                sha256=sha256_file(local),
                bytes=local.stat().st_size,
            )
        )
    for local in sorted(paths.hc_pdf.rglob("*.pdf")):
        rel = local.relative_to(paths.hc_pdf).as_posix()
        entries.append(
            SnapshotEntry(
                doc_id=f"HC-{local.stem}",
                kind="judgment",
                reason="high_court",
                source="aws_hc",
                source_url=f"{HC_BUCKET}/data/pdf/{rel}",
                local_path=local.relative_to(paths.data_dir).as_posix(),
                sha256=sha256_file(local),
                bytes=local.stat().st_size,
            )
        )
    for local in sorted(paths.statutes.glob("*.json")):
        record = json.loads(local.read_text(encoding="utf-8"))
        entries.append(
            SnapshotEntry(
                doc_id=f"ACT-{record['key']}",
                kind="statute",
                reason="statute",
                source="hf_legal_docs",
                source_url=f"{HF_LEGAL_DOCS}/data/{record['source_shard']}#doc_id={record['source_doc_id']}",
                local_path=local.relative_to(paths.data_dir).as_posix(),
                sha256=record["text_sha256"],
                bytes=len(record["text"].encode("utf-8")),
            )
        )
    content = hashlib.sha256(
        "".join(f"{e.doc_id}:{e.sha256}\n" for e in entries).encode()
    ).hexdigest()
    counts = Counter(e.reason for e in entries)
    return SnapshotManifest(
        snapshot_id=f"{config.name}-{content[:10]}",
        slice_name=config.name,
        slice_config_sha256=sha256_file(config_path),
        created_at=datetime.now(UTC),
        counts={**dict(sorted(counts.items())), "total": len(entries)},
        notes=notes,
        documents=entries,
    )


def write_manifest(manifest: SnapshotManifest, paths: Paths) -> Path:
    out = paths.snapshots / f"{manifest.snapshot_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():  # same content -> same ID; keep the first created_at
        out.write_text(manifest.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return out


def load_scan_index(paths: Paths) -> dict[str, dict[str, Any]]:
    return {r["path"]: r for r in read_scan(paths)}


def rmtree_tmp(paths: Paths) -> None:
    if paths.tmp.exists():
        shutil.rmtree(paths.tmp)


__all__ = [
    "CitationIndex",
    "Fetcher",
    "Paths",
    "ScSelection",
    "SnapshotManifest",
    "build_manifest",
    "case_type",
    "fetch_hc_metadata",
    "fetch_sc",
    "fetch_sc_metadata",
    "fetch_statutes",
    "hc_candidates",
    "load_scan_index",
    "numbered_paragraphs",
    "pdf_text",
    "read_sc_metadata",
    "read_scan",
    "scan_record",
    "scan_sc_year",
    "screen_hc",
    "select_sc",
    "write_manifest",
]
