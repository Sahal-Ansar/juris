"""Download the small source samples for PLAN 1.1 into data/raw/samples/.

Public sources, no credentials. Files already on disk are skipped, so re-running
is cheap. Picks are evenly spaced over each listing, so they're stable across runs
(as long as the upstream listing doesn't change).

Usage: uv run scripts/fetch_samples.py
"""

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from juris.config import get_settings

OUT = get_settings().data_dir / "raw" / "samples"
UA = {"User-Agent": "juris-research/0.1 (dataset reconnaissance)"}

SC_BUCKET = "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com"
HC_BUCKET = "https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com"
HF = "https://huggingface.co/datasets"
HF_VIEWER = "https://datasets-server.huggingface.co"


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        data: bytes = r.read()
        return data


def save(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(get(url))
    print(f"{dest.stat().st_size / 1e6:7.2f} MB  {dest.relative_to(OUT)}")


def s3_keys(bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = ""
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        xml = get(f"{bucket}/?{urllib.parse.urlencode(query)}").decode()
        keys += re.findall(r"<Key>(.*?)</Key>", xml)
        match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not match:
            return sorted(keys)
        token = match.group(1)


def spread(items: list[str], n: int) -> list[str]:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def supreme_court() -> None:
    out = OUT / "aws_sc"
    save(f"{SC_BUCKET}/metadata/parquet/year=2023/metadata.parquet", out / "2023_metadata.parquet")
    for year, n in ((2023, 50), (1960, 5)):
        pdfs = [
            k for k in s3_keys(SC_BUCKET, f"data/pdf/year={year}/english/") if k.endswith(".pdf")
        ]
        for key in spread(pdfs, n):
            name = key.rsplit("/", 1)[1]
            save(f"{SC_BUCKET}/{key}", out / f"pdf_{year}" / name)
            stem = name.removesuffix("_EN.pdf")
            save(
                f"{SC_BUCKET}/metadata/json/year={year}/{stem}.json",
                out / f"json_{year}" / f"{stem}.json",
            )


def high_court() -> None:
    out = OUT / "aws_hc"
    part = "year=2023/court=17_21/bench=meghalaya"
    save(f"{HC_BUCKET}/metadata/parquet/{part}/metadata.parquet", out / "meghalaya_2023.parquet")
    pdfs = [k for k in s3_keys(HC_BUCKET, f"data/pdf/{part}/") if k.endswith(".pdf")]
    for key in spread(pdfs, 20):
        save(f"{HC_BUCKET}/{key}", out / "pdf_meghalaya_2023" / key.rsplit("/", 1)[1])
    save(
        "https://raw.githubusercontent.com/vanga/indian-high-court-judgments/main/opendata/docs/high_courts.csv",
        out / "high_courts.csv",
    )


def hugging_face() -> None:
    cases = "KanoonGPT/indian-case-laws"
    docs = "KanoonGPT/indian-legal-documents"
    save(
        f"{HF}/{cases}/resolve/main/sample/v1/indian_case_laws_sample_v1.parquet",
        OUT / "hf_case_laws" / "sample_v1.parquet",
    )
    for name, dataset in (("hf_case_laws", cases), ("hf_legal_docs", docs)):
        save(f"{HF}/{dataset}/resolve/main/README.md", OUT / name / "README.md")
        save(f"{HF_VIEWER}/info?dataset={dataset}", OUT / name / "viewer_info.json")
        splits = json.loads(get(f"{HF_VIEWER}/splits?dataset={dataset}"))
        (OUT / name / "viewer_splits.json").write_text(json.dumps(splits, indent=2), "utf-8")
    config = "default"
    save(
        f"{HF_VIEWER}/first-rows?dataset={docs}&config={config}&split=train",
        OUT / "hf_legal_docs" / "first_rows.json",
    )


if __name__ == "__main__":
    supreme_court()
    high_court()
    hugging_face()
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"done: {total / 1e6:.1f} MB in {OUT}")
