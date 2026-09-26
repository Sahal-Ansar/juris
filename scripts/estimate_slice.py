"""Estimate the size of a corpus slice from the PLAN 1.1 samples.

Usage: uv run scripts/estimate_slice.py [slice_name]
Needs: uv run scripts/fetch_samples.py (and the per-year counts in rows_per_year.json).
"""

import json
import logging
import math
import statistics
import sys

from pypdf import PdfReader

from juris.config import get_settings
from juris.ingest.slice import act_signal, cited_sc_authorities, is_core, load_slice

SAMPLES = get_settings().data_dir / "raw" / "samples" / "aws_sc"


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def main(name: str) -> None:
    logging.disable(logging.WARNING)  # pypdf chatter about malformed PDFs
    config = load_slice(name).supreme_court
    per_year: dict[str, int] = json.loads((SAMPLES / "rows_per_year.json").read_text("utf-8"))
    lo_year, hi_year = config.years
    population = sum(n for y, n in per_year.items() if lo_year <= int(y) <= hi_year)

    pdfs = sorted(SAMPLES.glob("pdf_*/*.pdf"))
    core_docs, cites = 0, []
    for pdf in pdfs:
        text = "\n".join((page.extract_text() or "") for page in PdfReader(pdf).pages)
        found = cited_sc_authorities(text)
        cites.append(sum(len(found[style]) for style in config.one_hop.resolve_via))
        if is_core(act_signal(text), config.core):
            core_docs += 1

    n = len(pdfs)
    rate = core_docs / n
    lo, hi = wilson(core_docs, n)
    core = round(rate * population)
    # One-hop: resolvable citations per judgment times the core size, before de-duplication.
    # Popular precedents repeat heavily, so this is an upper bound and the cap usually binds.
    raw_one_hop = round(core * statistics.mean(cites))
    one_hop = min(config.one_hop.max_judgments, raw_one_hop)
    selected = core + one_hop
    distractors = max(
        config.distractors.min, round(config.distractors.fraction_of_selected * selected)
    )
    total = selected + distractors
    low_b, high_b = config.size_bounds

    print(f"population (SC {lo_year}-{hi_year}): {population}")
    print(f"sample: {n} PDFs, core matches {core_docs} ({rate:.1%}, 95% CI {lo:.1%}-{hi:.1%})")
    print(f"core estimate: {core} (CI {round(lo * population)}-{round(hi * population)})")
    print(
        f"one-hop: mean resolvable citations/judgment {statistics.mean(cites):.1f}, "
        f"raw upper bound {raw_one_hop}, capped {one_hop}"
    )
    print(f"distractors: {distractors}")
    verdict = "within" if low_b <= total <= high_b else "OUTSIDE"
    print(f"estimated total SC judgments: {total} ({verdict} bounds {low_b}-{high_b})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mvp_contract")
