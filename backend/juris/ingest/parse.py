"""Parse snapshot documents into raw page text and cleaned text (PLAN 3.2).

Judgments: the PDF text layer via pdfium (old SC volumes carry the source's own OCR layer).
Statutes: the text provided by the source. Documents with almost no text per page are
flagged ``needs_ocr``; no OCR engine runs (none of the MVP snapshot needs one, D-020).
"""

import json
from pathlib import Path
from typing import Any

import pypdfium2

from juris.ingest.clean import Profile, clean, latin_ratio, ocr_noise

MIN_CHARS_PER_PAGE = 200  # below this a page is treated as having no usable text layer
MIN_LATIN_RATIO = 0.85


def pdf_pages(path: Path) -> list[str]:
    pdf = pypdfium2.PdfDocument(path)
    try:
        return [page.get_textpage().get_text_range() for page in pdf]
    finally:
        pdf.close()


def profile_for(doc_id: str) -> Profile:
    if doc_id.startswith("SC-"):
        return "sc"
    if doc_id.startswith("HC-"):
        return "hc"
    return "statute"


def parse_entry(entry: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    """One snapshot entry -> a parse record (raw pages, clean text, offsets, quality)."""
    record: dict[str, Any] = {
        "doc_id": entry["doc_id"],
        "kind": entry["kind"],
        "reason": entry["reason"],
        "source": entry["source"],
    }
    local = data_dir / entry["local_path"]
    try:
        if entry["kind"] == "statute":
            pages = [json.loads(local.read_text(encoding="utf-8"))["text"]]
            record["text_source"] = "provided"
        else:
            pages = pdf_pages(local)
            record["text_source"] = "text_layer"
    except Exception as exc:
        return {**record, "error": f"{type(exc).__name__}: {exc}"}

    raw_text = "\f".join(pages)
    chars = sum(len(p.strip()) for p in pages)
    result = clean(pages, profile_for(entry["doc_id"]))
    ratio = latin_ratio(result.text)
    return {
        **record,
        "pages": len(pages),
        "raw_chars": len(raw_text),
        "clean_chars": len(result.text),
        "needs_ocr": entry["kind"] != "statute" and chars / max(len(pages), 1) < MIN_CHARS_PER_PAGE,
        "latin_ratio": round(ratio, 4),
        "language": "en" if ratio >= MIN_LATIN_RATIO else "non_en",
        "ocr_noise": round(ocr_noise(result.text), 4),
        "stats": dict(sorted(result.stats.items())),
        "page_starts": result.page_starts,
        "runs": result.runs,
        "raw_text": raw_text,
        "clean_text": result.text,
    }
