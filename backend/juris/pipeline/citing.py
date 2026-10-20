"""Helpers for citations written by a model: the pinpoint of a chunk, and evidence IDs in
the prompt's form. Shared by B1/B2 (6.3-6.4) and Counsel (7.5)."""

import re

from juris.models import Chunk


def pinpoint(chunk: Chunk, header: str | None) -> str:
    """Where in the document the chunk is: a section, paragraph(s), page(s), else its header."""
    if chunk.statute is not None:
        return f"s. {chunk.statute.section}"
    if chunk.para_start is not None and chunk.para_end is not None:
        if chunk.para_start == chunk.para_end:
            return f"para {chunk.para_start}"
        return f"paras {chunk.para_start}-{chunk.para_end}"
    if chunk.page_start is not None and chunk.page_end is not None:
        if chunk.page_start == chunk.page_end:
            return f"p. {chunk.page_start}"
        return f"pp. {chunk.page_start}-{chunk.page_end}"
    return header or chunk.section or "chunk"


_LOOSE_ID = re.compile(r"^E\s*-?\s*0*(\d+)$", re.IGNORECASE)


def clean_evidence_id(raw: str) -> str:
    """The ID in the prompt's form: brackets, spaces, case and a missing hyphen are tolerated
    and the number is zero-padded to three digits, so "E-3", "E3", "e-03" and "[E-003]" are
    all "E-003". Anything else is returned stripped, and stays malformed."""
    text = raw.strip().strip("[]()").strip()
    loose = _LOOSE_ID.match(text)
    return f"E-{int(loose.group(1)):03d}" if loose else text
