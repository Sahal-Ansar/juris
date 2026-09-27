"""Paragraph-aware chunking (PLAN 3.7).

Judgments: consecutive paragraphs (3.3) are packed greedily into chunks of at most
``max_text`` tokens, never across an opinion ``part`` or a front/headnote/body boundary. A
paragraph longer than that is split at sentence boundaries into windows that overlap by about
one sentence; a single sentence longer than a window (OCR run-ons, tables) is cut at the last
whitespace inside the limit and counted as a ``hard`` split. A chunk's ``text`` is the exact
slice ``clean_text[char_start:char_end]`` of the 3.2 clean text, so quotes can be matched.

Statutes: one chunk per section (3.5), sub-sections kept together; a section over the limit is
split at line boundaries (sub-sections, clauses, illustrations). Offsets index the section's
text (``statute_sections.text``).

``context_header`` ("Case title - Court - Year - paras 12-14") is prepended only for embedding.
Header, text and the model's two special tokens must fit ``limit`` (512: bge-m3 takes 8,192,
but multilingual-e5-large, the other candidate, takes 512).
"""

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tokenizers import Tokenizer

from juris.ingest.tokens import SPECIAL_TOKENS

DASH = chr(0x2014)
PARA = chr(0x00B6)
LIMIT = 512
MAX_TEXT = 480  # text tokens per chunk at most (the header takes the rest)
MIN_TAIL = 80  # a last chunk smaller than this joins the previous one when it fits
OVERLAP = 60  # tokens of trailing sentences repeated at the start of the next window
TITLE_CHARS = 90


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    context_header: str
    char_start: int
    char_end: int
    tokens: int  # text tokens, without header and special tokens
    embed_tokens: int  # header + text + special tokens: what the embedding model sees
    kind: str  # paragraphs | sentences | hard | section | section-lines
    section: str | None = None
    para_start: int | None = None  # paragraph seq (paragraphs.seq), inclusive
    para_end: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    statute_section_id: str | None = None
    part: int | None = None  # opinion part (3.3); not stored, used to keep parts apart

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ---- sentences ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "no", "nos", "s", "ss", "v", "vs", "rs", "art", "arts", "sec", "secs", "cl", "o", "r", "rr",
    "p", "pp", "para", "paras", "ltd", "co", "pvt", "corpn", "mr", "mrs", "ms", "dr", "shri",
    "smt", "j", "jj", "cj", "cji", "ors", "anr", "govt", "dt", "viz", "ie", "eg", "cf", "sr",
    "st", "sub", "vol", "ch", "ref", "ex", "exh", "hon", "ble", "addl", "asst", "dy", "supp",
    "etc", "ibid", "id", "op", "cit", "misc", "crl", "civ", "wp", "slp", "rfa", "fao",
}  # fmt: skip
_BOUNDARY = re.compile(
    "[.?!][\"'\u201d\u2019)\\]]*(?=\\s+[\"'\u201c\u2018(\\[]?[A-Z0-9])|\\n(?=\\s*\\S)"
)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence in ``text``; spans tile the text, so no text is lost."""
    cuts = []
    for m in _BOUNDARY.finditer(text):
        if m.group(0) != "\n":
            word = re.search(r"([A-Za-z.]+)\.$", text[max(0, m.start() - 12) : m.start() + 1])
            token = word.group(1).replace(".", "").lower() if word else ""
            if token in _ABBREVIATIONS or re.fullmatch(r"[a-z]", token):
                continue  # "s. 10", "v. State", "A. K. Sen"
        cuts.append(m.end())
    spans, start = [], 0
    for cut in cuts:
        while cut < len(text) and text[cut] in " \t\n":
            cut += 1
        if cut > start:
            spans.append((start, cut))
            start = cut
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    """Shrink a span so it neither starts nor ends with whitespace."""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


# ---- splitting one long unit -------------------------------------------------------------


def _hard_windows(text: str, offset: int, max_tokens: int, tok: Tokenizer) -> list[tuple[int, int]]:
    """Cut a run-on span at the last whitespace before ``max_tokens`` tokens."""
    enc = tok.encode(text, add_special_tokens=False)
    out, start_tok = [], 0
    while start_tok < len(enc.offsets):
        end_tok = min(len(enc.offsets), start_tok + max_tokens)
        s = enc.offsets[start_tok][0]
        e = enc.offsets[end_tok - 1][1]
        if end_tok < len(enc.offsets):
            space = text.rfind(" ", s, e)
            if space > s:
                e = space
                end_tok = next(i for i, o in enumerate(enc.offsets) if o[1] > e)
        out.append((offset + s, offset + e))
        start_tok = max(end_tok, start_tok + 1)
    return out


def split_windows(
    text: str, max_tokens: int, tok: Tokenizer, overlap: int = OVERLAP
) -> tuple[list[tuple[int, int]], bool]:
    """Sentence windows over ``text`` (offsets into it) and whether any hard cut was needed."""
    spans = sentence_spans(text)
    counts = [
        len(e.ids)
        for e in tok.encode_batch([text[s:e] for s, e in spans], add_special_tokens=False)
    ]
    windows: list[tuple[int, int]] = []
    hard = False
    i = 0
    while i < len(spans):
        if counts[i] > max_tokens:
            windows += _hard_windows(text[spans[i][0] : spans[i][1]], spans[i][0], max_tokens, tok)
            hard = True
            i += 1
            continue
        j, total = i, 0
        while j < len(spans) and total + counts[j] <= max_tokens:
            total += counts[j]
            j += 1
        windows.append((spans[i][0], spans[j - 1][1]))
        if j >= len(spans):
            break
        # step back over trailing sentences worth up to `overlap` tokens (but always advance)
        k, back = j, 0
        while k - 1 > i + 1 and back + counts[k - 1] <= overlap:
            k -= 1
            back += counts[k]
        i = k
    return [_trim(text, s, e) for s, e in windows if text[s:e].strip()], hard


# ---- judgments ---------------------------------------------------------------------------


def _short(title: str | None) -> str:
    t = re.sub(r"\s+", " ", (title or "").replace(" versus ", " v. ")).strip()
    return t if len(t) <= TITLE_CHARS else t[: TITLE_CHARS - 3].rstrip() + "..."


def judgment_header_base(meta: dict[str, Any]) -> str:
    year = (meta.get("decision_date") or "")[:4]
    parts = [_short(meta.get("title")), meta.get("court") or "", year]
    return f" {DASH} ".join(p for p in parts if p)


def _label(paras: Sequence[dict[str, Any]], piece: str = "") -> str:
    first, last = paras[0], paras[-1]
    section = first["section"]
    if section == "headnote":
        return "headnote"
    if section == "front":
        return "front matter"
    a, b = first["no"], last["no"]
    if first["numbering"] == "inferred":
        rng = a if a == b else f"{a}\u2013{b}"
        return f"paras {rng} (unnumbered){piece}"
    if a == b:
        return f"{PARA} {a}{piece}"
    return f"{PARA}{PARA} {a}\u2013{b}{piece}"


def chunk_judgment(
    doc_id: str,
    clean_text: str,
    paragraphs: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    tok: Tokenizer,
    limit: int = LIMIT,
) -> list[ChunkRecord]:
    """Chunks for one judgment; ``paragraphs`` are the 3.3 records in order (seq = index)."""
    base = judgment_header_base(meta)
    # budget with a generous label ("paras p-1234-p-1235 (part 12/12)") so every header fits
    worst = f"{base} {DASH} {PARA}{PARA} p-1234\u2013p-1235 (unnumbered) (part 12/12)"
    max_text = min(
        MAX_TEXT, limit - SPECIAL_TOKENS - len(tok.encode(worst, add_special_tokens=False).ids)
    )
    paras = [dict(p, seq=i) for i, p in enumerate(paragraphs)]
    counts = [
        len(e.ids)
        for e in tok.encode_batch(
            [clean_text[p["char_start"] : p["char_end"]] for p in paras],
            add_special_tokens=False,
        )
    ]
    out: list[ChunkRecord] = []

    def emit(group: list[dict[str, Any]], start: int, end: int, kind: str, piece: str = "") -> None:
        text = clean_text[start:end]
        label = _label(group, piece)
        header = f"{base} {DASH} {label}" if base else label
        n = len(tok.encode(text, add_special_tokens=False).ids)
        out.append(
            ChunkRecord(
                chunk_id=f"{doc_id}#c{len(out) + 1:04d}",
                doc_id=doc_id,
                text=text,
                context_header=header,
                char_start=start,
                char_end=end,
                tokens=n,
                embed_tokens=n
                + len(tok.encode(header, add_special_tokens=False).ids)
                + SPECIAL_TOKENS,
                kind=kind,
                section=group[0]["section"],
                part=group[0]["part"],
                para_start=group[0]["seq"],
                para_end=group[-1]["seq"],
                page_start=group[0]["page_start"],
                page_end=group[-1]["page_end"],
            )
        )

    def flush(group: list[dict[str, Any]]) -> None:
        if group:
            emit(group, group[0]["char_start"], group[-1]["char_end"], "paragraphs")

    group: list[dict[str, Any]] = []
    total = 0
    for p, n in zip(paras, counts, strict=True):
        boundary = group and (p["part"], p["section"]) != (group[0]["part"], group[0]["section"])
        if n > max_text:
            flush(group)
            group, total = [], 0
            text = clean_text[p["char_start"] : p["char_end"]]
            windows, hard = split_windows(text, max_text, tok)
            for k, (s, e) in enumerate(windows, 1):
                piece = f" (part {k}/{len(windows)})"
                emit(
                    [p],
                    p["char_start"] + s,
                    p["char_start"] + e,
                    "hard" if hard else "sentences",
                    piece,
                )
            continue
        # +1 for the newline joining two paragraphs
        if boundary or total + n + 1 > max_text:
            flush(group)
            group, total = [], 0
        group.append(p)
        total += n + 1
    flush(group)
    _merge_tails(out, clean_text, tok, max_text, base)
    return out


def _merge_tails(
    chunks: list[ChunkRecord], clean_text: str, tok: Tokenizer, max_text: int, base: str
) -> None:
    """Fold a small paragraph chunk into its predecessor when both come from the same run."""
    i = 1
    while i < len(chunks):
        prev, cur = chunks[i - 1], chunks[i]
        if (
            cur.kind == prev.kind == "paragraphs"
            and cur.tokens < MIN_TAIL
            and (cur.section, cur.part) == (prev.section, prev.part)
            and prev.para_end is not None
            and cur.para_start == prev.para_end + 1
            and prev.tokens + cur.tokens + 1 <= max_text
        ):
            text = clean_text[prev.char_start : cur.char_end]
            n = len(tok.encode(text, add_special_tokens=False).ids)
            if n <= max_text:
                header = prev.context_header.rsplit(f" {DASH} ", 1)
                # rebuild the paragraph label for the merged range
                label = _merge_label(prev.context_header, cur.context_header)
                prev.context_header = f"{header[0]} {DASH} {label}" if base else label
                prev.text, prev.char_end, prev.tokens = text, cur.char_end, n
                prev.para_end, prev.page_end = cur.para_end, cur.page_end
                prev.embed_tokens = (
                    n
                    + len(tok.encode(prev.context_header, add_special_tokens=False).ids)
                    + SPECIAL_TOKENS
                )
                del chunks[i]
                continue
        i += 1
    for k, c in enumerate(chunks, 1):  # keep IDs dense after merging
        c.chunk_id = f"{c.doc_id}#c{k:04d}"


def _merge_label(a: str, b: str) -> str:
    la, lb = a.rsplit(f" {DASH} ", 1)[-1], b.rsplit(f" {DASH} ", 1)[-1]
    if la in ("headnote", "front matter"):
        return la
    first = re.sub(rf"^{PARA}+ |^paras |\s*\(unnumbered\)$", "", la).split("\u2013")[0]
    last = re.sub(rf"^{PARA}+ |^paras |\s*\(unnumbered\)$", "", lb).split("\u2013")[-1]
    if la.startswith("paras"):
        return f"paras {first}\u2013{last} (unnumbered)"
    return f"{PARA}{PARA} {first}\u2013{last}"


# ---- statutes ----------------------------------------------------------------------------


def statute_header(section: dict[str, Any]) -> str:
    act = f"{section['act']}, {section['act_year']}"
    title = section.get("title") or ""
    rep = " [repealed]" if section.get("repealed") else ""
    return f" {DASH} ".join(x for x in (act, f"s. {section['section']}{rep}", title) if x)


def chunk_statute_section(
    section: dict[str, Any], tok: Tokenizer, limit: int = LIMIT
) -> list[ChunkRecord]:
    """One chunk per section; a section over the limit is split at line boundaries."""
    text = section["text"]
    act_id, no = section["act_id"], section["section"]
    header = statute_header(section)
    header_n = len(tok.encode(header + " (part 12/12)", add_special_tokens=False).ids)
    max_text = min(MAX_TEXT, limit - SPECIAL_TOKENS - header_n)
    base_id = f"ACT-{act_id}#s{no}"
    n = len(tok.encode(text, add_special_tokens=False).ids)
    if n <= max_text:
        spans, kind = [(0, len(text))], "section"
    else:
        spans, kind = _line_windows(text, max_text, tok), "section-lines"
    out = []
    for k, (s, e) in enumerate(spans, 1):
        piece = f" (part {k}/{len(spans)})" if len(spans) > 1 else ""
        h = header + piece
        t = text[s:e]
        m = len(tok.encode(t, add_special_tokens=False).ids)
        out.append(
            ChunkRecord(
                chunk_id=base_id if len(spans) == 1 else f"{base_id}-{k}",
                doc_id=f"ACT-{act_id}",
                text=t,
                context_header=h,
                char_start=s,
                char_end=e,
                tokens=m,
                embed_tokens=m + len(tok.encode(h, add_special_tokens=False).ids) + SPECIAL_TOKENS,
                kind=kind,
                section=f"s. {no}",
                page_start=section.get("page_start"),
                page_end=section.get("page_end"),
                statute_section_id=f"{act_id}:{no}",
            )
        )
    return out


def _line_windows(text: str, max_tokens: int, tok: Tokenizer) -> list[tuple[int, int]]:
    """Greedy windows of whole lines; a line over the limit falls back to sentence windows."""
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, pos + len(line)))
        pos += len(line) + 1
    counts = [
        len(e.ids)
        for e in tok.encode_batch([text[s:e] for s, e in lines], add_special_tokens=False)
    ]
    out: list[tuple[int, int]] = []
    start: int | None = None
    end, total = 0, 0
    for (s, e), n in zip(lines, counts, strict=True):
        if n > max_tokens:
            if start is not None:
                out.append((start, end))
                start, total = None, 0
            windows, _ = split_windows(text[s:e], max_tokens, tok)
            out += [(s + a, s + b) for a, b in windows]
            continue
        if start is not None and total + n + 1 > max_tokens:
            out.append((start, end))
            start, total = None, 0
        if start is None:
            start = s
        end, total = e, total + n + 1
    if start is not None:
        out.append((start, end))
    return [_trim(text, s, e) for s, e in out if text[s:e].strip()]
