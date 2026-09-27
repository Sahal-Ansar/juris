"""Clean extracted judgment/statute text while keeping a map back to the raw offsets.

Raw text is the page texts joined with ``\\f``. Cleaning works line by line: it drops page
furniture (repeated headers/footers, page numbers, SCR running heads, margin letters,
digital-signature blocks, symbol-only OCR debris), then fixes characters inside kept lines
(line-break hyphens, split ligatures, quotes, whitespace). Every clean character remembers
the raw character it came from; the map is stored as runs ``(clean_start, raw_start, length)``.

Line structure is kept (one output line per kept source line) so that paragraph
segmentation (PLAN 3.3) can still see line-initial paragraph numbers.
"""

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

Profile = Literal["sc", "hc", "statute"]

# Code points, not literals, so the source stays ASCII.
_SINGLE_QUOTES = (0x2018, 0x2019, 0x201A, 0x201B, 0x2032)
_DOUBLE_QUOTES = (0x201C, 0x201D, 0x201E, 0x201F, 0x2033, 0x00AB, 0x00BB)
_QUOTES = str.maketrans(
    {chr(c): "'" for c in _SINGLE_QUOTES} | {chr(c): '"' for c in _DOUBLE_QUOTES}
)
_SPACES = {chr(c) for c in (0x09, 0xA0, 0x2002, 0x2003, 0x2009, 0x200A, 0x202F, 0x3000)}
_DROP_CHARS = {chr(c) for c in (0x0D, 0x200B, 0x200C, 0x200D, 0xFEFF)}
_BREAK_HYPHENS = {chr(0xFFFE), chr(0x00AD)}  # pdfium's line-break hyphen; soft hyphen

_PAGE_NUMBER = re.compile(r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*\.?\s*$", re.I)
_SCR_HEAD = re.compile(r"(?i)(court\W{0,3}\s*repor|s\s*\.\s*c\s*\.\s*r\s*\.)")
# "SUPREME COURT REPORTS" running head, tolerant of OCR damage ("SUPl{EMl!:' COURT REPOR'r8").
_COURT_REPORTS = re.compile(r"(?i)\bsu\S{0,8}\s+cou\S{0,4}\s+rep")
# Badly damaged heads ("40 . tslfPREME COURT REPORTS [1963)"): "court rep..." plus a leading
# page number or a bracketed year, which body sentences don't have.
_COURT_REPORTS_DAMAGED = re.compile(
    r"(?i)^(?:\W*\d{1,4}\W.*c\W?ourt\W*rep|.*c\W?ourt\W*rep\S*\W*[\[(]\S{3,5}[\])])"
)
_MARGIN_LETTER = re.compile(r"^\s*[A-Ha-h]\s*$")
_TRAILING_MARGIN = re.compile(r"\s[A-H]\s*$")
# SCR margin letters at the start of a line ("E the Bank Guarantee ..."). "A" is also the
# article, so it is stripped only mid-sentence (previous line doesn't end a sentence).
_LEADING_MARGIN_BH = re.compile(r"^\s*[B-H]\s+[a-z]")
_LEADING_MARGIN = re.compile(r"^[A-H] (?=\S)")
_SENTENCE_END = set(".:;?!\"')]")
# Words the article "A" can never precede: "A and ...", "A was ..." are margin letters.
_NOT_AFTER_ARTICLE = frozenset(
    [
        "and",
        "as",
        "was",
        "were",
        "is",
        "are",
        "of",
        "to",
        "the",
        "in",
        "that",
        "by",
        "for",
        "on",
        "with",
        "which",
        "has",
        "had",
        "have",
        "be",
        "been",
        "it",
        "this",
        "these",
        "those",
        "not",
        "or",
        "but",
        "at",
        "from",
        "xxx",
        "also",
        "into",
        "upon",
    ]
)
_SIGNATURE_START = re.compile(
    r"(?i)(^\s*digitally\b|digitally\s+signed|signature\s+not\s+verified|signing\s+date)"
)
_SIGNATURE_END = re.compile(r"\d{2}:\d{2}:\d{2}")
_PICTURE = re.compile(r"^\s*==>\s*picture\b.*<==\s*$", re.I)
_WORD = re.compile(r"[A-Za-z]{2,}")
_LIGATURE_SPLIT = re.compile(r"\b([A-Za-z]*?(?:ffi|ffl|ff|fi|fl)) ([a-z]{2,})\b")
_LIGATURE_ALONE = {"fi", "fl", "ff", "ffi", "ffl", "Fi", "Fl", "Ff"}


@dataclass
class CleanResult:
    text: str
    runs: list[tuple[int, int, int]]
    page_starts: list[int]
    stats: Counter[str] = field(default_factory=Counter)

    def to_raw(self, clean_offset: int) -> int:
        """Raw offset of the character at ``clean_offset``."""
        starts = [r[0] for r in self.runs]
        i = bisect.bisect_right(starts, clean_offset) - 1
        clean_start, raw_start, length = self.runs[i]
        if not 0 <= clean_offset - clean_start < length:
            raise IndexError(clean_offset)
        return raw_start + (clean_offset - clean_start)

    def page_of(self, clean_offset: int) -> int:
        """1-based page number of a clean offset."""
        return bisect.bisect_right(self.page_starts, clean_offset)


class _Out:
    def __init__(self) -> None:
        self.chars: list[str] = []
        self.raw: list[int] = []

    def put(self, char: str, raw_index: int) -> None:
        self.chars.append(char)
        self.raw.append(raw_index)

    def runs(self) -> list[tuple[int, int, int]]:
        runs: list[tuple[int, int, int]] = []
        for i, r in enumerate(self.raw):
            if runs and runs[-1][1] + runs[-1][2] == r and runs[-1][0] + runs[-1][2] == i:
                start, raw_start, length = runs[-1]
                runs[-1] = (start, raw_start, length + 1)
            else:
                runs.append((i, r, 1))
        return runs


def _norm_key(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))


def _edge_indices(lines: list[str], k: int = 3) -> set[int]:
    """The first and last ``k`` content lines of a page. Debris ('...', '*') and single
    margin letters don't count, so a running head below them is still at the edge."""
    content = [i for i, line in enumerate(lines) if sum(c.isalnum() for c in line) >= 2]
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    # Short page numbers ("7") are edge lines too, though they aren't content lines.
    return set(content[:k] + content[-k:] + nonempty[:1] + nonempty[-1:])


def _vocabulary(raw_pages: list[str]) -> set[str]:
    """Words that occur intact in the document. Fragments of line-break-split words
    ('Bala|krishanaiya') are excluded, so they can't vouch for themselves."""
    text = "\n".join(raw_pages)
    intact = re.sub("[A-Za-z]+[" + "".join(sorted(_BREAK_HYPHENS)) + "][A-Za-z]+", " ", text)
    return {w.lower() for w in _WORD.findall(intact)}


def clean(raw_pages: list[str], profile: Profile) -> CleanResult:
    """Clean page texts. Offsets refer to ``"\\f".join(raw_pages)``."""
    stats: Counter[str] = Counter()
    vocab = _vocabulary(raw_pages)
    # Split the original page text so raw offsets stay aligned; CR chars are dropped later.
    page_lines = [p.split("\n") for p in raw_pages]

    # Repeated headers/footers: lines at a page edge whose digit-normalised form recurs
    # on at least 30% of pages (and 3 pages).
    edge_counts: Counter[str] = Counter()
    for lines in page_lines:
        edge_counts.update({_norm_key(lines[i]) for i in _edge_indices(lines)})
    threshold = max(3, int(0.3 * len(page_lines) + 0.999))
    repeated = {key for key, n in edge_counts.items() if n >= threshold and key}

    trailing_margin_doc = (
        profile == "sc"
        and sum(1 for lines in page_lines for line in lines if _TRAILING_MARGIN.search(line)) >= 5
    )
    leading_margin_doc = (
        profile == "sc"
        and sum(1 for lines in page_lines for line in lines if _LEADING_MARGIN_BH.match(line)) >= 5
    )

    out = _Out()
    page_starts: list[int] = []
    raw_offset = 0
    for page_no, lines in enumerate(page_lines):
        page_starts.append(len(out.chars))
        edges = _edge_indices(lines)
        signature = _signature_lines(lines)
        line_offset = raw_offset
        kept_any = False
        for i, line in enumerate(lines):
            start = line_offset
            line_offset += len(line) + 1
            reason = "signature" if i in signature else ""
            reason = reason or _drop_reason(line, i, page_no, edges, repeated, profile)
            if reason:
                stats[f"dropped_{reason}"] += 1
                continue
            text, raw_idx = _fix_line(line, start, vocab, trailing_margin_doc, stats)
            if leading_margin_doc and _LEADING_MARGIN.match(text):
                previous_end = next((c for c in reversed(out.chars) if c not in " \n"), ".")
                next_word = text[2:].split(" ", 1)[0].lower()
                if (
                    text[0] != "A"
                    or previous_end not in _SENTENCE_END
                    or next_word in _NOT_AFTER_ARTICLE
                ):
                    text, raw_idx = text[2:], raw_idx[2:]
                    stats["leading_margin_letter"] += 1
            if not text:
                continue
            if kept_any or (out.chars and out.chars[-1] != "\n"):
                out.put("\n", start - 1 if start > 0 else start)
            for ch, r in zip(text, raw_idx, strict=True):
                out.put(ch, r)
            kept_any = True
        raw_offset += len(raw_pages[page_no]) + 1  # + the "\f" separator
    text = "".join(out.chars)
    stats["lines_kept"] = text.count("\n") + (1 if text else 0)
    return CleanResult(text=text, runs=out.runs(), page_starts=page_starts, stats=stats)


def _signature_lines(lines: list[str], max_block: int = 8) -> set[int]:
    """Digital-signature stamps ("Digitally signed by ... 17:03:43 +0530"). A block starts at a
    signature phrase and ends at a timestamp, or at the end of the page, within ``max_block``
    lines, every line in it short. Otherwise it is ordinary text (e.g. a sentence that starts
    with 'Digitally')."""
    found: set[int] = set()
    last = max((k for k, line in enumerate(lines) if line.strip()), default=-1)
    i = 0
    while i < len(lines):
        if _SIGNATURE_START.search(lines[i]):
            for j in range(i, min(i + max_block, len(lines))):
                if len(lines[j].split()) > 6:
                    break
                if _SIGNATURE_END.search(lines[j]) or j == last:
                    found.update(range(i, j + 1))
                    i = j
                    break
        i += 1
    return found


def _drop_reason(
    line: str,
    index: int,
    page_no: int,
    edges: set[int],
    repeated: set[str],
    profile: Profile,
) -> str:
    stripped = line.strip()
    if not stripped:
        return "blank"
    if not any(c.isalnum() for c in stripped):
        return "symbols"
    if profile == "statute" and _PICTURE.match(line):
        return "picture"
    if index in edges:
        if _norm_key(line) in repeated:
            return "repeated_header"
        if _PAGE_NUMBER.match(line):
            return "page_number"
        first_line_of_doc = page_no == 0 and index == min(edges)  # the reporter citation
        if (
            profile == "sc"
            and not first_line_of_doc
            and len(stripped.split()) <= 12
            and _SCR_HEAD.search(stripped)
        ):
            return "scr_running_head"
    if (
        profile == "sc"
        and len(stripped.split()) <= 10
        and (_COURT_REPORTS.search(stripped) or _COURT_REPORTS_DAMAGED.search(stripped))
    ):
        return "scr_running_head"
    if profile == "sc" and _MARGIN_LETTER.match(line):
        return "margin_letter"
    return ""


def _fix_line(
    line: str, raw_start: int, vocab: set[str], trailing_margin: bool, stats: Counter[str]
) -> tuple[str, list[int]]:
    """Character fixes inside one kept line; returns the text and each char's raw index."""
    chars: list[str] = []
    idx: list[int] = []
    n = len(line)
    i = 0
    while i < n:
        ch = line[i]
        if ch in _DROP_CHARS:
            i += 1
            continue
        if ch in _BREAK_HYPHENS:
            left = re.search(r"[A-Za-z]+$", "".join(chars))
            right = re.match(r"[A-Za-z]+", line[i + 1 :])
            keep_hyphen = _keep_break_hyphen(
                left.group() if left else "", right.group() if right else "", vocab
            )
            stats["break_hyphen_kept" if keep_hyphen else "break_hyphen_joined"] += 1
            if keep_hyphen:
                chars.append("-")
                idx.append(raw_start + i)
            i += 1
            continue
        if ch in _SPACES or ch == " ":
            if chars and chars[-1] != " ":
                chars.append(" ")
                idx.append(raw_start + i)
            i += 1
            continue
        chars.append(ch.translate(_QUOTES))
        idx.append(raw_start + i)
        i += 1
    # trim spaces at both ends
    while chars and chars[-1] == " ":
        chars.pop()
        idx.pop()
    while chars and chars[0] == " ":
        chars.pop(0)
        idx.pop(0)
    if trailing_margin and len(chars) >= 3 and chars[-2] == " " and chars[-1] in "ABCDEFGH":
        del chars[-2:], idx[-2:]
        stats["trailing_margin_letter"] += 1
    return _join_ligatures(chars, idx, vocab, stats)


def _keep_break_hyphen(left: str, right: str, vocab: set[str]) -> bool:
    """A hyphen at a line break: join the halves if the joined word occurs intact in the
    document ('under|standing'); keep a real hyphen only when it doesn't and both halves are
    words ('plaintiff|appellant'); otherwise join ('Bala|krishanaiya')."""
    if not left or not right:
        return False
    if (left + right).lower() in vocab:
        return False
    return len(left) >= 3 and len(right) >= 3 and left.lower() in vocab and right.lower() in vocab


def _join_ligatures(
    chars: list[str], idx: list[int], vocab: set[str], stats: Counter[str]
) -> tuple[str, list[int]]:
    """'fi xed' -> 'fixed', 'eff ect' -> 'effect' (pdfium splits some ligatures)."""
    text = "".join(chars)
    drop: set[int] = set()
    for m in _LIGATURE_SPLIT.finditer(text):
        left, right = m.group(1), m.group(2)
        joined = (left + right).lower()
        # Join when the whole word occurs intact elsewhere ('effect'), or the fragment is a
        # bare ligature ('fi xed'). 'staff member' stays: 'staffmember' never occurs.
        if left in _LIGATURE_ALONE or joined in vocab:
            drop.add(m.start(2) - 1)
            stats["ligature_joined"] += 1
    if not drop:
        return text, idx
    keep = [i for i in range(len(chars)) if i not in drop]
    return "".join(chars[i] for i in keep), [idx[i] for i in keep]


def latin_ratio(text: str) -> float:
    """Share of letters that are basic Latin: a cheap English/non-English signal for Indian
    corpora, where non-English judgments are in Devanagari and other Indic scripts."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isascii()) / len(letters)


_NOISY = re.compile(r"[A-Za-z]+[!{}|~\\^`@#$%*<>\[\]]+[A-Za-z]+")


def ocr_noise(text: str) -> float:
    """Share of word tokens with symbols inside them (``SUPl{EMl!``): a rough OCR-quality score."""
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _NOISY.search(t)) / len(tokens)
