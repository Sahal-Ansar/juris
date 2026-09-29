"""The exact-quote rule (IDEA_final §6.1 check 2, §8 invariant 2).

A citation's quote must be a substring of its chunk's text after both are normalised the
same way: typographic quote marks become ASCII (the same code points as 3.2's cleaner), and
runs of whitespace become one space. Case is kept unless ``casefold`` is asked for. Nothing
else is forgiven: a changed word, an added ellipsis or a paraphrase fails.

Shared by the answer metrics (5.4) and the Citation Verifier (6.1), so both apply one rule.
"""

import re

_SINGLE_QUOTES = (0x2018, 0x2019, 0x201A, 0x201B, 0x2032)
_DOUBLE_QUOTES = (0x201C, 0x201D, 0x201E, 0x201F, 0x2033, 0x00AB, 0x00BB)
_QUOTES = str.maketrans(
    {chr(c): "'" for c in _SINGLE_QUOTES} | {chr(c): '"' for c in _DOUBLE_QUOTES}
)
_SPACE = re.compile(r"\s+")


def normalise_quote(text: str, *, casefold: bool = False) -> str:
    out = _SPACE.sub(" ", text.translate(_QUOTES)).strip()
    return out.casefold() if casefold else out


def quote_in_text(quote: str, text: str, *, casefold: bool = False) -> bool:
    """True if the normalised ``quote`` is non-empty and occurs in the normalised ``text``."""
    needle = normalise_quote(quote, casefold=casefold)
    return bool(needle) and needle in normalise_quote(text, casefold=casefold)
