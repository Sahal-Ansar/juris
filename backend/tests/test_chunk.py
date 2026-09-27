"""Chunking (PLAN 3.7).

The chunker takes the tokenizer as an argument, so these tests use a small in-memory
word-level tokenizer (one token per word or punctuation mark) with small limits: no model
download needed. One test checks the real bge-m3 tokenizer when it is present locally.
"""

import itertools
import re

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from juris.ingest.chunk import (
    ChunkRecord,
    chunk_judgment,
    chunk_statute_section,
    sentence_spans,
    split_windows,
)
from juris.ingest.tokens import SPECIAL_TOKENS, count_tokens, load_tokenizer, tokenizer_dir

PARA = chr(0x00B6)
DASH = chr(0x2014)
EN = chr(0x2013)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    t = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    t.pre_tokenizer = Whitespace()
    return t


def n_tokens(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False).ids)


# ---- sentences -----------------------------------------------------------------------------


def test_sentence_spans_respect_legal_abbreviations() -> None:
    text = (
        "The suit was filed under s. 10 of the Act. It relied on Ram v. State of U.P. and "
        "the view of A. K. Sen, J. The plaintiff paid Rs. 500 to the defendant. Appeal dismissed."
    )
    sentences = [text[s:e].strip() for s, e in sentence_spans(text)]
    assert sentences == [
        "The suit was filed under s. 10 of the Act.",
        "It relied on Ram v. State of U.P. and the view of A. K. Sen, J. The plaintiff paid "
        "Rs. 500 to the defendant.",
        "Appeal dismissed.",
    ]


def test_sentence_spans_tile_the_text() -> None:
    text = "One. Two!\nThree? Four"
    spans = sentence_spans(text)
    assert "".join(text[s:e] for s, e in spans) == text
    assert spans[0] == (0, 5)


# ---- judgments -----------------------------------------------------------------------------


def make_doc(
    paragraphs: list[tuple[str, str, str, int]],
) -> tuple[str, list[dict[str, object]]]:
    """(no, section, text, part) -> clean text and 3.3-style paragraph records."""
    clean, records = "", []
    for no, section, body, part in paragraphs:
        start = len(clean)
        clean += body + "\n"
        records.append(
            {
                "no": no,
                "part": part,
                "section": section,
                "numbering": "inferred" if no.startswith(("p-", "f-", "h-")) else "explicit",
                "char_start": start,
                "char_end": start + len(body),
                "page_start": 1,
                "page_end": 1,
                "text": body,
            }
        )
    return clean, records


META = {"title": "A versus B", "court": "Supreme Court of India", "decision_date": "2015-02-26"}


def sentences(k: int, words: int = 8) -> str:
    return " ".join(f"Sentence {i} " + "word " * (words - 3) + "ends." for i in range(k))


def check(chunks: list[ChunkRecord], clean: str, limit: int) -> None:
    for c in chunks:
        assert clean[c.char_start : c.char_end] == c.text  # exact provenance
        assert c.embed_tokens <= limit
        assert c.text == c.text.strip()


def test_short_paragraphs_are_packed_and_boundaries_respected(tok: Tokenizer) -> None:
    clean, paras = make_doc(
        [
            ("f-1", "front", "A v. B", 1),
            ("h-1", "headnote", "Held: the appeal fails.", 1),
            ("1", "body", sentences(2), 1),
            ("2", "body", sentences(2), 1),
            ("3", "body", sentences(2), 1),
            ("1", "body", sentences(1), 2),  # a separate opinion
        ]
    )
    chunks = chunk_judgment("SC-X", clean, paras, META, tok, limit=120)
    check(chunks, clean, 120)
    labels = [c.context_header.rsplit(f" {DASH} ", 1)[-1] for c in chunks]
    assert labels == ["front matter", "headnote", f"{PARA}{PARA} 1{EN}3", f"{PARA} 1"]
    assert chunks[2].para_start == 2 and chunks[2].para_end == 4
    assert chunks[0].context_header.startswith(f"A v. B {DASH} Supreme Court of India {DASH} 2015")
    assert [c.chunk_id for c in chunks] == [f"SC-X#c{i:04d}" for i in range(1, 5)]


def test_long_paragraph_is_split_between_sentences_with_overlap(tok: Tokenizer) -> None:
    clean, paras = make_doc([("7", "body", sentences(30), 1)])
    chunks = chunk_judgment("SC-X", clean, paras, META, tok, limit=80)
    check(chunks, clean, 80)
    assert len(chunks) > 3 and all(c.kind == "sentences" for c in chunks)
    assert chunks[0].context_header.endswith(f"{PARA} 7 (part 1/{len(chunks)})")
    for a, b in itertools.pairwise(chunks):
        assert b.char_start < a.char_end  # ~1 sentence of overlap
        assert clean[b.char_start - 1] in " \n"  # windows start at a sentence start
        assert re.search(r"ends\.$", a.text)  # ... and end at a sentence end


def test_run_on_sentence_is_cut_at_whitespace(tok: Tokenizer) -> None:
    clean, paras = make_doc([("1", "body", "word " * 300 + "end", 1)])
    chunks = chunk_judgment("SC-X", clean, paras, META, tok, limit=80)
    check(chunks, clean, 80)
    assert {c.kind for c in chunks} == {"hard"}
    assert "".join(c.text + " " for c in chunks).split() == clean.split()


def test_small_tail_joins_its_predecessor(tok: Tokenizer) -> None:
    clean, paras = make_doc([("1", "body", sentences(4), 1), ("2", "body", "Appeal dismissed.", 1)])
    chunks = chunk_judgment("SC-X", clean, paras, META, tok, limit=200)
    assert len(chunks) == 1 and chunks[0].para_end == 1


def test_inferred_paragraph_label(tok: Tokenizer) -> None:
    clean, paras = make_doc([("p-1", "body", sentences(1), 1), ("p-2", "body", sentences(1), 1)])
    chunks = chunk_judgment("SC-X", clean, paras, META, tok, limit=200)
    assert chunks[0].context_header.endswith(f"paras p-1{EN}p-2 (unnumbered)")


# ---- statutes ------------------------------------------------------------------------------


def statute(text: str, no: str = "10") -> dict[str, object]:
    return {
        "act_id": "contract_act",
        "act": "Indian Contract Act",
        "act_year": 1872,
        "section": no,
        "title": "What agreements are contracts",
        "text": text,
        "repealed": False,
        "page_start": 3,
        "page_end": 3,
    }


def test_statute_section_is_one_chunk(tok: Tokenizer) -> None:
    s = statute("10. What agreements are contracts.- All agreements are contracts.")
    (c,) = chunk_statute_section(s, tok)
    assert c.chunk_id == "ACT-contract_act#s10" and c.statute_section_id == "contract_act:10"
    assert c.text == s["text"] and (c.char_start, c.char_end) == (0, len(str(s["text"])))
    assert (
        c.context_header
        == f"Indian Contract Act, 1872 {DASH} s. 10 {DASH} What agreements are contracts"
    )


def test_long_statute_section_splits_between_lines(tok: Tokenizer) -> None:
    lines = ["73. Compensation.- When a contract has been broken, the party suffers."]
    lines += [
        f"({chr(97 + i)}) A contracts to sell goods to B and breaks the promise." for i in range(12)
    ]
    s = statute("\n".join(lines), "73")
    chunks = chunk_statute_section(s, tok, limit=70)
    assert len(chunks) > 1 and {c.kind for c in chunks} == {"section-lines"}
    assert [c.chunk_id for c in chunks] == [
        f"ACT-contract_act#s73-{k}" for k in range(1, len(chunks) + 1)
    ]
    for c in chunks:
        assert str(s["text"])[c.char_start : c.char_end] == c.text and c.embed_tokens <= 70
        assert c.text.split("\n")[0].startswith(("73.", "("))  # whole lines only


def test_split_windows_cover_every_sentence(tok: Tokenizer) -> None:
    text = sentences(12)
    windows, hard = split_windows(text, 30, tok)
    assert not hard
    covered = set()
    for s, e in windows:
        covered |= set(range(s, e))
        assert n_tokens(tok, text[s:e]) <= 30
    assert covered >= {i for i, ch in enumerate(text) if not ch.isspace()}


# ---- the real tokenizer, when downloaded -----------------------------------------------------


def test_bge_m3_tokenizer_counts() -> None:
    if not (tokenizer_dir() / "tokenizer.json").exists():
        pytest.skip("bge-m3 tokenizer not downloaded")
    assert count_tokens(["Specific performance of a contract."]) == [
        len(load_tokenizer().encode("Specific performance of a contract.").ids) - SPECIAL_TOKENS
    ]
