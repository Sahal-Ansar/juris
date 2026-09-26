"""Corpus slice definition (configs/corpus/*.yaml) and the text matchers behind it.

Used by PLAN 1.3 to estimate the slice from samples and by PLAN 3.1 to select it.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from juris.config import REPO_ROOT

CORPUS_CONFIGS_DIR = REPO_ROOT / "configs" / "corpus"

ActKey = Literal["contract_act", "specific_relief_act", "sale_of_goods_act"]

ACT_PATTERNS: dict[str, re.Pattern[str]] = {
    # "Contract Act" but not "Contract Labour (Regulation and Abolition) Act".
    "contract_act": re.compile(r"\b(?:Indian\s+)?Contract\s+Act\b", re.I),
    # Covers the 1963 Act and its 1877 predecessor.
    "specific_relief_act": re.compile(r"\bSpecific\s+Relief\s+Act\b", re.I),
    "sale_of_goods_act": re.compile(r"\bSale\s+of\s+Goods\s+Act\b", re.I),
}

# "Section 73", "Sections 73 and 74", "S. 16(c)", "u/s 10", "Ss. 55, 56" just before an Act name.
_SECTION = re.compile(
    r"\b(?:sections?|secs?\.?|ss?\.|u/s\.?)\s*"
    r"(\d+[A-Z]?(?:\s*\([^)]{1,6}\))*(?:\s*(?:,|and|&|to|-)\s*\d+[A-Z]?(?:\s*\([^)]{1,6}\))*)*)",
    re.I,
)
_SECTION_WINDOW = 90

CITATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "scr": re.compile(r"\[(\d{4})\]\s*(\d+)\s*S\.?\s?C\.?\s?R\.?\s*(\d+)"),
    "insc": re.compile(r"\b(\d{4})\s*INSC\s*(\d+)\b"),
    "scc": re.compile(r"\((\d{4})\)\s*(\d+)\s*SCC\s*(\d+)"),
    "air_sc": re.compile(r"\bAIR\s*(\d{4})\s*SC\s*(\d+)\b"),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CoreRule(_Strict):
    acts: list[ActKey]
    min_mentions: int = Field(ge=1)
    min_section_refs: int = Field(ge=1)


class OneHop(_Strict):
    resolve_via: list[Literal["scr", "insc", "scc", "air_sc"]]
    max_judgments: int = Field(ge=0)
    min_cited_by: int = Field(ge=1)


class Distractors(_Strict):
    fraction_of_selected: float = Field(ge=0, le=1)
    min: int = Field(ge=0)


class Adjustment(_Strict):
    acts: list[ActKey] | None = None
    min_mentions: int | None = None
    min_section_refs: int | None = None
    one_hop_max_judgments: int | None = None


class SupremeCourtSlice(_Strict):
    source: Literal["aws_sc"]
    years: tuple[int, int]
    core: CoreRule
    one_hop: OneHop
    distractors: Distractors
    size_bounds: tuple[int, int]
    if_too_large: list[Adjustment] = []
    if_too_small: list[Adjustment] = []


class Court(_Strict):
    name: str
    code: str


class ReasonedOnly(_Strict):
    min_pages: int
    min_numbered_paragraphs: int


class HighCourtSlice(_Strict):
    source: Literal["aws_hc"]
    courts: list[Court]
    years: tuple[int, int]
    candidate_case_types: list[str]
    core: CoreRule
    reasoned_only: ReasonedOnly
    max_judgments: int


class StatuteSpec(_Strict):
    key: ActKey
    title: str
    act_no: str
    note: str = ""
    optional: bool = False


class StatuteSlice(_Strict):
    source: Literal["hf_legal_docs"]
    acts: list[StatuteSpec]
    split: Literal["section"]
    verify_against: str


class SliceConfig(_Strict):
    name: str
    seed: int
    supreme_court: SupremeCourtSlice
    high_courts: HighCourtSlice
    statutes: StatuteSlice


def load_slice(name_or_path: str | Path, configs_dir: Path = CORPUS_CONFIGS_DIR) -> SliceConfig:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = configs_dir / f"{name_or_path}.yaml"
    return SliceConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ActSignal:
    mentions: dict[str, int]
    section_refs: dict[str, list[str]]


def act_signal(text: str) -> ActSignal:
    """Count mentions of each Act and collect section numbers cited just before it."""
    mentions: dict[str, int] = {}
    refs: dict[str, list[str]] = {}
    for key, pattern in ACT_PATTERNS.items():
        found = list(pattern.finditer(text))
        mentions[key] = len(found)
        sections: list[str] = []
        for match in found:
            window = text[max(0, match.start() - _SECTION_WINDOW) : match.start()]
            tail = list(_SECTION.finditer(window))
            # The section list must run right up to the Act name ("... of the Contract Act").
            if tail and re.fullmatch(
                r"[\s,()a-z]*(?:of\s+the)?\s*", window[tail[-1].end() :], re.I
            ):
                sections += re.findall(r"\d+[A-Z]?", tail[-1].group(1))
        refs[key] = sections
    return ActSignal(mentions=mentions, section_refs=refs)


def is_core(signal: ActSignal, rule: CoreRule) -> bool:
    mentions = sum(signal.mentions.get(a, 0) for a in rule.acts)
    section_refs = sum(len(signal.section_refs.get(a, [])) for a in rule.acts)
    return mentions >= rule.min_mentions or section_refs >= rule.min_section_refs


def cited_sc_authorities(text: str) -> dict[str, set[tuple[str, ...]]]:
    """Distinct SC citations in a judgment, per reporter style.

    SCR reprints carry their own citation in every page header ("[2023] 1 S.C.R. 809"),
    so the ``scr`` set includes the judgment's own year/volume; callers resolving the
    citation graph (PLAN 3.9) must drop those.
    """
    return {style: set(p.findall(text)) for style, p in CITATION_PATTERNS.items()}
