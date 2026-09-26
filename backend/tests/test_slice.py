import pytest

from juris.ingest.slice import (
    CoreRule,
    act_signal,
    cited_sc_authorities,
    is_core,
    load_slice,
)

RULE = CoreRule(acts=["contract_act", "specific_relief_act"], min_mentions=2, min_section_refs=1)


def test_repo_slice_config_is_valid() -> None:
    config = load_slice("mvp_contract")
    assert config.supreme_court.size_bounds == (2000, 5000)
    assert {a.key for a in config.statutes.acts} >= {"contract_act", "specific_relief_act"}


@pytest.mark.parametrize(
    ("text", "act", "sections"),
    [
        ("under Section 73 of the Indian Contract Act, 1872", "contract_act", ["73"]),
        ("Sections 73 and 74 of the Contract Act", "contract_act", ["73", "74"]),
        ("S. 16(c) of the Specific Relief Act, 1963", "specific_relief_act", ["16"]),
        ("u/s 10 of the Specific Relief Act", "specific_relief_act", ["10"]),
        ("Ss. 55, 56 of the Contract Act", "contract_act", ["55", "56"]),
    ],
)
def test_section_references(text: str, act: str, sections: list[str]) -> None:
    assert act_signal(text).section_refs[act] == sections


def test_contract_labour_act_is_not_the_contract_act() -> None:
    signal = act_signal("the Contract Labour (Regulation and Abolition) Act, 1970")
    assert signal.mentions["contract_act"] == 0


def test_section_far_from_act_is_not_a_reference() -> None:
    text = "Section 302 IPC was invoked. Much later, the parties relied on the Contract Act."
    assert act_signal(text).section_refs["contract_act"] == []


def test_core_rule() -> None:
    assert is_core(act_signal("Section 74 of the Contract Act"), RULE)
    assert is_core(act_signal("the Contract Act ... again the Indian Contract Act"), RULE)
    assert not is_core(act_signal("a passing mention of the Contract Act"), RULE)
    assert not is_core(act_signal("Sale of Goods Act and Sale of Goods Act"), RULE)


def test_citation_styles() -> None:
    text = (
        "relied on (2015) 4 SCC 136, [1964] 1 S.C.R. 515, AIR 1964 SC 1882 "
        "and 2023 INSC 1043, and again (2015) 4 SCC 136."
    )
    found = cited_sc_authorities(text)
    assert found["scc"] == {("2015", "4", "136")}
    assert found["scr"] == {("1964", "1", "515")}
    assert found["air_sc"] == {("1964", "1882")}
    assert found["insc"] == {("2023", "1043")}
