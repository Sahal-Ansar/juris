import hashlib
from pathlib import Path
from typing import Any

import pytest

from juris.ingest import acquire as a
from juris.ingest.slice import (
    Adjustment,
    CoreRule,
    Distractors,
    OneHop,
    SupremeCourtSlice,
    load_slice,
)

# ---- Fetcher --------------------------------------------------------------------------


def test_fetcher_downloads_once_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"judgment bytes")
    sha = hashlib.sha256(b"judgment bytes").hexdigest()
    fetcher = a.Fetcher(min_interval=0)
    dest = tmp_path / "out" / "doc.pdf"

    first = fetcher.download(source.as_uri(), dest, expect_sha256=sha)
    second = fetcher.download(source.as_uri(), dest, expect_sha256=sha)

    assert first.downloaded and not second.downloaded
    assert first.sha256 == second.sha256 == sha
    assert fetcher.requests == 1  # the second call never touched the source
    assert not list(dest.parent.glob("*.part"))


def test_fetcher_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="does not match"):
        a.Fetcher(min_interval=0).download(source.as_uri(), tmp_path / "d.pdf", "0" * 64)
    assert not (tmp_path / "d.pdf").exists() and not (tmp_path / "d.pdf.part").exists()


# ---- Text helpers ---------------------------------------------------------------------


def test_numbered_paragraphs_counts_the_longest_run() -> None:
    text = "1. Facts.\n2. More.\n3. Law.\n7. Stray.\n1. Order.\n2. Costs."
    assert a.numbered_paragraphs(text) == 3
    assert a.numbered_paragraphs("no numbers here") == 0


def test_unreadable_pdf_is_recorded_not_fatal() -> None:
    record = a.scan_record("year=2023/english/2023_1_1_5_EN.pdf", b"not a pdf")
    assert record["path"] == "2023_1_1_5"
    assert "error" in record and record["bytes"] == 9


# ---- Citation resolution ----------------------------------------------------------------


def rec(path: str, **kw: Any) -> dict[str, Any]:
    return {
        "path": path,
        "year": int(path[:4]),
        "chars": 1000,
        "mentions": {"contract_act": 0, "specific_relief_act": 0, "sale_of_goods_act": 0},
        "section_refs": {"contract_act": [], "specific_relief_act": [], "sale_of_goods_act": []},
        "scr": [],
        "insc": [],
        **kw,
    }


def test_citation_index_resolves_page_ranges_and_neutral_citations() -> None:
    meta = {"2019_5_100_130": {"case_id": "2019 INSC 77"}}
    index = a.CitationIndex.build(["1964_1_515_540", "2019_5_100_130", "2023_16_872_887"], meta)
    citing = rec(
        "2023_16_872_887",
        scr=[[1964, 1, 515], [1964, 1, 530], [1964, 2, 515], [2023, 16, 880]],
        insc=[[2019, 77], [2019, 999]],
    )
    # start page and a pinpoint inside the range both resolve; the wrong volume and the
    # judgment's own page header (2023/16/880) do not; unknown INSC numbers are ignored.
    assert index.resolve(citing, ["scr", "insc"]) == {"1964_1_515_540", "2019_5_100_130"}
    assert index.resolve(citing, ["insc"]) == {"2019_5_100_130"}


# ---- Selection ---------------------------------------------------------------------------


def sc_config(**over: Any) -> SupremeCourtSlice:
    base: dict[str, Any] = {
        "source": "aws_sc",
        "years": (1950, 2026),
        "core": CoreRule(
            acts=["contract_act", "specific_relief_act"], min_mentions=2, min_section_refs=1
        ),
        "one_hop": OneHop(resolve_via=["scr", "insc"], max_judgments=2, min_cited_by=2),
        "distractors": Distractors(fraction_of_selected=0.5, min=1),
        "size_bounds": (1, 100),
    }
    return SupremeCourtSlice(**{**base, **over})


def records() -> list[dict[str, Any]]:
    contract = {"contract_act": 3, "specific_relief_act": 0, "sale_of_goods_act": 0}
    passing = {"contract_act": 1, "specific_relief_act": 0, "sale_of_goods_act": 0}
    return [
        rec("2001_1_1_10", mentions=contract, scr=[[1990, 1, 5], [1991, 2, 3]]),
        rec("2002_1_1_10", mentions=contract, scr=[[1990, 1, 7], [1992, 3, 1]]),
        rec(
            "2003_1_1_10",
            section_refs={
                "contract_act": ["74"],
                "specific_relief_act": [],
                "sale_of_goods_act": [],
            },
            scr=[[1990, 1, 9], [1991, 2, 4]],
        ),
        rec("2004_1_1_10", mentions=passing),  # a passing mention is not core
        rec("1990_1_1_20"),
        rec("1991_2_1_20"),
        rec("1992_3_1_20"),
        *[rec(f"2010_{n}_1_10") for n in range(1, 9)],
        {"path": "2011_1_1_10", "year": 2011, "error": "PdfiumError"},
    ]


def test_select_sc_core_one_hop_and_distractors() -> None:
    selection = a.select_sc(sc_config(), records(), {}, seed=7)
    assert selection.core == ["2001_1_1_10", "2002_1_1_10", "2003_1_1_10"]
    # 1990 is cited by all three core judgments, 1991 by two, 1992 by one (< min_cited_by).
    assert selection.one_hop == ["1990_1_1_20", "1991_2_1_20"]
    assert selection.cited_by == {"1990_1_1_20": 3, "1991_2_1_20": 2}
    assert len(selection.distractors) == 3  # 50% of 5 selected, rounded half up
    assert not set(selection.distractors) & set(selection.core + selection.one_hop)
    assert selection.scan_errors == 1 and selection.total == 8
    # Seeded: the same inputs give the same distractors.
    assert a.select_sc(sc_config(), records(), {}, seed=7).distractors == selection.distractors


def test_select_sc_applies_size_guard_in_order() -> None:
    config = sc_config(
        size_bounds=(1, 4),
        if_too_large=[
            Adjustment(min_mentions=5, min_section_refs=5),
            Adjustment(one_hop_max_judgments=0),
        ],
    )
    # 3 core + 2 one-hop + 3 distractors = 8 > 4, so the first adjustment applies. It
    # leaves no core judgment at all, the total drops to the 1-distractor minimum, and the
    # second adjustment is not needed.
    selection = a.select_sc(config, records(), {}, seed=7)
    assert selection.core == [] and selection.one_hop == []
    assert len(selection.adjustments) == 1 and selection.core_rule.min_mentions == 5
    assert selection.total == 1

    # A milder first step keeps the section-ref judgment; its citations are too few for
    # one-hop, so it also stops after one step.
    milder = sc_config(size_bounds=(1, 4), if_too_large=[Adjustment(min_mentions=5)])
    selection = a.select_sc(milder, records(), {}, seed=7)
    assert selection.core == ["2003_1_1_10"] and selection.one_hop == []
    assert selection.total <= 4


def test_repo_slice_selection_config_loads() -> None:
    config = load_slice("mvp_contract")
    assert config.supreme_court.one_hop.resolve_via == ["scr", "insc"]


# ---- High Courts and statutes --------------------------------------------------------------


def test_case_type_and_pdf_key() -> None:
    assert a.case_type("CS(COMM)/123/2020 of A Vs B") == "CS(COMM)"
    assert a.case_type("no slash title") == ""
    row = {
        "pdf_link": "court/cnrorders/x/orders/DLHC01_1_2021-01-02.pdf",
        "year": "2021",
        "court": "7_26",
        "bench": "dhcdb",
    }
    assert a.hc_pdf_key(row) == "data/pdf/year=2021/court=7_26/bench=dhcdb/DLHC01_1_2021-01-02.pdf"


def test_statute_title_normalisation() -> None:
    assert a._norm_title("THE SPECIFIC RELIEF ACT, 1963") == a._norm_title(
        "The Specific Relief Act, 1963"
    )
    assert (
        a._norm_title("The Sale of Goods Act, 1930 (Act No. 3 of 1930)")
        == "the sale of goods act 1930"
    )


# ---- Manifest ---------------------------------------------------------------------------------


def test_manifest_id_depends_only_on_content(tmp_path: Path) -> None:
    paths = a.Paths(tmp_path)
    config = load_slice("mvp_contract")
    selection = a.ScSelection(
        core=["2001_1_1_10"],
        one_hop=[],
        distractors=["2010_1_1_10"],
        core_rule=config.supreme_court.core,
        one_hop_cap=0,
        scanned=2,
        in_years=2,
        scan_errors=0,
    )
    for path in ("2001_1_1_10", "2010_1_1_10"):
        pdf = paths.sc_pdf / f"year={path[:4]}" / f"{path}_EN.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(path.encode())
    config_path = tmp_path / "slice.yaml"
    config_path.write_text("x", encoding="utf-8")

    first = a.build_manifest(config, config_path, selection, paths, ["note"])
    second = a.build_manifest(config, config_path, selection, paths, ["note"])
    assert first.snapshot_id == second.snapshot_id
    assert first.counts == {"core": 1, "distractor": 1, "total": 2}
    assert [d.local_path for d in first.documents] == [
        "raw/aws_sc/pdf/year=2001/2001_1_1_10_EN.pdf",
        "raw/aws_sc/pdf/year=2010/2010_1_1_10_EN.pdf",
    ]
    out = a.write_manifest(first, paths)
    assert out.name == f"{first.snapshot_id}.json" and a.write_manifest(second, paths) == out

    (paths.sc_pdf / "year=2010" / "2010_1_1_10_EN.pdf").write_bytes(b"changed")
    assert (
        a.build_manifest(config, config_path, selection, paths, []).snapshot_id != first.snapshot_id
    )


def test_read_scan_dedupes_judgments_found_in_two_tars(tmp_path: Path) -> None:
    import gzip
    import json

    paths = a.Paths(tmp_path)
    paths.sc_scan.mkdir(parents=True)
    rows = {
        1950: [
            {"path": "1951_1_1_51", "year": 1950, "sha256": "x"},
            {"path": "1950_1_1_9", "year": 1950},
        ],
        1951: [
            {"path": "1951_1_1_51", "year": 1951, "sha256": "x"},
            {"path": "S_1951_1_5_9", "year": 1951},
        ],
    }
    for year, records in rows.items():
        with gzip.open(paths.sc_scan / f"year={year}.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in records)

    scan = a.read_scan(paths)
    assert [r["path"] for r in scan] == ["1950_1_1_9", "1951_1_1_51", "S_1951_1_5_9"]
    # The copy from the tar whose year matches the path's year wins.
    assert next(r for r in scan if r["path"] == "1951_1_1_51")["year"] == 1951


def test_supplementary_volume_paths() -> None:
    assert a.path_year("S_1959_1_979_1008") == 1959
    assert a.sc_local_pdf(a.Paths(Path("d")), "S_1959_1_979_1008").as_posix() == (
        "d/raw/aws_sc/pdf/year=1959/S_1959_1_979_1008_EN.pdf"
    )
    # Supp. volumes are not SCR page-range targets: (1959, 1) would collide with the
    # regular volume 1 of 1959.
    index = a.CitationIndex.build(["S_1959_1_979_1008"], {})
    assert index.scr == {}
