# Citation edges: hand-checked precision (PLAN 3.9)

**Done-when:** precision of resolved edges is at least 90% on a hand-checked sample of 50. Resolution rates per reporter style are in [citation_graph.md](citation_graph.md).

## Method

Resolved case edges were sampled with a fixed seed, stratified by reporter style and resolution route. For each edge, the citing context (about 160 characters either side) was read next to the title of the judgment the edge points to, taken from the AWS index. An edge counts as correct when the case named at the citation is that judgment. In `Case Law Reference` lists no name is printed, so there the report at that citation is taken as the cited case.

## Sample 1 (seed 39): before the fixes, 43 of 50 correct (86%)

The sample was 20 SCR direct, 5 INSC, 5 SCR page-range, 15 SCC alias, 3 AIR alias and 2 SCC OnLine alias edges. The seven wrong edges show three patterns:

- **Page range, 4 of 5 wrong.** An SCR page inside another report's pages was usually a misprint, not a pinpoint: "*Union of India v. H.S. Dhillon* (1972) 2 SCR 331" (for 33), "*I.R. Coelho* … 2001 (1) SCR 706" (for 2007), "*Sopan Sukhdeo Sable* … 2004 (1) SCR 100", and "[2019] 1 SCR 123" in a list. The range rule matched each of these to the report covering that page. The only range hit that was right was *Bandhua Mukti Morcha* "[1984] 2 SCR 79" (the report starts at 67).
- **Parallel members disagree, 1 wrong.** In "*Shamarao Parulekar v. District Magistrate* [1952] 1 SCR 683 : (1952) 2 SCC 1 : 1952 INSC 63", the printed INSC number is that of *Godavari Parulekar*.
- **Misprint or weak alias, 2 wrong.**
  - "*Gujarat Ambuja Cements* … 2005 (4) SCC 214 : 2000 (2) SCR 594": the source misprints the SCR citation, which really is *Monhinder Kaur*.
  - "*Kharbula Kuer* AIR (1963) SC 1203": the alias was learnt from a single printed pairing with *M.R. Balaji*.

**Fixes applied (D-027):**
1. SCR citations resolve by first page only; a page inside another report is left unresolved.
2. When the members of a parallel citation resolve to different judgments, all of them are left unresolved (`resolution = 'conflict'`, 17 citations).

Re-checked under these rules, 5 of the 7 are now unresolved, and the *Kharbula Kuer* alias no longer resolves. The one remaining wrong edge is the source misprint (2000 (2) SCR 594). That leaves **44 of 45 resolved edges correct (97.8%)**.

## Sample 2 (seed 2026): after the fixes, 50 of 50 correct (100%)

A fresh sample of 20 SCR, 5 INSC, 18 SCC, 5 AIR and 2 SCC OnLine resolved edges. Every edge names the judgment it points to. Examples:

| Citation in the text | Resolved to |
|---|---|
| *Samsher Singh v. State of Punjab* (1974) 2 SCC 831 : 1975 (1) SCR 814 | *Shamsher Singh & Anr v. State of Punjab* |
| *Cox and Kings Ltd v. SAP India* (2022) 8 SCC 1 [2022 INSC 523] | *Cox and Kings Limited v. SAP India* |
| *Central Inland Water Transport v. Brojo Nath Ganguly* [1986] 3 SCC 156 (alias) | *Central Inland Water Transport Corporation v. Brojo Nath Ganguly* |
| *Kharak Singh v. State of U.P.* AIR 1963 SC 1295 (alias) | *Kharak Singh v. State of U.P.* |
| *K.T. Moopil Nair v. State of Kerala* 1960 SCC OnLine SC 7 (alias) | *Kunnathat Thathunni Moopil Nair v. State of Kerala* |
| *Secretary, Irrigation Department v. G.C. Roy* (1992) 1 SCC 508 (alias) | *Secretary, Irrigation Department, Government of Orissa v. G. C. Roy* |

**Result: 50 of 50 (100%)**, above the 90% target. Over both samples under the final rules, 94 of 95 resolved edges are correct (98.9%).

## Statute edges (spot-check of 10)

Ten resolved statute edges were read in context, and all name the right Act and section:

| Section | Citing text |
|---|---|
| ICA s. 28 (×2) | "Section 28 of the Contract Act and Access to Justice"; "contrary to Sections 28 and 23 of the Contract Act" |
| ICA s. 176 | "his right under section 176 of the Indian Contract Act" |
| ICA s. 17 | "Section 17 of Indian Contract Act, 1872" |
| ICA s. 70 | "Section 70 of the Indian Contract Act" |
| SRA s. 42 | "Sections 36 to Section 42 of the Specific Relief Act, 1963" |
| SRA s. 17 | "s. 17 of the Specific Relief Act, 1963" |
| SRA s. 34 (×2) | two mentions of "section 34 of the Specific Relief Act" |
| SRA s. 10 | "Section 10 of the Specific Relief Act, 1963" |

**Fix applied (D-027):** in a judgment decided before the Act commenced, an undated reference ("the Specific Relief Act" in 1955) is to the older Act (the Specific Relief Act, 1877), so it stays unresolved. 65 mentions were affected.

## Known limitations

- **Misprints.** A printed citation that happens to be another report's first page resolves to that report.
- **Single-pairing aliases.** Most aliases (2,599 of 3,761) come from one printed pairing. A misprinted pairing can mislead them, although the conflict rule removes contradictions.
- **SCALE, JT, HC neutral citations** have no index to resolve against; they resolve only through aliases.
- **Treatment cues are weak signals.** They are taken from the citing sentence, and "relied on" or "followed" near several citations attaches to each of them. They are never verified treatment (D-013).
