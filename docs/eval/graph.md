# The evidence graph (PLAN 8.1)

Code: `backend/juris/graph/builder.py` (`build_graph`, `check_graph`), models
`backend/juris/models/graph.py`, `CaseView.graph`, `scripts/graph_stats.py`. Decision: D-061.

## What it is

A view over the Case Record, built from a `CaseView` and never stored on its own. `CaseView.graph`
is a derived field: it is built whenever the view is serialised (about 3 ms for the largest dev
run) and dropped when a view is read back, so it always matches the record. It is also exported
on its own (`schemas/evidence_graph.schema.json`) for the 9.1 `/runs/{id}/graph` endpoint.

| Node | ID | from |
|---|---|---|
| issue, position | `I-1`, `I-1.A` | S0 |
| claim, argument | `C-001`, `A-001` | S2, S5 |
| counterargument | `R-001` | S5 |
| objection | `O-001` | S6 |
| evidence | `E-001` | S1-S5 |
| document | the corpus ID (`SC-2015_1_627_666`) | a judgment's evidence |
| statute_section | document ID + `#s<section>` (`ACT-contract_act#s74`), as its chunks are named | a statute's evidence |

| Edge | from → to | attributes |
|---|---|---|
| `about_issue` | position, evidence → issue | |
| `supports` | claim → position | weight from the claim's status: verified/survives 1, weak/contested 0.5, unsupported/falls 0, proposed none |
| `supports` | argument → claim; argument → counterargument | |
| `supports` / `challenges` | document → argument | the precedent card's fit: directly 1, by analogy 0.5, does not support 0 (supports); contrary 1, distinguishable 0.5 (challenges) |
| `challenges` | counterargument → the claim it rebuts | |
| `cites` | claim → evidence, one per citation | best verification result over all gates, entailment label, pinpoint, the argument; weight verified 1, weak 0.5, unsupported/invalid 0, unchecked none |
| `quotes` | evidence → document or statute section | pinpoint |
| `targets` | objection → any record object | (the objection node says whether it was resolved) |
| `case_cites_case` | a later judgment → the judgment it treated | treatment (followed, distinguished, ...) |

Nodes carry a short label, the issues they belong to (for filtering), the position they argue
for, a claim's status, whether an objection was resolved, and whether a precedent's treatment
was verified.

An edge is kept only if both ends are nodes. Two kinds of record links have no node at one
end and are left out: an objection aimed at an ID that is not in the record (none in the dev
runs), and a treatment by a later judgment that the run never retrieved (1 of 3 in the
juris_full smoke runs). `check_graph` returns what breaks the rules: a node whose ID is not an
object of its type in the record, an edge with an end that is not a node, an edge between node
types the table above does not allow, a repeated node or edge.

## Check on real runs (2026-10-24)

`uv run scripts/graph_stats.py eval/results/*-dev-n5-s1 C:/juris-data/runs/cases`: the 35 runs
of the 7.12 smoke test (juris_full and six ablations, 5 dev items, local 7B) and two
`run_case.py` cases. **37/37 graphs validate**, with no problems.

| over 37 runs | count |
|---|---|
| nodes | evidence 1,718, document 943, claim 154, argument 154, position 116, objection 109, statute section 77, issue 58, counterargument 44 |
| edges | about_issue 1,932, quotes 1,718, supports 421, cites 186, targets 109, challenges 55, case_cites_case 8 |
| citations (`cites`) | unsupported 112, verified 65, weak 9, unchecked 0 |

The 5 juris_full runs: 496 nodes and 652 edges in all (57 to 170 nodes per run). Of 249
evidence nodes, only the cited ones (28 citations) link to a claim; the rest hang off their
issue and source. 8.2's checks (orphan claims, weak-only support, positions with no verified
support) read these edges.

The mock fixtures (`fixtures/runs/*.caseview.json`) now include the graph: `coercion_full`
has every node and edge type (62 nodes, 102 edges); `test_graph.py` checks it after every
event of the run.
