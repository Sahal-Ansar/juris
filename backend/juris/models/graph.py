"""The evidence graph: a view over the Case Record for the UI and for analysis (IDEA_final §8).

Built from a ``CaseView`` by ``juris.graph.builder.build_graph``; never stored or edited on its
own. Node IDs are Case Record IDs (``I-1``, ``I-1.A``, ``C-001``, ``E-003``, a document ID);
a statute section's ID is its document ID plus ``#s<section>``, as its chunks are named
(``ACT-contract_act#s74``).
"""

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints

from juris.models.common import (
    ArgumentId,
    ClaimStatus,
    EntailmentLabel,
    IssueId,
    JurisModel,
    PositionId,
    PrecedentFit,
    Treatment,
    VerificationStatus,
)

Text = Annotated[str, StringConstraints(min_length=1)]
Weight = Annotated[float, Field(ge=0.0, le=1.0)]


class NodeType(StrEnum):
    ISSUE = "issue"
    POSITION = "position"
    CLAIM = "claim"
    ARGUMENT = "argument"
    COUNTERARGUMENT = "counterargument"
    OBJECTION = "objection"
    EVIDENCE = "evidence"
    DOCUMENT = "document"
    STATUTE_SECTION = "statute_section"


class EdgeType(StrEnum):
    """Edges point from the node that acts to the node acted on.

    - ``about_issue``: position or evidence → issue
    - ``supports``: claim → position, argument → claim, argument → counterargument (its
      reasoning), document → argument (precedent fit: directly, by analogy, or not at all)
    - ``challenges``: counterargument → claim, document → argument (distinguishable, contrary)
    - ``cites``: claim → evidence, one per citation
    - ``quotes``: evidence → document or statute section (the evidence is a span of it)
    - ``targets``: objection → the record object it objects to
    - ``case_cites_case``: a later judgment → the judgment it treated (followed, overruled, ...)
    """

    ABOUT_ISSUE = "about_issue"
    SUPPORTS = "supports"
    CHALLENGES = "challenges"
    CITES = "cites"
    QUOTES = "quotes"
    TARGETS = "targets"
    CASE_CITES_CASE = "case_cites_case"


class GraphNode(JurisModel):
    id: Text
    type: NodeType
    label: Text = Field(description="Short display text; the full object is in the CaseView")
    issue_ids: list[IssueId] = Field(default_factory=list, description="For filtering by issue")
    position_id: PositionId | None = Field(
        default=None, description="Claims, arguments and counterarguments: the side they argue"
    )
    claim_status: ClaimStatus | None = Field(default=None, description="Claims only")
    resolved: bool | None = Field(default=None, description="Objections only")
    treatment_verified: bool | None = Field(
        default=None, description="Documents with a precedent card only"
    )


class GraphEdge(JurisModel):
    source: Text
    target: Text
    type: EdgeType
    weight: Weight | None = Field(
        default=None,
        description="Strength of the relation, 0-1: a citation's verification, a claim's "
        "status, a precedent's fit. None when unknown or not applicable",
    )
    verification: VerificationStatus | None = Field(
        default=None, description="cites: the citation's best result; None = not checked"
    )
    label: EntailmentLabel | None = Field(default=None, description="cites: entailment label")
    pinpoint: str | None = Field(default=None, description="cites and quotes")
    argument_id: ArgumentId | None = Field(
        default=None, description="cites: the argument the citation is in"
    )
    fit: PrecedentFit | None = Field(default=None, description="document → argument edges")
    treatment: Treatment | None = Field(default=None, description="case_cites_case")


class EvidenceGraph(JurisModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
