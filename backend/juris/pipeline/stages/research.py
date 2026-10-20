"""S1, research (PLAN 7.4): the Researcher builds and registers each issue's evidence pool.
Required, since counsel can cite only registered evidence; a case whose search found nothing
fails here.

The retrieval service is made on first use and shared by S1 and S5's targeted searches
(``researcher``), so the encoders load once per process.

Passthrough: each issue's question alone, ``evidence_per_issue_min`` chunks, no provisions.
"""

from collections.abc import Callable
from dataclasses import asdict

from juris.agents.researcher import IssuePool, Researcher, ResearchService
from juris.models.common import Stage
from juris.pipeline.budget import Spend
from juris.pipeline.orchestrator import StageContext


class ResearchFailed(RuntimeError):
    """No evidence could be registered for the case."""


class ResearchStage:
    stage = Stage.S1
    optional = False

    def __init__(
        self,
        service: Callable[[], ResearchService],
        *,
        candidate_k: int = 60,
        per_document: int = 3,
        cite_min: int = 3,
        rerank: bool = False,
    ) -> None:
        self._make_service = service
        self._options = (candidate_k, per_document, cite_min, rerank)
        self._researcher: Researcher | None = None

    @property
    def researcher(self) -> Researcher:
        if self._researcher is None:
            candidate_k, per_document, cite_min, rerank = self._options
            self._researcher = Researcher(
                self._make_service(),
                candidate_k=candidate_k,
                per_document=per_document,
                cite_min=cite_min,
                rerank=rerank,
            )
        return self._researcher

    def estimate(self, ctx: StageContext) -> Spend | None:
        return Spend()  # no model calls

    async def run(self, ctx: StageContext) -> None:
        params = ctx.params
        pools = self.researcher.research(
            ctx.record,
            stage=self.stage,
            min_items=params.evidence_per_issue_min,
            max_items=params.evidence_per_issue_max,
        )
        self._record(ctx, pools)

    async def passthrough(self, ctx: StageContext) -> None:
        researcher = self.researcher
        record = ctx.record
        n = ctx.params.evidence_per_issue_min
        for issue in record.view.issues.values():
            hits = researcher.service.search([issue.question], k=n, rerank=False)
            for hit in hits:
                record.register_evidence(hit, [issue.id], stage=self.stage, agent=researcher.name)
        if not record.view.evidence:
            raise ResearchFailed("the search found no evidence for any issue")

    def _record(self, ctx: StageContext, pools: list[IssuePool]) -> None:
        view = ctx.record.view
        if not view.evidence:
            raise ResearchFailed("the search found no evidence for any issue")
        values = ctx.run.values
        values["s1.evidence"] = float(len(view.evidence))
        values["s1.documents"] = float(len({e.document_id for e in view.evidence.values()}))
        values["s1.provisions_kept"] = float(sum(len(p.provisions_kept) for p in pools))
        values["s1.provisions_unconfirmed"] = float(
            sum(len(p.provisions_unconfirmed) for p in pools)
        )
        ctx.run.details["s1.pools"] = [asdict(p) for p in pools]
