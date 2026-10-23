"""The Juris system (IDEA_final §5 "P", PLAN 7.12): the orchestrator with every stage, as a
``System`` for the experiment runner (``kind: juris``) and ``scripts/run_case.py``.

The profile decides what runs: each stage ``on``, ``off`` or ``passthrough`` (``stages``), and
the deliberation parameters (``params``). The ablations (``configs/pipeline/no_*.yaml``) are
profiles: one role off or replaced by its passthrough.

The retrieval service is made once per process, on the CPU (the GPU belongs to the local LLM,
as for B1), and shared by S1, S3 and S5's searches. It is also the corpus the Case Record
reads documents and chunks from.
"""

from collections.abc import Callable
from typing import Protocol

from juris.agents.researcher import ResearchService
from juris.baselines.b1 import B1System
from juris.baselines.base import RunContext
from juris.models.common import Stage
from juris.pipeline.orchestrator import Orchestrator, StageRunner
from juris.pipeline.stages.cross_exam import CrossExamStage
from juris.pipeline.stages.final_verify import FinalVerificationStage
from juris.pipeline.stages.framing import IssueFramingStage
from juris.pipeline.stages.judge import JudgeStage
from juris.pipeline.stages.jury import JuryStage
from juris.pipeline.stages.opening import OpeningStage
from juris.pipeline.stages.precedent import PrecedentService, PrecedentStage
from juris.pipeline.stages.rebuttal import RebuttalStage
from juris.pipeline.stages.research import ResearchStage
from juris.pipeline.stages.verify import VerificationGate

# The ablation profiles in configs/pipeline/ (IDEA_final §5 "P-X"; PLAN 7.12).
ABLATIONS = (
    "no_opponent",
    "no_precedent_analyst",
    "no_rebuttal",
    "no_cross_exam",
    "no_jury",
    "no_verifier_gates",
)


class JurisService(ResearchService, PrecedentService, Protocol):
    """What the stages need from retrieval: search and lookups (S1, S5), judgment passages
    and citing cases (S3), and the records of a chunk (the Case Record)."""


def _local_service() -> JurisService:
    from juris.retrieval.service import RetrievalService

    return RetrievalService.local(device="cpu")


class JurisSystem:
    corpus_snapshot_id = B1System.corpus_snapshot_id  # the same slice and retriever as B0-B2
    embedding_model = B1System.embedding_model

    def __init__(self, service: Callable[[], JurisService] = _local_service) -> None:
        self._make_service = service
        self._service: JurisService | None = None

    def service(self) -> JurisService:
        if self._service is None:
            self._service = self._make_service()
        return self._service

    def stages(self, max_evidence_chars: int | None) -> list[StageRunner]:
        research = ResearchStage(self.service)
        return [
            IssueFramingStage(),
            research,
            OpeningStage(max_evidence_chars=max_evidence_chars),
            PrecedentStage(self.service),
            VerificationGate(Stage.S4),
            RebuttalStage(lambda: research.researcher, max_evidence_chars=max_evidence_chars),
            CrossExamStage(),
            VerificationGate(Stage.S7),
            JuryStage(),
            JudgeStage(),
            FinalVerificationStage(),
        ]

    async def run(self, ctx: RunContext) -> None:
        stages = self.stages(ctx.config.profile.params.max_evidence_chars)
        orchestrator = Orchestrator(
            stages,
            self.service,
            corpus_snapshot_id=self.corpus_snapshot_id,
            embedding_model=self.embedding_model,
        )
        await orchestrator.run(ctx)
