"""The agent framework (PLAN 7.2): one shape for every LLM-backed role.

An agent turns the part of the record it may see into a prompt, gets a structured output, and
writes it through the Case Record. ``run`` does it in this order:

1. **Prompt.** The system prompt is the shared header (``prompts/_shared.md``: the role, the
   rules every agent follows, the output schema). The user prompt is the role's template filled
   from ``input_view``, which says exactly what this agent may see. Jurors, for instance, get the
   frozen record without retrieval scores. Keeping it minimal limits cost and cross-talk.
2. **Check.** ``validate`` checks the output against the record without writing anything and
   returns ``Problem``s. Each one says where in the output it is: ``("claims", 2, "citations",
   0)``.
3. **Repair, once.** If there are problems, the previous output and the problems go back to the
   model (``prompts/_repair.md``, IDEA_final §8). This is separate from the gateway's own
   one-retry for output that does not match the schema.
4. **Drop.** Whatever still fails is dropped: the list item each problem points into. Then the
   output is checked again, since dropping a claim's only citation leaves a claim with none, and
   so on until it is clean. One ``validation_failed`` event records the errors and what was
   dropped. A problem that points at no list item (or an output that no longer fits its schema)
   drops the whole output, and ``run`` returns None.
5. **Apply.** ``apply`` writes the clean output through the Case Record. Since ``validate``
   already checked it, a rejection here is a bug and fails the stage.

``run`` is ``produce`` (steps 1-4) then ``apply``. A stage that runs several agents at once
(Counsel A and B) awaits their ``produce`` together and applies the outputs in a fixed order,
so the record's IDs do not depend on which call finished first. ``attempts`` keeps each output
the model gave and its problems (first, then after the repair), for the stage's statistics.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from juris.config import Role
from juris.events import CaseView
from juris.events.catalog import AgentCompletedPayload, AgentStartedPayload
from juris.llm.errors import StructuredOutputError
from juris.llm.schema import output_schema
from juris.llm.types import ChatMessage, LLMResult, Usage
from juris.models.common import Stage
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.prompts import PROMPTS_DIR, load_prompt

SHARED_PROMPT = "_shared"
REPAIR_PROMPT = "_repair"
WHOLE_OUTPUT = "the whole output"
_MAX_DROP_ROUNDS = 50  # each round removes at least one item, so this is only a guard

OutputPath = tuple[str | int, ...]


@dataclass(frozen=True)
class Problem:
    """Why part of an output is rejected, and where it is (``()``: the output as a whole)."""

    message: str
    path: OutputPath = ()

    def __str__(self) -> str:
        return f"{where(self.path)}: {self.message}" if self.path else self.message


def where(path: OutputPath) -> str:
    """``("claims", 2, "citations", 0)`` -> ``claims[2].citations[0]``."""
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else f".{part}" if out else part
    return out


class Agent[Out: BaseModel](ABC):
    """Subclasses set the class attributes and implement ``input_view`` and ``apply``."""

    role: ClassVar[Role]
    prompt: ClassVar[str]  # the role's template in ``prompts_dir``
    duty: ClassVar[str]  # one sentence for the shared header: what this agent does
    output_model: type[Out]
    max_tokens: ClassVar[int] = 4000
    prompts_dir: ClassVar[Path] = PROMPTS_DIR

    def __init__(self, name: str | None = None, *, seed_offset: int = 0) -> None:
        self.name = name or self.role.value  # e.g. "counsel_a"; names events and call tags
        self.seed_offset = seed_offset  # jurors differ by seed (IDEA_final §6.1)
        self.attempts: list[tuple[Out, list[Problem]]] = []  # the last run's outputs, checked

    @property
    def call_tags(self) -> dict[str, str]:
        """Extra tags on this agent's calls (e.g. the judgment a Precedent Analyst reads)."""
        return {}

    @abstractmethod
    def input_view(self, view: CaseView) -> dict[str, str]:
        """The role template's fields: everything this agent sees, and nothing more."""

    def validate(self, output: Out, record: CaseRecord) -> list[Problem]:
        """Problems with the output against the record, found without writing anything."""
        return []

    @abstractmethod
    def apply(self, output: Out, record: CaseRecord, *, stage: Stage) -> None:
        """Write the (validated) output through the Case Record."""

    # ---- the run ----------------------------------------------------------------------------

    def system_prompt(self) -> str:
        shared = load_prompt(SHARED_PROMPT)
        schema = json.dumps(output_schema(self.output_model), ensure_ascii=False)
        return Template(shared.body).substitute(
            role=self.role.value.replace("_", " "), duty=self.duty, schema=schema
        )

    async def run(self, ctx: StageContext) -> Out | None:
        """Prompt, check, repair once, drop what still fails, apply. None if nothing is left."""
        output = await self.produce(ctx)
        if output is not None:
            self.apply(output, ctx.record, stage=ctx.stage)
        return output

    async def produce(self, ctx: StageContext) -> Out | None:
        """Prompt, check, repair once, drop what still fails; write nothing but the agent and
        validation events. None if nothing is left."""
        record, emit = ctx.record, ctx.run.emitter.emit
        self.attempts = []
        model = ctx.run.config.model_for(self.role)
        emit(
            AgentStartedPayload(role=self.role, model=model.name), stage=ctx.stage, agent=self.name
        )
        template = load_prompt(self.prompt, self.prompts_dir)
        messages = [
            ChatMessage(
                role="user",
                content=Template(template.body).substitute(self.input_view(record.view)),
            )
        ]
        tags = {"prompt": f"{template.id}@v{template.version}"}
        usage, cached = Usage(), True
        output: Out | None = None
        try:
            result = await self._complete(ctx, messages, tags)
            usage, cached = usage + result.usage, cached and result.cached
            assert result.parsed is not None  # the gateway raises if it never validates
            output = result.parsed
            problems = self.validate(output, record)
            self.attempts.append((output, problems))
            if problems:
                messages += [
                    ChatMessage(role="assistant", content=result.text),
                    ChatMessage(role="user", content=self.repair_prompt(problems)),
                ]
                try:
                    repaired = await self._complete(ctx, messages, tags | {"repair": "record"})
                except StructuredOutputError:
                    pass  # keep the first output; its problems are dropped below
                else:
                    usage, cached = usage + repaired.usage, cached and repaired.cached
                    assert repaired.parsed is not None
                    output = repaired.parsed
                    problems = self.validate(output, record)
                    self.attempts.append((output, problems))
            if problems:
                output = self._drop(output, problems, ctx)
        except StructuredOutputError as exc:
            record.reject(
                [f"the output did not match the schema: {exc}"],
                stage=ctx.stage,
                agent=self.name,
                dropped=WHOLE_OUTPUT,
            )
            output = None
        emit(
            AgentCompletedPayload(
                role=self.role,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
                cached=cached,
            ),
            stage=ctx.stage,
            agent=self.name,
        )
        return output

    def repair_prompt(self, problems: Sequence[Problem]) -> str:
        template = load_prompt(REPAIR_PROMPT)
        return Template(template.body).substitute(problems="\n".join(f"- {p}" for p in problems))

    async def _complete(
        self, ctx: StageContext, messages: Sequence[ChatMessage], tags: dict[str, str]
    ) -> LLMResult[Out]:
        return await ctx.run.gateway.complete(
            messages,
            model=ctx.run.config.model_for(self.role),
            response_model=self.output_model,
            system=self.system_prompt(),
            seed=ctx.run.seed + self.seed_offset,
            max_tokens=self.max_tokens,
            tags={
                "agent": self.name,
                "role": self.role.value,
                "stage": ctx.stage.value,
                "item": ctx.run.item.id,
                **self.call_tags,
                **tags,
            },
        )

    def _drop(self, output: Out, problems: list[Problem], ctx: StageContext) -> Out | None:
        """Drop the items the problems point into until the output is clean (see the module
        docstring). Logs one ``validation_failed``."""
        errors = [str(p) for p in problems]
        dropped: list[str] = []
        current: Out | None = output
        for _ in range(_MAX_DROP_ROUNDS):
            if current is None or not problems:
                break
            items = [_item(p.path) for p in problems]
            if any(not item for item in items):
                current = None  # a problem with the output as a whole
                break
            data = current.model_dump(mode="json")
            for item in sorted(set(items), key=_order, reverse=True):  # last first: no shifts
                _remove(data, item)
                dropped.append(where(item))
            try:
                current = self.output_model.model_validate(data)
            except ValidationError as exc:
                errors.append(f"after dropping, the output no longer fits its schema: {exc}")
                current = None
                break
            problems = self.validate(current, ctx.record)
            errors += [str(p) for p in problems]
        ctx.record.reject(
            errors,
            stage=ctx.stage,
            agent=self.name,
            dropped=WHOLE_OUTPUT if current is None else ", ".join(dropped),
        )
        return current


def _item(path: OutputPath) -> OutputPath:
    """The path up to the last list index: the item to drop (``()`` if there is none)."""
    last = max((n for n, part in enumerate(path) if isinstance(part, int)), default=-1)
    return path[: last + 1]


def _order(path: OutputPath) -> tuple[str, ...]:
    """Sort key: list indices compare as numbers, so reversing removes the last item first."""
    return tuple(f"{p:010d}" if isinstance(p, int) else p for p in path)


def _remove(data: Any, path: OutputPath) -> None:
    for part in path[:-1]:
        data = data[part]
    del data[path[-1]]
