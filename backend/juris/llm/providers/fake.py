"""Deterministic provider for tests: canned responses keyed by a tag."""

from collections.abc import Mapping, Sequence

from juris.llm.types import ProviderRequest, ProviderResponse

Canned = str | ProviderResponse | Exception


class FakeProvider:
    """Returns the next canned response for ``request.tags[key_tag]``.

    A string becomes a response with token counts of 10 in / ``len(text)`` out.
    An exception is raised instead of answering. Every request is recorded in ``calls``.
    """

    def __init__(
        self,
        responses: Mapping[str, Sequence[Canned]],
        *,
        key_tag: str = "agent",
        name: str = "fake",
    ) -> None:
        self.name = name
        self.key_tag = key_tag
        self._queues = {key: list(items) for key, items in responses.items()}
        self.calls: list[ProviderRequest] = []

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.calls.append(request)
        key = request.tags.get(self.key_tag)
        queue = self._queues.get(key or "")
        if not queue:
            raise AssertionError(f"FakeProvider has no response left for {self.key_tag}={key!r}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, ProviderResponse):
            return item
        return ProviderResponse(
            text=item, input_tokens=10, output_tokens=len(item), stop_reason="end_turn"
        )
