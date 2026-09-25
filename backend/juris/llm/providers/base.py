from typing import Protocol

from juris.llm.types import ProviderRequest, ProviderResponse


class Provider(Protocol):
    """One LLM backend. Adapters raise ``juris.llm.errors`` exceptions, never SDK ones."""

    name: str

    async def complete(self, request: ProviderRequest) -> ProviderResponse: ...
