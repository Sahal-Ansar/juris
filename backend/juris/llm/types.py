"""Types shared by the gateway and the provider adapters."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from juris.config import ModelSpec


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str


class ProviderRequest(BaseModel):
    """Everything a provider needs for one call. ``tags`` are not part of the cache key."""

    model_config = ConfigDict(frozen=True)

    model: ModelSpec
    messages: tuple[ChatMessage, ...]
    system: str | None = None
    max_tokens: int = 16000
    temperature: float | None = None
    seed: int | None = None
    json_schema: dict[str, Any] | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        cost = (
            None
            if self.cost_usd is None or other.cost_usd is None
            else self.cost_usd + other.cost_usd
        )
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
        )


class LLMResult[T: BaseModel](BaseModel):
    """What ``LLMGateway.complete`` returns.

    ``usage`` is what this call spent: zero on a cache hit. ``provider_calls`` is 2
    when the structured-output repair retry ran.
    """

    text: str
    parsed: T | None = None
    model: ModelSpec
    usage: Usage
    cached: bool = False
    provider_calls: int = 0
    stop_reason: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
