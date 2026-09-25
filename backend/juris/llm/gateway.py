"""The single entry point for every model call: structured output, caching, cost, retries."""

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import overload

from pydantic import BaseModel, ValidationError

from juris.config import CacheMode, ModelSpec, Settings, TokenPrice
from juris.llm.cache import ResponseCache, cache_key
from juris.llm.cost import CallRecord, CostLedger, cost_usd
from juris.llm.errors import (
    ProviderError,
    RefusalError,
    StructuredOutputError,
    TransientProviderError,
)
from juris.llm.providers.anthropic import AnthropicProvider
from juris.llm.providers.base import Provider
from juris.llm.providers.openai_compat import OpenAICompatProvider
from juris.llm.schema import output_schema
from juris.llm.types import ChatMessage, LLMResult, ProviderRequest, ProviderResponse, Usage

REPAIR_PROMPT = (
    "Your previous answer did not match the required output schema.\n\n"
    "Previous answer:\n{previous}\n\n"
    "Validation errors:\n{errors}\n\n"
    "Reply again with only the corrected JSON."
)


class LLMGateway:
    def __init__(
        self,
        providers: Mapping[str, Provider],
        *,
        prices: Mapping[str, TokenPrice],
        cache: ResponseCache | None = None,
        cache_mode: CacheMode = "read_write",
        max_concurrency: int = 4,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        ledger: CostLedger | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._providers = dict(providers)
        self._prices = prices
        self._cache = cache if cache_mode != "off" else None
        self._cache_mode = cache_mode
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._sleep = sleep
        self.ledger = ledger or CostLedger()

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMGateway":
        providers: dict[str, Provider] = {
            "anthropic": AnthropicProvider(settings.anthropic_api_key),
            "openai": OpenAICompatProvider(settings.openai_api_key, settings.openai_base_url),
        }
        cache = None if settings.llm_cache == "off" else ResponseCache(settings.llm_cache_path)
        return cls(
            providers,
            prices=settings.llm_prices,
            cache=cache,
            cache_mode=settings.llm_cache,
            max_concurrency=settings.llm_max_concurrency,
            max_retries=settings.llm_max_retries,
        )

    @overload
    async def complete[T: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        *,
        model: ModelSpec,
        response_model: type[T],
        system: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int = 16000,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResult[T]: ...

    @overload
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: ModelSpec,
        response_model: None = None,
        system: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int = 16000,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResult[BaseModel]: ...

    async def complete[T: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        *,
        model: ModelSpec,
        response_model: type[T] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int = 16000,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResult[T] | LLMResult[BaseModel]:
        request = ProviderRequest(
            model=model,
            messages=tuple(messages),
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            json_schema=output_schema(response_model) if response_model else None,
            tags=dict(tags or {}),
        )
        key = cache_key(request)

        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                parsed = _try_parse(response_model, hit.text) if response_model else None
                if response_model is None or parsed is not None:
                    self._record(request, hit, cached=True)
                    return LLMResult[BaseModel](
                        text=hit.text,
                        parsed=parsed,
                        model=model,
                        usage=Usage(cost_usd=0.0),
                        cached=True,
                        stop_reason=hit.stop_reason,
                        tags=request.tags,
                    )

        response = await self._call(request)
        usage = self._record(request, response, cached=False)
        calls = 1
        parsed = None
        if response_model is not None:
            try:
                parsed = response_model.model_validate_json(response.text)
            except ValidationError as first_error:
                repair = request.model_copy(
                    update={
                        "messages": (
                            *request.messages,
                            ChatMessage(
                                role="user",
                                content=REPAIR_PROMPT.format(
                                    previous=response.text, errors=first_error
                                ),
                            ),
                        ),
                        "tags": {**request.tags, "repair": "1"},
                    }
                )
                response = await self._call(repair)
                usage = usage + self._record(repair, response, cached=False)
                calls = 2
                try:
                    parsed = response_model.model_validate_json(response.text)
                except ValidationError as second_error:
                    raise StructuredOutputError(
                        f"output still invalid after one repair: {second_error}", response.text
                    ) from second_error

        if self._cache is not None and self._cache_mode == "read_write":
            self._cache.put(key, response)
        return LLMResult[BaseModel](
            text=response.text,
            parsed=parsed,
            model=model,
            usage=usage,
            cached=False,
            provider_calls=calls,
            stop_reason=response.stop_reason,
            tags=request.tags,
        )

    async def _call(self, request: ProviderRequest) -> ProviderResponse:
        provider = self._providers.get(request.model.provider)
        if provider is None:
            raise ProviderError(f"unknown provider {request.model.provider!r}")
        for attempt in range(self._max_retries + 1):
            try:
                async with self._semaphore:
                    response = await provider.complete(request)
            except TransientProviderError as exc:
                if attempt == self._max_retries:
                    raise
                backoff = min(self._retry_base_delay * 2**attempt, self._retry_max_delay)
                await self._sleep(exc.retry_after or backoff * (0.5 + random.random() / 2))
                continue
            if response.stop_reason == "refusal":
                raise RefusalError(f"{request.model.name} declined the request")
            return response
        raise AssertionError("unreachable")

    def _record(
        self, request: ProviderRequest, response: ProviderResponse, *, cached: bool
    ) -> Usage:
        usd = (
            0.0
            if cached
            else cost_usd(
                request.model.name, response.input_tokens, response.output_tokens, self._prices
            )
        )
        self.ledger.add(
            CallRecord(
                provider=request.model.provider,
                model=request.model.name,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=usd,
                cached=cached,
                tags=request.tags,
            )
        )
        if cached:
            return Usage(cost_usd=0.0)
        return Usage(
            input_tokens=response.input_tokens, output_tokens=response.output_tokens, cost_usd=usd
        )


def _try_parse[T: BaseModel](model: type[T], text: str) -> T | None:
    try:
        return model.model_validate_json(text)
    except ValidationError:
        return None
