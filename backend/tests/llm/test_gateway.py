import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, SecretStr

from juris.config import CacheMode, ModelSpec, TokenPrice
from juris.llm import (
    ChatMessage,
    LLMGateway,
    ProviderError,
    RefusalError,
    StructuredOutputError,
    TransientProviderError,
)
from juris.llm.cache import ResponseCache
from juris.llm.providers import AnthropicProvider, FakeProvider
from juris.llm.types import ProviderRequest, ProviderResponse

MODEL = ModelSpec(provider="fake", name="fake-model")
PRICES = {"fake-model": TokenPrice(input=4.0, output=20.0)}
QUESTION = [ChatMessage(role="user", content="Is the contract void?")]
TAGS = {"agent": "judge", "run_id": "r1", "stage": "S9"}


class Verdict(BaseModel):
    leaning: str = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)


VALID = '{"leaning": "leaning_A", "confidence": 0.7}'
INVALID = '{"leaning": "A", "confidence": 3}'


def make_gateway(
    fake: FakeProvider,
    *,
    cache: ResponseCache | None = None,
    cache_mode: CacheMode = "read_write",
    sleeps: list[float] | None = None,
    max_retries: int = 3,
) -> LLMGateway:
    async def fake_sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return LLMGateway(
        {"fake": fake},
        prices=PRICES,
        cache=cache,
        cache_mode=cache_mode,
        max_retries=max_retries,
        sleep=fake_sleep,
    )


async def test_structured_output_parses() -> None:
    fake = FakeProvider({"judge": [VALID]})
    result = await make_gateway(fake).complete(
        QUESTION, model=MODEL, response_model=Verdict, tags=TAGS
    )

    assert result.parsed == Verdict(leaning="leaning_A", confidence=0.7)
    assert result.provider_calls == 1
    assert not result.cached
    schema = fake.calls[0].json_schema
    assert schema is not None
    assert schema["additionalProperties"] is False
    # Constraints the API rejects are stripped; Pydantic still enforces them.
    assert "minLength" not in schema["properties"]["leaning"]


async def test_invalid_output_gets_one_repair_retry() -> None:
    fake = FakeProvider({"judge": [INVALID, VALID]})
    result = await make_gateway(fake).complete(
        QUESTION, model=MODEL, response_model=Verdict, tags=TAGS
    )

    assert result.parsed is not None
    assert result.parsed.leaning == "leaning_A"
    assert result.provider_calls == 2
    repair = fake.calls[1]
    assert repair.messages[:-1] == tuple(QUESTION)
    assert repair.messages[-1].role == "user"
    assert INVALID in repair.messages[-1].content
    assert "confidence" in repair.messages[-1].content
    # Usage covers both calls.
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == len(INVALID) + len(VALID)


async def test_second_invalid_output_raises() -> None:
    fake = FakeProvider({"judge": [INVALID, INVALID]})
    with pytest.raises(StructuredOutputError) as info:
        await make_gateway(fake).complete(QUESTION, model=MODEL, response_model=Verdict, tags=TAGS)
    assert info.value.raw_text == INVALID
    assert len(fake.calls) == 2


async def test_cache_hit_makes_no_provider_call(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "llm.sqlite")
    fake = FakeProvider({"judge": [VALID, VALID]})
    gateway = make_gateway(fake, cache=cache)

    first = await gateway.complete(QUESTION, model=MODEL, response_model=Verdict, tags=TAGS)
    # Different tags, same request: still a hit.
    second = await gateway.complete(
        QUESTION, model=MODEL, response_model=Verdict, tags={"agent": "judge", "run_id": "r2"}
    )

    assert len(fake.calls) == 1
    assert second.cached and not first.cached
    assert second.parsed == first.parsed
    assert second.usage.cost_usd == 0.0
    assert gateway.ledger.total_tokens() == (10, len(VALID))

    # A different seed is a different request.
    await gateway.complete(QUESTION, model=MODEL, response_model=Verdict, seed=1, tags=TAGS)
    assert len(fake.calls) == 2


async def test_repaired_output_is_cached_under_the_original_request(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "llm.sqlite")
    fake = FakeProvider({"judge": [INVALID, VALID]})
    gateway = make_gateway(fake, cache=cache)

    await gateway.complete(QUESTION, model=MODEL, response_model=Verdict, tags=TAGS)
    again = await gateway.complete(QUESTION, model=MODEL, response_model=Verdict, tags=TAGS)

    assert again.cached and again.parsed is not None
    assert len(fake.calls) == 2


@pytest.mark.parametrize("mode", ["read_only", "off"])
async def test_cache_modes_that_do_not_write(tmp_path: Path, mode: CacheMode) -> None:
    cache = ResponseCache(tmp_path / "llm.sqlite")
    fake = FakeProvider({"judge": [VALID, VALID]})
    gateway = make_gateway(fake, cache=cache, cache_mode=mode)

    await gateway.complete(QUESTION, model=MODEL, tags=TAGS)
    await gateway.complete(QUESTION, model=MODEL, tags=TAGS)

    assert len(fake.calls) == 2


async def test_read_only_cache_still_serves_hits(tmp_path: Path) -> None:
    path = tmp_path / "llm.sqlite"
    await make_gateway(FakeProvider({"judge": [VALID]}), cache=ResponseCache(path)).complete(
        QUESTION, model=MODEL, tags=TAGS
    )
    fake = FakeProvider({"judge": []})
    result = await make_gateway(fake, cache=ResponseCache(path), cache_mode="read_only").complete(
        QUESTION, model=MODEL, tags=TAGS
    )
    assert result.cached and not fake.calls


async def test_cost_accounting() -> None:
    fake = FakeProvider(
        {
            "judge": [ProviderResponse(text=VALID, input_tokens=1000, output_tokens=500)],
            "juror": [ProviderResponse(text=VALID, input_tokens=2000, output_tokens=100)],
        }
    )
    gateway = make_gateway(fake)
    judge = await gateway.complete(QUESTION, model=MODEL, tags=TAGS)
    await gateway.complete(QUESTION, model=MODEL, seed=2, tags={"agent": "juror"})

    # 1000 * $4/M + 500 * $20/M
    assert judge.usage.cost_usd == pytest.approx(0.014)
    assert gateway.ledger.total_tokens() == (3000, 600)
    assert gateway.ledger.total_usd() == pytest.approx(0.014 + 0.010)
    assert [r.tags.get("stage") for r in gateway.ledger.records] == ["S9", None]


async def test_unpriced_model_has_unknown_cost() -> None:
    fake = FakeProvider({"judge": [VALID]})
    result = await make_gateway(fake).complete(
        QUESTION, model=ModelSpec(provider="fake", name="no-price"), tags=TAGS
    )
    assert result.usage.cost_usd is None


async def test_transient_errors_are_retried_with_backoff() -> None:
    sleeps: list[float] = []
    fake = FakeProvider(
        {
            "judge": [
                TransientProviderError("429", retry_after=7.0),
                TransientProviderError("503"),
                VALID,
            ]
        }
    )
    result = await make_gateway(fake, sleeps=sleeps).complete(QUESTION, model=MODEL, tags=TAGS)

    assert result.text == VALID
    assert len(fake.calls) == 3
    assert sleeps[0] == 7.0  # server-provided retry-after wins
    assert 0.5 <= sleeps[1] <= 2.0  # base 1s * 2**1, with jitter in [0.5, 1]


async def test_retries_give_up() -> None:
    fake = FakeProvider({"judge": [TransientProviderError("503")] * 3})
    with pytest.raises(TransientProviderError):
        await make_gateway(fake, max_retries=2, sleeps=[]).complete(
            QUESTION, model=MODEL, tags=TAGS
        )
    assert len(fake.calls) == 3


async def test_refusal_raises_and_is_not_cached(tmp_path: Path) -> None:
    refusal = ProviderResponse(text="", input_tokens=5, output_tokens=0, stop_reason="refusal")
    fake = FakeProvider({"judge": [refusal, VALID]})
    gateway = make_gateway(fake, cache=ResponseCache(tmp_path / "llm.sqlite"))

    with pytest.raises(RefusalError):
        await gateway.complete(QUESTION, model=MODEL, tags=TAGS)
    result = await gateway.complete(QUESTION, model=MODEL, tags=TAGS)
    assert not result.cached


async def test_concurrency_limit() -> None:
    active = 0
    peak = 0

    class SlowProvider:
        name = "fake"

        async def complete(self, request: ProviderRequest) -> ProviderResponse:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return ProviderResponse(text="ok", input_tokens=1, output_tokens=1)

    gateway = LLMGateway({"fake": SlowProvider()}, prices=PRICES, max_concurrency=2)
    await asyncio.gather(*(gateway.complete(QUESTION, model=MODEL, seed=i) for i in range(6)))
    assert peak == 2


async def test_unknown_provider() -> None:
    gateway = make_gateway(FakeProvider({}))
    with pytest.raises(ProviderError, match="unknown provider"):
        await gateway.complete(QUESTION, model=ModelSpec(provider="nope", name="x"))


async def test_anthropic_provider_without_key_fails_cleanly() -> None:
    provider = AnthropicProvider(api_key=None)
    request = ProviderRequest(model=ModelSpec(), messages=tuple(QUESTION))
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        await provider.complete(request)


def test_api_key_never_appears_in_repr() -> None:
    provider = AnthropicProvider(api_key=SecretStr("sk-ant-secret"))
    assert "sk-ant-secret" not in repr(vars(provider))
