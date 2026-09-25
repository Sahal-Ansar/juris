"""Gateway errors. Providers translate SDK exceptions into these."""


class LLMError(Exception):
    """Base class for gateway errors."""


class ProviderError(LLMError):
    """A non-retryable provider failure (bad request, auth, missing key)."""


class TransientProviderError(LLMError):
    """A retryable failure: rate limit, 5xx, timeout, connection error."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RefusalError(LLMError):
    """The model declined the request (``stop_reason == "refusal"``)."""


class StructuredOutputError(LLMError):
    """The output failed validation even after the one repair retry."""

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text
