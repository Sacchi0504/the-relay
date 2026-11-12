"""Groq Hosted Model Provider Implementation."""

import logging
import time
from typing import AsyncGenerator, Optional
import groq
from groq import AsyncGroq, APIError, RateLimitError, APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError

from app.core.config import settings
from app.core.models import (
    ErrorCategory,
    ExecutionStatus,
    ModelRequest,
    ModelResponse,
    ProviderSource,
)
from app.providers.base import BaseModelProvider

logger = logging.getLogger(__name__)


def classify_groq_error(e: Exception) -> tuple[ErrorCategory, ExecutionStatus]:
    """Map Groq API exceptions to normalized error categories and statuses."""
    if isinstance(e, AuthenticationError):
        return ErrorCategory.AUTHENTICATION_ERROR, ExecutionStatus.ERROR
    elif isinstance(e, RateLimitError):
        err_msg = str(e).lower()
        if "quota" in err_msg or "daily" in err_msg or "exceeded your current quota" in err_msg:
            return ErrorCategory.QUOTA_EXHAUSTED, ExecutionStatus.RATE_LIMITED
        return ErrorCategory.RATE_LIMIT_ERROR, ExecutionStatus.RATE_LIMITED
    elif isinstance(e, APITimeoutError):
        return ErrorCategory.TIMEOUT_ERROR, ExecutionStatus.TIMEOUT
    elif isinstance(e, APIConnectionError):
        return ErrorCategory.CONNECTION_ERROR, ExecutionStatus.ERROR
    elif isinstance(e, BadRequestError):
        return ErrorCategory.INVALID_REQUEST, ExecutionStatus.ERROR
    elif isinstance(e, APIError):
        return ErrorCategory.PROVIDER_ERROR, ExecutionStatus.ERROR
    return ErrorCategory.PROVIDER_ERROR, ExecutionStatus.ERROR


class GroqProvider(BaseModelProvider):
    """Real hosted inference using Groq API."""

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.default_model = default_model or settings.GROQ_MODEL
        self._client: Optional[AsyncGroq] = None
        if self.api_key:
            self._client = AsyncGroq(api_key=self.api_key, timeout=settings.REQUEST_TIMEOUT_SECONDS)

    @property
    def name(self) -> str:
        return "groq"

    @property
    def source(self) -> ProviderSource:
        return ProviderSource.REAL

    def get_client(self) -> AsyncGroq:
        if not self._client:
            if not self.api_key:
                raise AuthenticationError(
                    "GROQ_API_KEY is not set. Please provide a valid Groq API key in .env or environment.",
                    request=None,
                    body=None,
                )
            self._client = AsyncGroq(api_key=self.api_key, timeout=settings.REQUEST_TIMEOUT_SECONDS)
        return self._client

    async def generate(self, request: ModelRequest) -> ModelResponse:
        model_name = request.model or self.default_model
        t0 = time.perf_counter()
        first_token_ms: Optional[float] = None
        content_parts = []
        tokens_in = 0
        tokens_out = 0

        try:
            client = self.get_client()
            # We use streaming under the hood to capture first_token_ms accurately
            stream = await client.chat.completions.create(
                model=model_name,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - t0) * 1000.0
                        content_parts.append(delta.content)

                if hasattr(chunk, "usage") and chunk.usage:
                    tokens_in = chunk.usage.prompt_tokens or tokens_in
                    tokens_out = chunk.usage.completion_tokens or tokens_out

            total_ms = (time.perf_counter() - t0) * 1000.0
            full_content = "".join(content_parts)

            # Fallback estimation if usage wasn't emitted by stream
            if tokens_in == 0:
                prompt_text = " ".join(m.get("content", "") for m in request.messages)
                tokens_in = max(1, len(prompt_text.split()) * 4 // 3)
            if tokens_out == 0:
                tokens_out = max(1, len(full_content.split()) * 4 // 3)

            return ModelResponse(
                request_id=request.request_id,
                content=full_content,
                completion_status=ExecutionStatus.OK,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                first_token_ms=first_token_ms,
                total_ms=total_ms,
                provider=self.source,
                model_name=model_name,
                stream_chunks=content_parts,
            )

        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            err_category, exec_status = classify_groq_error(e)
            logger.warning("Groq request %s failed (%s): %s", request.request_id, err_category, e)
            return ModelResponse(
                request_id=request.request_id,
                content="",
                completion_status=exec_status,
                tokens_in=tokens_in,
                tokens_out=0,
                first_token_ms=None,
                total_ms=total_ms,
                raw_error=str(e),
                error_category=err_category,
                provider=self.source,
                model_name=model_name,
            )

    async def generate_stream(
        self, request: ModelRequest
    ) -> AsyncGenerator[str, None]:
        client = self.get_client()
        model_name = request.model or self.default_model
        stream = await client.chat.completions.create(
            model=model_name,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
