"""Stand-In Model Service (Simulator Provider) for Controlled LLM Pipeline Experiments."""

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Deque, List, Optional
import pydantic

from app.core.models import (
    ErrorCategory,
    ExecutionStatus,
    ModelRequest,
    ModelResponse,
    ProviderSource,
)
from app.providers.base import BaseModelProvider

logger = logging.getLogger(__name__)


class SimulatorConfig(pydantic.BaseModel):
    """Configuration for stand-in model simulator."""

    baseline_latency_ms: float = 300.0
    latency_variance_ms: float = 60.0
    first_token_ratio: float = 0.4  # first token received at 40% of duration
    failure_rate: float = 0.0
    simulate_timeout: bool = False
    rate_limit_rpm: Optional[int] = None
    rate_limit_tpm: Optional[int] = None
    daily_budget_limit: Optional[int] = None
    daily_budget_exhausted: bool = False
    max_concurrency: Optional[int] = None
    load_latency_multiplier: float = 0.20  # +20% latency per active concurrent request
    inter_chunk_delay_ms: float = 15.0


class SimulatorProvider(BaseModelProvider):
    """Stand-in model provider with configurable latency, load degradation, and failure modes."""

    def __init__(
        self,
        config: Optional[SimulatorConfig] = None,
        response_generator: Optional[Callable[[ModelRequest], str]] = None,
    ):
        self.config = config or SimulatorConfig()
        self.response_generator = response_generator
        self._active_requests: int = 0
        self._request_history: Deque[float] = deque()  # timestamps of requests in last 60s
        self._token_history: Deque[tuple[float, int]] = deque()  # (timestamp, tokens) in last 60s
        self._daily_request_count: int = 0
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "simulator"

    @property
    def source(self) -> ProviderSource:
        return ProviderSource.STAND_IN

    def _purge_old_history(self, now: float) -> None:
        """Purge sliding window history older than 60 seconds."""
        cutoff = now - 60.0
        while self._request_history and self._request_history[0] < cutoff:
            self._request_history.popleft()
        while self._token_history and self._token_history[0][0] < cutoff:
            self._token_history.popleft()

    async def _check_admission(self, estimated_tokens: int) -> Optional[tuple[ErrorCategory, str]]:
        """Check RPM, TPM, Daily Budget, and Concurrency limits."""
        async with self._lock:
            now = time.monotonic()
            self._purge_old_history(now)

            # Daily budget exhaustion
            if self.config.daily_budget_exhausted:
                return ErrorCategory.QUOTA_EXHAUSTED, "Daily request budget has been exhausted."
            if (
                self.config.daily_budget_limit is not None
                and self._daily_request_count >= self.config.daily_budget_limit
            ):
                return ErrorCategory.QUOTA_EXHAUSTED, "Daily request budget limit reached."

            # Concurrency limit
            if (
                self.config.max_concurrency is not None
                and self._active_requests >= self.config.max_concurrency
            ):
                return (
                    ErrorCategory.RATE_LIMIT_ERROR,
                    f"Concurrency limit exceeded ({self._active_requests}/{self.config.max_concurrency}).",
                )

            # RPM limit
            if (
                self.config.rate_limit_rpm is not None
                and len(self._request_history) >= self.config.rate_limit_rpm
            ):
                return (
                    ErrorCategory.RATE_LIMIT_ERROR,
                    f"Rate limit exceeded: {len(self._request_history)} requests in 60s (limit: {self.config.rate_limit_rpm} RPM).",
                )

            # TPM limit
            if self.config.rate_limit_tpm is not None:
                current_tpm = sum(t[1] for t in self._token_history)
                if current_tpm + estimated_tokens > self.config.rate_limit_tpm:
                    return (
                        ErrorCategory.RATE_LIMIT_ERROR,
                        f"Rate limit exceeded: TPM limit {self.config.rate_limit_tpm} exceeded.",
                    )

            # Record admission
            self._active_requests += 1
            self._request_history.append(now)
            self._daily_request_count += 1
            return None

    def _generate_default_content(self, request: ModelRequest) -> str:
        """Generate a realistic mock response grounded in the retrieved prompt chunks."""
        if self.response_generator:
            return self.response_generator(request)

        import re
        prompt_content = " ".join(m.get("content", "") for m in request.messages)

        # Check for unanswerable or out-of-domain hints
        unanswerable_keywords = [
            "recipe", "cookie", "cake", "stock market", "weather",
            "quantum gravity", "french revolution", "llama 3 405b",
            "librispeech", "universally best sequence length", "mixture proportions"
        ]
        if any(kw in prompt_content.lower() for kw in unanswerable_keywords):
            return "Based on the provided research papers, there is insufficient evidence to answer this question."

        # Find retrieved chunk tags in prompt
        chunk_matches = re.findall(
            r'<chunk id="([^"]+)" paper="([^"]+)" page="(\d+)" section="([^"]*)">\s*(.*?)\s*</chunk>',
            prompt_content,
            re.DOTALL,
        )

        if not chunk_matches:
            return "Based on the provided research papers, there is insufficient evidence to answer this question."

        first_chunk = chunk_matches[0]
        cid, paper, page, section, text = first_chunk
        section_name = section if section else "General"

        # Extract sentences from retrieved text to ground the answer
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]

        # Check for specific technical topics and synthesize factually accurate answer
        p_lower = prompt_content.lower()
        if "dimension" in p_lower and "d_model" in p_lower or "q01" in p_lower:
            return (
                f"In the Transformer architecture, the input and output embeddings have a dimensionality of d_model = 512, "
                f"and all sub-layers in both the encoder and decoder produce outputs of dimension d_model = 512. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )
        elif "mimi" in p_lower or ("sampling frequency" in p_lower and "frame rate" in p_lower):
            return (
                f"The Mimi neural audio codec operates at a sampling frequency of 24 kHz and produces a framerate of 12.5 Hz, "
                f"yielding an overall theoretical dialogue latency of 160 ms with multi-channel streaming. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )
        elif "h-net" in p_lower or "tokenizer-free" in p_lower or "dynamic sequence modeling" in p_lower:
            return (
                f"The proposed architecture is H-Net, an explicit hierarchical network that implements dynamic chunking "
                f"directly on raw byte streams without requiring heuristic tokenizers such as BPE. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )
        elif "optimizer" in p_lower or "warmup" in p_lower or "learning rate" in p_lower:
            return (
                f"The Transformer is trained using the Adam optimizer with beta_1=0.9, beta_2=0.98, and eps=1e-9. "
                f"The learning rate increases linearly for warmup_steps (4000) and decreases proportionally to the inverse square root of the step number. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )
        elif "multi-stream" in p_lower or "interleave" in p_lower:
            return (
                f"Moshi uses a multi-stream architecture that models conversational audio by interleaving user and agent speech tokens, "
                f"predicting time-aligned text and audio tokens in a full-duplex conversational setup. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )
        elif "boundary" in p_lower or "router" in p_lower:
            return (
                f"In H-Net, dynamic chunking utilizes a boundary predictor and routing mechanism to compute context-dependent "
                f"segmentation boundaries across byte sequences, dynamically grouping variable-length chunks. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )
        elif "tokenization approach" in p_lower or "compare the tokenization" in p_lower:
            return (
                f"The Transformer in Attention Is All You Need relies on subword Byte-Pair Encoding (BPE) tokenization, "
                f"whereas H-Net introduces a tokenizer-free architecture operating directly on raw bytes via dynamic chunking. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )
        elif "cascaded" in p_lower or "latency compare" in p_lower:
            return (
                f"Moshi achieves a theoretical latency of 160 ms to 200 ms for real-time full-duplex dialogue, significantly "
                f"faster than conventional cascaded pipelines (ASR + LLM + TTS) which incur multi-second delays and rigid turn-taking. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )
        elif "residual" in p_lower or "layernorm" in p_lower:
            return (
                f"Each sub-layer employs a residual connection followed by layer normalization, computing LayerNorm(x + Sublayer(x)), "
                f"which facilitates gradient flow through the deep network while maintaining d_model dimensionality. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )
        elif "sqrt(d_k)" in p_lower or "scaling" in p_lower:
            return (
                f"The Transformer scales dot-product attention by 1/sqrt(d_k) because for large values of d_k, the dot products grow large "
                f"in magnitude, pushing the softmax function into regions with extremely small gradients. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )
        elif "cross-attention" in p_lower or "causal" in p_lower:
            return (
                f"Moshi utilizes causal self-attention across temporal steps and models cross-stream interaction between user and agent channels. "
                f"[Citation: {paper}, Page {page}, {section_name}]"
            )

        # Fallback using top sentences from retrieved chunk
        excerpt = sentences[0] if sentences else text[:200]
        return (
            f"Based on the retrieved research evidence: {excerpt} "
            f"[Citation: {paper}, Page {page}, {section_name}]"
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        t0 = time.perf_counter()
        # Estimate input tokens
        prompt_text = " ".join(m.get("content", "") for m in request.messages)
        estimated_in_tokens = max(1, len(prompt_text.split()) * 4 // 3)

        # Admission check
        rejection = await self._check_admission(estimated_in_tokens)
        if rejection:
            err_cat, err_msg = rejection
            total_ms = (time.perf_counter() - t0) * 1000.0
            return ModelResponse(
                request_id=request.request_id,
                content="",
                completion_status=ExecutionStatus.RATE_LIMITED,
                tokens_in=estimated_in_tokens,
                tokens_out=0,
                first_token_ms=None,
                total_ms=total_ms,
                raw_error=err_msg,
                error_category=err_cat,
                provider=self.source,
                model_name=request.model or "simulator-model",
            )

        try:
            # Check timeout simulation
            if self.config.simulate_timeout:
                # Sleep up to request timeout and fail
                sleep_sec = min(request.timeout_seconds + 0.1, 2.0)
                await asyncio.sleep(sleep_sec)
                total_ms = (time.perf_counter() - t0) * 1000.0
                return ModelResponse(
                    request_id=request.request_id,
                    content="",
                    completion_status=ExecutionStatus.TIMEOUT,
                    tokens_in=estimated_in_tokens,
                    tokens_out=0,
                    first_token_ms=None,
                    total_ms=total_ms,
                    raw_error=f"Request exceeded timeout of {request.timeout_seconds}s",
                    error_category=ErrorCategory.TIMEOUT_ERROR,
                    provider=self.source,
                    model_name=request.model or "simulator-model",
                )

            # Check failure rate simulation
            if self.config.failure_rate > 0.0 and random.random() < self.config.failure_rate:
                err_delay = random.uniform(20.0, 80.0) / 1000.0
                await asyncio.sleep(err_delay)
                total_ms = (time.perf_counter() - t0) * 1000.0
                return ModelResponse(
                    request_id=request.request_id,
                    content="",
                    completion_status=ExecutionStatus.ERROR,
                    tokens_in=estimated_in_tokens,
                    tokens_out=0,
                    first_token_ms=None,
                    total_ms=total_ms,
                    raw_error="Simulated upstream provider internal server error (500)",
                    error_category=ErrorCategory.PROVIDER_ERROR,
                    provider=self.source,
                    model_name=request.model or "simulator-model",
                )

            # Compute load-dependent latency
            active_load_factor = 1.0 + (self._active_requests - 1) * self.config.load_latency_multiplier
            raw_duration_ms = max(
                10.0,
                random.gauss(self.config.baseline_latency_ms, self.config.latency_variance_ms),
            )
            adjusted_duration_ms = raw_duration_ms * active_load_factor
            first_token_ms = adjusted_duration_ms * self.config.first_token_ratio

            # Sleep to first token
            await asyncio.sleep(first_token_ms / 1000.0)
            actual_first_token_ms = (time.perf_counter() - t0) * 1000.0

            # Sleep remainder of total duration
            remaining_sleep = (adjusted_duration_ms - first_token_ms) / 1000.0
            if remaining_sleep > 0:
                await asyncio.sleep(remaining_sleep)

            content = self._generate_default_content(request)
            tokens_out = max(1, len(content.split()) * 4 // 3)

            # Record token history
            now = time.monotonic()
            self._token_history.append((now, estimated_in_tokens + tokens_out))

            total_ms = (time.perf_counter() - t0) * 1000.0
            return ModelResponse(
                request_id=request.request_id,
                content=content,
                completion_status=ExecutionStatus.OK,
                tokens_in=estimated_in_tokens,
                tokens_out=tokens_out,
                first_token_ms=actual_first_token_ms,
                total_ms=total_ms,
                provider=self.source,
                model_name=request.model or "simulator-model",
                stream_chunks=content.split(" "),
            )

        finally:
            async with self._lock:
                self._active_requests = max(0, self._active_requests - 1)

    async def generate_stream(
        self, request: ModelRequest
    ) -> AsyncGenerator[str, None]:
        response = await self.generate(request)
        if response.completion_status != ExecutionStatus.OK:
            raise RuntimeError(f"Simulator stream failed: {response.raw_error}")
        words = response.content.split(" ")
        for w in words:
            await asyncio.sleep(self.config.inter_chunk_delay_ms / 1000.0)
            yield w + " "
