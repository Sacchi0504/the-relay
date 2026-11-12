"""Resource management: Admission control, Rate limiting, Bounded Queue, Retries, and Circuit Breaker."""

import asyncio
import logging
import random
import time
from collections import deque
from enum import Enum
from typing import Callable, Coroutine, Deque, Dict, Optional, Tuple, TypeVar

from app.core.config import settings
from app.core.models import ErrorCategory, ExecutionStatus, ModelRequest, ModelResponse

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Protects downstream provider from repeated server failures (500s / timeouts)."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 15.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_state_change = time.monotonic()
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            if self.state == CircuitState.OPEN:
                if now - self.last_state_change >= self.recovery_timeout_seconds:
                    logger.info("Circuit breaker transitioning from OPEN to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = now
                    return True
                return False
            return True

    async def record_success(self) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker probe succeeded. Transitioning to CLOSED.")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_state_change = time.monotonic()

    async def record_failure(self, error_category: Optional[ErrorCategory]) -> None:
        # Rate-limits and auth errors do not count as provider infrastructure outages
        if error_category in (
            ErrorCategory.RATE_LIMIT_ERROR,
            ErrorCategory.QUOTA_EXHAUSTED,
            ErrorCategory.AUTHENTICATION_ERROR,
            ErrorCategory.INVALID_REQUEST,
        ):
            return

        async with self._lock:
            self.failure_count += 1
            now = time.monotonic()
            if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                logger.warning(
                    "Circuit breaker tripped to OPEN! (failures: %d, state: %s)",
                    self.failure_count,
                    self.state,
                )
                self.state = CircuitState.OPEN
                self.last_state_change = now


class LocalRateLimiter:
    """Sliding-window local rate limiter for RPM, TPM, and daily budget."""

    def __init__(
        self,
        rpm_limit: int = 30,
        tpm_limit: int = 6000,
        rpd_limit: int = 1000,
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self._request_timestamps: Deque[float] = deque()
        self._token_log: Deque[Tuple[float, int]] = deque()
        self._daily_requests = 0
        self._lock = asyncio.Lock()

    def _purge_old(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()
        while self._token_log and self._token_log[0][0] < cutoff:
            self._token_log.popleft()

    async def check_and_reserve(self, estimated_tokens: int) -> Tuple[bool, Optional[str]]:
        async with self._lock:
            now = time.monotonic()
            self._purge_old(now)

            if self._daily_requests >= self.rpd_limit:
                return False, f"Daily limit reached ({self._daily_requests}/{self.rpd_limit} RPD)"

            if len(self._request_timestamps) >= self.rpm_limit:
                return False, f"Local RPM limit exceeded ({len(self._request_timestamps)}/{self.rpm_limit} RPM)"

            current_tpm = sum(tok for _, tok in self._token_log)
            if current_tpm + estimated_tokens > self.tpm_limit:
                return False, f"Local TPM limit exceeded ({current_tpm + estimated_tokens}/{self.tpm_limit} TPM)"

            # Reserve
            self._request_timestamps.append(now)
            self._token_log.append((now, estimated_tokens))
            self._daily_requests += 1
            return True, None

    async def update_actual_tokens(self, estimated: int, actual: int) -> None:
        async with self._lock:
            diff = actual - estimated
            if diff != 0:
                now = time.monotonic()
                self._token_log.append((now, diff))


class AdmissionController:
    """Controls bounded queueing, bounded concurrency, and wait-time limits."""

    def __init__(
        self,
        max_concurrency: int = settings.MAX_CONCURRENCY,
        max_queue_size: int = settings.MAX_QUEUE_SIZE,
        queue_timeout_seconds: float = settings.QUEUE_TIMEOUT_SECONDS,
        rate_limiter: Optional[LocalRateLimiter] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.max_concurrency = max_concurrency
        self.max_queue_size = max_queue_size
        self.queue_timeout_seconds = queue_timeout_seconds
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rate_limiter = rate_limiter or LocalRateLimiter()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._current_queue_depth = 0
        self._peak_queue_depth = 0
        self._lock = asyncio.Lock()

    @property
    def peak_queue_length(self) -> int:
        return self._peak_queue_depth

    async def acquire_slot(
        self, estimated_tokens: int, deadline_seconds: Optional[float] = None
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Attempt admission. Returns (admitted: bool, queue_wait_ms: float, error_reason: Optional[str]).
        """
        # Check circuit breaker
        if not await self.circuit_breaker.can_execute():
            return False, 0.0, "Circuit breaker is OPEN. Upstream provider unavailable."

        # Check queue capacity
        async with self._lock:
            if self._current_queue_depth >= self.max_queue_size:
                return False, 0.0, f"Admission queue is full ({self._current_queue_depth}/{self.max_queue_size})"
            self._current_queue_depth += 1
            if self._current_queue_depth > self._peak_queue_depth:
                self._peak_queue_depth = self._current_queue_depth

        t_start_wait = time.perf_counter()
        wait_limit = min(
            self.queue_timeout_seconds,
            deadline_seconds if deadline_seconds is not None else self.queue_timeout_seconds,
        )

        try:
            # Wait for concurrency slot
            await asyncio.wait_for(self.semaphore.acquire(), timeout=wait_limit)
        except asyncio.TimeoutError:
            async with self._lock:
                self._current_queue_depth = max(0, self._current_queue_depth - 1)
            wait_ms = (time.perf_counter() - t_start_wait) * 1000.0
            return False, wait_ms, f"Queue wait limit of {wait_limit:.1f}s exceeded before slot acquired."

        queue_wait_ms = (time.perf_counter() - t_start_wait) * 1000.0

        # Now check rate limiter
        admitted, reason = await self.rate_limiter.check_and_reserve(estimated_tokens)
        if not admitted:
            self.semaphore.release()
            async with self._lock:
                self._current_queue_depth = max(0, self._current_queue_depth - 1)
            return False, queue_wait_ms, f"Rate limiter rejected: {reason}"

        return True, queue_wait_ms, None

    def release_slot(self) -> None:
        self.semaphore.release()
        self._current_queue_depth = max(0, self._current_queue_depth - 1)


class RetryManager:
    """Handles bounded retries with exponential backoff, jitter, and Retry-After support."""

    NON_RETRYABLE_ERRORS = {
        ErrorCategory.AUTHENTICATION_ERROR,
        ErrorCategory.INVALID_REQUEST,
        ErrorCategory.QUOTA_EXHAUSTED,
    }

    def __init__(
        self,
        max_retries: int = settings.MAX_RETRIES,
        base_backoff: float = settings.BASE_BACKOFF_SECONDS,
        max_backoff: float = settings.MAX_BACKOFF_SECONDS,
    ):
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff

    def compute_backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        if retry_after is not None and retry_after > 0:
            return min(self.max_backoff, retry_after)
        # Exponential backoff + full jitter
        exponential = self.base_backoff * (2 ** attempt)
        jittered = random.uniform(0.5, 1.0) * exponential
        return min(self.max_backoff, jittered)

    def is_retryable(self, error_category: Optional[ErrorCategory]) -> bool:
        if error_category is None:
            return False
        return error_category not in self.NON_RETRYABLE_ERRORS

