"""Tests for AdmissionController, CircuitBreaker, and RetryManager."""

import asyncio
import pytest
from app.core.models import ErrorCategory
from app.core.resource_manager import (
    AdmissionController,
    CircuitBreaker,
    CircuitState,
    LocalRateLimiter,
    RetryManager,
)


@pytest.mark.asyncio
async def test_circuit_breaker_transitions():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=0.1)
    assert cb.state == CircuitState.CLOSED
    assert await cb.can_execute() is True

    # 1. Non-infrastructure failures do not trip the breaker
    await cb.record_failure(ErrorCategory.RATE_LIMIT_ERROR)
    await cb.record_failure(ErrorCategory.AUTHENTICATION_ERROR)
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0

    # 2. Upstream provider errors increment failure count and trip
    await cb.record_failure(ErrorCategory.PROVIDER_ERROR)
    await cb.record_failure(ErrorCategory.TIMEOUT_ERROR)
    await cb.record_failure(ErrorCategory.PROVIDER_ERROR)
    assert cb.state == CircuitState.OPEN
    assert await cb.can_execute() is False

    # 3. Wait for recovery timeout -> transitions to HALF_OPEN
    await asyncio.sleep(0.12)
    assert await cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # 4. Successful execution closes circuit breaker
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


@pytest.mark.asyncio
async def test_admission_controller_queue_capacity():
    # Concurrency 1, max queue size 2
    admission = AdmissionController(
        max_concurrency=1,
        max_queue_size=2,
        queue_timeout_seconds=0.1,
    )

    # First request acquires slot immediately
    admitted1, _, _ = await admission.acquire_slot(estimated_tokens=100)
    assert admitted1 is True

    # Second request will wait in queue
    task2 = asyncio.create_task(admission.acquire_slot(estimated_tokens=100))
    await asyncio.sleep(0.01)

    # Third request will also wait in queue (queue full at 2)
    task3 = asyncio.create_task(admission.acquire_slot(estimated_tokens=100))
    await asyncio.sleep(0.01)

    # Fourth request exceeds max queue capacity -> rejected immediately
    admitted4, _, reason4 = await admission.acquire_slot(estimated_tokens=100)
    assert admitted4 is False
    assert "Admission queue is full" in reason4

    # Release slot 1, allowing task 2 to acquire
    admission.release_slot()
    admitted2, _, _ = await task2
    assert admitted2 is True
    admission.release_slot()

    admitted3, _, _ = await task3
    admission.release_slot()


def test_retry_manager_eligibility():
    rm = RetryManager(max_retries=2, base_backoff=1.0)

    # Non-retryable
    assert rm.is_retryable(ErrorCategory.AUTHENTICATION_ERROR) is False
    assert rm.is_retryable(ErrorCategory.INVALID_REQUEST) is False
    assert rm.is_retryable(ErrorCategory.QUOTA_EXHAUSTED) is False

    # Retryable
    assert rm.is_retryable(ErrorCategory.PROVIDER_ERROR) is True
    assert rm.is_retryable(ErrorCategory.TIMEOUT_ERROR) is True
    assert rm.is_retryable(ErrorCategory.RATE_LIMIT_ERROR) is True

    # Backoff with retry-after header
    assert rm.compute_backoff(0, retry_after=5.0) == 5.0

    # Exponential jittered backoff bounds
    bo0 = rm.compute_backoff(0)
    assert 0.5 <= bo0 <= 1.0
    bo1 = rm.compute_backoff(1)
    assert 1.0 <= bo1 <= 2.0

