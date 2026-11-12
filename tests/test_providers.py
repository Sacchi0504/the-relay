"""Tests for model provider abstractions (Groq error mapping and Simulator behaviours)."""

import pytest
from groq import AuthenticationError, RateLimitError, APITimeoutError, BadRequestError
from app.core.models import (
    ErrorCategory,
    ExecutionStatus,
    ModelRequest,
    ProviderSource,
)
from app.providers.groq_provider import classify_groq_error, GroqProvider
from app.providers.simulator_provider import SimulatorConfig, SimulatorProvider


def test_groq_error_classification():
    import httpx
    dummy_resp = httpx.Response(401, request=httpx.Request("POST", "https://api.groq.com"))
    dummy_rl_resp = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))

    auth_err = AuthenticationError("Invalid API key", response=dummy_resp, body=None)
    cat, status = classify_groq_error(auth_err)
    assert cat == ErrorCategory.AUTHENTICATION_ERROR
    assert status == ExecutionStatus.ERROR

    rl_err = RateLimitError("Rate limit exceeded 30 RPM", response=dummy_rl_resp, body=None)
    cat, status = classify_groq_error(rl_err)
    assert cat == ErrorCategory.RATE_LIMIT_ERROR
    assert status == ExecutionStatus.RATE_LIMITED

    quota_err = RateLimitError("Exceeded your current quota", response=dummy_rl_resp, body=None)
    cat, status = classify_groq_error(quota_err)
    assert cat == ErrorCategory.QUOTA_EXHAUSTED
    assert status == ExecutionStatus.RATE_LIMITED

    timeout_err = APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))
    cat, status = classify_groq_error(timeout_err)
    assert cat == ErrorCategory.TIMEOUT_ERROR
    assert status == ExecutionStatus.TIMEOUT


@pytest.mark.asyncio
async def test_simulator_baseline_latency():
    config = SimulatorConfig(baseline_latency_ms=80.0, latency_variance_ms=5.0)
    sim = SimulatorProvider(config=config)
    req = ModelRequest(request_id="test_sim_lat", messages=[{"role": "user", "content": "test"}], model="sim")
    resp = await sim.generate(req)
    assert resp.completion_status == ExecutionStatus.OK
    assert resp.total_ms >= 60.0
    assert resp.first_token_ms is not None
    assert resp.first_token_ms < resp.total_ms
    assert resp.provider == ProviderSource.STAND_IN


@pytest.mark.asyncio
async def test_simulator_rate_limiting():
    # Set limit to 2 RPM
    config = SimulatorConfig(baseline_latency_ms=10.0, rate_limit_rpm=2)
    sim = SimulatorProvider(config=config)
    req1 = ModelRequest(request_id="r1", messages=[{"role": "user", "content": "hi"}], model="sim")
    req2 = ModelRequest(request_id="r2", messages=[{"role": "user", "content": "hi"}], model="sim")
    req3 = ModelRequest(request_id="r3", messages=[{"role": "user", "content": "hi"}], model="sim")

    res1 = await sim.generate(req1)
    res2 = await sim.generate(req2)
    res3 = await sim.generate(req3)

    assert res1.completion_status == ExecutionStatus.OK
    assert res2.completion_status == ExecutionStatus.OK
    assert res3.completion_status == ExecutionStatus.RATE_LIMITED
    assert res3.error_category == ErrorCategory.RATE_LIMIT_ERROR


@pytest.mark.asyncio
async def test_simulator_daily_budget_exhausted():
    config = SimulatorConfig(daily_budget_exhausted=True)
    sim = SimulatorProvider(config=config)
    req = ModelRequest(request_id="r_budget", messages=[{"role": "user", "content": "hi"}], model="sim")
    resp = await sim.generate(req)
    assert resp.completion_status == ExecutionStatus.RATE_LIMITED
    assert resp.error_category == ErrorCategory.QUOTA_EXHAUSTED


@pytest.mark.asyncio
async def test_simulator_simulated_failure():
    config = SimulatorConfig(failure_rate=1.0)
    sim = SimulatorProvider(config=config)
    req = ModelRequest(request_id="r_fail", messages=[{"role": "user", "content": "hi"}], model="sim")
    resp = await sim.generate(req)
    assert resp.completion_status == ExecutionStatus.ERROR
    assert resp.error_category == ErrorCategory.PROVIDER_ERROR


@pytest.mark.asyncio
async def test_simulator_simulated_timeout():
    config = SimulatorConfig(simulate_timeout=True)
    sim = SimulatorProvider(config=config)
    req = ModelRequest(
        request_id="r_to",
        messages=[{"role": "user", "content": "hi"}],
        model="sim",
        timeout_seconds=0.1,
    )
    resp = await sim.generate(req)
    assert resp.completion_status == ExecutionStatus.TIMEOUT
    assert resp.error_category == ErrorCategory.TIMEOUT_ERROR


@pytest.mark.asyncio
async def test_simulator_streaming():
    config = SimulatorConfig(baseline_latency_ms=20.0, inter_chunk_delay_ms=5.0)
    sim = SimulatorProvider(config=config)
    req = ModelRequest(
        request_id="r_stream",
        messages=[{"role": "user", "content": "Attention Is All You Need"}],
        model="sim",
    )
    chunks = []
    async for chunk in sim.generate_stream(req):
        chunks.append(chunk)
    assert len(chunks) > 1
    full_text = "".join(chunks)
    assert len(full_text) > 20
