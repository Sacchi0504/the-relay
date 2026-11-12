"""Definition of mandatory load test scenarios and retrieval experiment matrices."""

import asyncio
import logging
from typing import List, Tuple

from app.core.config import settings
from app.core.models import RetrievalConfig
from app.providers.simulator_provider import SimulatorConfig

logger = logging.getLogger(__name__)

# Three required retrieval configurations
RETRIEVAL_CONFIGS = [
    RetrievalConfig(
        name="small_k3_norerank",
        chunk_size_words=250,
        chunk_overlap_words=50,
        top_k=3,
        use_reranking=False,
    ),
    RetrievalConfig(
        name="medium_k5_norerank",  # Baseline
        chunk_size_words=500,
        chunk_overlap_words=100,
        top_k=5,
        use_reranking=False,
    ),
    RetrievalConfig(
        name="large_k8_rerank",
        chunk_size_words=900,
        chunk_overlap_words=150,
        top_k=8,
        use_reranking=True,
    ),
]


def get_scenario_simulator_config(scenario_name: str) -> SimulatorConfig:
    """Returns tailored simulator configuration for each mandatory scenario."""
    if scenario_name == "steady_load":
        return SimulatorConfig(
            baseline_latency_ms=250.0,
            latency_variance_ms=40.0,
            first_token_ratio=0.35,
            rate_limit_rpm=30,
            rate_limit_tpm=6000,
            max_concurrency=4,
            load_latency_multiplier=0.10,
        )
    elif scenario_name == "sudden_burst":
        return SimulatorConfig(
            baseline_latency_ms=260.0,
            latency_variance_ms=50.0,
            first_token_ratio=0.35,
            rate_limit_rpm=30,
            rate_limit_tpm=6000,
            max_concurrency=4,
            load_latency_multiplier=0.25,  # Latency rises sharply with burst queue depth
        )
    elif scenario_name == "sustained_overload":
        return SimulatorConfig(
            baseline_latency_ms=280.0,
            latency_variance_ms=60.0,
            first_token_ratio=0.40,
            rate_limit_rpm=20,  # Tight RPM to trigger rate limits
            rate_limit_tpm=4000,
            max_concurrency=3,
            load_latency_multiplier=0.35,
        )
    elif scenario_name == "limit_halved":
        return SimulatorConfig(
            baseline_latency_ms=250.0,
            latency_variance_ms=40.0,
            first_token_ratio=0.35,
            rate_limit_rpm=15,  # Halved from 30 RPM
            rate_limit_tpm=3000,  # Halved from 6000 TPM
            max_concurrency=2,
            load_latency_multiplier=0.20,
        )
    elif scenario_name == "daily_budget_exhausted":
        return SimulatorConfig(
            daily_budget_exhausted=True,  # Immediate quota rejection
        )
    else:
        return SimulatorConfig()

