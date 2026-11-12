"""Observability and execution tracing for The Relay pipeline stages."""

import logging
import time
from typing import Optional
from app.core.models import DerivedTimings, StageTimestamps

logger = logging.getLogger(__name__)


class RequestTracer:
    """Manages high-resolution stage timestamps and computes derived observability metrics."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.timestamps = StageTimestamps(request_received_at=time.perf_counter())

    def mark_queue_start(self) -> None:
        self.timestamps.queue_admission_started_at = time.perf_counter()

    def mark_queue_complete(self) -> None:
        self.timestamps.queue_admission_completed_at = time.perf_counter()

    def mark_retrieval_start(self) -> None:
        self.timestamps.retrieval_started_at = time.perf_counter()

    def mark_retrieval_complete(self) -> None:
        self.timestamps.retrieval_completed_at = time.perf_counter()

    def mark_model_start(self) -> None:
        self.timestamps.model_execution_started_at = time.perf_counter()

    def mark_first_token(self) -> None:
        if self.timestamps.first_token_received_at is None:
            self.timestamps.first_token_received_at = time.perf_counter()

    def mark_model_complete(self) -> None:
        self.timestamps.model_generation_completed_at = time.perf_counter()

    def mark_validation_complete(self) -> None:
        self.timestamps.validation_completed_at = time.perf_counter()

    def mark_response_complete(self) -> None:
        self.timestamps.response_completed_at = time.perf_counter()

    def derive_timings(self) -> DerivedTimings:
        """Derive stage durations in milliseconds."""
        ts = self.timestamps

        queue_wait = (
            (ts.queue_admission_completed_at - ts.queue_admission_started_at) * 1000.0
            if ts.queue_admission_completed_at > ts.queue_admission_started_at
            else 0.0
        )
        retrieval_time = (
            (ts.retrieval_completed_at - ts.retrieval_started_at) * 1000.0
            if ts.retrieval_completed_at > ts.retrieval_started_at
            else 0.0
        )
        ttft = (
            (ts.first_token_received_at - ts.model_execution_started_at) * 1000.0
            if ts.first_token_received_at and ts.first_token_received_at > ts.model_execution_started_at
            else None
        )
        model_time = (
            (ts.model_generation_completed_at - ts.model_execution_started_at) * 1000.0
            if ts.model_generation_completed_at > ts.model_execution_started_at
            else 0.0
        )
        val_time = (
            (ts.validation_completed_at - ts.model_generation_completed_at) * 1000.0
            if ts.validation_completed_at > ts.model_generation_completed_at
            else 0.0
        )
        e2e = (
            (ts.response_completed_at - ts.request_received_at) * 1000.0
            if ts.response_completed_at > ts.request_received_at
            else 0.0
        )

        return DerivedTimings(
            queue_wait_ms=queue_wait,
            retrieval_ms=retrieval_time,
            time_to_first_token_ms=ttft,
            model_execution_ms=model_time,
            validation_ms=val_time,
            end_to_end_ms=e2e,
        )

