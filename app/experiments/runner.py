"""Experiment execution engine for retrieval benchmarks and load scenarios."""

import asyncio
import csv
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.models import (
    AnswerGrade,
    ExecutionStatus,
    PipelineResult,
    ProviderSource,
    RetrievalConfig,
)
from app.core.resource_manager import AdmissionController, LocalRateLimiter, RetryManager
from app.experiments.scenarios import RETRIEVAL_CONFIGS, get_scenario_simulator_config
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.groq_provider import GroqProvider
from app.providers.simulator_provider import SimulatorConfig, SimulatorProvider
from app.retrieval.ingestion import DocumentStore
from app.retrieval.search import ResearchRetriever

logger = logging.getLogger(__name__)

RESULTS_FILE = settings.RESULTS_DIR / "results.csv"

CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]


def init_results_csv() -> None:
    """Initialize results.csv with standard header if it does not exist."""
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def append_result_to_csv(res: PipelineResult) -> None:
    """Append a single request result to results.csv."""
    init_results_csv()
    first_token_str = (
        f"{res.first_token_ms:.2f}" if res.first_token_ms is not None else ""
    )
    total_ms_str = f"{res.total_ms:.2f}"
    grade_str = res.answer_grade.value if res.answer_grade else ""
    status_str = res.status.value
    source_str = res.source.value

    row = [
        res.request_id,
        res.scenario,
        source_str,
        res.start_time,
        first_token_str,
        total_ms_str,
        status_str,
        res.retries,
        res.tokens_in,
        res.tokens_out,
        res.retrieval_setting,
        grade_str,
    ]

    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def grade_answer(res: PipelineResult, q_meta: dict) -> AnswerGrade:
    """Evaluate correctness of answer against expected ground truth."""
    if res.status != ExecutionStatus.OK:
        return None

    expected_refusal = q_meta.get("expected_refusal", False)
    if expected_refusal:
        if res.is_refusal:
            return AnswerGrade.REFUSED_CORRECTLY
        else:
            return AnswerGrade.REFUSED_WRONGLY

    # For expected answerable questions:
    if res.is_refusal:
        return AnswerGrade.REFUSED_WRONGLY

    # Must have citations
    if not res.citations:
        return AnswerGrade.WRONG

    # Check target paper if specified
    target_paper = q_meta.get("paper_target")
    if target_paper and target_paper not in ("comparison", "general"):
        cited_papers = {c["paper_id"] for c in res.citations}
        if target_paper not in cited_papers:
            return AnswerGrade.WRONG

    # Check answer text for expected technical keywords
    answer_lower = res.answer.lower()
    expected_keywords = q_meta.get("expected_answer_keywords", [])
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    if expected_keywords and matches == 0:
        return AnswerGrade.WRONG

    return AnswerGrade.CORRECT


class ExperimentRunner:
    """Coordinates benchmark experiments across configurations and load scenarios."""

    def __init__(self, doc_store: Optional[DocumentStore] = None):
        self.doc_store = doc_store or DocumentStore()
        self.questions: List[dict] = []
        self._load_questions()

    def _load_questions(self) -> None:
        q_file = settings.EVALUATIONS_DIR / "questions.json"
        if not q_file.exists():
            raise FileNotFoundError(f"Questions file missing at {q_file}")
        with open(q_file, "r", encoding="utf-8") as f:
            self.questions = json.load(f)
        logger.info("Loaded %d evaluation questions.", len(self.questions))

    async def run_retrieval_benchmarks(
        self, provider_type: str = "stand-in"
    ) -> Dict[str, dict]:
        """Test and compare at least 3 retrieval configurations."""
        logger.info("Starting Retrieval Quality Benchmarks across 3 configurations...")
        results_by_config = {}

        for config in RETRIEVAL_CONFIGS:
            logger.info("Benchmarking Retrieval Config: %s ...", config.name)
            retriever = ResearchRetriever(self.doc_store, config)

            if provider_type == "groq" and settings.GROQ_API_KEY:
                provider = GroqProvider()
                admission = AdmissionController()
            else:
                # Retrieval benchmark evaluates quality and latency across all questions
                sim_config = SimulatorConfig(
                    baseline_latency_ms=180.0,
                    latency_variance_ms=30.0,
                    first_token_ratio=0.35,
                    rate_limit_rpm=None,
                    rate_limit_tpm=None,
                    max_concurrency=8,
                )
                provider = SimulatorProvider(config=sim_config)
                local_limiter = LocalRateLimiter(
                    rpm_limit=1000, tpm_limit=1000000, rpd_limit=100000
                )
                admission = AdmissionController(
                    max_concurrency=8,
                    max_queue_size=50,
                    rate_limiter=local_limiter,
                )

            orchestrator = PipelineOrchestrator(
                retriever=retriever,
                provider=provider,
                admission_controller=admission,
                scenario_name=f"retrieval_bench_{config.name}",
            )

            config_results = []
            for q in self.questions:
                res = await orchestrator.execute(q["question"])
                # Evaluate grade
                res.answer_grade = grade_answer(res, q)
                append_result_to_csv(res)
                config_results.append(res)
                await asyncio.sleep(0.05)

            results_by_config[config.name] = self._compute_metrics(config_results)

        return results_by_config

    async def run_load_scenarios(self) -> Dict[str, dict]:
        """Run all 5 mandatory load testing scenarios using calibrated stand-in simulator."""
        logger.info("Starting 5 Mandatory Load Scenarios...")
        scenarios = [
            ("steady_load", 12, 1.2),  # 12 requests, 1.2s delay
            ("sudden_burst", 16, 0.05),  # 16 requests arriving within 1s
            ("sustained_overload", 25, 0.1),  # 25 requests exceeding 20 RPM limit
            ("limit_halved", 18, 0.2),  # 18 requests under 15 RPM halved limit
            ("daily_budget_exhausted", 8, 0.05),  # 8 requests under exhausted daily budget
        ]

        scenario_metrics = {}
        # Use baseline retrieval config for load testing
        baseline_config = RETRIEVAL_CONFIGS[1]  # medium_k5_norerank
        retriever = ResearchRetriever(self.doc_store, baseline_config)

        for sc_name, num_requests, interval in scenarios:
            logger.info("Executing Scenario: %s (%d requests)...", sc_name, num_requests)
            sim_config = get_scenario_simulator_config(sc_name)
            provider = SimulatorProvider(config=sim_config)

            # Custom rate limiter for halved limit scenario
            rpm = sim_config.rate_limit_rpm or 30
            tpm = sim_config.rate_limit_tpm or 6000
            rate_limiter = LocalRateLimiter(rpm_limit=rpm, tpm_limit=tpm)
            admission = AdmissionController(
                max_concurrency=sim_config.max_concurrency or settings.MAX_CONCURRENCY,
                max_queue_size=settings.MAX_QUEUE_SIZE,
                rate_limiter=rate_limiter,
            )

            orchestrator = PipelineOrchestrator(
                retriever=retriever,
                provider=provider,
                admission_controller=admission,
                scenario_name=sc_name,
            )

            async def send_req(idx: int, q_dict: dict) -> PipelineResult:
                await asyncio.sleep(idx * interval)
                res = await orchestrator.execute(q_dict["question"])
                res.answer_grade = grade_answer(res, q_dict)
                append_result_to_csv(res)
                return res

            tasks = [
                send_req(i, self.questions[i % len(self.questions)])
                for i in range(num_requests)
            ]
            results = await asyncio.gather(*tasks)

            metrics = self._compute_metrics(results)
            metrics["peak_queue_length"] = admission.peak_queue_length
            scenario_metrics[sc_name] = metrics

        return scenario_metrics

    def _compute_metrics(self, results: List[PipelineResult]) -> dict:
        total = len(results)
        if total == 0:
            return {}

        ok_count = sum(1 for r in results if r.status == ExecutionStatus.OK)
        rl_count = sum(1 for r in results if r.status == ExecutionStatus.RATE_LIMITED)
        to_count = sum(1 for r in results if r.status == ExecutionStatus.TIMEOUT)
        err_count = sum(1 for r in results if r.status == ExecutionStatus.ERROR)

        correct_count = sum(
            1 for r in results if r.answer_grade == AnswerGrade.CORRECT
        )
        wrong_count = sum(1 for r in results if r.answer_grade == AnswerGrade.WRONG)
        refused_corr_count = sum(
            1 for r in results if r.answer_grade == AnswerGrade.REFUSED_CORRECTLY
        )
        refused_wrong_count = sum(
            1 for r in results if r.answer_grade == AnswerGrade.REFUSED_WRONGLY
        )

        total_latencies = [r.total_ms for r in results]
        first_token_latencies = [
            r.first_token_ms for r in results if r.first_token_ms is not None
        ]

        tokens_in_list = [r.tokens_in for r in results if r.status == ExecutionStatus.OK]
        tokens_out_list = [r.tokens_out for r in results if r.status == ExecutionStatus.OK]
        retries_list = [r.retries for r in results]

        return {
            "total_requests": total,
            "success_rate_pct": (ok_count / total) * 100.0,
            "rate_limited_pct": (rl_count / total) * 100.0,
            "timeout_pct": (to_count / total) * 100.0,
            "error_pct": (err_count / total) * 100.0,
            "correct_and_cited_pct": (correct_count / total) * 100.0,
            "wrong_pct": (wrong_count / total) * 100.0,
            "correctly_refused_pct": (refused_corr_count / total) * 100.0,
            "latency_p50_ms": float(np.percentile(total_latencies, 50)) if total_latencies else 0.0,
            "latency_p95_ms": float(np.percentile(total_latencies, 95)) if total_latencies else 0.0,
            "latency_p99_ms": float(np.percentile(total_latencies, 99)) if total_latencies else 0.0,
            "first_token_p50_ms": (
                float(np.percentile(first_token_latencies, 50))
                if first_token_latencies
                else None
            ),
            "first_token_p95_ms": (
                float(np.percentile(first_token_latencies, 95))
                if first_token_latencies
                else None
            ),
            "avg_tokens_in": float(np.mean(tokens_in_list)) if tokens_in_list else 0.0,
            "avg_tokens_out": float(np.mean(tokens_out_list)) if tokens_out_list else 0.0,
            "retries_per_request": float(np.mean(retries_list)) if retries_list else 0.0,
        }
