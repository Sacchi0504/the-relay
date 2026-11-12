"""Pipeline Orchestrator implementing all 6 processing stages with bounded loops."""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.models import (
    AnswerGrade,
    ChunkMetadata,
    DerivedTimings,
    ErrorCategory,
    EvidenceAssessmentDecision,
    ExecutionStatus,
    ModelRequest,
    ModelResponse,
    PipelineResult,
    ProviderSource,
    RetrievalConfig,
    StageTimestamps,
    ValidationDecision,
)
from app.core.resource_manager import AdmissionController, RetryManager
from app.pipeline.prompts import (
    SYSTEM_PROMPT,
    format_regeneration_prompt,
    format_user_prompt,
)
from app.providers.base import BaseModelProvider
from app.retrieval.search import ResearchRetriever
from app.validation.validator import CitationValidator

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 4000
MIN_RELEVANCE_SCORE = 1.0


class InputValidationError(Exception):
    pass


class PipelineOrchestrator:
    """Orchestrates Stages 1 through 6 with full observability and resource controls."""

    def __init__(
        self,
        retriever: ResearchRetriever,
        provider: BaseModelProvider,
        admission_controller: Optional[AdmissionController] = None,
        retry_manager: Optional[RetryManager] = None,
        validator: Optional[CitationValidator] = None,
        scenario_name: str = "default",
    ):
        self.retriever = retriever
        self.provider = provider
        self.admission_controller = admission_controller or AdmissionController()
        self.retry_manager = retry_manager or RetryManager()
        self.validator = validator or CitationValidator()
        self.scenario_name = scenario_name

    def _validate_input(self, question: str) -> str:
        """Stage 1: Input validation."""
        if not isinstance(question, str):
            raise InputValidationError("Question must be a string.")
        cleaned = question.strip()
        if not cleaned:
            raise InputValidationError("Question cannot be empty or whitespace.")
        if len(cleaned) > MAX_QUESTION_LENGTH:
            raise InputValidationError(
                f"Question exceeds maximum permitted length of {MAX_QUESTION_LENGTH} characters."
            )
        return cleaned

    def _assess_evidence(
        self, question: str, chunks: List[ChunkMetadata]
    ) -> EvidenceAssessmentDecision:
        """Stage 3: Evidence assessment based on relevance score and keyword coverage."""
        if not chunks:
            return EvidenceAssessmentDecision.INSUFFICIENT

        max_score = max(c.score for c in chunks)
        if max_score < MIN_RELEVANCE_SCORE:
            return EvidenceAssessmentDecision.RETRY_RETRIEVAL

        # Significant query terms coverage check using English stop words
        import re
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        query_terms = [
            w.lower()
            for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", question)
            if w.lower() not in ENGLISH_STOP_WORDS
        ]

        if query_terms:
            top_chunks_text = " ".join(c.chunk_text.lower() for c in chunks[:3])
            matched_terms = [qt for qt in query_terms if qt in top_chunks_text]
            coverage = len(matched_terms) / len(query_terms)
            if coverage < 0.4 or len(matched_terms) < 2:
                logger.info(
                    "Evidence assessed as INSUFFICIENT (keyword coverage %.1f%%: matched %s of %s)",
                    coverage * 100,
                    matched_terms,
                    query_terms,
                )
                return EvidenceAssessmentDecision.INSUFFICIENT

        return EvidenceAssessmentDecision.SUFFICIENT

    async def execute(
        self,
        question: str,
        request_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> PipelineResult:
        req_id = request_id or uuid.uuid4().hex[:12]
        t_start = time.perf_counter()
        req_timeout = timeout_seconds or settings.REQUEST_TIMEOUT_SECONDS
        deadline = t_start + req_timeout

        timestamps = StageTimestamps(request_received_at=t_start)
        retries_count = 0
        tokens_in = 0
        tokens_out = 0
        first_token_ms: Optional[float] = None

        # ---------------------------------------------------------
        # STAGE 1: Request Intake and Query Processing
        # ---------------------------------------------------------
        try:
            clean_question = self._validate_input(question)
        except InputValidationError as e:
            t_end = time.perf_counter()
            timestamps.response_completed_at = t_end
            return PipelineResult(
                request_id=req_id,
                question=question,
                answer="",
                is_refusal=True,
                refusal_reason=str(e),
                status=ExecutionStatus.ERROR,
                source=self.provider.source,
                scenario=self.scenario_name,
                retrieval_setting=self.retriever.config.name,
                total_ms=(t_end - t_start) * 1000.0,
                error_category=ErrorCategory.INVALID_REQUEST,
                raw_error=str(e),
                timestamps=timestamps,
                timings=DerivedTimings(end_to_end_ms=(t_end - t_start) * 1000.0),
            )

        # ---------------------------------------------------------
        # ADMISSION CONTROL & QUEUEING
        # ---------------------------------------------------------
        timestamps.queue_admission_started_at = time.perf_counter()
        estimated_prompt_tokens = max(50, len(clean_question.split()) * 2 + 1000)
        remaining_sec = max(0.1, deadline - time.perf_counter())

        admitted, queue_wait_ms, reject_reason = await self.admission_controller.acquire_slot(
            estimated_tokens=estimated_prompt_tokens,
            deadline_seconds=remaining_sec,
        )
        timestamps.queue_admission_completed_at = time.perf_counter()

        if not admitted:
            t_end = time.perf_counter()
            timestamps.response_completed_at = t_end
            status = (
                ExecutionStatus.TIMEOUT
                if "Queue wait limit" in (reject_reason or "")
                else ExecutionStatus.RATE_LIMITED
            )
            err_cat = (
                ErrorCategory.TIMEOUT_ERROR
                if status == ExecutionStatus.TIMEOUT
                else ErrorCategory.RATE_LIMIT_ERROR
            )
            return PipelineResult(
                request_id=req_id,
                question=clean_question,
                answer="",
                is_refusal=True,
                refusal_reason=reject_reason,
                status=status,
                source=self.provider.source,
                scenario=self.scenario_name,
                retrieval_setting=self.retriever.config.name,
                total_ms=(t_end - t_start) * 1000.0,
                error_category=err_cat,
                raw_error=reject_reason,
                timestamps=timestamps,
                timings=DerivedTimings(
                    queue_wait_ms=queue_wait_ms,
                    end_to_end_ms=(t_end - t_start) * 1000.0,
                ),
            )

        try:
            # ---------------------------------------------------------
            # STAGE 2 & 3: Research Retrieval & Evidence Assessment
            # ---------------------------------------------------------
            timestamps.retrieval_started_at = time.perf_counter()
            retrieved_chunks = self.retriever.search(clean_question)
            evidence_status = self._assess_evidence(clean_question, retrieved_chunks)

            # Bounded retrieval loop if evidence was weak on attempt 1
            if (
                evidence_status == EvidenceAssessmentDecision.RETRY_RETRIEVAL
                and settings.MAX_RETRIEVAL_ATTEMPTS > 1
            ):
                logger.info(
                    "Attempt 1 evidence weak. Expanding query in bounded retrieval loop..."
                )
                expanded_query = f"{clean_question} architecture model parameters evaluation"
                second_chunks = self.retriever.search(expanded_query)
                if second_chunks:
                    retrieved_chunks = second_chunks
                    evidence_status = self._assess_evidence(clean_question, retrieved_chunks)

            timestamps.retrieval_completed_at = time.perf_counter()
            retrieval_ms = (
                timestamps.retrieval_completed_at - timestamps.retrieval_started_at
            ) * 1000.0

            # If evidence is still insufficient, refuse correctly WITHOUT unnecessary model call
            if (
                evidence_status
                in (
                    EvidenceAssessmentDecision.INSUFFICIENT,
                    EvidenceAssessmentDecision.RETRY_RETRIEVAL,
                )
                or not retrieved_chunks
            ):
                t_end = time.perf_counter()
                timestamps.response_completed_at = t_end
                refusal_text = (
                    "Based on the provided research papers, there is insufficient evidence to answer this question."
                )
                return PipelineResult(
                    request_id=req_id,
                    question=clean_question,
                    answer=refusal_text,
                    is_refusal=True,
                    refusal_reason="Insufficient evidence found in approved research papers.",
                    retrieved_chunks=retrieved_chunks,
                    status=ExecutionStatus.OK,
                    source=self.provider.source,
                    scenario=self.scenario_name,
                    retrieval_setting=self.retriever.config.name,
                    answer_grade=AnswerGrade.REFUSED_CORRECTLY,
                    total_ms=(t_end - t_start) * 1000.0,
                    timestamps=timestamps,
                    timings=DerivedTimings(
                        queue_wait_ms=queue_wait_ms,
                        retrieval_ms=retrieval_ms,
                        end_to_end_ms=(t_end - t_start) * 1000.0,
                    ),
                )

            # ---------------------------------------------------------
            # STAGE 4 & 5: Answer Generation & Answer Validation
            # ---------------------------------------------------------
            val_attempt = 1
            max_val_attempts = settings.MAX_VALIDATION_ATTEMPTS
            final_answer = ""
            final_citations = []
            final_status = ExecutionStatus.OK
            error_cat: Optional[ErrorCategory] = None
            raw_err: Optional[str] = None
            is_refusal = False
            refusal_reason = None

            generation_prompt = format_user_prompt(clean_question, retrieved_chunks)

            while val_attempt <= max_val_attempts:
                # Check overall deadline
                now = time.perf_counter()
                if now >= deadline:
                    final_status = ExecutionStatus.TIMEOUT
                    error_cat = ErrorCategory.TIMEOUT_ERROR
                    raw_err = "Overall request deadline exceeded during generation."
                    break

                timestamps.model_execution_started_at = time.perf_counter()
                model_req = ModelRequest(
                    request_id=f"{req_id}_v{val_attempt}",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": generation_prompt},
                    ],
                    model=settings.GROQ_MODEL,
                    timeout_seconds=max(1.0, deadline - time.perf_counter()),
                )

                # Execute with retries if provider returns transient failure
                model_resp = await self._execute_model_with_retries(model_req, deadline)
                retries_count += model_resp.retries_occurred
                tokens_in += model_resp.tokens_in
                tokens_out += model_resp.tokens_out

                if first_token_ms is None and model_resp.first_token_ms is not None:
                    first_token_ms = model_resp.first_token_ms
                    timestamps.first_token_received_at = (
                        timestamps.model_execution_started_at + (first_token_ms / 1000.0)
                    )

                timestamps.model_generation_completed_at = time.perf_counter()

                if model_resp.completion_status != ExecutionStatus.OK:
                    final_status = model_resp.completion_status
                    error_cat = model_resp.error_category
                    raw_err = model_resp.raw_error
                    break

                # STAGE 5: Answer Validation
                val_result = self.validator.validate_answer(
                    answer=model_resp.content,
                    retrieved_chunks=retrieved_chunks,
                    attempt_number=val_attempt,
                    max_attempts=max_val_attempts,
                )
                timestamps.validation_completed_at = time.perf_counter()

                if val_result.passed:
                    final_answer = model_resp.content
                    final_citations = val_result.citations
                    is_refusal = val_result.is_refusal
                    break
                elif val_result.decision == ValidationDecision.RETRY_GENERATION:
                    logger.info(
                        "Validation failed attempt %d: %s. Retrying generation...",
                        val_attempt,
                        val_result.failure_reasons,
                    )
                    generation_prompt = format_regeneration_prompt(
                        clean_question, retrieved_chunks, val_result.failure_reasons
                    )
                    val_attempt += 1
                else:
                    # REJECTED after max attempts
                    logger.warning(
                        "Validation permanently failed after %d attempts: %s",
                        val_attempt,
                        val_result.failure_reasons,
                    )
                    final_answer = (
                        "Answer refused: Response could not be validated against research evidence."
                    )
                    is_refusal = True
                    refusal_reason = f"Citation validation failed: {'; '.join(val_result.failure_reasons)}"
                    break

            # ---------------------------------------------------------
            # STAGE 6: Response Completion and Measurement
            # ---------------------------------------------------------
            t_end = time.perf_counter()
            timestamps.response_completed_at = t_end

            model_exec_ms = (
                (timestamps.model_generation_completed_at - timestamps.model_execution_started_at) * 1000.0
                if timestamps.model_execution_started_at > 0
                else 0.0
            )
            val_ms = (
                (timestamps.validation_completed_at - timestamps.model_generation_completed_at) * 1000.0
                if timestamps.validation_completed_at > 0
                else 0.0
            )

            # Determine answer grade
            grade = None
            if final_status == ExecutionStatus.OK:
                if is_refusal:
                    grade = AnswerGrade.REFUSED_CORRECTLY
                else:
                    grade = AnswerGrade.CORRECT

            timings = DerivedTimings(
                queue_wait_ms=queue_wait_ms,
                retrieval_ms=retrieval_ms,
                time_to_first_token_ms=first_token_ms,
                model_execution_ms=model_exec_ms,
                validation_ms=val_ms,
                end_to_end_ms=(t_end - t_start) * 1000.0,
            )

            return PipelineResult(
                request_id=req_id,
                question=clean_question,
                answer=final_answer,
                is_refusal=is_refusal,
                refusal_reason=refusal_reason,
                citations=final_citations,
                retrieved_chunks=retrieved_chunks,
                status=final_status,
                source=self.provider.source,
                scenario=self.scenario_name,
                retrieval_setting=self.retriever.config.name,
                answer_grade=grade,
                retries=retries_count,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                first_token_ms=first_token_ms,
                total_ms=(t_end - t_start) * 1000.0,
                timestamps=timestamps,
                timings=timings,
                error_category=error_cat,
                raw_error=raw_err,
            )

        finally:
            self.admission_controller.release_slot()

    async def _execute_model_with_retries(
        self, request: ModelRequest, deadline: float
    ) -> ModelResponse:
        attempt = 0
        retries_done = 0
        while True:
            resp = await self.provider.generate(request)
            if resp.completion_status == ExecutionStatus.OK:
                await self.admission_controller.circuit_breaker.record_success()
                resp.retries_occurred = retries_done
                return resp

            await self.admission_controller.circuit_breaker.record_failure(resp.error_category)

            # Check if retryable
            if not self.retry_manager.is_retryable(resp.error_category):
                logger.warning("Error %s is non-retryable. Halting.", resp.error_category)
                resp.retries_occurred = retries_done
                return resp

            if attempt >= self.retry_manager.max_retries:
                logger.warning("Max retries (%d) reached for request %s.", attempt, request.request_id)
                resp.retries_occurred = retries_done
                return resp

            backoff = self.retry_manager.compute_backoff(attempt)
            if time.perf_counter() + backoff >= deadline:
                logger.warning("Deadline would be exceeded during backoff sleep. Aborting retry.")
                resp.completion_status = ExecutionStatus.TIMEOUT
                resp.error_category = ErrorCategory.TIMEOUT_ERROR
                resp.retries_occurred = retries_done
                return resp

            logger.info(
                "Retrying request %s in %.2fs (attempt %d/%d)...",
                request.request_id,
                backoff,
                attempt + 1,
                self.retry_manager.max_retries,
            )
            await asyncio.sleep(backoff)
            attempt += 1
            retries_done += 1
