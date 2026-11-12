"""Core data models and schemas for The Relay."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import pydantic


class ErrorCategory(str, Enum):
    AUTHENTICATION_ERROR = "authentication_error"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMIT_ERROR = "rate_limit_error"
    QUOTA_EXHAUSTED = "quota_exhausted"
    TIMEOUT_ERROR = "timeout_error"
    CONNECTION_ERROR = "connection_error"
    PROVIDER_ERROR = "provider_error"
    INCOMPLETE_STREAM = "incomplete_stream"
    VALIDATION_ERROR = "validation_error"


class ExecutionStatus(str, Enum):
    OK = "ok"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    ERROR = "error"


class ProviderSource(str, Enum):
    REAL = "real"
    STAND_IN = "stand-in"


class AnswerGrade(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    REFUSED_CORRECTLY = "refused_correctly"
    REFUSED_WRONGLY = "refused_wrongly"


class EvidenceAssessmentDecision(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    RETRY_RETRIEVAL = "retry_retrieval"
    CONFLICTING = "conflicting"


class ValidationDecision(str, Enum):
    PASSED = "passed"
    RETRY_GENERATION = "retry_generation"
    REJECTED = "rejected"


class ChunkMetadata(pydantic.BaseModel):
    paper_id: str
    paper_title: str
    arxiv_id: str
    version: int
    page_number: int
    section_heading: str
    chunk_id: str
    chunk_text: str
    content_hash: str
    score: float = 0.0


class RetrievalConfig(pydantic.BaseModel):
    name: str = "default"
    chunk_size_words: int = 500
    chunk_overlap_words: int = 100
    top_k: int = 5
    use_reranking: bool = False
    reranker_type: str = "bm25_lexical"  # bm25_lexical, tfidf_hybrid


class ModelRequest(pydantic.BaseModel):
    request_id: str
    messages: List[Dict[str, str]]
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    stream: bool = False
    system_prompt: Optional[str] = None


class ModelResponse(pydantic.BaseModel):
    request_id: str
    content: str = ""
    completion_status: ExecutionStatus = ExecutionStatus.OK
    tokens_in: int = 0
    tokens_out: int = 0
    first_token_ms: Optional[float] = None
    total_ms: float = 0.0
    raw_error: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    provider: ProviderSource = ProviderSource.REAL
    model_name: str = ""
    retries_occurred: int = 0
    stream_chunks: List[str] = field(default_factory=list)


class StageTimestamps(pydantic.BaseModel):
    request_received_at: float = 0.0
    queue_admission_started_at: float = 0.0
    queue_admission_completed_at: float = 0.0
    retrieval_started_at: float = 0.0
    retrieval_completed_at: float = 0.0
    model_execution_started_at: float = 0.0
    first_token_received_at: Optional[float] = None
    model_generation_completed_at: float = 0.0
    validation_completed_at: float = 0.0
    response_completed_at: float = 0.0


class DerivedTimings(pydantic.BaseModel):
    queue_wait_ms: float = 0.0
    retrieval_ms: float = 0.0
    time_to_first_token_ms: Optional[float] = None
    model_execution_ms: float = 0.0
    validation_ms: float = 0.0
    end_to_end_ms: float = 0.0


class PipelineResult(pydantic.BaseModel):
    request_id: str
    question: str
    answer: str
    is_refusal: bool = False
    refusal_reason: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[ChunkMetadata] = field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.OK
    source: ProviderSource = ProviderSource.REAL
    scenario: str = "default"
    retrieval_setting: str = "default"
    answer_grade: Optional[AnswerGrade] = None
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    first_token_ms: Optional[float] = None
    total_ms: float = 0.0
    start_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    timestamps: Optional[StageTimestamps] = None
    timings: Optional[DerivedTimings] = None
    error_category: Optional[ErrorCategory] = None
    raw_error: Optional[str] = None

