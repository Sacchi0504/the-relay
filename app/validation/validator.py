"""Citation and Evidence Validation for Research Responses."""

import logging
import re
from typing import Dict, List, Optional, Tuple
from app.core.models import ChunkMetadata, ValidationDecision

logger = logging.getLogger(__name__)

APPROVED_PAPER_IDS = {
    "attention_is_all_you_need",
    "moshi_realtime_dialogue",
    "dynamic_chunking_hierarchical",
}

# Regex to capture citations like [Citation: paper_id, Page X, Section] or [paper_id, Page X]
CITATION_PATTERN = re.compile(
    r"\[(?:Citation:\s*)?([a-zA-Z0-9_\-\.]+),\s*(?:Page|p\.|p)\s*(\d+)(?:,\s*([^\]]+))?\]",
    re.IGNORECASE,
)

REFUSAL_MARKERS = [
    "insufficient evidence",
    "cannot be answered",
    "does not contain information",
    "not mentioned in the provided",
    "not supported by the approved",
    "refuse to answer",
    "no evidence found",
    "not found in the research papers",
]


class ValidationResult:
    def __init__(
        self,
        decision: ValidationDecision,
        citations: List[Dict],
        failure_reasons: List[str],
        is_refusal: bool = False,
    ):
        self.decision = decision
        self.citations = citations
        self.failure_reasons = failure_reasons
        self.is_refusal = is_refusal

    @property
    def passed(self) -> bool:
        return self.decision == ValidationDecision.PASSED


class CitationValidator:
    """Validates source citations, page numbers, and claim grounding."""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def check_is_refusal(self, text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in REFUSAL_MARKERS)

    def extract_citations(self, text: str) -> List[Dict]:
        matches = CITATION_PATTERN.findall(text)
        citations = []
        for match in matches:
            paper_id = match[0].strip().lower()
            # Normalize paper id shortcuts if any
            if "attention" in paper_id:
                paper_id = "attention_is_all_you_need"
            elif "moshi" in paper_id:
                paper_id = "moshi_realtime_dialogue"
            elif "chunk" in paper_id or "hierarchical" in paper_id or "h-net" in paper_id:
                paper_id = "dynamic_chunking_hierarchical"

            page_num = int(match[1].strip())
            section = match[2].strip() if match[2] else ""
            citations.append(
                {
                    "raw": f"[{match[0]}, Page {match[1]}]",
                    "paper_id": paper_id,
                    "page_number": page_num,
                    "section": section,
                }
            )
        return citations

    def validate_answer(
        self,
        answer: str,
        retrieved_chunks: List[ChunkMetadata],
        attempt_number: int = 1,
        max_attempts: int = 2,
    ) -> ValidationResult:
        if not answer or not answer.strip():
            return ValidationResult(
                decision=ValidationDecision.RETRY_GENERATION if attempt_number < max_attempts else ValidationDecision.REJECTED,
                citations=[],
                failure_reasons=["Empty or whitespace response"],
            )

        # Check if the model explicitly and correctly refused
        if self.check_is_refusal(answer):
            return ValidationResult(
                decision=ValidationDecision.PASSED,
                citations=[],
                failure_reasons=[],
                is_refusal=True,
            )

        citations = self.extract_citations(answer)
        failure_reasons = []

        if not citations:
            failure_reasons.append("Answer contains no citations to supporting research.")
        else:
            # Map of valid (paper_id, page_number) in retrieved chunks
            retrieved_pages = {(c.paper_id, c.page_number) for c in retrieved_chunks}
            retrieved_papers = {c.paper_id for c in retrieved_chunks}

            for cit in citations:
                pid = cit["paper_id"]
                page = cit["page_number"]

                # 1. Validate paper identifier against approved library
                if pid not in APPROVED_PAPER_IDS:
                    failure_reasons.append(
                        f"Fabricated or unapproved paper_id cited: '{pid}'. "
                        f"Must be one of {sorted(APPROVED_PAPER_IDS)}."
                    )
                    continue

                # 2. Validate paper is in retrieved context
                if pid not in retrieved_papers:
                    failure_reasons.append(
                        f"Cited paper '{pid}' was not among the retrieved context passages."
                    )
                    continue

                # 3. Validate page exists within reasonable vicinity (exact or +- 1 page)
                matching_page = any(
                    c_pid == pid and abs(c_page - page) <= 1 for (c_pid, c_page) in retrieved_pages
                )
                if not matching_page:
                    valid_pages_for_paper = [
                        c.page_number for c in retrieved_chunks if c.paper_id == pid
                    ]
                    failure_reasons.append(
                        f"Cited page {page} for '{pid}' does not match retrieved chunk pages: {valid_pages_for_paper}."
                    )

        # Determine decision
        if failure_reasons:
            logger.warning(
                "Answer validation failed on attempt %d: %s",
                attempt_number,
                "; ".join(failure_reasons),
            )
            decision = (
                ValidationDecision.RETRY_GENERATION
                if attempt_number < max_attempts
                else ValidationDecision.REJECTED
            )
            return ValidationResult(
                decision=decision,
                citations=citations,
                failure_reasons=failure_reasons,
                is_refusal=False,
            )

        return ValidationResult(
            decision=ValidationDecision.PASSED,
            citations=citations,
            failure_reasons=[],
            is_refusal=False,
        )

