"""Functional tests for input validation, retrieval, chunking, citation validation, and refusal."""

import pytest
from app.core.models import (
    AnswerGrade,
    ChunkMetadata,
    ExecutionStatus,
    RetrievalConfig,
    ValidationDecision,
)
from app.pipeline.orchestrator import InputValidationError, PipelineOrchestrator
from app.providers.simulator_provider import SimulatorProvider
from app.retrieval.chunking import DocumentChunker
from app.retrieval.ingestion import DocumentStore, PageDocument
from app.retrieval.search import ResearchRetriever
from app.validation.validator import CitationValidator


@pytest.fixture(scope="session")
def doc_store():
    return DocumentStore()


@pytest.fixture
def retriever(doc_store):
    config = RetrievalConfig(name="test_med", chunk_size_words=500, top_k=3)
    return ResearchRetriever(doc_store, config)


@pytest.fixture
def validator():
    return CitationValidator()


def test_input_validation():
    retriever_mock = None
    provider_mock = SimulatorProvider()
    orchestrator = PipelineOrchestrator(retriever=retriever_mock, provider=provider_mock)

    # Empty question
    with pytest.raises(InputValidationError):
        orchestrator._validate_input("   ")

    # Oversized question
    with pytest.raises(InputValidationError):
        orchestrator._validate_input("a" * 5000)

    # Valid question
    clean = orchestrator._validate_input("  What is self-attention?  ")
    assert clean == "What is self-attention?"


def test_chunking_metadata_preservation():
    chunker = DocumentChunker(chunk_size_words=100, chunk_overlap_words=20)
    page = PageDocument(
        paper_id="attention_is_all_you_need",
        paper_title="Attention Is All You Need",
        arxiv_id="1706.03762",
        version=7,
        page_number=3,
        text=" ".join(["word"] * 250),
        section_heading="Model Architecture",
    )
    chunks = chunker.chunk_pages([page])
    assert len(chunks) > 1
    for c in chunks:
        assert c.paper_id == "attention_is_all_you_need"
        assert c.page_number == 3
        assert c.version == 7
        assert c.section_heading == "Model Architecture"
        assert c.content_hash is not None
        assert len(c.content_hash) == 64


def test_retrieval_relevance(retriever):
    results = retriever.search("Transformer multi-head attention", top_k=3)
    assert len(results) > 0
    papers = {c.paper_id for c in results}
    assert "attention_is_all_you_need" in papers
    assert results[0].score > 0.0


def test_citation_validator_valid(validator):
    chunk = ChunkMetadata(
        paper_id="attention_is_all_you_need",
        paper_title="Attention Is All You Need",
        arxiv_id="1706.03762",
        version=7,
        page_number=4,
        section_heading="Multi-Head Attention",
        chunk_id="att_p4_c1",
        chunk_text="Multi-head attention projections...",
        content_hash="abc",
    )
    valid_answer = (
        "The model uses multi-head attention with d_k=64. "
        "[Citation: attention_is_all_you_need, Page 4, Multi-Head Attention]"
    )
    res = validator.validate_answer(valid_answer, retrieved_chunks=[chunk])
    assert res.passed is True
    assert len(res.citations) == 1
    assert res.citations[0]["paper_id"] == "attention_is_all_you_need"
    assert res.citations[0]["page_number"] == 4


def test_citation_validator_fabricated_paper(validator):
    chunk = ChunkMetadata(
        paper_id="attention_is_all_you_need",
        paper_title="Attention Is All You Need",
        arxiv_id="1706.03762",
        version=7,
        page_number=4,
        section_heading="Multi-Head Attention",
        chunk_id="att_p4_c1",
        chunk_text="Multi-head attention...",
        content_hash="abc",
    )
    # Fabricated citation to unapproved paper
    invalid_answer = "This is a claim. [Citation: fictional_bert_2019, Page 2, Introduction]"
    res = validator.validate_answer(invalid_answer, retrieved_chunks=[chunk], attempt_number=2, max_attempts=2)
    assert res.passed is False
    assert res.decision == ValidationDecision.REJECTED
    assert any("Fabricated or unapproved paper_id" in err for err in res.failure_reasons)


def test_citation_validator_page_mismatch(validator):
    chunk = ChunkMetadata(
        paper_id="attention_is_all_you_need",
        paper_title="Attention Is All You Need",
        arxiv_id="1706.03762",
        version=7,
        page_number=4,
        section_heading="Multi-Head Attention",
        chunk_id="att_p4_c1",
        chunk_text="Multi-head attention...",
        content_hash="abc",
    )
    # Citing page 12 when only page 4 was retrieved
    invalid_answer = "This is a claim. [Citation: attention_is_all_you_need, Page 12, Conclusion]"
    res = validator.validate_answer(invalid_answer, retrieved_chunks=[chunk], attempt_number=2, max_attempts=2)
    assert res.passed is False
    assert any("does not match retrieved chunk pages" in err for err in res.failure_reasons)


@pytest.mark.asyncio
async def test_end_to_end_correct_refusal(retriever):
    provider = SimulatorProvider()
    orchestrator = PipelineOrchestrator(retriever=retriever, provider=provider)
    # Query completely unrelated to research papers
    result = await orchestrator.execute("How do you bake sourdough bread with wild yeast?")
    assert result.status == ExecutionStatus.OK
    assert result.is_refusal is True
    assert "insufficient evidence" in result.answer.lower()
    assert result.answer_grade == AnswerGrade.REFUSED_CORRECTLY

