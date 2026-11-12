"""Executes live research Q&A benchmarks against the real Groq API and logs to results.csv."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure root directory is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.core.models import ProviderSource, RetrievalConfig
from app.experiments.runner import append_result_to_csv, grade_answer, init_results_csv
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.groq_provider import GroqProvider
from app.retrieval.ingestion import DocumentStore
from app.retrieval.search import ResearchRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealGroqBenchmark")

# Subset of 6 representative questions across all papers and categories
REAL_PROBE_QUESTIONS = [
    {
        "id": "q01_factual_transformer_dim",
        "category": "direct_factual",
        "paper_target": "attention_is_all_you_need",
        "question": "In the Transformer architecture, what is the dimensionality d_model of the input and output embeddings?",
        "expected_answer_keywords": ["512", "d_model"],
        "expected_refusal": False,
        "target_page": 3,
    },
    {
        "id": "q02_factual_moshi_mimi",
        "category": "direct_factual",
        "paper_target": "moshi_realtime_dialogue",
        "question": "What is the sampling frequency and frame rate of the Mimi neural audio codec used in Moshi?",
        "expected_answer_keywords": ["24 kHz", "12.5 Hz", "Mimi"],
        "expected_refusal": False,
        "target_page": 2,
    },
    {
        "id": "q03_factual_hnet_name",
        "category": "direct_factual",
        "paper_target": "dynamic_chunking_hierarchical",
        "question": "What is the name of the tokenizer-free hierarchical architecture proposed for dynamic sequence modeling?",
        "expected_answer_keywords": ["H-Net", "hierarchical"],
        "expected_refusal": False,
        "target_page": 1,
    },
    {
        "id": "q07_comparison_tokenization",
        "category": "comparison",
        "paper_target": "comparison",
        "question": "Compare the tokenization approach of the Transformer in Attention Is All You Need with H-Net in Dynamic Chunking.",
        "expected_answer_keywords": ["byte-pair encoding", "BPE", "tokenizer-free", "raw bytes", "H-Net"],
        "expected_refusal": False,
        "target_page": 1,
    },
    {
        "id": "q10_technical_similar_residual_norm",
        "category": "similar_terminology",
        "paper_target": "attention_is_all_you_need",
        "question": "What is the role of residual connections and layer normalization around each sub-layer in the Transformer?",
        "expected_answer_keywords": ["LayerNorm", "residual", "d_model"],
        "expected_refusal": False,
        "target_page": 3,
    },
    {
        "id": "q12_unanswerable_cookie_recipe",
        "category": "unanswerable",
        "paper_target": "out_of_domain",
        "question": "What is the recipe for baking chocolate chip cookies in an electric oven at 350 degrees Fahrenheit?",
        "expected_answer_keywords": ["insufficient evidence"],
        "expected_refusal": True,
        "target_page": None,
    },
]


async def main():
    logger.info("================================================================")
    logger.info("RUNNING LIVE REAL-PROVIDER BENCHMARK VIA GROQ API")
    logger.info("Selected Model: %s", settings.GROQ_MODEL)
    logger.info("================================================================")

    init_results_csv()
    doc_store = DocumentStore()
    config = RetrievalConfig(
        name="small_k3_norerank",
        chunk_size_words=250,
        top_k=3,
        use_reranking=False,
    )
    retriever = ResearchRetriever(doc_store, config)
    provider = GroqProvider(default_model=settings.GROQ_MODEL)

    orchestrator = PipelineOrchestrator(
        retriever=retriever,
        provider=provider,
        scenario_name="real_groq_live",
    )

    print("\n" + "=" * 80)
    print("LIVE GROQ INFERENCE EXECUTION ON RESEARCH QUESTIONS")
    print("=" * 80)

    for i, q in enumerate(REAL_PROBE_QUESTIONS, 1):
        print(f"\n--- [Question {i}/{len(REAL_PROBE_QUESTIONS)}] [{q['id']}] ---")
        print(f"Query: {q['question']}")

        result = await orchestrator.execute(q["question"])
        result.source = ProviderSource.REAL
        result.answer_grade = grade_answer(result, q)
        append_result_to_csv(result)

        print(f"Status: {result.status.value.upper()} | Grade: {result.answer_grade.value if result.answer_grade else 'N/A'}")
        if result.is_refusal:
            print(f"Refusal: {result.answer}")
        else:
            print(f"Answer snippet: {result.answer[:180]}...")
            if result.citations:
                print(f"Citations: {result.citations}")

        ttft_str = f"{result.first_token_ms:.1f}ms" if result.first_token_ms else "N/A"
        print(
            f"Metrics: TTFT: {ttft_str} | Total: {result.total_ms:.1f}ms | "
            f"Tokens: {result.tokens_in} in, {result.tokens_out} out"
        )

        # Gentle inter-request spacing to stay well within free tier limits
        await asyncio.sleep(2.0)

    print("\n" + "=" * 80)
    print("All live real Groq requests recorded to results/results.csv")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
