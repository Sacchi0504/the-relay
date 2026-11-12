"""Main CLI entry point for The Relay research question answering pipeline."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure root directory is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.core.models import RetrievalConfig
from app.experiments.runner import append_result_to_csv, init_results_csv
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.groq_provider import GroqProvider
from app.providers.simulator_provider import SimulatorProvider
from app.retrieval.ingestion import DocumentStore
from app.retrieval.search import ResearchRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TheRelay")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The Relay: Research-Grounded Multi-Stage LLM Pipeline"
    )
    parser.add_argument(
        "--question", "-q", type=str, help="Research engineering question to ask."
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Launch interactive CLI query loop."
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=settings.PROVIDER_TYPE,
        choices=["groq", "stand-in", "simulator"],
        help="Inference provider (groq or stand-in).",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=settings.GROQ_MODEL, help="Model identifier."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Chunk size in words."
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Number of passages to retrieve."
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Enable TF-IDF / section heading reranker."
    )
    return parser


async def run_single_query(orchestrator: PipelineOrchestrator, question: str):
    print(f"\n[Question]: {question}")
    print("-" * 70)
    result = await orchestrator.execute(question)
    append_result_to_csv(result)

    print(f"[Status]: {result.status.value.upper()}")
    if result.is_refusal:
        print(f"[Refusal]: {result.answer}")
        if result.refusal_reason:
            print(f"[Reason]: {result.refusal_reason}")
    else:
        print(f"[Answer]:\n{result.answer}")
        if result.citations:
            print("\n[Citations Verified]:")
            for c in result.citations:
                sec = f", {c['section']}" if c.get("section") else ""
                print(f"  * {c['paper_id']} (Page {c['page_number']}{sec})")

    print("-" * 70)
    print(
        f"[Metrics]: Latency: {result.total_ms:.1f}ms (TTFT: {result.first_token_ms or 0:.1f}ms) | "
        f"Tokens: {result.tokens_in} in, {result.tokens_out} out | "
        f"Retrieved: {len(result.retrieved_chunks)} chunks"
    )
    if result.timings:
        print(
            f"[Stage Breakdown]: Queue: {result.timings.queue_wait_ms:.1f}ms | "
            f"Retrieval: {result.timings.retrieval_ms:.1f}ms | "
            f"Model: {result.timings.model_execution_ms:.1f}ms | "
            f"Validation: {result.timings.validation_ms:.1f}ms"
        )
    print("-" * 70)


async def main():
    parser = build_parser()
    args = parser.parse_args()

    init_results_csv()
    doc_store = DocumentStore()
    config = RetrievalConfig(
        name=f"custom_k{args.top_k}",
        chunk_size_words=args.chunk_size,
        top_k=args.top_k,
        use_reranking=args.rerank,
    )
    retriever = ResearchRetriever(doc_store, config)

    if args.provider in ("groq", "real") and settings.GROQ_API_KEY:
        provider = GroqProvider(default_model=args.model)
        logger.info("Using real Groq provider (%s)", args.model)
    else:
        provider = SimulatorProvider()
        logger.info("Using stand-in simulator provider")

    orchestrator = PipelineOrchestrator(
        retriever=retriever,
        provider=provider,
        scenario_name="cli_interactive" if args.interactive else "cli_single",
    )

    if args.question:
        await run_single_query(orchestrator, args.question)
    elif args.interactive:
        print("\n=======================================================")
        print(" Welcome to The Relay Interactive Research Assistant")
        print(" Approved Papers: Attention Is All You Need, Moshi, H-Net")
        print(" Type 'exit' or 'quit' to end session.")
        print("=======================================================\n")
        while True:
            try:
                user_q = input("\nEnter research question: ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit", "q"):
                    break
                await run_single_query(orchestrator, user_q)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nSession ended.")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())

