"""Runner script for executing retrieval benchmarks and all 5 mandatory load scenarios."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
import pandas as pd

# Ensure workspace root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.experiments.runner import ExperimentRunner, init_results_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("BenchmarkRunner")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", default=True, help="Start with fresh results.csv")
    args = parser.parse_args()

    logger.info("================================================================")
    logger.info("THE RELAY: BENCHMARK SUITE AND LOAD EXPERIMENT EXECUTION")
    logger.info("================================================================")

    if args.fresh and settings.RESULTS_FILE.exists():
        settings.RESULTS_FILE.unlink()
        logger.info("Cleared previous results.csv for clean reproducibility audit.")

    init_results_csv()
    runner = ExperimentRunner()

    # 1. Retrieval Benchmarks across 3 configurations
    logger.info("\n--- PHASE 1: RETRIEVAL QUALITY AND COST BENCHMARKS ---")
    retrieval_metrics = await runner.run_retrieval_benchmarks(provider_type="stand-in")

    # 2. Five Mandatory Load Testing Scenarios
    logger.info("\n--- PHASE 2: FIVE MANDATORY LOAD TESTING SCENARIOS ---")
    scenario_metrics = await runner.run_load_scenarios()

    # Save summary metrics to JSON
    summary = {
        "retrieval_benchmarks": retrieval_metrics,
        "load_scenarios": scenario_metrics,
    }
    summary_path = settings.RESULTS_DIR / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary metrics to %s", summary_path)

    # Print summary tables to stdout
    print("\n" + "=" * 80)
    print("TABLE 1: RETRIEVAL CONFIGURATION PERFORMANCE & COST COMPARISON")
    print("=" * 80)
    retrieval_rows = []
    for cfg_name, m in retrieval_metrics.items():
        retrieval_rows.append(
            {
                "Config": cfg_name,
                "Requests": m.get("total_requests", 0),
                "Correct & Cited (%)": f"{m.get('correct_and_cited_pct', 0.0):.1f}%",
                "Wrong (%)": f"{m.get('wrong_pct', 0.0):.1f}%",
                "Correct Refusal (%)": f"{m.get('correctly_refused_pct', 0.0):.1f}%",
                "Tokens/Answer": f"{m.get('avg_tokens_out', 0.0):.1f}",
                "First Token p50 (ms)": (
                    f"{m.get('first_token_p50_ms', 0.0):.1f}"
                    if m.get("first_token_p50_ms") is not None
                    else "N/A"
                ),
                "Total Latency p50 (ms)": f"{m.get('latency_p50_ms', 0.0):.1f}",
                "Total Latency p95 (ms)": f"{m.get('latency_p95_ms', 0.0):.1f}",
            }
        )
    df_retrieval = pd.DataFrame(retrieval_rows)
    print(df_retrieval.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 2: FIVE MANDATORY LOAD SCENARIOS BENCHMARK RESULTS")
    print("=" * 80)
    scenario_rows = []
    for sc_name, m in scenario_metrics.items():
        scenario_rows.append(
            {
                "Scenario": sc_name,
                "Requests": m.get("total_requests", 0),
                "Success (%)": f"{m.get('success_rate_pct', 0.0):.1f}%",
                "Rate Limited (%)": f"{m.get('rate_limited_pct', 0.0):.1f}%",
                "Timeout (%)": f"{m.get('timeout_pct', 0.0):.1f}%",
                "Error (%)": f"{m.get('error_pct', 0.0):.1f}%",
                "Retries/Req": f"{m.get('retries_per_request', 0.0):.2f}",
                "Peak Queue": m.get("peak_queue_length", 0),
                "Latency p50 (ms)": f"{m.get('latency_p50_ms', 0.0):.1f}",
                "Latency p95 (ms)": f"{m.get('latency_p95_ms', 0.0):.1f}",
                "Latency p99 (ms)": f"{m.get('latency_p99_ms', 0.0):.1f}",
            }
        )
    df_scenarios = pd.DataFrame(scenario_rows)
    print(df_scenarios.to_string(index=False))
    print("=" * 80)
    print(f"\nAll individual request records preserved in: {settings.RESULTS_DIR / 'results.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
