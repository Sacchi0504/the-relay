# The Relay: Research-Grounded Multi-Stage LLM Pipeline

> **A resilient, observable question-answering pipeline operating under real-world operational constraints.**

The Relay demonstrates robust execution of multi-stage LLM applications under variable traffic, provider rate limits, token restrictions, timeouts, and quota exhaustion. Grounded strictly in an approved research library of three exact paper versions, the pipeline retrieves evidence, bounds query loops, avoids unnecessary model calls, verifies citations, and refuses unsupported queries.

---

## Architecture Overview

The system strictly enforces logical boundaries across 6 processing stages:

```
[User Question]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Intake & Query Processing                     │
│ - Validate input format and enforce length bounds       │
│ - Assign unique request ID and record start timestamp  │
│ - Acquire admission slot via bounded queue & semaphore │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Research Retrieval                            │
│ - Search local BM25 index over approved papers         │
│ - Preserve page, section, chunk ID & content hash      │
│ - Optional TF-IDF + section heading reranking          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Evidence Assessment                           │
│ - Calculate query term coverage over top passages      │
│ - If weak: bounded retrieval loop (query reformulation)│
│ - If unsupported: REFUSE directly (0 tokens wasted!)   │
└──────────────────────────┬─────────────────────────────┘
                           │ (Sufficient Evidence)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Answer Generation                             │
│ - Format prompt with sanitized XML context blocks      │
│ - Execute model call via GroqProvider or Simulator     │
│ - Stream first token, record TTFT and token usage      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 5: Answer Validation                             │
│ - Validate paper IDs, page numbers & claim grounding   │
│ - If invalid: trigger bounded regeneration attempt (<=2)│
│ - If still invalid: refuse response cleanly            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 6: Response Completion & Observability           │
│ - Calculate stage latency breakdown and derived timings │
│ - Append row to results.csv with execution status      │
│ - Deliver validated response to user                   │
└────────────────────────────────────────────────────────┘
```

---

## Approved Research Library

Ingestion downloads the exact specified versions and verifies SHA-256 checksums recorded in `papers/manifest.json`:

| Research Paper | arXiv ID | Version | Pages | SHA-256 Checksum |
|---|---|---|---|---|
| **Attention Is All You Need** | `1706.03762` | 7 | 15 | `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697` |
| **Moshi: A Speech-Text Foundation Model for Real-Time Dialogue** | `2410.00037` | 1 | 66 | `a687144cc77bea86b0c60ae800ccb1d3e0816033fa5c7960388fe29a0dbfcfac` |
| **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling** | `2507.07955` | 2 | 39 | `04a2b32e6e82853cfc23c5db132750874a86cde99219ec6f664128f4aa83122f` |

No external search or outside research papers are permitted.

---

## Quickstart & Installation

### Prerequisites
- Python 3.12+
- Virtual environment

### Setup
```bash
# 1. Clone or navigate to workspace
cd d:\Relay

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download and verify approved papers
python scripts/download_papers.py
```

### Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Configure your Groq API key (optional for simulator experiments):
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
RELAY_PROVIDER=stand-in  # "groq" or "stand-in"
```

---

## CLI Usage

### Ask a Single Question
```bash
python app/main.py --question "In the Transformer architecture, what is the dimensionality d_model?"
```

### Interactive Session
```bash
python app/main.py --interactive
```

---

## Running Benchmarks & Tests

### Execute Full Test Suite
```bash
pytest -v
```

### Run All Retrieval Benchmarks and Load Scenarios
```bash
python scripts/run_benchmarks.py
```
This executes:
1. 3 Retrieval Configurations on 16 test questions.
2. 5 Mandatory Load Scenarios (`steady_load`, `sudden_burst`, `sustained_overload`, `limit_halved`, `daily_budget_exhausted`).
3. Saves all 127 individual request rows to `results/results.csv`.
4. Outputs summary tables and writes `results/summary_metrics.json`.

### Generate Engineering Report
```bash
python scripts/generate_report.py
```
Outputs the complete report at `report/ENGINEERING_REPORT.md`.

---

## Summary Results

### Table 1: Retrieval Configuration Trade-Offs
| Config | Requests | Correct & Cited (%) | Wrong (%) | Correct Refusal (%) | Tokens/Answer | TTFT p50 (ms) | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|
| `small_k3_norerank` | 16 | **62.5%** | 12.5% | 25.0% | **37.5** | **73.7** | **193.5** |
| `medium_k5_norerank` | 16 | 43.8% | 25.0% | 25.0% | 37.2 | 75.0 | 200.8 |
| `large_k8_rerank` | 16 | 18.8% | 12.5% | 25.0% | 22.1 | 60.4 | 173.1 |

*Conclusion:* `small_k3_norerank` (250 words, top_k=3) delivers the highest accuracy and evidence grounding with minimal token consumption.

### Table 2: Five Mandatory Load Scenarios
| Scenario | Requests | Success (%) | Rate Limited (%) | Timeout (%) | Retries/Req | Peak Queue | Latency p50 (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|---|
| `steady_load` | 12 | 16.7% | 83.3% | 0.0% | 0.50 | 3 | 0.2 | 2307.4 |
| `sudden_burst` | 16 | 31.2% | 68.8% | 0.0% | 0.00 | **8** | 74.6 | 565.2 |
| `sustained_overload` | 25 | 12.0% | 88.0% | 0.0% | 0.00 | 3 | 0.1 | 430.2 |
| `limit_halved` | 18 | 5.6% | 94.4% | 0.0% | 0.11 | 2 | 0.1 | 556.2 |
| `daily_budget_exhausted` | 8 | 0.0% | 100.0% | 0.0% | **0.00** | 1 | 1.8 | 2.5 |

*Conclusion:* The admission controller, bounded queue, and retry policies reliably protect upstream capacity and cleanly refuse non-retryable errors without amplification.

---

## Strict Engineering Compliance
- ✅ Exact paper versions downloaded and checksummed (SHA-256).
- ✅ Bounded execution loops (retrieval $\le 2$, regeneration $\le 2$).
- ✅ Zero hardcoded API keys.
- ✅ Stand-in simulator calibrated and explicitly identified.
- ✅ 100% reproducible results recomputed from `results/results.csv`.

