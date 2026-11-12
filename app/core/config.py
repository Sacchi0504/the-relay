"""Configuration management for The Relay pipeline."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    # Groq API configuration
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # Provider selection: "groq" or "stand-in" (simulator)
    PROVIDER_TYPE: str = os.getenv("RELAY_PROVIDER", "groq").lower()

    # Resource management & Admission Control
    MAX_CONCURRENCY: int = int(os.getenv("RELAY_MAX_CONCURRENCY", "4"))
    MAX_QUEUE_SIZE: int = int(os.getenv("RELAY_MAX_QUEUE_SIZE", "20"))
    REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("RELAY_REQUEST_TIMEOUT_SECONDS", "60.0"))
    QUEUE_TIMEOUT_SECONDS: float = float(os.getenv("RELAY_QUEUE_TIMEOUT_SECONDS", "30.0"))

    # Retries
    MAX_RETRIES: int = int(os.getenv("RELAY_MAX_RETRIES", "2"))
    BASE_BACKOFF_SECONDS: float = float(os.getenv("RELAY_BASE_BACKOFF_SECONDS", "1.0"))
    MAX_BACKOFF_SECONDS: float = float(os.getenv("RELAY_MAX_BACKOFF_SECONDS", "16.0"))

    # Bounded Pipeline Loops
    MAX_RETRIEVAL_ATTEMPTS: int = int(os.getenv("RELAY_MAX_RETRIEVAL_ATTEMPTS", "2"))
    MAX_VALIDATION_ATTEMPTS: int = int(os.getenv("RELAY_MAX_VALIDATION_ATTEMPTS", "2"))

    # Paths
    PAPERS_DIR: Path = BASE_DIR / "papers"
    MANIFEST_PATH: Path = PAPERS_DIR / "manifest.json"
    RESULTS_DIR: Path = BASE_DIR / "results"
    RESULTS_FILE: Path = RESULTS_DIR / "results.csv"
    EVALUATIONS_DIR: Path = BASE_DIR / "evaluations"
    CONFIGS_DIR: Path = BASE_DIR / "configs"
    REPORT_DIR: Path = BASE_DIR / "report"

    # Known Groq free-tier limits (documented)
    # Llama-3.3-70b-versatile: 30 RPM, 6,000 TPM, 1,000 RPD
    # Llama-3.1-8b-instant: 30 RPM, 6,000 TPM, 14,400 RPD
    GROQ_RPM_LIMIT: int = int(os.getenv("GROQ_RPM_LIMIT", "30"))
    GROQ_TPM_LIMIT: int = int(os.getenv("GROQ_TPM_LIMIT", "6000"))
    GROQ_RPD_LIMIT: int = int(os.getenv("GROQ_RPD_LIMIT", "1000"))


settings = Settings()
