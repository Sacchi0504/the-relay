"""Download and verify the approved research papers."""

import hashlib
import json
import logging
from pathlib import Path
import httpx
import pymupdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

APPROVED_PAPERS = [
    {
        "paper_id": "attention_is_all_you_need",
        "paper_title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "version": 7,
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "filename": "1706.03762v7.pdf",
    },
    {
        "paper_id": "moshi_realtime_dialogue",
        "paper_title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037",
        "version": 1,
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "filename": "2410.00037v1.pdf",
    },
    {
        "paper_id": "dynamic_chunking_hierarchical",
        "paper_title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955",
        "version": 2,
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "filename": "2507.07955v2.pdf",
    },
]

PAPERS_DIR = Path("papers")


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_papers(force: bool = False) -> dict:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    manifest_path = PAPERS_DIR / "manifest.json"

    headers = {
        "User-Agent": "TheRelayResearchPipeline/2.0 (Engineering Evaluation; mailto:research@example.com)"
    }

    with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as client:
        for paper in APPROVED_PAPERS:
            dest_path = PAPERS_DIR / paper["filename"]
            logger.info("Processing %s (arXiv:%sv%s)...", paper["paper_title"], paper["arxiv_id"], paper["version"])

            if dest_path.exists() and not force:
                logger.info("File %s already exists. Verifying integrity...", paper["filename"])
            else:
                logger.info("Downloading from %s ...", paper["url"])
                resp = client.get(paper["url"])
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                logger.info("Downloaded %d bytes to %s", len(resp.content), dest_path)

            # Verify it is a valid PDF
            try:
                doc = pymupdf.open(dest_path)
                page_count = len(doc)
                doc.close()
                logger.info("Verified valid PDF with %d pages.", page_count)
            except Exception as e:
                logger.error("Failed to parse PDF %s: %s", dest_path, e)
                raise RuntimeError(f"Corrupted or invalid PDF: {dest_path}") from e

            checksum = compute_sha256(dest_path)
            manifest[paper["paper_id"]] = {
                "paper_id": paper["paper_id"],
                "paper_title": paper["paper_title"],
                "arxiv_id": paper["arxiv_id"],
                "version": paper["version"],
                "filename": paper["filename"],
                "file_path": str(dest_path.as_posix()),
                "sha256": checksum,
                "page_count": page_count,
            }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved papers manifest to %s", manifest_path)
    return manifest


if __name__ == "__main__":
    download_papers()
