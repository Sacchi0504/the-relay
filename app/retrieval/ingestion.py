"""PDF Document Ingestion and Text Extraction with Section Metadata."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pymupdf

from app.core.config import settings

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised when document ingestion or verification fails."""
    pass


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class PageDocument:
    def __init__(
        self,
        paper_id: str,
        paper_title: str,
        arxiv_id: str,
        version: int,
        page_number: int,
        text: str,
        section_heading: str = "General",
        blocks: Optional[List[Dict]] = None,
    ):
        self.paper_id = paper_id
        self.paper_title = paper_title
        self.arxiv_id = arxiv_id
        self.version = version
        self.page_number = page_number
        self.text = text
        self.section_heading = section_heading
        self.blocks = blocks or []


class DocumentStore:
    """Loads approved research papers, verifies integrity, and provides extracted pages."""

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = manifest_path or settings.MANIFEST_PATH
        self.manifest: Dict[str, dict] = {}
        self.pages: List[PageDocument] = []
        self._load_and_verify()

    def _load_and_verify(self) -> None:
        if not self.manifest_path.exists():
            raise IngestionError(
                f"Manifest file not found at {self.manifest_path}. Run download_papers.py first."
            )

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        for paper_id, meta in self.manifest.items():
            file_path = Path(meta["file_path"])
            if not file_path.exists():
                raise IngestionError(f"Paper PDF file missing: {file_path}")

            # Verify checksum against manifest to prevent silent replacement
            actual_checksum = compute_sha256(file_path)
            expected_checksum = meta["sha256"]
            if actual_checksum != expected_checksum:
                raise IngestionError(
                    f"Checksum mismatch for {paper_id} ({file_path}): "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )

            self._extract_paper(paper_id, meta, file_path)

        logger.info(
            "Successfully ingested %d papers with %d total pages.",
            len(self.manifest),
            len(self.pages),
        )

    def _extract_paper(self, paper_id: str, meta: dict, file_path: Path) -> None:
        try:
            doc = pymupdf.open(file_path)
        except Exception as e:
            raise IngestionError(f"Failed to open PDF {file_path}: {e}") from e

        # Extract TOC: list of [lvl, title, page_num]
        toc = doc.get_toc()
        # Build page-to-heading mapping from TOC (page numbers in TOC are 1-based)
        page_headings: Dict[int, str] = {}
        current_heading = "Abstract"
        toc_idx = 0

        for p_num in range(1, len(doc) + 1):
            # Advance TOC entries that apply up to this page
            while toc_idx < len(toc) and toc[toc_idx][2] <= p_num:
                current_heading = toc[toc_idx][1].strip()
                toc_idx += 1
            page_headings[p_num] = current_heading

        for p_idx in range(len(doc)):
            page_num = p_idx + 1
            page = doc[p_idx]
            page_text = page.get_text("text").strip()

            heading = page_headings.get(page_num, "General")

            # Check if first few lines on page introduce a new heading
            lines = [l.strip() for l in page_text.split("\n") if l.strip()]
            if lines and page_num == 1:
                heading = "Title & Abstract"

            page_doc = PageDocument(
                paper_id=paper_id,
                paper_title=meta["paper_title"],
                arxiv_id=meta["arxiv_id"],
                version=meta["version"],
                page_number=page_num,
                text=page_text,
                section_heading=heading,
            )
            self.pages.append(page_doc)

        doc.close()

