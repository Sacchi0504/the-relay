"""Text chunking and metadata preservation for research papers."""

import hashlib
import re
from typing import List
from app.core.models import ChunkMetadata
from app.retrieval.ingestion import PageDocument


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of chunk text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentChunker:
    """Splits ingested pages into research chunks with full source metadata."""

    def __init__(self, chunk_size_words: int = 500, chunk_overlap_words: int = 100):
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = max(0, min(chunk_overlap_words, chunk_size_words - 50))

    def chunk_pages(self, pages: List[PageDocument]) -> List[ChunkMetadata]:
        chunks: List[ChunkMetadata] = []

        for page in pages:
            page_text = page.text
            if not page_text or len(page_text.strip()) < 30:
                continue

            words = page_text.split()
            if len(words) <= self.chunk_size_words:
                # Page fits in a single chunk
                chunk_id = f"{page.paper_id}_p{page.page_number}_c1"
                chunk_text = " ".join(words)
                chunks.append(
                    ChunkMetadata(
                        paper_id=page.paper_id,
                        paper_title=page.paper_title,
                        arxiv_id=page.arxiv_id,
                        version=page.version,
                        page_number=page.page_number,
                        section_heading=page.section_heading,
                        chunk_id=chunk_id,
                        chunk_text=chunk_text,
                        content_hash=compute_content_hash(chunk_text),
                    )
                )
            else:
                # Sliding window of words
                step = self.chunk_size_words - self.chunk_overlap_words
                c_idx = 1
                for start in range(0, len(words), step):
                    chunk_words = words[start : start + self.chunk_size_words]
                    if len(chunk_words) < 30 and start > 0:
                        # Skip tiny trailing fragment
                        continue

                    chunk_id = f"{page.paper_id}_p{page.page_number}_c{c_idx}"
                    chunk_text = " ".join(chunk_words)
                    chunks.append(
                        ChunkMetadata(
                            paper_id=page.paper_id,
                            paper_title=page.paper_title,
                            arxiv_id=page.arxiv_id,
                            version=page.version,
                            page_number=page.page_number,
                            section_heading=page.section_heading,
                            chunk_id=chunk_id,
                            chunk_text=chunk_text,
                            content_hash=compute_content_hash(chunk_text),
                        )
                    )
                    c_idx += 1

        return chunks

