"""Research retrieval implementation with BM25, TF-IDF, and Reranking."""

import math
import re
from typing import Dict, List, Optional
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from app.core.models import ChunkMetadata, RetrievalConfig
from app.retrieval.chunking import DocumentChunker
from app.retrieval.ingestion import DocumentStore


def tokenize(text: str) -> List[str]:
    """Simple lowercase alphanumeric tokenizer."""
    return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())


class ResearchRetriever:
    """Manages chunk index and executes lexical and reranked searches."""

    def __init__(self, doc_store: DocumentStore, config: RetrievalConfig):
        self.doc_store = doc_store
        self.config = config
        self.chunker = DocumentChunker(
            chunk_size_words=config.chunk_size_words,
            chunk_overlap_words=config.chunk_overlap_words,
        )
        self.chunks: List[ChunkMetadata] = self.chunker.chunk_pages(doc_store.pages)
        self.corpus_tokens = [tokenize(c.chunk_text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

        # Prepare TF-IDF if reranking is enabled
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        if config.use_reranking and self.chunks:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words="english", token_pattern=r"(?u)\b\w+\b"
            )
            corpus_texts = [f"{c.section_heading} {c.chunk_text}" for c in self.chunks]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[ChunkMetadata]:
        if not self.chunks or not self.bm25:
            return []

        k = top_k or self.config.top_k
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        bm25_scores = self.bm25.get_scores(query_tokens)

        # Basic filtering and ranking
        ranked_indices = np.argsort(bm25_scores)[::-1]

        if not self.config.use_reranking:
            results = []
            for idx in ranked_indices[:k]:
                score = float(bm25_scores[idx])
                if score <= 0.0:
                    continue
                chunk = self.chunks[idx].model_copy()
                chunk.score = score
                results.append(chunk)
            return results

        # Optional Reranking: Reciprocal Rank Fusion of BM25 and TF-IDF + Section Heading Boost
        candidate_count = min(len(self.chunks), max(k * 3, 20))
        top_candidates = ranked_indices[:candidate_count]

        # Calculate TF-IDF cosine similarity for top candidates
        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix[top_candidates] * query_vec.T).toarray().ravel()

        bm25_subset = bm25_scores[top_candidates]
        # Normalize BM25 and TF-IDF scores
        max_bm25 = np.max(bm25_subset) if len(bm25_subset) > 0 and np.max(bm25_subset) > 0 else 1.0
        max_tfidf = np.max(tfidf_scores) if len(tfidf_scores) > 0 and np.max(tfidf_scores) > 0 else 1.0

        reranked_scores = []
        for i, idx in enumerate(top_candidates):
            norm_bm25 = bm25_subset[i] / max_bm25
            norm_tfidf = tfidf_scores[i] / max_tfidf

            # Boost if query terms appear in the section heading
            heading = self.chunks[idx].section_heading.lower()
            heading_boost = 0.0
            for qt in query_tokens:
                if len(qt) > 3 and qt in heading:
                    heading_boost += 0.15

            combined_score = 0.5 * norm_bm25 + 0.35 * norm_tfidf + heading_boost
            reranked_scores.append((combined_score, idx))

        reranked_scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in reranked_scores[:k]:
            if score <= 0.0:
                continue
            chunk = self.chunks[idx].model_copy()
            chunk.score = float(score)
            results.append(chunk)

        return results

