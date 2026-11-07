"""Hybrid retrieval: vector leg + keyword leg, fused with RRF."""

import re

from rag.retrieval.fusion import rrf
from rag.store import keyword_search, vector_search


# E31: 30 chunks answer the most questions of the golden set, with a pool of 100 candidates per search.
DEFAULT_K = 30
DEFAULT_CANDIDATES = 100


class HybridRetriever:
    def __init__(self, conn, embedder):
        self.conn = conn
        self.embedder = embedder

    def search(self, query: str, k: int = DEFAULT_K, candidates: int = DEFAULT_CANDIDATES) -> list[dict]:
        """Top-k chunks. `candidates` is how many each leg contributes before fusion."""
        if not re.search(r"\w", query):
            return []
        query_vector = self.embedder.embed([query], kind="query")[0]
        legs = [
            vector_search(self.conn, query_vector, candidates),
            keyword_search(self.conn, query, candidates),
        ]
        return rrf(legs)[:k]

    def search_many(self, queries: list[str], k: int = DEFAULT_K, candidates: int = DEFAULT_CANDIDATES) -> list[dict]:
        """Search each phrasing (e.g. the same question in Spanish and English) and fuse the rankings.

        The embedding model matches within a language far better than across languages.
        """
        rankings = [self.search(query, k=candidates, candidates=candidates) for query in queries]
        return rrf(rankings)[:k]
