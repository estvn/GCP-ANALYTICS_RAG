"""Hybrid retrieval: vector leg + keyword leg, fused with RRF."""

import re

from rag.retrieval.fusion import rrf
from rag.store import keyword_search, vector_search


class HybridRetriever:
    def __init__(self, conn, embedder):
        self.conn = conn
        self.embedder = embedder

    def search(self, query: str, k: int = 5, candidates: int = 30) -> list[dict]:
        """Top-k chunks. `candidates` is how many each leg contributes before fusion."""
        if not re.search(r"\w", query):
            return []
        query_vector = self.embedder.embed([query], kind="query")[0]
        legs = [
            vector_search(self.conn, query_vector, candidates),
            keyword_search(self.conn, query, candidates),
        ]
        return rrf(legs)[:k]
