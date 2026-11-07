"""Vector retrieval: embed the question and take the nearest chunks."""

import re

from rag.store import vector_search


class VectorRetriever:
    def __init__(self, conn, embedder):
        self.conn = conn
        self.embedder = embedder

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Top-k chunks by cosine distance to the query."""
        if not re.search(r"\w", query):
            return []
        query_vector = self.embedder.embed([query], kind="query")[0]
        return vector_search(self.conn, query_vector, k)
