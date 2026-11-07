"""Keyword retrieval: full-text match on the words that discriminate between chunks."""

from rag.store import keyword_search


class KeywordRetriever:
    def __init__(self, conn):
        self.conn = conn

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Top-k chunks sharing a discriminating word with the query."""
        return keyword_search(self.conn, query, k)
