"""Offline pipeline: course files -> markdown -> chunks -> embeddings -> database."""

from rag.ingestion.chunk import chunk_markdown
from rag.ingestion.parse import convert_corpus
from rag.ingestion.pipeline import ingest

__all__ = ["chunk_markdown", "convert_corpus", "ingest"]
