"""Offline pipeline: course files -> markdown -> chunks -> embeddings -> database."""

from rag.ingestion.chunk import StructuralChunker, chunk_markdown
from rag.ingestion.parse import convert_corpus
from rag.ingestion.pipeline import ingest

__all__ = ["StructuralChunker", "chunk_markdown", "convert_corpus", "ingest"]
