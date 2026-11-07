"""Offline pipeline: course files -> markdown -> chunks -> embeddings -> database."""

from rag.ingestion.parse import convert_corpus
from rag.ingestion.pipeline import ingest
from rag.ingestion.structural import StructuralChunker, chunk_markdown

__all__ = ["StructuralChunker", "chunk_markdown", "convert_corpus", "ingest"]
