"""Offline pipeline: course files -> markdown -> chunks -> embeddings -> database."""

from rag.ingestion.parse import convert_corpus
from rag.ingestion.pipeline import ingest
from rag.ingestion.semantic import SemanticChunker

__all__ = ["SemanticChunker", "convert_corpus", "ingest"]
