"""Offline pipeline: course files -> markdown -> chunks -> embeddings -> database."""

from rag.ingestion.parse import convert_corpus
from rag.ingestion.pipeline import ingest
from rag.ingestion.structure_then_semantic import StructureThenSemanticChunker

__all__ = ["StructureThenSemanticChunker", "convert_corpus", "ingest"]
