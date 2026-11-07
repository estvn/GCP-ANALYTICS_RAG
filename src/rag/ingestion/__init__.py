"""Offline pipeline: course files -> markdown -> chunks -> embeddings -> database."""

from rag.ingestion.parse import convert_corpus
from rag.ingestion.pipeline import ingest
from rag.ingestion.structure_semantic_merge import StructureSemanticMergeChunker

__all__ = ["StructureSemanticMergeChunker", "convert_corpus", "ingest"]
