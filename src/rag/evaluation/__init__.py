"""Measure retrieval quality against a golden set."""

from rag.evaluation.golden import Evidence, Question, load_golden
from rag.evaluation.metrics import evaluate, is_relevant

__all__ = ["Evidence", "Question", "evaluate", "is_relevant", "load_golden"]
