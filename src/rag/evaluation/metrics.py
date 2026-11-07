"""Retrieval metrics. A question is answered when a top-k chunk contains one of its evidence snippets."""

from collections.abc import Callable

from rag.evaluation.golden import Question


def _squash(text: str) -> str:
    return " ".join(text.split())


def is_relevant(hit: dict, question: Question) -> bool:
    body = _squash(hit["text"])
    return any(hit["source"] == e.source and _squash(e.text) in body for e in question.relevant)


def evaluate(search: Callable[[str, int], list[dict]], questions: list[Question], k: int = 5) -> dict:
    """hit_rate: share of questions answered in the top k. mrr: mean of 1/rank of the first answer (0 beyond k)."""
    reciprocal_ranks, misses = [], []
    for question in questions:
        hits = search(question.text, k)
        rank = next((i for i, hit in enumerate(hits[:k], start=1) if is_relevant(hit, question)), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if rank is None:
            misses.append(question.id)
    n = len(questions)
    return {
        "n": n,
        "hit_rate": sum(r > 0 for r in reciprocal_ranks) / n,
        "mrr": sum(reciprocal_ranks) / n,
        "misses": misses,
    }
