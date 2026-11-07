"""Reciprocal Rank Fusion: merge several rankings without comparing their raw scores."""

from collections.abc import Sequence

RRF_K = 60  # damping constant from the original RRF paper


def rrf(rankings: list[list[dict]], k: int = RRF_K, weights: Sequence[float] | None = None) -> list[dict]:
    """Each hit scores weight / (k + position) in every list it appears in. Hits are keyed by "id".

    `weights` holds one weight per list (None = all 1, plain RRF).
    """
    weights = [1] * len(rankings) if weights is None else weights
    if len(weights) != len(rankings):
        raise ValueError(f"{len(weights)} weights for {len(rankings)} rankings")
    merged: dict = {}
    scores: dict = {}
    for weight, ranking in zip(weights, rankings):
        for position, hit in enumerate(ranking, start=1):
            fields = merged.setdefault(hit["id"], {})
            for key, value in hit.items():
                if fields.get(key) is None:
                    fields[key] = value
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + weight / (k + position)
    return [{**merged[i], "score": scores[i]} for i in sorted(merged, key=scores.get, reverse=True)]
