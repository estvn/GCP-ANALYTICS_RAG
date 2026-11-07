"""Reciprocal Rank Fusion: merge several rankings without comparing their raw scores."""

RRF_K = 60  # damping constant from the original RRF paper


def rrf(rankings: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Each hit scores 1 / (k + position) in every list it appears in. Hits are keyed by "id"."""
    merged: dict = {}
    scores: dict = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            fields = merged.setdefault(hit["id"], {})
            for key, value in hit.items():
                if fields.get(key) is None:
                    fields[key] = value
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (k + position)
    return [{**merged[i], "score": scores[i]} for i in sorted(merged, key=scores.get, reverse=True)]
