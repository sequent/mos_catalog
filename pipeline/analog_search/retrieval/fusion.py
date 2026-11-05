"""Reciprocal Rank Fusion: combines ranked lists without tuning score scales."""
from __future__ import annotations


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """`rankings` are lists of ids, best first. Score(id) = Σ weight / (k + rank), rank starting at 1.

    Returns (id, score) sorted by score desc; ties are broken by id for determinism.
    """
    score: dict[int, float] = {}
    weights = weights or [1.0] * len(rankings)
    for w, ranking in zip(weights, rankings):
        for rank, i in enumerate(ranking, start=1):
            score[i] = score.get(i, 0.0) + w / (k + rank)
    return sorted(score.items(), key=lambda t: (-t[1], t[0]))
