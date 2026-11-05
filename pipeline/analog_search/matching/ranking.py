"""B5: ranking INSIDE the compatibility class. Scores never change the class or the verdict.

Text relevance is built from the word score (reliable on numbers) and the meaning similarity (reliable on synonyms),
NOT from the RRF place of B2: the stage-4 measurement showed that the RRF order is worse than the word score on numbers.
"""
from __future__ import annotations

from .rules import Rules


def _minmax(values: list[float | None]) -> list[float]:
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values)
    lo, hi = min(present), max(present)
    if hi - lo < 1e-12:
        return [1.0 if v is not None else 0.0 for v in values]
    return [0.0 if v is None else (v - lo) / (hi - lo) for v in values]


def rank(items: list[dict], rules: Rules, pair_boosts: dict[str, float] | None = None) -> list[dict]:
    """`items`: dicts with keys `compat` (Compatibility), `bm25_score`, `dense_score`, `index`.

    Adds `score_parts` and `score`, returns the items sorted by (class order, score desc, index).
    """
    ranking = rules.policy["ranking"]
    w = ranking["score"]["weights"]
    mix = ranking["score"]["text_mix"]
    order = {k: i for i, k in enumerate(ranking["order"])}
    bm = [it.get("bm25_score") for it in items]
    bm_max = max([b for b in bm if b is not None] or [0.0])
    bm_norm = [(b / bm_max if (b is not None and bm_max > 0) else 0.0) for b in bm]
    dn_norm = _minmax([it.get("dense_score") for it in items])
    boosts = pair_boosts or {}
    for it, b, d in zip(items, bm_norm, dn_norm):
        text = mix["bm25"] * b + mix["dense"] * d
        attr = it["compat"].attribute_coverage
        boost = boosts.get(it.get("id"), 0.0) * ranking["score"].get("pair_boost", 0.0)
        it["score_parts"] = {"attribute_coverage": round(attr, 4), "text_relevance": round(text, 4),
                             "bm25_norm": round(b, 4), "dense_norm": round(d, 4), "pair_boost": round(boost, 4)}
        it["score"] = round(w["attribute_coverage"] * attr + w["text_relevance"] * text + boost, 4)
    class_first = ranking.get("class_first", True)             # False only for the ablation «score only»
    return sorted(items, key=lambda it: ((order.get(it["compat"].klass, len(order)) if class_first else 0), -it["score"], it["index"]))
