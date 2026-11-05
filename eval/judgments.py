"""Independent judgment of technical errors.

The old metric `key_differs` read the attributes that OUR extractor produced: when the extractor missed a value, it answered «no error».
Here the critical values of a candidate are written by a PERSON (transcribed from the xlsx while looking at the original name) in
`eval/judgments.yaml`, and the comparison below uses nothing from analog_search: not the extractor, not the rules, not the normaliser.
The result has three values: `conflict`, `no_conflict`, `unknown` (no judgment is not «no error»).
"""
from __future__ import annotations

from pathlib import Path

import yaml

JUDGMENTS_PATH = Path(__file__).resolve().parent / "judgments.yaml"


def _same(a, b, order_sensitive: bool) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        x, y = (list(a), list(b)) if order_sensitive else (sorted(a), sorted(b))
        return all(_same(i, j, True) for i, j in zip(x, y))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def constraint_verdict(constraint: dict, critical_values: dict) -> str:
    """One required constraint of the query against the critical values of a candidate: conflict | no_conflict | unknown.

    constraint: {attribute, expected, comparison: equal | allowed_values, allowed_values: [...], order_sensitive: bool}"""
    attr = constraint["attribute"]
    if attr not in critical_values:
        return "unknown"                                    # nobody wrote the candidate's value: this is NOT «no conflict»
    have = critical_values[attr]
    order = bool(constraint.get("order_sensitive", False))
    if constraint.get("comparison", "equal") == "allowed_values":
        ok = any(_same(have, v, order) for v in constraint.get("allowed_values", [])) or _same(have, constraint["expected"], order)
    else:
        ok = _same(have, constraint["expected"], order)
    return "no_conflict" if ok else "conflict"


def pair_verdict(constraints: list[dict], pair: dict | None) -> str:
    """The verdict for a (query, candidate) pair from the written judgment. A conflict in any constraint wins; one unknown makes
    the whole pair unknown unless a conflict is already certain."""
    if pair is None:
        return "unknown"
    if pair.get("relevance") == "unacceptable":
        return "conflict"
    verdicts = [constraint_verdict(c, pair.get("critical_values", {})) for c in constraints]
    if "conflict" in verdicts:
        return "conflict"
    if not verdicts or "unknown" in verdicts:
        return "unknown" if pair.get("relevance") != "acceptable" else "no_conflict"
    return "no_conflict"


class Judgments:
    def __init__(self, data: dict | None):
        data = data or {}
        self.version = data.get("version")
        self.constraints = {qid: (v or {}).get("constraints", []) for qid, v in (data.get("queries") or {}).items()}
        self.pairs = {(p["query_id"], p["candidate_id"]): p for p in (data.get("pairs") or [])}

    @classmethod
    def load(cls, path: Path | str = JUDGMENTS_PATH) -> "Judgments":
        p = Path(path)
        return cls(yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None)

    def verdict(self, query_id: str, candidate_id: str | None) -> str:
        if candidate_id is None:
            return "unknown"
        return pair_verdict(self.constraints.get(query_id, []), self.pairs.get((query_id, candidate_id)))
