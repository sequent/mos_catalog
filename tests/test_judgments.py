"""the technical error is judged from values written by people, never by our extractor."""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from judgments import Judgments, constraint_verdict, pair_verdict  # noqa: E402


def test_the_comparator_imports_nothing_of_the_system():
    """The independence is structural: judgments.py must not use the extractor, the rules or the normaliser."""
    tree = ast.parse((ROOT / "eval" / "judgments.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and "analog_search" in m for m in imported)


# ---------------------------------------------------------------- a documented example
POWER_100 = [{"attribute": "power_kw", "expected": 0.1, "comparison": "equal"}]


def test_a_300_w_lamp_for_a_100_w_query_is_a_conflict_even_if_the_extractor_is_broken(monkeypatch):
    import analog_search.attributes as attrs
    monkeypatch.setattr(attrs, "extract_attributes", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken extractor")))
    pair = {"query_id": "X07", "candidate_id": "row-000331", "critical_values": {"power_kw": 0.3}}
    assert pair_verdict(POWER_100, pair) == "conflict"
    assert pair_verdict(POWER_100, {"critical_values": {"power_kw": 0.1}}) == "no_conflict"


def test_a_value_nobody_wrote_is_unknown_not_no_conflict():
    assert constraint_verdict(POWER_100[0], {}) == "unknown"
    assert pair_verdict(POWER_100, {"critical_values": {}}) == "unknown"
    assert pair_verdict(POWER_100, None) == "unknown"                                  # no judgment at all


def test_a_certain_conflict_wins_over_an_unknown_value():
    cs = POWER_100 + [{"attribute": "dims", "expected": [600, 600]}]
    assert pair_verdict(cs, {"critical_values": {"power_kw": 0.3}}) == "conflict"      # dims unknown, power conflicts
    assert pair_verdict(cs, {"critical_values": {"power_kw": 0.1}}) == "unknown"       # power fine, dims unknown


# ---------------------------------------------------------------- sides, allowed values, the decision of a person
def test_the_order_of_sides_conflicts_when_it_matters_and_not_when_it_does_not():
    ordered = {"attribute": "dims", "expected": [600, 400], "order_sensitive": True}
    free = {"attribute": "dims", "expected": [600, 400], "order_sensitive": False}
    assert constraint_verdict(ordered, {"dims": [400, 600]}) == "conflict"
    assert constraint_verdict(free, {"dims": [400, 600]}) == "no_conflict"
    assert constraint_verdict(ordered, {"dims": [600, 400]}) == "no_conflict"


def test_a_step_allowed_by_the_reference_method_is_not_an_error():
    cam = {"attribute": "resolution_mp", "expected": 3, "comparison": "allowed_values", "allowed_values": [2, 3, 4]}
    assert constraint_verdict(cam, {"resolution_mp": 2}) == "no_conflict"
    assert constraint_verdict(cam, {"resolution_mp": 8}) == "conflict"


def test_the_decision_of_a_person_counts():
    assert pair_verdict([], {"relevance": "unacceptable"}) == "conflict"
    assert pair_verdict(POWER_100, {"relevance": "unacceptable", "critical_values": {"power_kw": 0.1}}) == "conflict"
    assert pair_verdict(POWER_100, {"relevance": "acceptable", "critical_values": {}}) == "no_conflict"
    assert pair_verdict([], {"relevance": "unknown"}) == "unknown"


def test_numbers_are_compared_as_numbers_not_as_strings():
    assert constraint_verdict({"attribute": "length_mm", "expected": 3000}, {"length_mm": 3000.0}) == "no_conflict"
    assert constraint_verdict({"attribute": "ip", "expected": "IP66"}, {"ip": "IP65"}) == "conflict"


def test_the_judgments_file_is_loaded_and_a_missing_one_is_empty(tmp_path):
    j = Judgments.load(tmp_path / "nothing.yaml")
    assert j.verdict("Q01", "row-000001") == "unknown" and j.verdict("Q01", None) == "unknown"
    p = tmp_path / "j.yaml"
    p.write_text("version: 1\nqueries:\n  X07:\n    constraints:\n      - {attribute: power_kw, expected: 0.1}\n"
                 "pairs:\n  - {query_id: X07, candidate_id: row-000331, critical_values: {power_kw: 0.3}}\n", encoding="utf-8")
    j = Judgments.load(p)
    assert j.verdict("X07", "row-000331") == "conflict" and j.verdict("X07", "row-000999") == "unknown"


def test_the_shipped_judgments_file_is_valid_and_honestly_empty():
    j = Judgments.load()
    assert j.version == 1 and j.pairs == {}                                            # nothing is claimed that nobody judged
