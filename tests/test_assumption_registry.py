"""every assumption has an id, a source and a status; nobody is credited with an approval nobody gave;
the register points at real places of the configuration and the README shows it."""
import copy
from pathlib import Path

import pytest
import yaml

from analog_search.matching import load_rules
from analog_search.matching.rules import ASSUMPTION_STATUSES, ASSUMPTIONS_PATH, RULES_PATH, load_assumptions

ROOT = Path(__file__).resolve().parents[1]
REG = yaml.safe_load(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))["assumptions"]
RULES = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
DICT = yaml.safe_load((ROOT / "config" / "dictionaries.yaml").read_text(encoding="utf-8"))


def test_every_assumption_is_complete():
    ids = [a["id"] for a in REG]
    assert len(ids) == len(set(ids)) and ids == sorted(ids)
    for a in REG:
        assert a["statement"].strip() and a["source"].strip() and a["status"] in ASSUMPTION_STATUSES, a["id"]
        assert "affected_types" in a and "examples" in a and a["config_refs"], a["id"]


def test_nobody_is_credited_with_an_approval_that_nobody_gave():
    """Nothing in the shipped register is approved by a specialist: the file says so honestly."""
    assert all(a["status"] == "author_assumption" for a in REG)
    assert not any(a.get("approved_by") or a.get("approved_on") for a in REG)


def test_an_approval_needs_a_name_and_a_date(tmp_path):
    raw = yaml.safe_load(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))
    raw["assumptions"][0]["status"] = "specialist_approved"
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="approved_by и approved_on"):
        load_assumptions(p)
    raw["assumptions"][0].update(approved_by="Иванов И. И.", approved_on="2026-11-10")
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert load_assumptions(p)["A01"]["status"] == "specialist_approved"


def test_an_unknown_status_is_refused(tmp_path):
    raw = yaml.safe_load(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))
    raw["assumptions"][1]["status"] = "approved"
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="status"):
        load_assumptions(p)


def resolve(ref):
    kind, _, path = ref.partition(":")
    if kind == "rules":
        return next((r for r in RULES["rules"] if r["id"] == path), None) is not None
    if kind in ("dims_order", "defaults", "policy"):
        node = RULES[kind]
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True
    if kind == "dictionaries":
        node = DICT
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True
    return False


@pytest.mark.parametrize("a", REG, ids=[a["id"] for a in REG])
def test_every_reference_points_at_a_real_place(a):
    for ref in a["config_refs"]:
        assert resolve(ref), f"{a['id']}: {ref} does not exist in the configuration"


def test_every_rule_names_an_assumption_that_exists():
    ids = {a["id"] for a in REG}
    for r in RULES["rules"]:
        assert r.get("assumption") in ids, r["id"]


def test_an_unknown_assumption_in_a_rule_stops_the_load(tmp_path):
    raw = copy.deepcopy(RULES)
    raw["rules"][0]["assumption"] = "A99"
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="A99"):
        load_rules(p)


def test_the_readme_shows_every_assumption_with_its_status():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for a in REG:
        assert f"| {a['id']} |" in readme, a["id"]
    assert "author_assumption" in readme and "specialist_approved" in readme
