"""config/rules.yaml: validation (loud errors) and the automatic check of every `examples` entry.

This is the check a specialist runs after editing a rule: every example «query / candidate / expect» of every rule
is executed through the real comparison and must give the stated status of the rule's characteristic.
"""
import copy
from pathlib import Path

import pytest
import yaml

from analog_search import process
from analog_search.matching import compare, load_rules
from analog_search.matching.rules import RULES_PATH

RULES = load_rules()
EXAMPLES = [(r.id, r.attribute, e) for r in RULES.rules for e in r.examples]
EXPECT_TO_STATUS = {"match": "match", "conflict": "conflict", "flag": "flag", "substitute": "substitute"}


@pytest.mark.parametrize("rule_id,attribute,example", EXAMPLES, ids=[f"{r}:{e['expect']}:{e['query']}" for r, _, e in EXAMPLES])
def test_rule_examples(rule_id, attribute, example):
    comp = compare(process(example["query"]), process(example["candidate"]), RULES)
    got = {c.name: c.status for c in comp.comparisons}
    assert attribute in got, f"характеристика {attribute} не найдена в запросе «{example['query']}»: {got}"
    assert got[attribute] == EXPECT_TO_STATUS[example["expect"]], (rule_id, example, got)


def test_every_rule_has_examples():
    assert all(r.examples for r in RULES.rules), "у каждого правила должны быть примеры «да/нет»"


def test_current_file_is_valid_and_marked_as_assumption():
    assert RULES.version and RULES.status in ("assumption", "approved_by_specialist")


# ---------------------------------------------------------------- the loader must be loud
def _write(tmp_path, mutate):
    raw = copy.deepcopy(yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")))
    mutate(raw)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return p


def test_unknown_mode_is_rejected(tmp_path):
    p = _write(tmp_path, lambda r: r["rules"][0].update(mode="strict"))
    with pytest.raises(ValueError, match="неизвестный режим"):
        load_rules(p)


def test_ladder_needs_steps(tmp_path):
    def mutate(r):
        rule = next(x for x in r["rules"] if x["mode"] == "ladder")
        del rule["ladder"]
    with pytest.raises(ValueError, match="ступеней"):
        load_rules(_write(tmp_path, mutate))


def test_duplicate_ids_are_rejected(tmp_path):
    p = _write(tmp_path, lambda r: r["rules"][1].update(id=r["rules"][0]["id"]))
    with pytest.raises(ValueError, match="повторяется"):
        load_rules(p)


def test_type_in_both_dims_lists_is_rejected(tmp_path):
    p = _write(tmp_path, lambda r: r["dims_order"]["ordered"].append("ворота"))
    with pytest.raises(ValueError, match="обоих списках"):
        load_rules(p)


def test_missing_type_relation_is_rejected(tmp_path):
    p = _write(tmp_path, lambda r: r["policy"]["type"]["relation"].pop("same"))
    with pytest.raises(ValueError, match="policy.type.relation"):
        load_rules(p)


def test_changing_a_rule_changes_the_result(tmp_path):
    """The specialist's workflow: edit the file, the behaviour changes, no code is touched."""
    q, c = process("камера 4 Мп"), process("видеокамера 2Мп")
    assert {x.name: x.status for x in compare(q, c, RULES).comparisons}["resolution_mp"] == "conflict"
    relaxed = load_rules(_write(tmp_path, lambda r: next(x for x in r["rules"] if x["id"] == "camera_resolution").update(max_steps=2)))
    assert {x.name: x.status for x in compare(q, c, relaxed).comparisons}["resolution_mp"] == "substitute"
