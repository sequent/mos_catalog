"""the rule that applies to a pair is chosen by the type of the QUERY, never by the order of the file and never
by a candidate of another known family.
"""
import copy
import itertools
import random

import pytest
import yaml

from analog_search import process
from analog_search.matching import compare, load_rules
from analog_search.matching.rules import RULES_PATH, TypeContext

BASE = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
RULES = load_rules()


def load(tmp_path, mutate):
    raw = copy.deepcopy(BASE)
    mutate(raw)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_rules(p)


def rows(q, c, rules=RULES):
    comp = compare(process(q, query=True), process(c), rules)
    return {x.name: x for x in comp.comparisons}


# ---------------------------------------------------------------- the steps
def test_steps_go_from_the_query_to_the_candidate_only_when_the_query_has_no_type():
    assert list(TypeContext("кабель контрольный", "кабель", "воздуховод", "воздуховод").steps()) == [
        ("тип запроса", "кабель контрольный"), ("семейство запроса", "кабель")]
    assert list(TypeContext(None, None, "ворота откатные", "ворота").steps())[0][1] == "ворота откатные"
    assert list(TypeContext(None, None, None, None).steps()) == []


# ---------------------------------------------------------------- the candidate of another family does not decide
def test_a_candidate_of_another_known_family_does_not_set_the_order_of_sides():
    """Gates are `sorted`. A query about a cabinet must not inherit it from a gate: the order is unknown → «проверить»."""
    r = rows("шкаф 400х700", "Ворота откатные 700х400 мм")["dims"]
    assert r.status == "match_reordered" and r.rule_id is None


def test_the_rule_of_the_query_type_applies_with_its_scope():
    r = rows("ворота откатные 6000х2250", "Ворота откатные 2250х6000 мм")["dims"]
    assert r.status == "match" and r.rule_id == "gate_size" and r.rule_scope == "тип запроса"


def test_the_candidate_is_used_only_when_the_query_has_no_type():
    r = rows("6000х2250", "Ворота откатные 2250х6000 мм")["dims"]
    assert r.status == "match" and r.rule_id == "gate_size" and r.rule_scope == "тип кандидата (тип запроса неизвестен)"


def test_a_wire_is_compared_by_the_cable_rule():
    r = rows("провод ПВ 3х1.5", "Провод ПВ 3х2,5")["cores_section"]
    assert r.status == "conflict" and r.rule_id == "cable_cores_section" and r.rule_scope in ("тип запроса", "семейство запроса")


def test_a_generic_order_policy_does_not_leak_from_a_listed_type_to_a_generic_query():
    """«решетка» (generic) and «решетка жалюзийная» (listed as ordered): the query does not know the order → check, not conflict."""
    r = rows("решетка 500х600 мм", "Решетка жалюзийная 600х500 мм")["dims"]
    assert r.status == "match_reordered"
    r = rows("решетка жалюзийная 500х600 мм", "Решетка жалюзийная 600х500 мм")["dims"]
    assert r.status == "conflict"                                              # both know the type: ordered


# ---------------------------------------------------------------- specific beats general; equals are refused
def two_rules(raw):
    base = next(r for r in raw["rules"] if r["id"] == "cable_cores_section")
    for r in raw["rules"]:
        if r["id"] in ("cable_cores_section",):
            r["applies_to"] = ["кабель"]
    raw["rules"].append({"id": "control_cable_cores", "applies_to": ["кабель контрольный"], "attribute": "cores_section", "mode": "flag", "title": "x"})
    return base


def test_the_rule_of_a_type_beats_the_rule_of_its_family(tmp_path):
    r = load(tmp_path, two_rules)
    control = rows("контрольный кабель КВВГнг 5х2.5", "Кабель контрольный КВВГнг 5х1,5", r)["cores_section"]
    assert control.rule_id == "control_cable_cores" and control.status == "flag"
    general = rows("кабель 5х2.5", "Кабель контрольный КВВГнг 5х1,5", r)["cores_section"]
    assert general.rule_id == "cable_cores_section" and general.status == "conflict"


def test_two_rules_of_the_same_specificity_are_refused_with_both_ids(tmp_path):
    def add(raw):
        raw["rules"].append({"id": "second_section_rule", "applies_to": ["кабель"], "attribute": "cores_section", "mode": "flag", "title": "x"})
    with pytest.raises(ValueError, match="cable_cores_section.*second_section_rule|second_section_rule.*cable_cores_section"):
        load(tmp_path, add)


# ---------------------------------------------------------------- the order of the file does not matter
PAIRS = [
    ("кабель КВВГнг-LS 5*2.5", "Кабель контрольный КВВГнг(A)-LS 5х1,5"), ("камера 3 Мп", "Видеокамера уличная 2Мп"),
    ("ворота откатные 6000х2250", "Ворота откатные 2250х6000 мм"), ("шкаф 400х700", "Ворота откатные 700х400 мм"),
    ("кабель контрольный КВВГнг 5х2.5 экранированный", "Кабель контрольный КВВГЭнг 5х2,5"), ("провод 3х1.5", "Провод ПВ 3х2,5"),
    ("видеокамера 4 Мп", "Видеокамера 5Мп"), ("воздуховод 500х300", "Воздуховод 300х500"),
]


def snapshot(rules):
    out = []
    for q, c in PAIRS:
        comp = compare(process(q, query=True), process(c), rules)
        out.append((comp.klass, tuple(sorted((x.name, x.status, x.rule_id, x.rule_scope) for x in comp.comparisons))))
    return out


def test_shuffling_the_rules_in_the_file_changes_nothing(tmp_path):
    expected = snapshot(RULES)
    rnd = random.Random(7)
    for i in range(6):
        def shuffle(raw, rnd=rnd):
            rnd.shuffle(raw["rules"])
            for r in raw["rules"]:
                rnd.shuffle(r["applies_to"])
        assert snapshot(load(tmp_path, shuffle)) == expected


def test_every_permutation_of_two_overlapping_family_rules_gives_one_answer(tmp_path):
    answers = set()
    for order in itertools.permutations(range(2)):
        def reorder(raw, order=order):
            two_rules(raw)
            keep = [r for r in raw["rules"] if r["attribute"] != "cores_section"]
            sect = [r for r in raw["rules"] if r["attribute"] == "cores_section"]
            raw["rules"] = keep + [sect[i] for i in order]
        r = load(tmp_path, reorder)
        answers.add(tuple(snapshot(r)))
    assert len(answers) == 1


def test_the_scope_reaches_the_answer_of_the_engine():
    from analog_search.matching.explain import explain
    q, c = process("ворота откатные 6000х2250", query=True), process("Ворота откатные 2250х6000 мм")
    row = next(x for x in explain(compare(q, c, RULES), q, c)["rows"] if x["attribute"] == "dims")
    assert row["rule"] == "gate_size" and row["rule_scope"] == "тип запроса"
