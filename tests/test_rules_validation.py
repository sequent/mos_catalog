"""the configuration file is a contract. Whatever the specialist can write must either work or stop the load
with the path of the field and the values that are allowed: nothing is silently ignored.
"""
import copy
import re
from pathlib import Path

import pytest
import yaml

from analog_search.matching import compare, load_rules
from analog_search.matching.rules import ATTRIBUTE_MODES, RULES_PATH
from analog_search import process

ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))


def load(tmp_path, mutate):
    raw = copy.deepcopy(BASE)
    mutate(raw)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_rules(p)


def rule(raw, rid):
    return next(r for r in raw["rules"] if r["id"] == rid)


def test_the_shipped_configuration_is_valid():
    assert load_rules().version


# ---------------------------------------------------------------- the load stops, naming the field
@pytest.mark.parametrize("mutate,message", [
    (lambda r: rule(r, "cable_cores_section").update(attribute="cores_sectoin"), "неизвестная характеристика «cores_sectoin»"),
    (lambda r: r["defaults"]["per_attribute"].update(materail="exact"), "defaults.per_attribute: неизвестная характеристика «materail»"),
    (lambda r: r["defaults"]["per_attribute"].update(material="flag"), "defaults.per_attribute.material: режим «flag» не поддержан; работают: exact, ignore"),
    (lambda r: r["defaults"]["per_attribute"].update(material="ladder"), "material"),
    (lambda r: rule(r, "gate_size").update(mode="min_ge"), "для «dims» режим «min_ge» не поддержан"),
    (lambda r: rule(r, "cable_cores_section").update(mode="ladder", ladder=[1, 2], max_steps=1), "для «cores_section» режим «ladder» не поддержан"),
    (lambda r: rule(r, "cable_cores_section").update(mode="min_ge"), "режим «min_ge» не поддержан"),
    (lambda r: rule(r, "camera_resolution").update(mode="whatever"), "неизвестный режим"),
    (lambda r: rule(r, "camera_resolution").update(applies_to=["видеокамеры"]), "applies_to: неизвестные типы (их нет в config/dictionaries.yaml): видеокамеры"),
    (lambda r: rule(r, "camera_resolution").update(colour="x"), "неизвестные поля"),
    (lambda r: rule(r, "camera_resolution").update(dims="sorted"), "dims относится только к характеристике dims"),
    (lambda r: r["dims_order"]["sorted"].append("несуществующий"), "dims_order.sorted: неизвестные типы"),
    (lambda r: r["policy"]["composite"].update(max_compatibility="compatible"), "policy.composite.max_compatibility: допустимо unverified или null"),
    (lambda r: r["policy"]["composite"].update(max_compatibility=True), "policy.composite.max_compatibility"),
    (lambda r: r["policy"]["ui"].update(review_table_limit=-1), "policy.ui.review_table_limit: нужно целое число ≥ 0"),
    (lambda r: r["policy"]["ui"].update(review_rows_when_found=1.5), "policy.ui.review_rows_when_found"),
    (lambda r: r["policy"]["not_found"].update(unconfirmed_limit=-3), "policy.not_found.unconfirmed_limit"),
    (lambda r: r["policy"]["ranking"]["score"]["weights"].update(text_relevance=-0.1), "policy.ranking.score.weights.text_relevance"),
    (lambda r: r["policy"]["ranking"]["score"]["weights"].update(attribute_coverage=float("inf")), "policy.ranking.score.weights.attribute_coverage"),
    (lambda r: r["policy"]["ranking"]["score"]["weights"].pop("text_relevance"), "нет обязательного поля «text_relevance»"),
    (lambda r: r["policy"]["ranking"].update(order=["compatible", "unknown_class"]), "policy.ranking.order: неизвестные классы"),
    (lambda r: r["policy"]["verdict"]["map"].pop("incompatible"), "policy.verdict.map: нет обязательного поля «incompatible»"),
    (lambda r: r["policy"]["verdict"]["map"].update(incompatible="delete"), "policy.verdict.map.incompatible: недопустимый вердикт"),
    (lambda r: r["policy"]["word_coverage"].update(min_for_confirmed=1.5), "policy.word_coverage.min_for_confirmed"),
    (lambda r: r["policy"]["word_coverage"].update(type_word_required="yes"), "policy.word_coverage.type_word_required: нужно true или false"),
    (lambda r: r["policy"]["evidence"].update(weak_attributes=["colour"]), "policy.evidence.weak_attributes: неизвестные характеристики"),
    (lambda r: r["policy"]["evidence"].update(confirm_by_type=["несуществующий тип"]), "confirm_by_type"),
    (lambda r: r["policy"]["unparsed_constraints"].update(blocking_reasons=["number_with_unit", "oops"]), "неизвестные причины: oops"),
    (lambda r: r["policy"]["ui"]["status_lines"].update(found="Найдено"), "должно быть {n}"),
    (lambda r: r["policy"]["type"]["relation"].update(same="maybe"), "policy.type.relation.same"),
    (lambda r: r["policy"]["type"]["relation"].pop("unknown_either_side"), "нет записей для unknown_either_side"),
])
def test_a_mistake_stops_the_load_with_the_path_of_the_field(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        load(tmp_path, mutate)


def test_a_typo_in_a_policy_key_is_not_ignored(tmp_path):
    with pytest.raises(ValueError, match="policy.ui: неизвестные поля"):
        load(tmp_path, lambda r: r["policy"]["ui"].update(review_table_limt=3))
    with pytest.raises(ValueError, match="policy: неизвестные поля"):
        load(tmp_path, lambda r: r["policy"].update(feedbak={}))


@pytest.mark.parametrize("removed", [("policy.type", "hard_drop"), ("policy.composite", "searchable"), ("policy.not_found", "min_text_relevance"),
                                     ("policy.ranking", "explain"), ("policy", "services"), ("policy", "feedback"), ("defaults", "unknown_type")])
def test_the_promises_that_nothing_kept_are_gone_and_cannot_come_back_silently(tmp_path, removed):
    """These fields once appeared in the file but no code read them; writing them again must be an error, not a no-op."""
    path, key = removed

    def add(raw):
        node = raw
        for part in path.split("."):
            node = node[part]
        node[key] = {} if key in ("services", "feedback") else 1
    with pytest.raises(ValueError, match="неизвестные поля"):
        load(tmp_path, add)


# ---------------------------------------------------------------- what is allowed really works
def test_the_ladder_tolerance_changes_the_answer(tmp_path):
    q, c = process("камера 4 Мп", query=True), process("Видеокамера уличная 2Мп")
    assert {x.name: x.status for x in compare(q, c, load_rules()).comparisons}["resolution_mp"] == "conflict"
    r = load(tmp_path, lambda raw: rule(raw, "camera_resolution").update(max_steps=2))
    assert {x.name: x.status for x in compare(q, c, r).comparisons}["resolution_mp"] == "substitute"


def test_ignore_removes_only_the_chosen_comparison(tmp_path):
    r = load(tmp_path, lambda raw: rule(raw, "cable_cores_section").update(mode="ignore"))
    comp = compare(process("кабель КВВГнг-LS 5*2.5", query=True), process("Кабель контрольный КВВГнг(A)-LS 5х1,5"), r)
    names = {x.name for x in comp.comparisons}
    assert "cores_section" not in names and "fire_class" in names


@pytest.mark.parametrize("attr,mode", [(a, m) for a, modes in ATTRIBUTE_MODES.items() for m in modes if m in ("exact", "flag", "ignore", "info", "ambiguous")][:60])
def test_every_mode_the_registry_allows_loads(tmp_path, attr, mode):
    load(tmp_path, lambda r: r["defaults"]["per_attribute"].update({attr: mode}))


# ---------------------------------------------------------------- no promise without a reader
def _leaf_keys(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, dict) and k not in ("relation", "map", "status_lines", "weights", "text_mix", "per_attribute"):
                _leaf_keys(v, out)
            elif isinstance(v, dict):
                out.add(k)
                out.update(v.keys())
            else:
                out.add(k)


def test_every_field_of_the_policy_has_a_reader_in_the_code():
    """A field shown to the specialist as a switch must be read by some module (not only by the loader that validates it)."""
    keys: set[str] = set()
    _leaf_keys(BASE["policy"], keys)
    _leaf_keys({"defaults": BASE["defaults"]}, keys)
    sources = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "pipeline" / "analog_search").rglob("*.py")
                        if p.name != "rules.py")
    unread = sorted(k for k in keys if k not in sources and not k.startswith("_"))
    assert not unread, f"поля без потребителя в коде: {unread}"
