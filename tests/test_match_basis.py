"""what a positive answer rests on, and what the answer says it does not know.

`same` is a historical machine name; the user must never read it as «the same product» unless the whole name is the same.
"""
import re
from pathlib import Path

import pytest

from analog_search import process
from analog_search.matching import compare, load_rules

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules()
BUILT = (ROOT / "data" / "processed" / "catalog_v2.jsonl").exists() and (ROOT / "data" / "index" / "dense.npy").exists() \
    and (ROOT / "models" / "multilingual-e5-small" / "onnx" / "model.onnx").exists()
real = pytest.mark.skipif(not BUILT, reason="catalog, index or model are not built")


def comp(q, c):
    return compare(process(q, query=True), process(c), RULES)


# ---------------------------------------------------------------- the basis of a match
def test_an_incomplete_query_matches_by_its_constraints_only():
    for c in ("Труба ПНД 25 мм", "Труба ПВХ 25 мм"):
        r = comp("труба 25 мм", c)
        assert r.klass == "compatible" and r.match_basis == "request_constraints"


def test_the_same_normalised_name_is_the_only_identity_ground():
    r = comp("Ворота откатные 6000х2250 мм", "Ворота откатные 6000х2250 мм")
    assert r.match_basis == "normalized_name"
    r = comp("ворота ОТКАТНЫЕ  6000x2250 мм", "Ворота откатные 6000х2250 мм")          # case, spacing and the sign of × are normalised
    assert r.match_basis == "normalized_name"


def test_a_name_made_equal_only_by_synonyms_is_not_identity():
    """`search_text` applies meaning-synonyms («камера» → «видеокамера»); identity must not."""
    r = comp("камера 2 Мп", "видеокамера 2 Мп")
    assert r.match_basis == "request_constraints"


def test_a_kit_never_rests_on_its_name():
    r = comp("Шкаф управления IP66 в составе:", "Шкаф управления IP66 в составе:")
    assert r.klass != "compatible" or r.match_basis != "normalized_name"


def test_the_thermometer_assumption_is_a_basis_of_its_own():
    r = comp("инфракрасный термометр", "Инфракрасный термометр (пирометр) Fluke 561")
    assert r.klass == "compatible" and r.match_basis == "type_assumption"


def test_a_substitute_names_the_rule_and_is_not_a_match():
    r = comp("камера 3 Мп", "Видеокамера уличная 2Мп")
    assert r.klass == "compatible_with_substitution"
    row = next(x for x in r.comparisons if x.name == "resolution_mp")
    assert row.status == "substitute" and row.rule_id == "camera_resolution"


def test_no_basis_for_what_is_not_a_positive_answer():
    assert comp("кабель КВВГнг 5*2.5", "Кабель КВВГнг 5х1,5").match_basis is None          # incompatible
    assert comp("светильник IP66 50 Гц", "Светильник IP66 60 Гц").match_basis is None      # unverified


# ---------------------------------------------------------------- what the user reads
def item(engine_answer, name_part):
    return next(x for x in engine_answer["table"] if name_part in x["name"])


@pytest.fixture(scope="module")
def engine():
    from analog_search.engine import SearchEngine
    return SearchEngine.build()


@real
def test_labels_on_the_screen_are_honest(engine):
    a = engine.search("труба гофра ПВХ 25 мм")
    confirmed = [x for x in a["table"] if x["verdict"] == "same"]
    assert confirmed and all(x["verdict_text"] == "совпадает по указанным параметрам" for x in confirmed)
    assert all(x["match_basis"] == "request_constraints" and x["match_note"] for x in confirmed)


@real
def test_the_whole_name_as_a_query_gives_the_identity_label(engine):
    a = engine.search("Ворота откатные 2250х6000 мм")
    top = a["table"][0]
    assert top["name"] == "Ворота откатные 2250х6000 мм" and top["match_basis"] == "normalized_name"
    assert top["verdict_text"] == "та же позиция по данным названия" and top["match_note"] is None


@real
def test_the_thermometer_card_says_what_was_not_checked(engine):
    top = engine.search("Инфракрасный термометр любого производителя")["table"][0]
    assert top["match_basis"] == "type_assumption"
    assert top["match_note"] == "Подобрано по типу; диапазон и точность не проверялись; действует допущение автора."


@real
def test_a_substitute_is_not_listed_among_the_matches(engine):
    top = engine.search("камера 3 Мп")["table"][0]
    assert top["verdict"] == "substitute" and top["verdict_text"] == "допустимая замена"
    ex = top["explanation"]
    assert "Допустимо по правилу: разрешение" in ex and "camera_resolution" in ex
    assert "Совпало: разрешение" not in ex and "Допустимая замена:" not in ex


@real
@pytest.mark.parametrize("query", ["кабель КВВГнг-LS 7*1.5", "камера IP уличная 2 Мп вариофокальная", "ворота откатные 6000х2250", "труба гофра ПВХ 25 мм",
                                   "ИБП 12В для улицы", "трансформатор силовой сухой 1000 кВА", "тройник 25х30 мм любой цвет", "клапан противопожарный EI60 200х200"])
def test_the_identity_phrase_is_never_used_without_the_ground(engine, query):
    a = engine.search(query)
    for x in a["table"] + a["results"] + a["unconfirmed"]:
        if re.search(r"та же позиция", x["verdict_text"], re.I):
            assert x["match_basis"] == "normalized_name"


# ---------------------------------------------------------------- the remainder shown to the user (T06, first part)
@real
def test_the_answer_lists_what_could_not_be_parsed(engine):
    q = engine.search("ноутбук 8 Гб")["query"]
    assert q["unparsed_constraints"] == [{"raw": "8 Гб", "reason": "number_with_unit", "severity": "blocking"}]
    assert engine.search("светильник светодиодный 100 Вт IP66")["query"]["unparsed_constraints"] == []


@real
def test_an_unparsed_requirement_is_the_reason_of_the_check(engine):
    top = engine.search("ноутбук 8 Гб")["table"][0]
    assert top["verdict"] == "check" and top["needs_review"]
    assert "8 Гб" in top["explanation"] and "не удалось разобрать" in top["explanation"]


def test_the_page_shows_the_remainder_and_the_note():
    src = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "unparsed_constraints" in src and "match_note" in src and "Не удалось разобрать" in src
