"""The requirements of a query: its subject, a model and a series, the number of keys, units of length, the key of a whole name. Values are written by hand; the expected answer is never computed by the extractor.
Everything here uses synthetic names or neutral examples: no id of a holdout query or of a reference row appears in the code under test."""
import json
import re
from pathlib import Path

import pytest

from analog_search import process
from analog_search.matching import compare, load_rules
from analog_search.matching.compat import _name_key, _other_model_of_the_same_series, query_subject

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules()


def attrs(text, query=True):
    return {a["name"]: a["value"] for a in process(text, query=query)["attributes"]}


def comp(q, c):
    return compare(process(q, query=True), process(c), RULES)


def status(q, c, attribute):
    return {x.name: x.status for x in comp(q, c).comparisons}.get(attribute)


# ---------------------------------------------------------------- units of length are never lost
@pytest.mark.parametrize("text", ["Крышка лотка L=3000 мм", "Крышка лотка L 3000 мм", "Крышка лотка L: 3000 мм", "Крышка лотка L=3000 mm", "Крышка лотка L 3000 mm",
                                  "Крышка лотка L=300 cm", "Крышка лотка L=300 см", "Крышка лотка L=3 m", "Крышка лотка L=3 м", "Крышка лотка L 3 m"])
def test_every_equivalent_spelling_of_a_length_is_3000_mm(text):
    assert attrs(text)["length_mm"] == 3000


@pytest.mark.parametrize("text,value", [("Крышка лотка L=3000 cm", 30000), ("Крышка лотка L=3000 см", 30000), ("Крышка лотка L=3000 m", 3000000)])
def test_the_unit_decides_the_scale_it_is_not_dropped(text, value):
    assert attrs(text)["length_mm"] == value


def test_a_length_in_centimetres_conflicts_with_the_same_number_of_millimetres():
    cover = "Крышка лотка, L=3000 мм «Стандарт KLS 300»"
    assert status("Крышка лотка L=3000 cm", cover, "length_mm") == "conflict"
    assert status("Крышка лотка L=3000 мм", cover, "length_mm") == "match"
    assert status("Крышка лотка L=300 cm", cover, "length_mm") == "match"
    assert status("Крышка лотка L=3 m", cover, "length_mm") == "match"


@pytest.mark.parametrize("text", ["Крышка лотка L=3000 дм", "Крышка лотка L 3000"])
def test_an_unknown_unit_or_no_unit_without_a_sign_is_not_defaulted_to_millimetres(text):
    p = process(text, query=True)
    assert "length_mm" not in {a["name"] for a in p["attributes"]}
    assert [(u["raw"], u["severity"]) for u in p["unparsed_constraints"]] == [("3000", "blocking")]


def test_the_evidence_keeps_the_whole_construct_with_its_unit():
    a = next(x for x in process("Крышка лотка L=3000 cm", query=True)["attributes"] if x["name"] == "length_mm")
    assert a["raw"] == "L=3000 см"


def test_latin_units_do_not_damage_codes_or_derived_units():
    from analog_search.normalize import normalize_text as n
    assert n("SKAT-5m") == "SKAT-5m" and n("10W-40 масло") == "10W-40 масло" and n("кабель 2.5 mm2") == "кабель 2.5 mm2"
    assert n("шкаф 5 m") == "шкаф 5 м" and n("труба 25 mm") == "труба 25 мм"


# ---------------------------------------------------------------- decoration of the name does not break «the whole name is the same»
SAND = "Песок природный II класс, средний, круглые сита"


@pytest.mark.parametrize("query", [SAND, "Песок природный II класс средний круглые сита", "песок природный  II класс;средний;круглые сита"])
def test_the_same_name_with_other_decoration_is_one_ground_of_the_match(query):
    c = comp(query, SAND)
    assert c.klass == "compatible" and c.match_basis == "normalized_name"


@pytest.mark.parametrize("query", ["Песок природный I класс средний круглые сита", "Песок природный III класс средний круглые сита",
                                   "Песок природный II класс крупный круглые сита", "Песок природный II класс средний квадратные сита"])
def test_a_different_property_is_not_made_equal_by_removing_punctuation(query):
    assert comp(query, SAND).klass != "compatible"


@pytest.mark.parametrize("a,b", [("Лист 1.5 мм", "Лист 15 мм"), ("Датчик -10 °C", "Датчик 10 °C"), ("Реле 10±1 А", "Реле 10 А"), ("Стол 1,5 м", "Стол 15 м")])
def test_numbers_signs_and_tolerances_keep_the_keys_apart(a, b):
    assert _name_key(process(a)) != _name_key(process(b))


def test_a_plain_name_without_parameters_is_confirmed_only_by_the_whole_name():
    assert comp("Мешок, цементный", "Мешок цементный").klass == "compatible"
    assert comp("Мешок цементный", "Мешок бумажный").klass != "compatible"


# ---------------------------------------------------------------- the number of keys
@pytest.mark.parametrize("text,n", [("выкл. 1-кл. открытой установки", 1), ("Выключатель одноклавишный", 1), ("Выключатель двухклавишный", 2),
                                    ("Выключатель трёхклавишный", 3), ("выкл. 2 кл. 16А", 2), ("выключатель 3 клавиши", 3)])
def test_the_number_of_keys(text, n):
    assert attrs(text)["switch_gang_count"] == n


def test_a_class_abbreviation_is_not_a_number_of_keys():
    assert "switch_gang_count" not in attrs("прибор класс точности 1 кл.")


def test_one_key_and_two_keys_conflict_unknown_is_not_a_match():
    one = "выкл. одноклавишный открытой установки 250 В"
    assert status(one, "Выключатель двухклавишный открытой установки 250 В", "switch_gang_count") == "conflict"
    assert status(one, "Выключатель одноклавишный открытой установки 250 В", "switch_gang_count") == "match"
    assert status(one, "Выключатель открытой установки 250 В", "switch_gang_count") == "no_data"
    assert comp(one, "Выключатель открытой установки 250 В").klass != "compatible"


# ---------------------------------------------------------------- the subject of the query is not replaced by common parameters
def test_a_switch_is_a_known_type_and_the_abbreviation_is_expanded():
    p = process("выкл. 250 В 16 А", query=True)
    assert p["type"] == "выключатель" and p["type_specific"]


def test_a_socket_on_the_same_voltage_and_current_is_not_a_switch():
    c = comp("выкл. одноклавишный 250 В, 16 А, IP20", "Розетка одноместная открытой установки 250В, 16А, IP 20")
    assert c.klass == "incompatible" and c.type_relation == "different_both_specific"


def test_pipes_are_not_a_gauge_and_the_steel_of_the_object_is_not_the_steel_of_the_device():
    q = "ультразвуковой толщиномер для контроля толщины стальных труб"
    assert "material" not in attrs(q)
    assert attrs("толщиномер стальной") .get("material") == ["сталь"]                 # the device itself made of steel stays a material
    assert attrs("труба стальная 25 мм")["material"] == ["сталь"]
    assert comp(q, "Трубы стальные электросварные, наружный диаметр 530 мм, толщина стенки 7 мм").klass in ("incompatible", "unverified")


def test_the_subject_is_the_first_word_that_is_not_an_adjective_or_a_code():
    q = lambda t: (query_subject(process(t, query=True)) or (None,))[0]               # noqa: E731
    assert q("ультразвуковой толщиномер для контроля") == "толщиномер"
    assert q("TOR-1 тестер масла") == "тестер"
    assert q("настенный оптический кросс 16 портов") == "кросс"
    assert q("патч-корд оптический") is None                                           # a compound noun: the plain-word check does not judge it


def test_an_unknown_type_query_is_not_confirmed_by_a_thing_that_does_not_name_the_subject():
    c = comp("анализатор вибрации 220 В IP65", "Блок питания 220 В IP65 в металлическом корпусе")
    assert c.klass != "compatible" and any("анализатор" in r for r in c.reasons)       # the missing subject is named


def test_the_subject_guard_does_not_block_a_correct_unknown_type_positive():
    assert comp("электродрель Makita HP1630K", "Электродрель ручная Makita HP1630K").klass == "compatible"


# ---------------------------------------------------------------- a model, a series, a manufacturer
def test_another_model_of_the_same_hyphenated_series_is_a_conflict():
    assert status("выключатель BA10-001B", "Выключатель BA10-006B", "model_codes") == "conflict"
    assert status("манометр МО-05", "Манометр МО-06", "model_codes") == "conflict"
    assert status("выключатель BA10-001B", "Выключатель BA10-001B белый", "model_codes") == "match"


def test_a_model_written_with_a_suffix_or_in_another_series_is_not_a_conflict():
    assert _other_model_of_the_same_series(["DS-2CD1021"], ["DS-2CD1021-I"]) is None
    assert _other_model_of_the_same_series(["XNV-6081RE"], ["XNP-6400RW"]) is None
    assert _other_model_of_the_same_series(["BA10-001B"], []) is None


@pytest.mark.parametrize("token", ["1-кл", "исп.56", "IP 20", "ГОСТ 8639-82"])
def test_service_tokens_have_no_series(token):
    assert _other_model_of_the_same_series([token], ["1-кл.", "исп.57", "IP 44", "ГОСТ 8732-78"]) is None


def test_a_four_character_model_counts_only_for_the_same_type_and_never_as_a_unit():
    assert status("манометр МО-05", "Манометр МО-05 цифровой", "model_codes") == "match"
    assert status("прибор МО-05", "Датчик МО-05", "model_codes") != "match"            # different words, no shared type: a short code alone is not enough
    assert "ip20" not in {c for c in __import__("analog_search.matching.compat", fromlist=["x"]).strong_codes({"ip20", "dn50", "rj45"}, True)}


def test_a_brand_alone_never_confirms_and_another_series_is_a_conflict():
    req = lambda q, c: next((x for x in comp(q, c).comparisons if x.name == "series"), None)        # noqa: E731
    assert req("воздушный клапан JKL 700х400", "Воздушный клапан JKL-700x400").status == "info"
    assert req("воздушный клапан JKL 700х400", "Воздушный клапан BN-Z-200").status == "conflict"
    assert req("воздушный клапан JKL 700х400", "Воздушный клапан 700х400").status == "no_data"
    assert req("воздушный клапан 700х400", "Воздушный клапан BN-Z-200") is None
    assert req("пирометр Fluke любого производителя", "Пирометр Raytek") is None            # the user withdrew the requirement
    assert comp("фен Makita", "Электродрель Makita HP1630K").klass != "compatible"           # the brand is not the thing


def test_a_series_conflict_is_a_reject_not_a_check():
    c = comp("воздушный клапан JKL 700х400", "Воздушный клапан с площадкой под электропривод BN-Z-200")
    assert c.klass == "incompatible"


def test_a_code_inside_a_hyphenated_model_is_not_a_series_requirement():
    assert next((x for x in comp("камера XNV-6081RE", "Видеокамера XNP-6400RW").comparisons if x.name == "series"), None) is None


# ---------------------------------------------------------------- the engine: the trace changes nothing, no reference id is in the code
BUILT = (ROOT / "data" / "processed" / "catalog_v2.jsonl").exists() and (ROOT / "data" / "index" / "dense.npy").exists()


@pytest.mark.skipif(not BUILT, reason="catalog or index is not built")
def test_the_trace_is_extra_and_describes_every_candidate_once():
    from analog_search.engine import SearchEngine
    eng = SearchEngine.build()
    plain, traced = eng.search("труба стальная 25 мм"), eng.search("труба стальная 25 мм", trace=True)
    strip = lambda a: {k: v for k, v in a.items() if k not in ("trace", "diagnostics")}          # noqa: E731
    assert strip(plain) == strip(traced) and "trace" not in plain
    t = traced["trace"]
    ids = [r["id"] for r in t["ranked"]]
    assert len(ids) == len(set(ids)) and set(ids) <= set(t["b2_candidates"]) and t["pool"] == "main"
    assert {r["id"] for r in traced["table"]} == {r["id"] for r in t["ranked"] if r["on_screen"]}
    assert {r["id"] for r in traced["results"]} == {r["id"] for r in t["ranked"] if r["in_results"]}
