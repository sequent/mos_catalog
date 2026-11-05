"""units written in another case must give the same value, and a number with a unit that nothing
explains must never be silently dropped from the decision.

Reproduced defect: «светильник светодиодный 100 вт IP66» (lowercase «вт») lost the power and was confirmed by a 300 Вт lamp.
"""
from pathlib import Path

import pytest

from analog_search import process
from analog_search.matching import compare, load_rules

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules()
BUILT = (ROOT / "data" / "processed" / "catalog_v2.jsonl").exists() and (ROOT / "data" / "index" / "dense.npy").exists() \
    and (ROOT / "models" / "multilingual-e5-small" / "onnx" / "model.onnx").exists()
real = pytest.mark.skipif(not BUILT, reason="catalog, index or model are not built")


def val(text, name, query=True):
    return [a["value"] for a in process(text, query=query)["attributes"] if a["name"] == name and a["source"] == "main"]


def klass(q, c):
    return compare(process(q, query=True), process(c), RULES).klass


# ---------------------------------------------------------------- part A: units in any case
@pytest.mark.parametrize("text", ["светильник 100 Вт", "светильник 100 вт", "светильник 100Вт", "светильник 100 ВТ", "светильник 100 W", "светильник 100W"])
def test_watt_in_any_spelling(text):
    assert val(text, "power_kw") == [0.1]


@pytest.mark.parametrize("text", ["насос 1.5 кВт", "насос 1.5 квт", "насос 1,5 КВТ", "насос 1.5 kW", "насос 1.5 кВТ"])
def test_kilowatt_in_any_spelling(text):
    assert val(text, "power_kw") == [1.5]


@pytest.mark.parametrize("text", ["ИБП 3 кВА", "ИБП 3 ква", "ИБП 3 КВА", "ИБП 3 kVA", "ИБП 3 кВа"])
def test_kilovolt_ampere_in_any_spelling(text):
    assert val(text, "apparent_power_kva") == [3]


@pytest.mark.parametrize("text", ["клапан EI60", "клапан ei60", "клапан EI 60", "клапан ei 60"])
def test_fire_resistance_in_any_case(text):
    assert val(text, "fire_resistance_ei") == [60]


@pytest.mark.parametrize("text", ["фланец Ду80", "фланец ду80", "фланец ДУ80", "фланец DN80", "фланец dn80", "фланец ДУ 80"])
def test_nominal_bore_in_any_case(text):
    assert val(text, "nominal_bore_mm") == [80]


@pytest.mark.parametrize("text,name,expected", [
    ("шкаф 6 kV", "voltage_v", [6000]), ("реле 10 kA", "current_a", 10000), ("труба 1.6 MPa", "pressure_mpa", 1.6),
    ("груз 5 kg", "mass_kg", 5), ("груз 5 КГ", "mass_kg", 5), ("шкаф 25 ММ", "bare_mm", 25), ("труба 25 ММ", "diameter_mm", 25),
])
def test_other_units_in_other_spellings(text, name, expected):
    got = val(text, name)
    assert got and (got[0] == expected or got == [expected])


def test_prefix_case_is_not_flattened():
    """мВт (milliwatt) must not be turned into МВт (megawatt): an unclear unit stays unparsed instead of becoming a wrong value."""
    assert val("датчик 5 мВт", "power_kw") == []
    assert val("генератор 5 МВт", "power_kw") == [5000]


def test_a_catalog_row_in_lowercase_is_read_the_same_way():
    assert val("Светильник светодиодный, 100 вт, IP66", "power_kw", query=False) == [0.1]


# ---------------------------------------------------------------- part B: the remainder must not disappear
def unparsed(text, query=True):
    return process(text, query=query)["unparsed_constraints"]


def test_remainder_is_reported_with_a_reason_and_severity():
    u = unparsed("реле IP66 50 Гц")
    assert [(x["raw"], x["reason"], x["severity"]) for x in u] == [("50 Гц", "number_with_unit", "blocking")]
    assert u[0]["source"] == "main" and u[0]["span"][0] < u[0]["span"][1]


def test_measurement_label_without_unit_is_blocking():
    u = unparsed("светильник IP66 мощность 100")
    assert [(x["reason"], x["severity"]) for x in u] == [("measurement_label", "blocking")]


def test_a_bare_number_is_only_informational():
    u = unparsed("светильник IP66 12")
    assert [(x["reason"], x["severity"]) for x in u] == [("bare_number", "informational")]


def test_explained_numbers_are_not_remainder():
    assert unparsed("светильник светодиодный 100 Вт IP66") == []
    assert unparsed("кабель КВВГнг-LS 5х1,5") == []


def test_model_codes_and_references_are_not_remainder():
    assert unparsed("Источник SKAT-V.12DC-18") == []
    assert unparsed("огнетушитель ОП-5") == []
    assert unparsed("УПС-02 исп.56") == []


def test_alt_fragment_never_blocks():
    assert [x for x in unparsed("Сталь полосовая 100х10 мм (прим. 60 Гц)") if x["severity"] == "blocking"] == []


def test_ordinary_words_after_a_number_are_not_units():
    assert [x for x in unparsed("комплект 5 в сборе") if x["severity"] == "blocking"] == []
    assert [x for x in unparsed("щит 3 шт") if x["severity"] == "blocking"] == []


def test_unparsed_query_constraint_caps_the_class_at_unverified():
    # without the guard IP66 alone would make this «compatible», though 50 Гц against 60 Гц was never compared
    assert klass("светильник IP66 50 Гц", "Светильник IP66 60 Гц") == "unverified"
    c = compare(process("светильник IP66 50 Гц", query=True), process("Светильник IP66 60 Гц"), RULES)
    assert any("50 Гц" in r for r in c.reasons)


def test_the_same_unparsed_value_on_the_candidate_resolves_it():
    assert klass("светильник IP66 50 Гц", "Светильник IP66 50 Гц") == "compatible"


def test_a_known_conflict_still_wins():
    assert klass("светильник IP66 50 Гц 100 Вт", "Светильник IP66 60 Гц 300 Вт") == "incompatible"


def test_bare_number_does_not_cap_the_class():
    assert klass("светильник IP66 12", "Светильник IP66") == "compatible"


def test_unparsed_candidate_remainder_does_not_block_the_query():
    assert klass("светильник IP66", "Светильник IP66 50 Гц") == "compatible"


# ---------------------------------------------------------------- the reproduced defect
@real
@pytest.mark.parametrize("q", ["светильник светодиодный 100 вт IP66", "светильник светодиодный 100 Вт IP66", "светильник светодиодный 100Вт IP66"])
def test_a_300_w_lamp_is_not_confirmed_for_100_w(q):
    from analog_search.engine import SearchEngine
    a = SearchEngine.build().search(q)
    assert a["status"] == "not_found"
    assert not any("300 Вт" in x["name"] for x in a["results"])
