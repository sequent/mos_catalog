"""The four assumptions of the author, one test group each. Every assumption lives in a config file (rules.yaml or
dictionaries.yaml), so each test also checks that SWITCHING THE ASSUMPTION OFF changes the answer.

1. cable cross-section must match            → rules.yaml: cable_cores_section (exact)
2. camera resolution: ±1 step                → rules.yaml: camera_resolution (ladder, max_steps 1)
3. infrared thermometer, any manufacturer    → rules.yaml: policy.evidence.confirm_by_type + dictionaries.yaml: any_phrases, type
4. tee 25×30 mm, any colour                  → dictionaries.yaml: colors, any_phrases; rules.yaml: defaults.per_attribute.color
5. «металлическая» = «стальная»              → dictionaries.yaml: materials (металл), generic_materials
6. синонимы типов                            → dictionaries.yaml: synonyms.replace (датчик/извещатель, огнезадерживающий, дооснащение, Хуавей)
"""
import copy
from pathlib import Path

import pytest
import yaml

from analog_search import process
from analog_search.engine import SearchEngine
from analog_search.matching import compare, load_rules
from analog_search.matching.rules import RULES_PATH

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules()
BUILT = (ROOT / "data" / "processed" / "catalog_v2.jsonl").exists() and (ROOT / "data" / "index" / "dense.npy").exists() \
    and (ROOT / "models" / "multilingual-e5-small" / "onnx" / "model.onnx").exists()
real = pytest.mark.skipif(not BUILT, reason="catalog, index or model are not built")


def attr_status(query, candidate, attribute, rules=RULES):
    comp = compare(process(query, query=True), process(candidate), rules)
    return {c.name: c.status for c in comp.comparisons}.get(attribute)


def klass(query, candidate, rules=RULES):
    return compare(process(query, query=True), process(candidate), rules).klass


def rules_with(mutate, tmp_path):
    raw = copy.deepcopy(yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")))
    mutate(raw)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_rules(p)


@pytest.fixture(scope="module")
def engine():
    return SearchEngine.build()


# ---------------------------------------------------------------- 1. cross-section must match
def test_cross_section_must_match():
    cand = "Кабель контрольный КВВГнг(A)-LS 5х1,5"
    assert attr_status("кабель КВВГнг-LS 5*1.5", cand, "cores_section") == "match"
    assert attr_status("кабель КВВГнг-LS 5*2.5", cand, "cores_section") == "conflict"
    assert klass("кабель КВВГнг-LS 5*2.5", cand) == "incompatible"          # not even shown as a variant
    assert attr_status("кабель КВВГнг-LS 7*1.5", cand, "cores_section") == "conflict"       # number of cores too


def test_cross_section_assumption_can_be_relaxed(tmp_path):
    rules = rules_with(lambda r: next(x for x in r["rules"] if x["id"] == "cable_cores_section").update(mode="flag"), tmp_path)
    cand = "Кабель контрольный КВВГнг(A)-LS 5х1,5"
    assert attr_status("кабель КВВГнг-LS 5*2.5", cand, "cores_section", rules) == "flag"
    assert klass("кабель КВВГнг-LS 5*2.5", cand, rules) == "unverified"


# ---------------------------------------------------------------- 2. camera resolution ±1 step
@pytest.mark.parametrize("q,c,expected", [
    ("камера 3 Мп", "Видеокамера уличная 2Мп", "substitute"),       # one step down
    ("камера 2 Мп", "Видеокамера уличная 3Мп", "substitute"),       # one step up
    ("камера 4 Мп", "Видеокамера уличная 2Мп", "conflict"),         # two steps
    ("камера 5 Мп", "Видеокамера уличная 8Мп", "substitute"),       # next step of the scale (5 → 8), not «±1 Мп»
    ("камера 3 Мп", "Видеокамера уличная 3Мп", "match"),
])
def test_camera_resolution_one_step(q, c, expected):
    assert attr_status(q, c, "resolution_mp") == expected


def test_camera_resolution_tolerance_can_be_changed(tmp_path):
    rules = rules_with(lambda r: next(x for x in r["rules"] if x["id"] == "camera_resolution").update(max_steps=2), tmp_path)
    assert attr_status("камера 4 Мп", "Видеокамера уличная 2Мп", "resolution_mp", rules) == "substitute"


# ---------------------------------------------------------------- 3. infrared thermometer, any manufacturer
def test_phrase_any_manufacturer_is_cut_out_and_reported():
    q = process("Инфракрасный термометр любого производителя", query=True)
    assert [a["attribute"] for a in q["assumptions"]] == ["manufacturer"]
    assert "любого" not in q["search_text"] and "производител" not in q["search_text"]
    assert q["type"] == "термометр инфракрасный" and q["type_specific"]
    assert process("Инфракрасный термометр любого производителя")["assumptions"] == []           # a catalog row is never edited


def test_pyrometer_is_the_same_type():
    assert process("Пирометр Fluke 561")["type"] == "термометр инфракрасный"


@real
def test_infrared_thermometer_any_manufacturer_is_found(engine):
    a = engine.search("Инфракрасный термометр любого производителя")
    assert a["status"] == "found" and a["query"]["assumptions"][0]["label"] == "производитель"
    assert [x["name"] for x in a["results"]] == ["Инфракрасный термометр (пирометр) Fluke 561"]
    assert a["results"][0]["verdict"] == "same" and "Подобрано по типу; диапазон и точность не проверялись" in a["results"][0]["explanation"]


@real
def test_type_only_confirmation_is_a_switchable_assumption(engine, tmp_path):
    rules = rules_with(lambda r: r["policy"]["evidence"].update(confirm_by_type=[]), tmp_path)
    off = SearchEngine(engine.records, engine.retriever, rules).search("Инфракрасный термометр любого производителя")
    assert off["status"] == "not_found"                                        # the assumption off: nothing to confirm with


@real
def test_type_only_confirmation_does_not_confirm_other_types(engine):
    a = engine.search("Тепловизор FLIR")
    assert all(x["name"] != "Инфракрасный термометр (пирометр) Fluke 561" for x in a["results"])
    assert not any("термометр" in x["name"].lower() for x in a["results"])


def test_type_only_confirmation_needs_a_known_type(tmp_path):
    with pytest.raises(Exception, match="confirm_by_type"):
        rules_with(lambda r: r["policy"]["evidence"].update(confirm_by_type=["несуществующий тип"]), tmp_path)


def test_type_only_confirmation_does_not_override_a_conflict():
    """If the query names a characteristic and it differs, the type alone must not rescue the candidate."""
    comp = compare(process("инфракрасный термометр 500 °C", query=True), process("Инфракрасный термометр (пирометр) 1000 °C"), RULES)
    assert [(x.name, x.status) for x in comp.comparisons] == [("temperature_c", "conflict")]
    assert comp.klass == "incompatible" and not comp.type_confirmed
    # no characteristic named in the query: the type alone confirms
    comp = compare(process("инфракрасный термометр", query=True), process("Инфракрасный термометр (пирометр) Fluke 561"), RULES)
    assert comp.klass == "compatible" and comp.type_confirmed


# ---------------------------------------------------------------- 4. tee 25×30 mm, any colour
def test_phrase_any_colour_is_cut_out_and_reported():
    q = process("Тройник 25х30 мм любой цвет", query=True)
    assert [a["attribute"] for a in q["assumptions"]] == ["color"]
    assert "любой" not in q["search_text"] and not [a for a in q["attributes"] if a["name"] == "color"]


def test_colour_is_a_characteristic():
    assert [a["value"] for a in process("Тройник 25х30 мм, цвет белый, NTAN")["attributes"] if a["name"] == "color"] == [["белый"]]
    assert [a["value"] for a in process("тройник красный", query=True)["attributes"] if a["name"] == "color"] == [["красный"]]


CAND = "Тройник 25х30 мм, цвет белый, NTAN"


def test_colour_same_differs_and_unstated():
    assert attr_status("тройник 25х30 мм белый", CAND, "color") == "match"
    assert attr_status("тройник 25х30 мм красный", CAND, "color") == "flag"
    assert attr_status("тройник 25х30 мм красный", "Тройник 25х30 мм NTAN", "color") == "no_data"
    assert klass("тройник 25х30 мм любой цвет", CAND) == "compatible"
    assert klass("тройник 25х30 мм красный", CAND) == "unverified"          # shown to be checked, not «та же позиция»
    assert klass("тройник 25х30 мм белый", CAND) == "compatible"


def test_size_still_decides_for_any_colour():
    assert klass("тройник 25х30 мм любой цвет", "Тройник 25х17 IM") == "incompatible"
    assert klass("тройник 25х30 мм любой цвет", "Тройник 60х40 мм, цвет белый, NTAN") == "incompatible"


def test_colour_alone_is_not_evidence():
    """«белый» must not make an unrelated white thing a candidate (colour is a weak attribute, like «уличный»)."""
    c = compare(process("тройник 25х30 мм белый", query=True), process("Сфера предупреждения для ВЛ МПЗУ-300М белый"), RULES)
    assert c.confirmed_strong == 0 and c.klass == "unverified" and not c.weak_evidence


def test_colour_mode_is_changeable(tmp_path):
    rules = rules_with(lambda r: r["defaults"]["per_attribute"].update(color="exact"), tmp_path)
    assert attr_status("тройник 25х30 мм красный", CAND, "color", rules) == "conflict"
    assert klass("тройник 25х30 мм красный", CAND, rules) == "incompatible"
    rules = rules_with(lambda r: r["defaults"]["per_attribute"].update(color="ignore"), tmp_path)
    assert attr_status("тройник 25х30 мм красный", CAND, "color", rules) is None
    assert klass("тройник 25х30 мм красный", CAND, rules) == "compatible"


@real
def test_tee_any_colour_is_found(engine):
    a = engine.search("Тройник 25х30 мм любой цвет")
    assert a["status"] == "found" and a["query"]["assumptions"][0]["label"] == "цвет"
    assert [x["name"] for x in a["results"]] == [CAND] and a["results"][0]["verdict"] == "same"


@real
def test_tee_of_another_colour_needs_a_check(engine):
    a = engine.search("тройник 25х30 мм красный")
    assert a["status"] == "not_found" and a["display_status"] == "review"
    assert [x["name"] for x in a["table"]] == [CAND] and a["table"][0]["needs_review"]


# ---------------------------------------------------------------- 5. «металлическая» = «стальная»
DOOR = "Дверь противопожарная металлическая однопольная ДПМ-01/30, размером 1000х2100 мм"
PIPE = "Труба металлическая для водосточных систем, покрытие полиэстер, диаметр 150 мм, длина 1000 мм"


@pytest.mark.parametrize("q,c,expected", [
    ("дверь стальная ДПМ-01/30", DOOR, "match"),                              # the assumption
    ("дверь металлическая ДПМ-01/30", "Дверь стальная ДПМ-01/30", "match"),   # and the other way round
    ("труба металлическая 150 мм", PIPE, "match"),
    ("труба алюминиевая 150 мм", PIPE, "no_data"),                            # «металлическая» does not exclude aluminium
    ("труба медная 150 мм", PIPE, "no_data"),
    ("труба из нержавеющей стали 150 мм", PIPE, "match"),                   # stainless steel is steel
    ("труба ПВХ 150 мм", PIPE, "conflict"),                                   # a plastic is not a metal
    ("труба стальная 150 мм", "Труба полиэтиленовая 150 мм", "conflict"),
    ("труба алюминиевая 150 мм", "Труба стальная 150 мм", "conflict"),        # two named metals still differ
])
def test_metallic_means_steel(q, c, expected):
    assert attr_status(q, c, "material") == expected


def test_metallic_assumption_is_a_dictionary_entry(tmp_path):
    """Remove `generic_materials` and «металлическая» is no longer steel: a steel query meets only «нет данных» ... or conflict."""
    from analog_search.config import load_dictionaries
    assert load_dictionaries().generic_materials["металл"]["same_as"] == frozenset({"сталь"})
    gen = load_dictionaries().generic_materials
    saved = dict(gen)
    try:
        gen.clear()
        assert attr_status("дверь стальная ДПМ-01/30", DOOR, "material") == "conflict"        # without the assumption: «сталь» ≠ «металл»
    finally:
        gen.update(saved)
    assert attr_status("дверь стальная ДПМ-01/30", DOOR, "material") == "match"


@real
def test_steel_door_is_confirmed_by_the_metallic_one(engine):
    a = engine.search("дверь стальная противопожарная одностворчатая ДПМ-01/30 1000*2100")
    assert a["status"] == "found" and [x["name"] for x in a["results"]] == [DOOR]


@real
def test_aluminium_pipe_query_is_not_rejected_by_a_metallic_pipe(engine):
    a = engine.search("труба алюминиевая водосточная диаметр 150 мм длина 1 м")
    assert a["status"] == "not_found" and any(x["name"] == PIPE for x in a["table"])
    assert all(x["needs_review"] for x in a["table"])


# ---------------------------------------------------------------- 6. synonyms of types and words
@pytest.mark.parametrize("text,expected_type", [
    ("датчик пожарный тепловой ИПТ-Ех", "извещатель"),
    ("датчик охранный вибрационный", "извещатель"),
    ("Извещатель пожарный тепловой ИПТ-Ех", "извещатель"),
    ("датчик температуры воздуха", "датчик"),                       # an ordinary sensor is NOT a detector
    ("датчик воды", "датчик"),
    ("огнезадерживающий клапан EI60", "клапан противопожарный"),
    ("Клапаны противопожарные квадратные с электроприводом", "клапан противопожарный"),
])
def test_type_synonyms(text, expected_type):
    assert process(text, query=True)["type"] == expected_type


def test_word_synonyms_in_the_search_text():
    t = process("дооснащение действующих мультиплексоров Хуавей", query=True)["search_text"]
    assert "доукомплектация" in t and "существующих" in t and "huawei" in t
    assert "дооснащени" not in t and "хуавей" not in t


def test_synonyms_are_switched_off_by_removing_them(monkeypatch):
    import dataclasses
    from analog_search.config import load_dictionaries
    d = load_dictionaries()
    off = dataclasses.replace(d, replace=[(rx, to) for rx, to in d.replace if to not in ("извещатель", "противопожарный", "доукомплектация", "существующих", "huawei")])
    monkeypatch.setattr("analog_search.normalize.load_dictionaries", lambda *a, **k: off)
    assert process("датчик пожарный тепловой ИПТ-Ех", query=True)["type"] == "датчик"
    assert process("огнезадерживающий клапан EI60", query=True)["type"] == "клапан"
    assert "huawei" not in process("Хуавей", query=True)["search_text"]


@real
def test_fire_detector_asked_as_a_sensor_is_shown_for_checking(engine):
    a = engine.search("датчик пожарный тепловой ИПТ-Ех")
    assert a["status"] == "not_found"                                  # the code «ИПТ-Ех» has no digits: not a model code, nothing to confirm
    assert a["table"][0]["id"] == "row-000646" and a["table"][0]["needs_review"]
    assert all(x["type"] == "извещатель" for x in a["table"])           # no ordinary sensors in the list


@real
def test_ordinary_sensor_is_not_mixed_with_detectors(engine):
    a = engine.search("датчик температуры воздуха")
    assert a["table"] and a["table"][0]["type"] == "датчик" and a["table"][0]["name"] == "Датчик температуры наружного воздуха"
    assert not any(x["type"] == "извещатель" for x in a["table"])


@real
def test_fire_damper_by_its_other_name_is_confirmed(engine):
    a = engine.search("огнезадерживающий клапан EI60 200х200 с электроприводом")
    assert a["status"] == "found" and {x["id"] for x in a["results"]} >= {"row-000357", "row-001031"}
    assert all(x["type"] == "клапан противопожарный" for x in a["results"])


@real
def test_kit_upgrade_asked_with_other_words_is_shown_but_never_confirmed(engine):
    a = engine.search("дооснащение действующих мультиплексоров Хуавей на ПС 220кВ, комплект")
    assert a["status"] == "not_found" and a["table"][0]["id"] == "row-000723"
    assert a["table"][0]["needs_review"] and a["table"][0]["composite"]          # a kit: «та же позиция» is impossible by policy


# ---------------------------------------------------------------- 7. the noun of the query must be in the candidate (defect 7.1 of the log)
PODSTAVKA = "Пожарная подставка раструбная ППР из высокопрочного чугуна, диаметр 200 мм"
ZADVIZHKA = "Задвижка клиновая с выдвижным шпинделем, номинальное давление 1,6 МПа, Ду200"


def test_a_different_thing_with_the_same_number_is_not_confirmed():
    """«задвижка диаметр 200 мм» matched a fire stand «… диаметр 200 мм» at word coverage exactly 0.5."""
    comp = compare(process("задвижка диаметр 200 мм", query=True), process(PODSTAVKA), RULES)
    assert comp.word_coverage == 0.5 and comp.missing_words == ["задвижка"]
    assert comp.klass == "unverified" and any("слова типа" in r for r in comp.reasons)
    assert comp.weak_evidence is False                                         # and it is not even listed for checking


def test_the_rule_can_be_switched_off(tmp_path):
    rules = rules_with(lambda r: r["policy"]["word_coverage"].update(type_word_required=False), tmp_path)
    assert klass("задвижка диаметр 200 мм", PODSTAVKA, rules) == "compatible"       # the defect, visible when the rule is off


def test_a_thing_named_by_its_mark_keeps_its_type_without_the_noun():
    """A cable written as «ВВГнг(A)-LS 5х6» has no word «кабель», but the dictionary gives it the type кабель."""
    comp = compare(process("кабель ВВГнг-LS 5x6", query=True), process("ВВГнг(A)-LS 5х6"), RULES)
    assert not any("слова типа" in r for r in comp.reasons)


def test_the_noun_present_in_the_candidate_does_not_block():
    comp = compare(process("задвижка диаметр 200 мм", query=True), process("Задвижка клиновая, диаметр 200 мм"), RULES)
    assert comp.klass == "compatible"


@real
def test_probe_query_no_longer_confirms_the_fire_stand(engine):
    a = engine.search("задвижка диаметр 200 мм")
    assert a["status"] == "not_found"
    assert all(x["type"] == "задвижка" for x in a["table"]) and a["table"]
