"""End-to-end behaviour of the whole scheme B1→B7.

Part 1 uses a tiny synthetic catalog and works without the model (`mode="bm25"`).
Part 2 uses the real catalog and the real model (skipped when they are not built): the assignment's example queries,
invariants on random queries and a self-search sanity check.
"""
import random
from pathlib import Path

import pytest

from analog_search import process
from analog_search.engine import SearchEngine
from analog_search.matching import load_rules
from analog_search.retrieval import Retriever
from analog_search.retrieval.dense import DEFAULT_MODEL_DIR

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "processed" / "catalog_v2.jsonl"
INDEX_PATH = ROOT / "data" / "index" / "dense.npy"


# ================================================================== part 1: synthetic, no model
def _rec(text, i, price=100):
    r = process(text)
    r.update({"id": f"row-{i:06d}", "raw_name": text, "price": price})
    return r


SMALL = [
    "Кабель контрольный КВВГнг(A)-LS 7х1,5", "Кабель контрольный КВВГЭнг(A)-LS 7х1,5", "Кабель контрольный КВВГнг(A)-LS 5х1,5",
    "Ворота откатные 2250х6000 мм", "Гофрированная труба из ПВХ с протяжкой d=25 мм", "Труба ПНД гибкая гофр. д.25мм",
    "Шкаф управления IP66 в составе:", "Шкаф управления навесной IP66",
    "Перевозка грузов I класса автомобилями бортовыми на расстояние до 7 км",
    "Видеокамера уличная корпусная 2Мп", "Видеокамера уличная корпусная 3Мп", "Видеокамера уличная корпусная 8Мп",
]


@pytest.fixture(scope="module")
def small_engine():
    recs = [_rec(t, i) for i, t in enumerate(SMALL)]
    return SearchEngine(recs, Retriever(recs), load_rules(), mode="bm25")


def names(items):
    return [x["name"] for x in items]


def test_found_with_explanation_and_price(small_engine):
    a = small_engine.search("ворота откатные 6000х2250")
    assert a["status"] == "found" and a["results"][0]["name"] == "Ворота откатные 2250х6000 мм"
    top = a["results"][0]
    assert top["verdict"] == "same" and top["price"] == 100 and "Совпало" in top["explanation"]
    assert {"attribute_coverage", "text_relevance"} <= set(top["score_parts"])
    assert top["comparison"][0]["attribute"] == "type"


def test_numbers_decide_between_siblings(small_engine):
    a = small_engine.search("кабель КВВГнг 5*1.5")
    assert a["status"] == "found" and a["results"][0]["name"].endswith("5х1,5")
    assert not any(x["name"].endswith("7х1,5") for x in a["results"])          # conflict → not in the answer


def test_screen_difference_gives_not_found_with_unconfirmed(small_engine):
    a = small_engine.search("кабель КВВГнг-LS 7*1.5")
    assert a["status"] == "found" and a["results"][0]["name"] == "Кабель контрольный КВВГнг(A)-LS 7х1,5"
    b = small_engine.search("кабель КВВГЭнг-LS 5*1.5")                           # only «Э» differs from the 5×1.5 row
    assert b["status"] == "not_found" and not b["results"]
    assert any(x["name"].endswith("5х1,5") for x in b["unconfirmed"])


def test_ladder_substitute(small_engine):
    a = small_engine.search("камера 3 Мп")
    verdicts = {x["name"]: x["verdict"] for x in a["results"]}
    assert verdicts["Видеокамера уличная корпусная 3Мп"] == "same"
    assert verdicts["Видеокамера уличная корпусная 2Мп"] == "substitute"
    assert "Видеокамера уличная корпусная 8Мп" not in verdicts
    assert a["results"][0]["verdict"] == "same"                                   # class order before score


def test_composite_is_never_confirmed(small_engine):
    a = small_engine.search("шкаф управления IP66")
    assert [x["name"] for x in a["results"]] == ["Шкаф управления навесной IP66"]
    comp = next(x for x in a["unconfirmed"] if x["name"].endswith("в составе:"))      # the kit is shown, but only as «проверить»
    assert comp["verdict"] == "check" and any("комплект" in r for r in comp["reasons"])
    assert all(not x["composite"] for x in a["results"])


def test_material_difference_is_not_confirmed(small_engine):
    a = small_engine.search("труба гофра ПВХ 25 мм")
    assert a["results"][0]["name"].startswith("Гофрированная труба из ПВХ")
    assert not any("ПНД" in x["name"] for x in a["results"])


def test_nothing_relevant_gives_empty_not_found(small_engine):
    a = small_engine.search("огнетушитель порошковый ОП-5")
    assert a["status"] == "not_found" and not a["results"] and not a["unconfirmed"] and a["nearest_rejected"] is None


def test_services_are_a_separate_pool(small_engine):
    s = small_engine.search("перевозка грузов на расстояние до 7 км")
    assert s["query"]["pool"] == "service"
    assert all(small_engine.records[int(x["id"][4:])]["is_service"] for x in s["results"] + s["unconfirmed"])
    e = small_engine.search("кабель КВВГнг 7*1.5")
    assert e["query"]["pool"] == "main" and not any("Перевозка" in x["name"] for x in e["results"] + e["unconfirmed"])


def test_codes_are_shown_as_written(small_engine):
    """The matching form of a code (look-alike letters mapped) must never reach the user."""
    a = small_engine.search("кабель КВВГнг-LS 7*1.5")
    shown = " ".join(str(x["value"]) for x in a["query"]["attributes"])
    assert "КВВГнг" in shown
    rows = {r["attribute"]: r for r in a["results"][0]["comparison"]}
    assert rows["cable_mark"]["query"] == "КВВГнг" and "КВВГнг" in rows["cable_mark"]["candidate"]


def test_answer_carries_the_rules_version(small_engine):
    a = small_engine.search("ворота откатные 6000х2250")
    assert a["rules_version"] and a["rules_status"] in ("assumption", "approved_by_specialist")
    assert a["diagnostics"]["candidates"] > 0


def test_search_is_deterministic(small_engine):
    a, b = small_engine.search("камера 3 Мп"), small_engine.search("камера 3 Мп")
    assert names(a["results"]) == names(b["results"]) and [x["score"] for x in a["results"]] == [x["score"] for x in b["results"]]


# ================================================================== part 2: real catalog and model
real = pytest.mark.skipif(not (CATALOG_PATH.exists() and INDEX_PATH.exists() and (DEFAULT_MODEL_DIR / "onnx" / "model.onnx").exists()),
                          reason="catalog, index or model are not built (see README)")


@pytest.fixture(scope="module")
def engine():
    return SearchEngine.build()


@real
def test_assignment_example_queries(engine):
    # 1. cable 7*1.5: the catalog has only the screened КВВГЭнг (and a FRLS one) → not found, honest candidate shown
    a = engine.search("кабель КВВГнг-LS 7*1.5")
    assert a["status"] == "not_found" and not a["results"]
    assert any(x["name"].endswith("КВВГЭнг(A)-LS 7х1,5") for x in a["unconfirmed"])
    assert not any("FRLS 7х1,5" in x["name"] for x in a["results"] + a["unconfirmed"])       # fire class differs → rejected
    # 2. camera: the two outdoor 2 Mp cameras with a motorised lens
    a = engine.search("камера IP уличная 2 Мп вариофокальная")
    assert a["status"] == "found" and a["results"][0]["name"].startswith("Видеокамера уличная корпусная 2Мп")
    assert all(x["verdict"] in ("same", "substitute") for x in a["results"])
    # 3. gate: sides in another order
    a = engine.search("ворота откатные 6000х2250")
    assert a["status"] == "found" and a["results"][0]["name"] == "Ворота откатные 2250х6000 мм"
    # 4. corrugated pipe: PVC pipes confirmed, the PND pipe (other material) only unconfirmed
    a = engine.search("труба гофра ПВХ 25 мм")
    assert a["status"] == "found" and all("ПНД" not in x["name"] for x in a["results"])
    assert any("ПВХ" in x["name"] for x in a["results"])
    # 5. UPS: voltage is hidden inside the model code → not confirmable
    a = engine.search("ИБП 12В для улицы")
    assert a["status"] == "not_found" and "SKAT-V.12DC-18" in a["unconfirmed"][0]["name"]
    # 6. transformer 1000 kVA: only 1250 kVA exists → not found, the nearest explains the difference
    a = engine.search("трансформатор силовой сухой 1000 кВА")
    assert a["status"] == "not_found" and "ТСЗ-1250" in a["nearest_rejected"]["name"] and not a["unconfirmed"]
    assert "полная мощность" in a["nearest_rejected"]["explanation"]
    # 7. fire extinguisher: nothing in the catalog → honest «not found», no random rows
    a = engine.search("огнетушитель порошковый ОП-5")
    assert a["status"] == "not_found" and not a["results"] and not a["unconfirmed"] and a["nearest_rejected"] is None


@real
def test_invariants_on_random_queries(engine):
    rng = random.Random(11)
    pool = [r for r in engine.records if not (r["is_service"] or r["is_adjustment"])]
    for rec in rng.sample(pool, 60):
        a = engine.search(rec["raw_name"])
        assert len(a["results"]) <= 5 and len(a["unconfirmed"]) <= 3
        assert (a["status"] == "found") == bool(a["results"])
        assert all(x["verdict"] in ("same", "substitute") for x in a["results"])
        assert all(x["verdict"] == "check" for x in a["unconfirmed"])
        assert all(not x["composite"] for x in a["results"]), "a composite item must never be confirmed"
        for x in a["results"]:
            assert not any(r["status"] == "conflict" for r in x["comparison"]), x["name"]
        order = [x["compatibility"] for x in a["results"]]
        assert order == sorted(order, key=["compatible", "compatible_with_substitution"].index), "class before score"
        ids = [x["id"] for x in a["results"] + a["unconfirmed"]]
        assert len(ids) == len(set(ids))
        if a["status"] == "found":
            assert a["nearest_rejected"] is None
        assert not any(engine.records[int(x["id"][4:]) - 2]["is_service"] for x in a["results"] + a["unconfirmed"])


@real
def test_self_search_returns_the_row_itself(engine):
    """Searching a row by its own name must confirm it (sanity check, not a benchmark)."""
    rng = random.Random(3)
    pool = [r for r in engine.records if not (r["is_service"] or r["is_adjustment"] or r["composite"])]
    sample = rng.sample(pool, 150)
    ok = 0
    for rec in sample:
        a = engine.search(rec["raw_name"])
        ok += any(x["name"].lower() == rec["raw_name"].lower() for x in a["results"])
    assert ok / len(sample) >= 0.95, f"self-search confirmed only {ok}/{len(sample)}"


@real
def test_service_query_stays_in_the_service_pool(engine):
    a = engine.search("Перевозка грузов I класса автомобилями бортовыми грузоподъемностью до 5 т на расстояние до 7 км")
    assert a["query"]["pool"] == "service"
    assert all("еревозк" in x["name"] or "огрузк" in x["name"] or "азгрузк" in x["name"] for x in a["results"] + a["unconfirmed"])


# ================================================================== the main table of the screen (decision of the author)
def test_review_rows_are_in_the_main_table_with_an_honest_status_line(small_engine):
    a = small_engine.search("кабель КВВГЭнг-LS 5*1.5")        # only «Э» (screen) differs from the 5×1.5 row
    assert a["status"] == "found" or a["status"] == "not_found"
    b = small_engine.search("кабель КВВГнг-LS 7*1.5 экранированный")
    assert b["display_status"] in ("review", "not_found", "found")


def test_display_status_review_has_flagged_rows_only(small_engine):
    a = small_engine.search("кабель КВВГЭнг-LS 5*1.5")
    assert a["status"] == "not_found" and a["display_status"] == "review"
    assert a["status_line"] == "Подтверждённых подходящих позиций не найдено. Ниже варианты для проверки"
    assert a["table"] and all(x["needs_review"] and x["verdict"] == "check" for x in a["table"])
    assert any(x["name"].endswith("5х1,5") for x in a["table"])
    assert a["results"] == []                                   # nothing is CONFIRMED: «status» stays honest


def test_display_status_found_lists_confirmed_first(small_engine):
    a = small_engine.search("ворота откатные 6000х2250")
    assert a["display_status"] == "found" and a["status_line"].startswith("Найдено подходящих позиций: 1")
    assert a["table"][0]["name"] == "Ворота откатные 2250х6000 мм" and not a["table"][0]["needs_review"]
    flags = [x["needs_review"] for x in a["table"]]
    assert flags == sorted(flags), "confirmed rows must come before the rows that need checking"


def test_display_status_not_found_has_an_empty_table(small_engine):
    a = small_engine.search("огнетушитель порошковый ОП-5")
    assert a["display_status"] == "not_found" and a["status_line"] == "Не найдено" and a["table"] == []


def test_review_in_main_table_can_be_switched_off(small_engine, tmp_path):
    import copy
    import yaml
    from analog_search.matching.rules import RULES_PATH
    raw = copy.deepcopy(yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")))
    raw["policy"]["ui"]["review_in_main_table"] = False
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    eng = SearchEngine(small_engine.records, small_engine.retriever, load_rules(p), "bm25")
    a = eng.search("кабель КВВГЭнг-LS 5*1.5")
    assert a["display_status"] == "not_found" and a["table"] == [] and a["unconfirmed"]        # the list still exists, the table is empty
