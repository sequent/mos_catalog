"""one limit of the main table, 1 ≤ limit ≤ 5, in the engine, the CLI and the API.

Reproduced defect: `search("светильник светодиодный 100 Вт IP66", top_k=1)` returned TWO rows in `table` (review branch ignored top_k).
"""
from pathlib import Path

import pytest

from analog_search.engine import MAX_TOP_K, SearchEngine

ROOT = Path(__file__).resolve().parents[1]
BUILT = (ROOT / "data" / "processed" / "catalog_v2.jsonl").exists() and (ROOT / "data" / "index" / "dense.npy").exists() \
    and (ROOT / "models" / "multilingual-e5-small" / "onnx" / "model.onnx").exists()
pytestmark = pytest.mark.skipif(not BUILT, reason="catalog, index or model are not built")

# one query for each state of the answer
SCENARIOS = {
    "only_confirmed": "ворота откатные 6000х2250",
    "confirmed_and_review": "труба гофра ПВХ 25 мм",
    "only_review": "светильник светодиодный 100 Вт IP66",
    "empty": "огнетушитель порошковый ОП-5",
}


@pytest.fixture(scope="module")
def engine():
    return SearchEngine.build()


def test_the_scenarios_really_are_in_the_state_they_claim(engine):
    st = {k: engine.search(q) for k, q in SCENARIOS.items()}
    assert st["only_confirmed"]["status"] == "found" and all(not x["needs_review"] for x in st["only_confirmed"]["table"])
    assert st["confirmed_and_review"]["status"] == "found" and any(x["needs_review"] for x in st["confirmed_and_review"]["table"])
    assert st["only_review"]["status"] == "not_found" and st["only_review"]["display_status"] == "review" and len(st["only_review"]["table"]) > 1
    assert st["empty"]["display_status"] == "not_found" and st["empty"]["table"] == []


@pytest.mark.parametrize("limit", [1, 3, 5])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_the_table_never_exceeds_the_limit(engine, scenario, limit):
    a = engine.search(SCENARIOS[scenario], top_k=limit)
    assert len(a["table"]) <= limit
    assert len(a["results"]) <= limit
    confirmed = [x for x in a["table"] if not x["needs_review"]]
    assert [x["id"] for x in confirmed] == [x["id"] for x in a["results"]][: len(confirmed)]      # order of the confirmed ones kept
    flags = [x["needs_review"] for x in a["table"]]
    assert flags == sorted(flags)                                                                 # confirmed rows first


def test_a_reproduced_defect(engine):
    a = engine.search("светильник светодиодный 100 Вт IP66", top_k=1)
    assert len(a["table"]) == 1


@pytest.mark.parametrize("scenario,expected", [("only_confirmed", "found"), ("confirmed_and_review", "found"), ("only_review", "review"), ("empty", "not_found")])
def test_statuses_do_not_depend_on_the_limit(engine, scenario, expected):
    for limit in (1, 3, 5):
        a = engine.search(SCENARIOS[scenario], top_k=limit)
        assert a["display_status"] == expected
        assert (a["status"] == "found") == bool(a["results"])                   # strict status: found only with a confirmed row
        if expected == "review":
            assert a["status"] == "not_found" and a["results"] == [] and a["table"]


def test_review_status_line_says_what_is_true(engine):
    a = engine.search(SCENARIOS["only_review"])
    assert a["status_line"] == "Подтверждённых подходящих позиций не найдено. Ниже варианты для проверки"


def test_review_rows_can_be_switched_off(engine, tmp_path):
    import copy
    import yaml
    from analog_search.matching import load_rules
    from analog_search.matching.rules import RULES_PATH
    raw = copy.deepcopy(yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")))
    raw["policy"]["ui"]["review_in_main_table"] = False
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    eng = SearchEngine(engine.records, engine.retriever, load_rules(p))
    a = eng.search(SCENARIOS["confirmed_and_review"], top_k=5)
    assert all(not x["needs_review"] for x in a["table"])


@pytest.mark.parametrize("bad", [0, -1, 6, 20])
def test_the_engine_itself_refuses_a_limit_outside_1_to_5(engine, bad):
    with pytest.raises(ValueError, match="top_k"):
        engine.search("ворота откатные 6000х2250", top_k=bad)


def test_the_limit_is_five():
    assert MAX_TOP_K == 5
