"""The control queries are data too: this file checks the data and the evidence that a «no answer» query has no answer.

1. Structure of eval/queries.yaml (ids, splits, answer rows exist in the dataset, the status of the set).
2. ABSENCE OF MATCHES with patterns in the ORIGINAL xlsx, independent of our system: every pattern group must be able to
   match (a synthetic witness) and must not be too wide (a near miss); a broken pattern is caught by mutation tests below.
   This shows that nothing in the xlsx matches the patterns; it does NOT prove that no semantic analogue exists: that is the decision
   of an expert (`expert_decision`).
3. The system, run on every «no answer» query, must not confirm anything («не найдено»), and whatever is shown is marked «требует проверки».
"""
import re
from pathlib import Path

import pytest
import yaml

from analog_search.source import read_source_rows

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "База_позиций_тестовое.xlsx"
SPEC = yaml.safe_load((ROOT / "eval" / "queries.yaml").read_text(encoding="utf-8"))
QUERIES = SPEC["queries"]
NO_ANSWER = [q for q in QUERIES if not q["answer_rows"]]
FLAGS = re.I


def matches_all(patterns, text):
    return all(re.search(p, text, FLAGS) for p in patterns)


@pytest.fixture(scope="module")
def dataset():
    """{row id: raw name}, read straight from the original spreadsheet (row id = Excel row number)."""
    return {f"row-{n:06d}": name for n, name, _ in read_source_rows(XLSX)}


# ---------------------------------------------------------------- the set and its status
def test_structure_of_the_query_file():
    ids = [q["id"] for q in QUERIES]
    assert len(ids) == len(set(ids))
    assert all(q["split"] in ("dev", "test", "holdout") for q in QUERIES)
    assert all(q["query"].strip() and "answer_rows" in q and "answer_note" in q for q in QUERIES)
    assert len(QUERIES) >= 20 and len(NO_ANSWER) >= 3, "the assignment: ≥ 20 queries, ≥ 3 without an answer"


def test_the_set_declares_what_it_is():
    """The report takes its headers from here: no header may claim «the system was not tuned» for a set that was used."""
    assert SPEC["usage"] == "regression" and SPEC["independence"] == "not_independent" and SPEC["set_name"]
    assert set(SPEC["splits"]) == {"dev", "test", "holdout"}
    assert all("использован при исправлениях" in v for v in SPEC["splits"].values())


def test_answer_rows_exist_in_the_dataset(dataset):
    assert not [(q["id"], r) for q in QUERIES for r in q["answer_rows"] if r not in dataset]


def test_no_query_is_a_verbatim_dataset_row(dataset):
    low = {v.lower().strip(): k for k, v in dataset.items()}
    assert not [q["id"] for q in QUERIES if q["query"].lower().strip() in low]


def test_the_old_field_name_is_gone():
    assert not any("absent_proof" in q for q in QUERIES)


# ---------------------------------------------------------------- the evidence of absence
def test_every_no_answer_query_carries_complete_checks():
    for q in NO_ANSWER:
        assert q.get("absence_checks"), q["id"]
        assert "expert_decision" in q, q["id"]                                      # the field exists; empty until an expert fills it
        for g in q["absence_checks"]:
            assert g["patterns"] and g["synthetic_witness"] and g["near_miss"], (q["id"], g)
        assert q.get("variants_checked"), q["id"]


GROUPS = [(q["id"], i, g) for q in NO_ANSWER for i, g in enumerate(q["absence_checks"])]
GROUP_IDS = [f"{a}#{b}" for a, b, _ in GROUPS]


@pytest.mark.parametrize("qid,i,group", GROUPS, ids=GROUP_IDS)
def test_a_pattern_group_can_match(qid, i, group):
    """A group that cannot match anything proves nothing. The synthetic witness must match EVERY pattern of the group."""
    for p in group["patterns"]:
        re.compile(p)                                                               # a valid regular expression
        assert re.search(p, group["synthetic_witness"], FLAGS), f"{qid}: «{p}» does not match the witness «{group['synthetic_witness']}»"
    assert matches_all(group["patterns"], group["synthetic_witness"])


@pytest.mark.parametrize("qid,i,group", GROUPS, ids=GROUP_IDS)
def test_a_pattern_group_is_not_too_wide(qid, i, group):
    """The near miss is close to the witness but must NOT satisfy the whole group (otherwise the group would match everything)."""
    assert not matches_all(group["patterns"], group["near_miss"]), f"{qid}: the group also matches the near miss «{group['near_miss']}»"


@pytest.mark.parametrize("qid,i,group", GROUPS, ids=GROUP_IDS)
def test_nothing_in_the_original_dataset_matches_the_group(qid, i, group, dataset):
    """Independent of our code: no xlsx row matches ALL patterns of the group."""
    hits = [rid for rid, name in dataset.items() if matches_all(group["patterns"], name)]
    assert not hits, f"{qid}: the group {group['patterns']} matches {hits[:3]}: the query may have an answer"


@pytest.mark.parametrize("q", NO_ANSWER, ids=[q["id"] for q in NO_ANSWER])
def test_the_recorded_variants_are_the_current_facts(q, dataset):
    """The notes about broader spellings (other × signs, units, synonyms) are checked against the xlsx, so they cannot go stale."""
    for v in q["variants_checked"]:
        n = sum(1 for name in dataset.values() if re.search(v["pattern"], name, FLAGS))
        assert n == v["rows_matched"], f"{q['id']}: {v['pattern']} matches {n} rows, recorded {v['rows_matched']}"
        if v["rows_matched"]:
            assert v["note"], f"{q['id']}: matches found must be explained (why they are not an answer)"


# ---------------------------------------------------------------- mutation tests: a broken check must be noticed
def test_a_pattern_with_a_wrong_unit_is_caught_by_the_witness():
    g = next(g for qid, _, g in GROUPS if qid == "X07")
    broken = [g["patterns"][0], g["patterns"][1].replace("Вт", "Wt")]
    assert not matches_all(broken, g["synthetic_witness"])                          # the witness test would fail on it


def test_a_pattern_with_a_stray_anchor_is_caught_by_the_witness():
    g = next(g for qid, _, g in GROUPS if qid == "X10" and g["patterns"][1] == "распашн")
    broken = [g["patterns"][0], "^" + g["patterns"][1]]
    assert not matches_all(broken, g["synthetic_witness"])


def test_a_too_wide_pattern_is_caught_by_the_near_miss():
    g = next(g for qid, _, g in GROUPS if qid == "X08")
    wide = [g["patterns"][0], ".*"]                                                 # the second pattern made as wide as possible
    assert matches_all(wide, g["synthetic_witness"])                                # the witness test alone would not notice it
    assert matches_all(wide, g["near_miss"])                                        # …but the near miss satisfies the wide group: the test above rejects it
    assert not matches_all(g["patterns"], g["near_miss"])                           # while the real group does not


def test_a_group_that_matches_a_real_row_is_caught_by_the_dataset_check(dataset):
    """Control: a pattern that DOES occur in the xlsx is reported, so an empty result is meaningful."""
    assert any(matches_all(["Ворота откатные"], name) for name in dataset.values())


# ---------------------------------------------------------------- the system under test
CATALOG = ROOT / "data" / "processed" / "catalog_v2.jsonl"
INDEX = ROOT / "data" / "index" / "dense.npy"
MODEL = ROOT / "models" / "multilingual-e5-small" / "onnx" / "model.onnx"
real = pytest.mark.skipif(not (CATALOG.exists() and INDEX.exists() and MODEL.exists()), reason="catalog, index or model are not built")


@pytest.fixture(scope="module")
def engine():
    from analog_search.engine import SearchEngine
    return SearchEngine.build()


@real
@pytest.mark.parametrize("q", NO_ANSWER, ids=[q["id"] for q in NO_ANSWER])
def test_system_says_not_found_where_there_is_no_answer(q, engine):
    a = engine.search(q["query"])
    assert a["status"] == "not_found", f"{q['id']}: confirmed {[x['name'] for x in a['results']]}"
    assert a["results"] == []
    assert a["display_status"] in ("review", "not_found")
    assert all(x["needs_review"] for x in a["table"]), "everything shown under «not found» must be marked as needing a check"
    assert not any(x["verdict"] in ("same", "substitute") for x in a["table"])


@real
def test_all_no_answer_queries_get_no_confirmed_result(engine):
    """No confirmed result on this FIXED set (it was used while changes were made, so it is a regression check, not an independent estimate)."""
    honest = sum(engine.search(q["query"])["status"] == "not_found" for q in NO_ANSWER)
    assert honest == len(NO_ANSWER), f"no confirmed result only on {honest} of {len(NO_ANSWER)}"
