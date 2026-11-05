"""the measurement is a protocol, its numbers have definitions, runs are separate artifacts."""
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import evaluate as ev  # noqa: E402
from judgments import Judgments  # noqa: E402

BUILT = (ROOT / "data" / "processed" / "catalog_v2.jsonl").exists() and (ROOT / "data" / "index" / "dense.npy").exists() \
    and (ROOT / "models" / "multilingual-e5-small" / "onnx" / "model.onnx").exists()
real = pytest.mark.skipif(not BUILT, reason="catalog, index or model are not built")


# ---------------------------------------------------------------- importing is free of side effects
def test_importing_the_module_builds_nothing_and_writes_nothing():
    summary = ROOT / "reports" / "quality" / "quality.md"
    before = (summary.stat().st_mtime_ns, hashlib.sha256(summary.read_bytes()).hexdigest()) if summary.exists() else None
    runs = ROOT / "reports" / "quality" / "runs"
    before_runs = sorted(p.name for p in runs.iterdir()) if runs.exists() else []
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(ROOT / 'eval')!r}); import evaluate; print('ok')"],
                       capture_output=True, text=True, cwd="/")
    assert r.returncode == 0 and "ok" in r.stdout
    assert time.perf_counter() - t0 < 10                                              # no engine, no model, no queries
    assert ((summary.stat().st_mtime_ns, hashlib.sha256(summary.read_bytes()).hexdigest()) if summary.exists() else None) == before
    assert (sorted(p.name for p in runs.iterdir()) if runs.exists() else []) == before_runs


# ---------------------------------------------------------------- the status of the set comes from the file
def test_the_report_never_claims_independence_the_file_does_not_state():
    class Rules:
        version, status = "x", "assumption"
    text = " ".join(ev.status_lines({"set_name": "s", "usage": "regression", "independence": "not_independent", "reviewed_by": None}, 40, 26, Rules()))
    assert "внутренняя регрессия" in text and "не указан" in text
    text = " ".join(ev.status_lines({"set_name": "s", "usage": "independent", "independence": "independent", "reviewed_by": "Иванов И."}, 20, 12, Rules()))
    assert "внутренняя регрессия" not in text and "Иванов И." in text


def test_no_hard_coded_claim_about_tuning_is_left_in_the_generator():
    src = (ROOT / "eval" / "evaluate.py").read_text(encoding="utf-8")
    for phrase in ("по ним система не настраивалась", "система на них не запускалась и не настраивалась"):
        assert phrase not in src


# ---------------------------------------------------------------- the definitions of the metrics, on a controlled example
class FakeCtx:
    """An answer where the expected row is only 6th in the union of the diagnostic lists."""
    def __init__(self, answers):
        self._answers = answers

    def candidates(self, text, mode, n=50):
        return ["r1", "r2", "r3", "r4", "r5", "r6", "r7"][:n]

    def answer(self, eng, text):
        return self._answers[text]


def row(i):
    return {"id": f"r{i}"}


def test_a_row_that_is_only_in_the_diagnostic_lists_is_not_a_display_hit():
    answers = {"q": {"results": [], "table": [row(1), row(2), row(3), row(4), row(5)], "unconfirmed": [row(6)]},
               "n": {"results": [], "table": [], "unconfirmed": []}}
    queries = [{"id": "Q", "query": "q", "answer_rows": ["r6"]}, {"id": "N", "query": "n", "answer_rows": []}]
    m, cand, _ = ev.separate_metrics(FakeCtx(answers), queries, Judgments(None), eng=object())
    assert m["display_hit@5"] == 0 and m["confirmed_hit@5"] == 0                      # r6 is not on the screen
    assert m["candidate_hit@50"] == 1                                                  # but B2 had it
    assert m["negative_empty_display"] == 1 and m["negative_refusal"] == 1


def test_unknown_does_not_disappear_from_the_technical_errors():
    answers = {"q": {"results": [row(1)], "table": [row(1)], "unconfirmed": []}}
    queries = [{"id": "Q", "query": "q", "answer_rows": ["r1"]}]
    m, _, _ = ev.separate_metrics(FakeCtx(answers), queries, Judgments(None), eng=object())
    assert m["technical_error@1_results"] == {"conflict": 0, "no_conflict": 0, "unknown": 1, "first_rows": 1}


def test_a_judged_conflict_is_counted_and_a_row_outside_the_list_is_not_called_wrong():
    answers = {"q": {"results": [row(9)], "table": [row(9)], "unconfirmed": []}}
    queries = [{"id": "Q", "query": "q", "answer_rows": ["r1"]}]
    judg = Judgments({"version": 1, "queries": {"Q": {"constraints": [{"attribute": "power_kw", "expected": 0.1}]}},
                      "pairs": [{"query_id": "Q", "candidate_id": "r9", "critical_values": {"power_kw": 0.3}}]})
    m, _, _ = ev.separate_metrics(FakeCtx(answers), queries, judg, eng=object())
    assert m["technical_error@1_results"]["conflict"] == 1
    assert m["confirmed_top1_outside_answer_rows"] == 1 and m["confirmed_top1_in_answer_rows"] == 0   # reported as «not assessed»


def test_the_separated_table_prints_a_denominator_for_every_number():
    answers = {"q": {"results": [row(1)], "table": [row(1)], "unconfirmed": []}, "n": {"results": [], "table": [], "unconfirmed": []}}
    queries = [{"id": "Q", "query": "q", "answer_rows": ["r1"]}, {"id": "N", "query": "n", "answer_rows": []}]
    m, _, _ = ev.separate_metrics(FakeCtx(answers), queries, Judgments(None), eng=object())
    lines = ev.separate_table({s: m for s in ev.SPLITS})
    assert all("/" in c or "%" in c or c.strip().isdigit() or "·" in c for l in lines[2:] for c in l.split("|")[2:-1] if c.strip())


# ---------------------------------------------------------------- runs are separate, repeatable artifacts
@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    d = tmp_path_factory.mktemp("mini")
    q = {"usage": "regression", "independence": "not_independent", "set_name": "mini", "reviewed_by": None,
         "splits": {"dev": "тестовый набор: СВОЙ ТЕКСТ ПРО ИСПОЛЬЗОВАНИЕ", "test": "t", "holdout": "h"},
         "queries": [
             {"id": "A1", "split": "dev", "query": "ворота откатные 6000х2250", "answer_rows": ["row-000159"], "answer_note": "", "key": None},
             {"id": "A2", "split": "dev", "query": "огнетушитель порошковый ОП-5", "answer_rows": [], "answer_note": "", "key": None}]}
    p = d / "q.yaml"
    p.write_text(yaml.safe_dump(q, allow_unicode=True), encoding="utf-8")
    return d, p


@real
def test_a_run_writes_its_own_directory_with_inputs_and_a_manifest(mini):
    d, qpath = mini
    summary = ROOT / "reports" / "quality" / "results.json"
    before = hashlib.sha256(summary.read_bytes()).hexdigest()
    ev.main(["--queries", str(qpath), "--output-dir", str(d / "runs"), "--run-id", "r1", "--no-ablations"])
    run = d / "runs" / "r1"
    assert {p.name for p in run.iterdir()} >= {"quality.md", "results.json", "manifest.json", "queries.yaml", "rules.yaml", "dictionaries.yaml", "judgments.yaml"}
    report = (run / "quality.md").read_text(encoding="utf-8")
    assert "СВОЙ ТЕКСТ ПРО ИСПОЛЬЗОВАНИЕ" in report and "internal-regression" not in report and "mini" in report
    m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert len(m["rules_sha256"]) == 64 and m["catalog_manifest"]["catalog_sha256"] and m["dense_meta"]["spec"]["revision"] and "commit" in m["git"]
    assert hashlib.sha256(summary.read_bytes()).hexdigest() == before                      # the summary is untouched without --update-summary


@real
def test_a_run_is_never_overwritten_and_two_runs_are_separate(mini):
    d, qpath = mini
    with pytest.raises(SystemExit):
        ev.main(["--queries", str(qpath), "--output-dir", str(d / "runs"), "--run-id", "r1", "--no-ablations"])
    ev.main(["--queries", str(qpath), "--output-dir", str(d / "runs"), "--run-id", "r2", "--no-ablations"])
    a = json.loads((d / "runs" / "r1" / "results.json").read_text(encoding="utf-8"))
    b = json.loads((d / "runs" / "r2" / "results.json").read_text(encoding="utf-8"))
    assert a["separate"] == b["separate"] and a["run_id"] == "r1" and b["run_id"] == "r2"   # the same snapshot gives the same metrics


# ---------------------------------------------------------------- the blind sheets
def test_the_blind_sheet_hides_the_system_and_is_reproducible(tmp_path):
    import make_blind_sheets as bs
    spec = {"queries": [{"id": "Q1", "query": "q", "answer_rows": ["row-1"]}]}
    by_id = {f"row-{i}": {"source_row": i, "raw_name": f"имя {i}", "price": 10 * i} for i in (1, 2, 3)}
    a = bs.build_rows(spec, {"Q1": ["row-2", "row-3", "row-2"]}, by_id, seed=1)
    b = bs.build_rows(spec, {"Q1": ["row-2", "row-3", "row-2"]}, by_id, seed=1)
    assert a == b and sorted(r[2] for r in a) == ["row-1", "row-2", "row-3"]            # duplicates removed, expected row included
    assert not any(w in " ".join(bs.COLUMNS).lower() for w in ("score", "verdict", "system", "baseline", "rank"))
    assert all(r[6:] == ["", "", "", "", ""] for r in a)                                # nothing pre-filled


def test_the_protocol_and_the_template_exist_and_agree():
    assert "один итоговый прогон" in (ROOT / "eval" / "PROTOCOL.md").read_text(encoding="utf-8").lower()
    t = yaml.safe_load((ROOT / "eval" / "independent_template.yaml").read_text(encoding="utf-8"))
    assert t["usage"] == "independent" and t["independence"] == "independent" and "frozen_commit" in t
