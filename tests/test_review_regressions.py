"""Regressions found by the external review: F1 spaced «в»/«а», F2 strict numbers in the rules, F3 model files and the index at start,
F5 metric definitions and speed, F6 the protocol of an independent run, S3 the same written value."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from analog_search import process
from analog_search.engine import SearchEngine
from analog_search.matching import compare, load_rules
from analog_search.matching.rules import RULES_PATH
from analog_search.normalize import normalize_text
from analog_search.retrieval import embed_catalog
from analog_search.retrieval.dense import embedding_spec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import evaluate as ev  # noqa: E402
import runguard  # noqa: E402
from judgments import Judgments  # noqa: E402

BASE = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- F1: «380 в» and «10 а» are not bare numbers
@pytest.mark.parametrize("text,expected", [
    ("светильник 380 в IP66", "светильник 380 В IP66"), ("светильник 10 а IP66", "светильник 10 А IP66"),
    ("выкл. 250 в, 10 а, ip20", "выкл. 250 В, 10 А, ip20"), ("светильник 380 В IP66", "светильник 380 В IP66"),
    ("светильник 380в IP66", "светильник 380в IP66"),
    ("кабель 100 в комплекте", "кабель 100 в комплекте"), ("2 а также", "2 а также"),               # a preposition / a conjunction stay words
])
def test_a_spaced_lowercase_volt_or_ampere_is_a_unit_unless_a_word_follows(text, expected):
    assert normalize_text(text) == expected


def attrs(text, query=True):
    return {a["name"]: a["value"] for a in process(text, query=query)["attributes"]}


@pytest.mark.parametrize("text", ["светильник светодиодный 380 в IP66", "светильник светодиодный 380 В IP66", "светильник светодиодный 380в IP66"])
def test_the_voltage_is_extracted_in_every_spelling(text):
    assert attrs(text)["voltage_v"] == [380]


def test_a_lowercase_ampere_after_a_space_is_a_current():
    assert attrs("светильник светодиодный 10 а IP66")["current_a"] == 10


def _compare(q, c):
    rules = load_rules()
    return compare(process(q, query=True), process(c), rules)


def test_380_volts_is_not_confirmed_by_a_220_volt_lamp_in_any_spelling():
    for q in ("светильник светодиодный 380 в IP66", "светильник светодиодный 380 В IP66"):
        comp = _compare(q, "Светильник светодиодный Un~220В, Pn=35Вт, IP66")
        assert comp.klass == "incompatible", q


def test_10_a_is_not_confirmed_by_16_a():
    assert _compare("светильник светодиодный 10 а IP66", "Светильник светодиодный 16 А IP66").klass == "incompatible"
    assert _compare("светильник светодиодный 10 а IP66", "Светильник светодиодный 10 А IP66").klass == "compatible"


@pytest.mark.skipif(not (ROOT / "data" / "index" / "dense.npy").exists(), reason="index is not built")
def test_the_380_volt_query_confirms_no_220_volt_lamp_on_the_real_catalog():
    eng = SearchEngine.build()
    a = eng.search("светильник светодиодный 380 в IP66")
    assert a["status"] != "found" and not a["results"]
    assert all(x["verdict"] != "same" for x in a["table"])
    b = eng.search("светильник светодиодный 220 в IP66")
    assert b["status"] == "found"                               # the same spelling for the right value still finds lamps


# ---------------------------------------------------------------- S3: the same written value is a match of the notation, and the answer says so
def test_the_same_unparsed_value_on_both_sides_is_reported_as_matched_by_text():
    comp = _compare("реле IP66 50 Гц", "Реле IP66 50 Гц")
    assert comp.klass == "compatible" and comp.text_matched == ["50 Гц"]
    other = _compare("реле IP66 50 Гц", "Реле IP66 60 Гц")
    assert other.klass == "unverified" and other.text_matched == [] and other.unparsed == ["50 Гц"]


def test_the_explanation_names_the_notation_only_match():
    from analog_search.matching.explain import explain
    q, c = process("реле IP66 50 Гц", query=True), process("Реле IP66 50 Гц")
    text = explain(compare(q, c, load_rules()), q, c)["text"]
    assert "50 Гц" in text and "по написанию" in text


# ---------------------------------------------------------------- F2: numbers of the rules file are validated when it loads
def load(tmp_path, mutate):
    raw = copy.deepcopy(BASE)
    mutate(raw)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_rules(p)


def cam(raw):
    return next(r for r in raw["rules"] if r["mode"] == "ladder")


@pytest.mark.parametrize("mutate,message", [
    (lambda r: cam(r).update(max_steps=1.9), "max_steps: нужно целое число"),
    (lambda r: cam(r).update(max_steps=True), "max_steps: нужно целое число"),
    (lambda r: cam(r).update(max_steps="1"), "max_steps: нужно целое число"),
    (lambda r: cam(r).update(max_steps=-1), "max_steps: нужно целое число"),
    (lambda r: cam(r).update(max_steps=float("nan")), "max_steps"),
    (lambda r: cam(r).update(ladder=[1, float("nan"), 3]), r"ladder\[1\]: нужно конечное число"),
    (lambda r: cam(r).update(ladder=[1, float("inf"), 3]), r"ladder\[1\]"),
    (lambda r: cam(r).update(ladder=[1, "два", 3]), r"ladder\[1\]"),
    (lambda r: cam(r).update(ladder=[1, True, 3]), "ladder"),
    (lambda r: cam(r).update(ladder="1 2 3"), "ladder"),
    (lambda r: r["policy"].pop("evidence"), "policy: нет обязательного поля «evidence»"),
    (lambda r: r["policy"].pop("composite"), "policy: нет обязательного поля «composite»"),
])
def test_a_bad_number_or_a_missing_block_stops_the_load_with_the_path(tmp_path, mutate, message):
    with pytest.raises(Exception, match=message):
        load(tmp_path, mutate)


def test_the_shipped_camera_rule_and_a_whole_step_still_load(tmp_path):
    assert load(tmp_path, lambda r: None).version
    assert load(tmp_path, lambda r: cam(r).update(max_steps=2)).version


# ---------------------------------------------------------------- F3: the content of the model files is checked; the server never embeds
def _model_dir(tmp_path, content=b"weights-0001"):
    d = tmp_path / "model"
    (d / "onnx").mkdir(parents=True)
    files = {}
    for name in ("onnx/model.onnx", "onnx/tokenizer.json"):
        (d / name).write_bytes(content)
        files[name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    (d / "model_manifest.json").write_text(json.dumps({"repo": "r", "revision": "rev", "files": files}), encoding="utf-8")
    return d


def test_the_spec_accepts_files_that_match_the_manifest(tmp_path):
    assert embedding_spec(_model_dir(tmp_path), 256)["revision"] == "rev"


def test_a_same_size_change_of_the_tokenizer_is_detected(tmp_path):
    d = _model_dir(tmp_path)
    (d / "onnx" / "tokenizer.json").write_bytes(b"weights-0002")                 # four bytes replaced, the size is the same
    with pytest.raises(ValueError, match="SHA-256"):
        embedding_spec(d, 256)


def test_a_missing_or_resized_file_is_still_refused(tmp_path):
    d = _model_dir(tmp_path)
    (d / "onnx" / "model.onnx").write_bytes(b"short")
    with pytest.raises(FileNotFoundError):
        embedding_spec(d, 256)


class CountingEncoder:
    dim = 4

    def __init__(self):
        self.calls = 0

    def spec(self):
        return {"model": "m", "revision": "r", "weights_sha256": "w", "tokenizer_sha256": "t", "max_length": 256}

    def encode_passages(self, texts):
        self.calls += 1
        return np.ones((len(texts), 4), dtype=np.float32)


RECS = []
for i, n in enumerate(("Труба ПВХ 25 мм", "Кабель КВВГнг 5х1,5")):
    r = process(n)
    r["id"] = f"row-{i:06d}"
    RECS.append(r)


def test_a_missing_index_stops_the_start_with_the_command_and_zero_embedding_calls(tmp_path):
    enc = CountingEncoder()
    with pytest.raises(RuntimeError, match="build_index.py"):
        embed_catalog(RECS, enc, tmp_path)
    assert enc.calls == 0 and not (tmp_path / "dense.npy").exists()


def test_a_stale_index_stops_the_start_and_is_not_rewritten(tmp_path):
    embed_catalog(RECS, CountingEncoder(), tmp_path, rebuild=True)
    before = (tmp_path / "dense.npy").read_bytes()
    enc = CountingEncoder()
    with pytest.raises(RuntimeError, match="build_index.py"):
        embed_catalog(list(reversed(RECS)), enc, tmp_path)
    assert enc.calls == 0 and (tmp_path / "dense.npy").read_bytes() == before


def test_only_an_explicit_rebuild_embeds(tmp_path):
    enc = CountingEncoder()
    embed_catalog(RECS, enc, tmp_path, rebuild=True)
    assert enc.calls == 1
    embed_catalog(RECS, enc, tmp_path)                                           # a valid index is simply loaded
    assert enc.calls == 1


def test_a_matrix_of_another_dtype_or_size_is_refused(tmp_path):
    embed_catalog(RECS, CountingEncoder(), tmp_path, rebuild=True)
    meta = json.loads((tmp_path / "dense_meta.json").read_text(encoding="utf-8"))
    enc = CountingEncoder()
    enc.dim = 384                                                                  # the contract says 384, the matrix has 4
    with pytest.raises(RuntimeError, match="размерность"):
        embed_catalog(RECS, enc, tmp_path)
    assert meta["dim"] == 4


def test_the_engine_start_never_asks_for_a_rebuild(monkeypatch):
    import analog_search.engine as engine
    import analog_search.manifest as manifest
    seen = {}

    def fake_embed(records, encoder, **kw):
        seen.update(kw)
        raise RuntimeError("stop")
    monkeypatch.setattr(manifest, "require_fresh_catalog", lambda *a, **k: None)
    monkeypatch.setattr(engine, "load_records", lambda *a, **k: RECS)
    monkeypatch.setattr(engine, "E5Encoder", lambda: object())
    monkeypatch.setattr(engine, "embed_catalog", fake_embed)
    with pytest.raises(RuntimeError, match="stop"):
        engine.SearchEngine.build()
    assert not seen.get("rebuild") and not seen.get("force")


def test_only_the_preparation_script_allows_a_rebuild():
    assert "rebuild=True" in (ROOT / "scripts" / "build_index.py").read_text(encoding="utf-8")
    assert "rebuild" not in (ROOT / "pipeline" / "analog_search" / "engine.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- F5: precision over all confirmations; speed measured the same way
class FakeCtx:
    def __init__(self, answers):
        self._answers = answers

    def candidates(self, text, mode, n=50):
        return []

    def answer(self, eng, text):
        return self._answers[text]


def _r(i):
    return {"id": f"r{i}"}


def test_a_confirmation_on_a_no_answer_query_counts_in_the_precision():
    answers = {"p": {"results": [_r(1)], "table": [_r(1)], "unconfirmed": []},
               "n": {"results": [_r(7)], "table": [_r(7)], "unconfirmed": []}}
    queries = [{"id": "P", "query": "p", "answer_rows": ["r1"]}, {"id": "N", "query": "n", "answer_rows": []}]
    m, _, _ = ev.separate_metrics(FakeCtx(answers), queries, Judgments(None), eng=object())
    assert m["confirmed_precision@1"] == {"acceptable": 1, "unacceptable": 1, "unassessed": 0, "total": 2}          # 1 of 2, not 1 of 1
    assert (m["confirmed_top1_in_answer_rows"], m["confirmed_top1_total"]) == (1, 1)                               # the narrower, older number


def test_a_confirmed_row_outside_the_list_is_unassessed_until_people_judge_it():
    answers = {"p": {"results": [_r(9)], "table": [_r(9)], "unconfirmed": []}}
    queries = [{"id": "P", "query": "p", "answer_rows": ["r1"]}]
    m, _, _ = ev.separate_metrics(FakeCtx(answers), queries, Judgments(None), eng=object())
    assert m["confirmed_precision@1"] == {"acceptable": 0, "unacceptable": 0, "unassessed": 1, "total": 1}
    judg = Judgments({"version": 1, "queries": {"P": {"constraints": [{"attribute": "power_kw", "expected": 0.1}]}},
                      "pairs": [{"query_id": "P", "candidate_id": "r9", "critical_values": {"power_kw": 0.3}}]})
    m, _, _ = ev.separate_metrics(FakeCtx(answers), queries, judg, eng=object())
    assert m["confirmed_precision@1"]["unacceptable"] == 1


def test_the_timing_measures_every_mode_with_the_same_work_and_no_cache():
    calls = {"words": 0, "full": 0, "b2": 0}

    class Eng:
        def __init__(self, key):
            self.key = key

        def search(self, text):
            calls[self.key] += 1

    class Retr:
        def search(self, text, **kw):
            calls["b2"] += 1

    class Ctx:
        engine_words, engine = Eng("words"), Eng("full")
    Ctx.engine.retriever = Retr()
    qs = [{"query": "труба 25 мм"}, {"query": "кабель 5х1,5"}]
    t = ev.timing_experiment(Ctx, qs, repeats=3)
    per_mode = (1 + 3) * len(qs)                                                    # one warm-up pass + 3 timed passes, in EVERY mode
    assert calls == {"words": per_mode, "full": per_mode, "b2": 2 * per_mode}
    assert set(t["modes"]) == {"B2, без E5", "B2, с E5", "вся цепочка, без E5", "вся цепочка, с E5"}
    assert all(v["calls"] == 3 * len(qs) and v["p50_ms"] <= v["p95_ms"] for v in t["modes"].values()) and t["answer_cache"] is False


def test_the_e5_experiment_no_longer_compares_cached_and_uncached_seconds():
    src = (ROOT / "eval" / "evaluate.py").read_text(encoding="utf-8")
    assert '"seconds": round(' not in src and "['seconds']" not in src


# ---------------------------------------------------------------- F6: an independent run is checked BEFORE any search
CLEAN = {"commit": "a" * 40, "tracked_changes": False, "untracked_files": {}}


def good_spec():
    return {"set_name": "s", "independence": "independent", "reviewed_by": "Иванов И.", "frozen_commit": "a" * 12,
            "queries": [{"id": "Q1", "split": "holdout", "query": "q", "answer_rows": []}]}


def pre(spec, snap, tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    return runguard.preflight(spec, f, snap, tmp_path / "ledger.jsonl")


def test_a_reviewed_frozen_clean_set_passes(tmp_path):
    assert pre(good_spec(), CLEAN, tmp_path)["effective_independence"] == "independent"


@pytest.mark.parametrize("mutate,snap,message", [
    (lambda s: s.update(reviewed_by=None), CLEAN, "reviewed_by"),
    (lambda s: s.pop("frozen_commit"), CLEAN, "frozen_commit"),
    (lambda s: s.update(frozen_commit="b" * 12), CLEAN, "не совпадает"),
    (lambda s: None, {**CLEAN, "tracked_changes": True}, "не чистое"),
    (lambda s: None, {**CLEAN, "untracked_files": {"pipeline/x.py": "h"}}, "не чистое"),
    (lambda s: None, {"commit": None, "tracked_changes": None, "untracked_files": None}, "frozen_commit"),
    (lambda s: s.update(queries=[]), CLEAN, "нет запросов"),
    (lambda s: s["queries"][0].pop("answer_rows"), CLEAN, "answer_rows"),
])
def test_an_independent_run_is_refused_for_each_missing_condition(tmp_path, mutate, snap, message):
    spec = good_spec()
    mutate(spec)
    with pytest.raises(SystemExit, match=message):
        pre(spec, snap, tmp_path)


def test_a_set_that_does_not_claim_independence_is_only_described(tmp_path):
    spec = {**good_spec(), "independence": "not_independent", "reviewed_by": None}
    assert pre(spec, {**CLEAN, "tracked_changes": True}, tmp_path)["effective_independence"] == "not_independent"


def test_a_second_run_under_another_run_id_is_not_a_new_blind_attempt(tmp_path):
    spec = good_spec()
    first = pre(spec, CLEAN, tmp_path)
    runguard.record_usage(first["queries_sha256"], "run-1", spec, CLEAN, "started", tmp_path / "ledger.jsonl")
    again = pre(spec, CLEAN, tmp_path)
    assert again["effective_independence"] == "regression_after_first_run" and again["first_run"]["run_id"] == "run-1"
    text = " ".join(ev.status_lines(spec, 1, 0, type("R", (), {"version": "x", "status": "y"})(), again))
    assert "Повторный прогон" in text and "run-1" in text and "внутренняя регрессия" in text


def test_a_changed_set_file_is_a_different_set(tmp_path):
    spec = good_spec()
    first = pre(spec, CLEAN, tmp_path)
    runguard.record_usage(first["queries_sha256"], "run-1", spec, CLEAN, "started", tmp_path / "ledger.jsonl")
    spec["queries"].append({"id": "Q2", "split": "holdout", "query": "z", "answer_rows": []})
    assert pre(spec, CLEAN, tmp_path)["first_run"] is None


def test_main_stops_before_the_engine_is_built_and_leaves_no_run_directory(tmp_path, monkeypatch):
    spec = {**good_spec(), "reviewed_by": None}
    q = tmp_path / "q.yaml"
    q.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(ev, "Context", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the engine must not be built")))
    with pytest.raises(SystemExit, match="reviewed_by"):
        ev.main(["--queries", str(q), "--output-dir", str(tmp_path / "runs"), "--run-id", "x", "--no-ablations"])
    assert not (tmp_path / "runs").exists()


def test_a_failed_run_keeps_a_manifest_that_says_so(tmp_path, monkeypatch):
    spec = {**good_spec(), "independence": "not_independent"}
    q = tmp_path / "q.yaml"
    q.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(ev, "Context", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        ev.main(["--queries", str(q), "--output-dir", str(tmp_path / "runs"), "--run-id", "x", "--no-ablations"])
    m = json.loads((tmp_path / "runs" / "x" / "manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "failed" and "boom" in m["error"] and m["inputs"]["queries"]["sha256"] and m["code_snapshot"]["commit"]


def test_the_answer_log_keeps_what_was_written_before_an_interruption(tmp_path):
    log = runguard.AnswerLog(tmp_path / "a.jsonl.gz")
    log.write("full", "q1", {"table": [{"comparison": [{"evidence": []}]}]})
    log.close()                                                                   # (flush happens at every write; close ends the gzip stream)
    got = runguard.read_answers(tmp_path / "a.jsonl.gz")
    assert got[0]["query"] == "q1" and "comparison" in got[0]["answer"]["table"][0]


def test_the_snapshot_records_untracked_executable_files(tmp_path):
    import subprocess
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "a@b"], ["git", "config", "user.name", "n"]):
        subprocess.run(cmd, cwd=tmp_path, check=True)
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "c"], cwd=tmp_path, check=True)
    assert runguard.is_clean(runguard.snapshot(root=tmp_path))
    (tmp_path / "pipeline" / "b.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "pipeline" / "a.py").write_text("x = 2\n", encoding="utf-8")
    out = tmp_path / "out"
    snap = runguard.snapshot(out, root=tmp_path)
    assert not runguard.is_clean(snap) and snap["tracked_changes"] and "pipeline/b.py" in snap["untracked_files"]
    assert "x = 2" in (out / "code.diff").read_text(encoding="utf-8")


# ---------------------------------------------------------------- F6: the blind sheet is built from saved results, nothing is searched
def test_the_blind_sheet_script_does_not_run_any_system():
    src = (ROOT / "eval" / "make_blind_sheets.py").read_text(encoding="utf-8")
    assert "Context(" not in src and "make_systems" not in src and "--from-run" in src


def test_the_first_rows_come_from_the_results_of_the_run():
    import make_blind_sheets as bs
    results = {"systems": {"a": {"all": {"per_query": {"Q1": {"top5": ["r1", "r2"]}}}}, "b": {"all": {"per_query": {"Q1": {"top5": []}}}}},
               "display": {"all": {"per_query": {"Q1": {"table": ["r3", "r4", "r5", "r6"]}}}}}
    got = bs.first_rows_from_results(results, {"queries": [{"id": "Q1"}]})
    assert got == {"Q1": ["r1", "r3", "r4", "r5"]}
