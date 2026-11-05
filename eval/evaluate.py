"""Quality check (part 3 of the assignment): control queries, the whole scheme vs the simplest word search.

What a run produces: a RUN DIRECTORY `reports/quality/runs/<run-id>/` with
  quality.md, results.json            the report and the numbers;
  manifest.json                       rules / dictionary / catalog / index hashes, library versions, settings (what produced the numbers);
  queries.yaml, judgments.yaml,       copies of the inputs, rules.yaml, dictionaries.yaml: the run can be repeated.
  rules.yaml, dictionaries.yaml
The summary `reports/quality/quality.md` and `results.json` are updated only by the explicit flag `--update-summary`; runs are never overwritten.

The status of the set (usage, independence, what happened to each split) is READ FROM THE QUERY FILE: the report never claims that a
set «was not tuned on» unless the file says so.

Metrics (every denominator is printed):
  candidate_hit@50 / candidate_recall@50   an expected row is among the 50 candidates of B2 (hit: any; recall: the share of the expected rows);
  confirmed_hit@1 / @5                     the first / first five rows of `results` (strictly confirmed rows);
  display_hit@1 / @5                       the first / first five rows of `table` (what the screen shows, confirmed first, then «к проверке»);
  confirmation_coverage                    queries with a non-empty `results` / all (also separately for answerable and no-answer queries);
  confirmed_precision@1                    over every query with a confirmed first row (answerable and no-answer): acceptable (in the list of
                                           expected rows, or judged no_conflict) / unacceptable (a confirmation on a no-answer query, or judged
                                           conflict) / unassessed. The older share «first confirmed row in the list, among answerable queries»
                                           is printed under its own name and does not see confirmations on no-answer queries;
  technical_error@1                        conflict / no_conflict / unknown for the first row, judged by PEOPLE (eval/judgments.yaml), independent
                                           of the extractor; unlabelled means unknown, not «no error»;
  negative_refusal / _empty_display / _review_display   no-answer queries: no confirmed row / empty screen / variants «к проверке».

Run: .venv/bin/python -I eval/evaluate.py [--run-id NAME] [--queries PATH] [--output-dir DIR] [--update-summary] [--compare RESULTS.json]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analog_search.engine import SearchEngine  # noqa: E402
from analog_search.manifest import DICTIONARY_PATH, run_manifest  # noqa: E402
from analog_search.matching import load_rules  # noqa: E402
from analog_search.matching.rules import RULES_PATH  # noqa: E402
from analog_search.normalize import code_skeleton, compact_code  # noqa: E402
from analog_search.record import process  # noqa: E402
from analog_search.retrieval import RawBM25Baseline  # noqa: E402
from judgments import JUDGMENTS_PATH, Judgments  # noqa: E402
import runguard  # noqa: E402

DEFAULT_QUERIES = ROOT / "eval" / "queries.yaml"
SUMMARY_DIR = ROOT / "reports" / "quality"
SPLITS = ("all", "dev", "test", "holdout")
SPLIT_NAMES = {"all": "Все", "dev": "Dev", "test": "Test", "holdout": "Holdout"}


# ------------------------------------------------------------------ helpers
def pct(a, b):
    return f"{a}/{b}" if b else "—"


def canon(v):
    return tuple(canon(x) for x in v) if isinstance(v, list) else (round(float(v), 9) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)


def key_differs(rec: dict, key: dict | None) -> bool:
    """OLD metric, kept for comparison with earlier runs: the row states the key characteristic EXTRACTED BY OUR EXTRACTOR and it is not the
    requested one. It is not independent: a value the extractor missed counts as «no difference». See technical_error@1 for the judged one."""
    if not key:
        return False
    vals = [a["value"] for a in rec["attributes"] if a["name"] == key["attribute"] and a["source"] == "main"]
    if not vals:
        return False
    if key["attribute"] == "model_codes":
        want = {compact_code(code_skeleton(x)) for x in key["value"]}
        return not (want & {c for v in vals for c in v})
    if key["attribute"] == "dims":
        return all(sorted(v) != sorted(key["value"]) for v in vals)
    return all(canon(v) != canon(key["value"]) for v in vals)


def split_queries(queries):
    return {"all": queries, **{s: [q for q in queries if q["split"] == s] for s in SPLITS[1:]}}


class Context:
    """Everything a run needs, built once (no global state: importing this module builds nothing)."""

    def __init__(self, rules_path=None, log: "runguard.AnswerLog | None" = None):
        self.log = log                                     # every full answer is written here as soon as it is obtained
        self.engine = SearchEngine.build(rules_path=rules_path)
        self.records = self.engine.records
        self.by_id = {r["id"]: r for r in self.records}
        self.baseline = RawBM25Baseline(self.records)
        self.engine_words = SearchEngine(self.records, self.engine.retriever, self.engine.rules, "bm25")     # the same chain without E5
        self._answers: dict[tuple[int, str], dict] = {}

    def answer(self, eng: SearchEngine, text: str) -> dict:
        key = (id(eng), text)
        if key not in self._answers:
            self._answers[key] = eng.search(text)
            if self.log is not None:
                self.log.write("words" if eng is self.engine_words else "full", text, self._answers[key])
        return self._answers[key]

    def candidates(self, text: str, mode: str, n: int = 50) -> list[str]:
        q = process(text, query=True)
        pool = "service" if q["is_service"] or q["is_adjustment"] else "main"
        return [self.records[c["index"]]["id"] for c in self.engine.retriever.search(text, top_k=n, mode=mode, pool=pool, q=q)]


# ------------------------------------------------------------------ systems: each returns (ordered ids, status, extra ids)
def make_systems(ctx: Context):
    def baseline(q):
        ids = [c["id"] for c in ctx.baseline.search(q, 5)]
        return ids, ("found" if ids else "not_found"), []

    def words(q):
        ids = [c["id"] for c in ctx.engine.retriever.search(q, 5, mode="bm25")]
        return ids, ("found" if ids else "not_found"), []

    def hybrid(q):
        ids = [c["id"] for c in ctx.engine.retriever.search(q, 5, mode="hybrid")]
        return ids, ("found" if ids else "not_found"), []

    def full(q, eng=None):
        a = ctx.answer(eng or ctx.engine, q) if eng is None else eng.search(q)
        return [x["id"] for x in a["results"]], a["status"], [x["id"] for x in a["unconfirmed"]]
    return {"1 простой поиск по словам (baseline)": baseline,
            "2 слова + нормализация и токены характеристик (B2 без смысла)": words,
            "3 слова + смысл, RRF (весь B2)": hybrid,
            "4 вся схема B1–B7": full}, full


def score_system(ctx, fn, queries, **kw):
    r = {"ans": 0, "hit5": 0, "hit1": 0, "hit5_lenient": 0, "wrong_key": 0, "wrong_top1": 0,
         "noans": 0, "honest": 0, "false_answer": 0, "per_query": {}}
    for q in queries:
        exp = set(q["answer_rows"])
        ids, status, extra = fn(q["query"], **kw)
        top = ids[0] if ids else None
        row = {"top5": ids, "status": status, "extra": extra, "top1": top}
        if exp:
            r["ans"] += 1
            r["hit5"] += bool(set(ids) & exp)
            r["hit1"] += bool(top and top in exp)
            r["hit5_lenient"] += bool((set(ids) | set(extra)) & exp)
        else:
            r["noans"] += 1
            r["honest"] += not ids
            r["false_answer"] += bool(ids)
        if top and top not in exp:
            r["wrong_top1"] += 1
            r["wrong_key"] += key_differs(ctx.by_id[top], q.get("key"))
            row["wrong"] = True
            row["wrong_key"] = key_differs(ctx.by_id[top], q.get("key"))
        row["rank"] = next((i for i, x in enumerate(ids, 1) if x in exp), None)
        r["per_query"][q["id"]] = row
    return r


def table(results):
    lines = ["| вариант | верный в топ-5 | верный на 1-м месте | 1-е место с другой важной характеристикой (по извлекателю, не независимо) | 1-е место чужое (любое) | «не найдено» там, где ответа нет | выдан ответ там, где его нет |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {pct(r['hit5'], r['ans'])} | {pct(r['hit1'], r['ans'])} | {r['wrong_key']} | {r['wrong_top1']} | "
                     f"{pct(r['honest'], r['noans'])} | {r['false_answer']} |")
    return lines


# ------------------------------------------------------------------ what the screen shows
def display_stats(ctx, queries):
    st = {"ans": 0, "shown": 0, "confirmed": 0, "noans": 0, "clean_refusal": 0, "review_shown": 0, "confirmed_on_noans": 0, "per_query": {}}
    for q in queries:
        a = ctx.answer(ctx.engine, q["query"])
        ids = [x["id"] for x in a["table"]]
        exp = set(q["answer_rows"])
        st["per_query"][q["id"]] = {"display_status": a["display_status"], "table": ids}
        if exp:
            st["ans"] += 1
            st["shown"] += bool(set(ids) & exp)
            st["confirmed"] += bool(set(x["id"] for x in a["results"]) & exp)
        else:
            st["noans"] += 1
            st["clean_refusal"] += not ids
            st["review_shown"] += bool(ids)
            st["confirmed_on_noans"] += bool(a["results"])
    return st


# ------------------------------------------------------------------ separated metrics
def separate_metrics(ctx, queries, judg: Judgments, eng=None, mode="hybrid"):
    eng = eng or ctx.engine
    pos, neg = [q for q in queries if q["answer_rows"]], [q for q in queries if not q["answer_rows"]]
    m = {"positive": len(pos), "negative": len(neg)}
    cand = {q["id"]: ctx.candidates(q["query"], mode) for q in queries}
    m["candidate_hit@50"] = sum(bool(set(cand[q["id"]]) & set(q["answer_rows"])) for q in pos)
    m["candidate_recall@50"] = (sum(len(set(cand[q["id"]]) & set(q["answer_rows"])) / len(q["answer_rows"]) for q in pos) / len(pos)) if pos else None
    ans = {q["id"]: ctx.answer(eng, q["query"]) for q in queries}
    for k in (1, 5):
        m[f"confirmed_hit@{k}"] = sum(bool({x["id"] for x in ans[q["id"]]["results"][:k]} & set(q["answer_rows"])) for q in pos)
        m[f"display_hit@{k}"] = sum(bool({x["id"] for x in ans[q["id"]]["table"][:k]} & set(q["answer_rows"])) for q in pos)
    m["confirmation_coverage_all"] = sum(bool(ans[q["id"]]["results"]) for q in queries)
    m["confirmation_coverage_positive"] = sum(bool(ans[q["id"]]["results"]) for q in pos)
    m["confirmation_coverage_negative"] = sum(bool(ans[q["id"]]["results"]) for q in neg)
    has_top = [q for q in pos if ans[q["id"]]["results"]]
    # OLD, narrower number (kept under its old keys so earlier runs stay comparable): among ANSWERABLE queries that have a confirmed row,
    # how often is the first one in the list of expected rows. It ignores every confirmation on a no-answer query.
    m["confirmed_top1_in_answer_rows"] = sum(ans[q["id"]]["results"][0]["id"] in q["answer_rows"] for q in has_top)
    m["confirmed_top1_outside_answer_rows"] = len(has_top) - m["confirmed_top1_in_answer_rows"]          # not assessed, not «wrong»
    m["confirmed_top1_total"] = len(has_top)
    # confirmed_precision@1 over EVERY non-empty `results` (answerable and no-answer queries): a confirmation on a no-answer query is a known error
    prec = {"acceptable": 0, "unacceptable": 0, "unassessed": 0}
    for q in queries:
        rows = ans[q["id"]]["results"]
        if not rows:
            continue
        rid = rows[0]["id"]
        if rid in q["answer_rows"]:
            prec["acceptable"] += 1
        elif not q["answer_rows"]:
            prec["unacceptable"] += 1                                         # agreed that nothing in the base answers the query
        else:
            v = judg.verdict(q["id"], rid)                                    # outside the list: only people's judgment decides
            prec["acceptable" if v == "no_conflict" else "unacceptable" if v == "conflict" else "unassessed"] += 1
    m["confirmed_precision@1"] = {**prec, "total": sum(prec.values())}
    for where in ("results", "table"):
        tally = {"conflict": 0, "no_conflict": 0, "unknown": 0}
        for q in queries:
            rows = ans[q["id"]][where]
            if rows:                                                          # only queries that have a first row are judged
                tally[judg.verdict(q["id"], rows[0]["id"])] += 1
        m[f"technical_error@1_{where}"] = {**tally, "first_rows": sum(bool(ans[q["id"]][where]) for q in queries)}
    m["negative_refusal"] = sum(not ans[q["id"]]["results"] for q in neg)
    m["negative_empty_display"] = sum(not ans[q["id"]]["table"] for q in neg)
    m["negative_review_display"] = sum(bool(ans[q["id"]]["table"]) for q in neg)
    return m, cand, ans


def separate_table(per_split: dict) -> list[str]:
    head = "| показатель (знаменатель) | " + " | ".join(SPLIT_NAMES[s] for s in SPLITS) + " |"
    lines = [head, "|---|" + "---|" * len(SPLITS)]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(per_split[s]) for s in SPLITS) + " |")
    row("candidate_hit@50: эталон среди 50 кандидатов B2 (с ответом)", lambda m: pct(m["candidate_hit@50"], m["positive"]))
    row("candidate_recall@50: средняя доля эталонных строк среди 50 (с ответом)", lambda m: "—" if m["candidate_recall@50"] is None else f"{100 * m['candidate_recall@50']:.0f} %")
    row("confirmed_hit@1: верная первой среди подтверждённых (с ответом)", lambda m: pct(m["confirmed_hit@1"], m["positive"]))
    row("confirmed_hit@5: верная в первых пяти подтверждённых (с ответом)", lambda m: pct(m["confirmed_hit@5"], m["positive"]))
    row("display_hit@1: верная первой в таблице на экране (с ответом)", lambda m: pct(m["display_hit@1"], m["positive"]))
    row("display_hit@5: верная в таблице на экране (с ответом)", lambda m: pct(m["display_hit@5"], m["positive"]))
    row("confirmation_coverage: запросов с подтверждённой строкой (все)", lambda m: pct(m["confirmation_coverage_all"], m["positive"] + m["negative"]))
    row("   … из них с ответом / без ответа", lambda m: f"{m['confirmation_coverage_positive']}/{m['positive']} · {m['confirmation_coverage_negative']}/{m['negative']}")
    row("confirmed_precision@1: приемлемых / неприемлемых / не оценено (все запросы с подтверждённой первой строкой)",
        lambda m: f"{m['confirmed_precision@1']['acceptable']} / {m['confirmed_precision@1']['unacceptable']} / {m['confirmed_precision@1']['unassessed']} "
                  f"({m['confirmed_precision@1']['total']})")
    row("доля запросов с ответом, у которых подтверждённая первая строка в списке эталонов (только положительные запросы; подтверждения на запросах без ответа не учитываются)",
        lambda m: pct(m["confirmed_top1_in_answer_rows"], m["confirmed_top1_total"]))
    for where, label in (("results", "подтверждённая первая"), ("table", "первая в таблице")):
        row(f"technical_error@1, {label}: конфликт / нет конфликта / не оценено (строк)",
            lambda m, w=where: f"{m[f'technical_error@1_{w}']['conflict']} / {m[f'technical_error@1_{w}']['no_conflict']} / {m[f'technical_error@1_{w}']['unknown']} ({m[f'technical_error@1_{w}']['first_rows']})")
    row("negative_refusal: без подтверждённой строки (без ответа)", lambda m: pct(m["negative_refusal"], m["negative"]))
    row("negative_empty_display: пустой экран (без ответа)", lambda m: pct(m["negative_empty_display"], m["negative"]))
    row("negative_review_display: на экране варианты «к проверке» (без ответа)", lambda m: pct(m["negative_review_display"], m["negative"]))
    return lines


# ------------------------------------------------------------------ the contribution of E5 (paired; same pool, dictionary, tokens, parameters)
def e5_experiment(ctx, queries, judg):
    out = {"stages": {}, "queries": {}}
    for label, mode, eng in (("без E5 (BM25 + токены)", "bm25", ctx.engine_words), ("с E5 (RRF, вес 0,5)", "hybrid", ctx.engine)):
        m, cand, ans = separate_metrics(ctx, queries, judg, eng=eng, mode=mode)
        out["stages"][label] = {"metrics": m}
        out["queries"][label] = {q["id"]: {"cand_hit": bool(set(cand[q["id"]]) & set(q["answer_rows"])) if q["answer_rows"] else None,
                                           "confirmed_hit5": bool({x["id"] for x in ans[q["id"]]["results"][:5]} & set(q["answer_rows"])) if q["answer_rows"] else None,
                                           "display_hit5": bool({x["id"] for x in ans[q["id"]]["table"][:5]} & set(q["answer_rows"])) if q["answer_rows"] else None,
                                           "confirmed_any": bool(ans[q["id"]]["results"])} for q in queries}
    return out


def e5_lines(exp, queries) -> list[str]:
    a, b = list(exp["stages"])
    ma, mb = exp["stages"][a]["metrics"], exp["stages"][b]["metrics"]
    n = ma["positive"]
    lines = ["| показатель | " + a + " | " + b + " |", "|---|---|---|",
             f"| candidate_hit@50 (с ответом) | {pct(ma['candidate_hit@50'], n)} | {pct(mb['candidate_hit@50'], n)} |",
             f"| candidate_recall@50 | {100 * ma['candidate_recall@50']:.0f} % | {100 * mb['candidate_recall@50']:.0f} % |",
             f"| confirmed_hit@5 в полной схеме (с ответом) | {pct(ma['confirmed_hit@5'], n)} | {pct(mb['confirmed_hit@5'], n)} |",
             f"| display_hit@5 в полной схеме (с ответом) | {pct(ma['display_hit@5'], n)} | {pct(mb['display_hit@5'], n)} |",
             f"| подтверждённая строка на запросах без ответа | {ma['confirmation_coverage_negative']}/{ma['negative']} | {mb['confirmation_coverage_negative']}/{mb['negative']} |",
             ""]
    qa, qb = exp["queries"][a], exp["queries"][b]
    for key, label in (("cand_hit", "в 50 кандидатах"), ("confirmed_hit5", "подтверждена верная"), ("display_hit5", "верная на экране")):
        won = [q["id"] for q in queries if q["answer_rows"] and qb[q["id"]][key] and not qa[q["id"]][key]]
        lost = [q["id"] for q in queries if q["answer_rows"] and qa[q["id"]][key] and not qb[q["id"]][key]]
        lines.append(f"- {label}: E5 добавляет {len(won)} ({', '.join(won) or '—'}), теряет {len(lost)} ({', '.join(lost) or '—'}).")
    lines.append("")
    return lines


# ------------------------------------------------------------------ speed: each mode measured the same way, with no answer cache
def _pctl(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))]


def timing_experiment(ctx, queries, repeats: int = 3) -> dict:
    """Milliseconds per query for B2 alone and for the whole chain, without and with E5. Same queries, same warm-up (one untimed pass of
    every call), same number of repeats, no answer cache: every timed call does the full work. Metrics are computed from saved answers,
    not from these calls."""
    texts = [q["query"] for q in queries]

    def b2(mode):
        def run(t):
            q = process(t, query=True)
            pool = "service" if q["is_service"] or q["is_adjustment"] else "main"
            ctx.engine.retriever.search(t, top_k=50, mode=mode, pool=pool, q=q)
        return run
    calls = {("B2", "без E5"): b2("bm25"), ("B2", "с E5"): b2("hybrid"),
             ("вся цепочка", "без E5"): ctx.engine_words.search, ("вся цепочка", "с E5"): ctx.engine.search}
    out = {"repeats": repeats, "queries": len(texts), "warm_up_passes": 1, "answer_cache": False, "modes": {}}
    for (part, mode), fn in calls.items():
        for t in texts:
            fn(t)                                                          # warm-up, not timed
        ms = []
        for _ in range(repeats):
            for t in texts:
                t0 = time.perf_counter()
                fn(t)
                ms.append(1000 * (time.perf_counter() - t0))
        out["modes"][f"{part}, {mode}"] = {"p50_ms": round(_pctl(ms, 0.5), 1), "p95_ms": round(_pctl(ms, 0.95), 1), "calls": len(ms)}
    return out


def timing_lines(t: dict) -> list[str]:
    lines = [f"Скорость (мс на запрос; {t['queries']} запросов × {t['repeats']} повторов, один непосчитанный прогрев, кэша ответов нет, обе ветки измерены одинаково; "
             "метрики качества считаются по сохранённым ответам, а не по этим вызовам):\n", "| что измерено | p50, мс | p95, мс |", "|---|---|---|"]
    lines += [f"| {k} | {v['p50_ms']} | {v['p95_ms']} |" for k, v in t["modes"].items()]
    return lines + [""]


# ------------------------------------------------------------------ the baseline WITH a refusal threshold (selected on dev only)
def threshold_baseline(ctx, by_split):
    def top1(q):
        r = ctx.baseline.search(q["query"], 5)
        return (r[0]["score"] if r else 0.0), [c["id"] for c in r]
    dev = by_split["dev"]
    scores = {q["id"]: top1(q) for q in by_split["all"]}
    neg_max = max([scores[q["id"]][0] for q in dev if not q["answer_rows"]] or [0.0])
    best = None
    for th in sorted({scores[q["id"]][0] for q in dev} | {neg_max + 1e-9}):
        if any(scores[q["id"]][0] >= th for q in dev if not q["answer_rows"]):
            continue                                                         # a false answer on a no-answer dev query
        hit = sum(bool(set(scores[q["id"]][1]) & set(q["answer_rows"])) for q in dev if q["answer_rows"] and scores[q["id"]][0] >= th)
        if best is None or hit >= best[1]:
            best = (th, hit)
    th = best[0]
    out = {"threshold": round(th, 3), "selected_on": f"dev ({len(dev)} запросов, {sum(1 for q in dev if not q['answer_rows'])} без ответа)", "per_split": {}}
    for s in SPLITS:
        qs = by_split[s]
        pos, neg = [q for q in qs if q["answer_rows"]], [q for q in qs if not q["answer_rows"]]
        out["per_split"][s] = {"hit5": sum(bool(set(scores[q["id"]][1]) & set(q["answer_rows"])) and scores[q["id"]][0] >= th for q in pos), "positive": len(pos),
                               "refused": sum(scores[q["id"]][0] < th for q in neg), "negative": len(neg)}
    return out


# ------------------------------------------------------------------ ablations
def rules_variant(mutate):
    raw = copy.deepcopy(yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")))
    mutate(raw)
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(raw, f, allow_unicode=True)
    f.close()
    return load_rules(f.name)


def set_rule_mode(raw, rid, mode):
    next(r for r in raw["rules"] if r["id"] == rid)["mode"] = mode


VARIANTS = {
    "полная схема (как есть)": (lambda r: None, "soft"),
    "тип — жёсткий фильтр (кандидаты другого/неизвестного типа отбрасываются)": (lambda r: None, "hard"),
    "комплекты без ограничения («та же позиция» возможна)": (lambda r: r["policy"]["composite"].update(max_compatibility=None), "soft"),
    "ранжирование только по баллу (без «класс прежде балла»)": (lambda r: r["policy"]["ranking"].update(class_first=False), "soft"),
    "без проверки слов запроса (порог 0)": (lambda r: r["policy"]["word_coverage"].update(min_for_confirmed=0.0), "soft"),
    "без требования подтверждённой характеристики": (lambda r: r["policy"]["evidence"].update(require_confirmed_attribute=False), "soft"),
    "порядок сторон у типов без правила: sorted": (lambda r: r["defaults"].update(dims_order_unlisted="sorted"), "soft"),
    "экран кабеля «Э» игнорируется (cable_screen: ignore)": (lambda r: set_rule_mode(r, "cable_screen", "ignore"), "soft"),
    "без защиты от неразобранных ограничений (blocking_reasons: [])": (lambda r: r["policy"]["unparsed_constraints"].update(blocking_reasons=[]), "soft"),
}


def probe_b(eng, probes):
    ok = wrong = 0
    for q, exp in probes:
        a = eng.search(q)
        ids = [x["id"] for x in a["results"]]
        ok += bool(set(ids) & exp)
        wrong += bool(ids) and ids[0] not in exp
    return ok / len(probes), wrong / len(probes)


# ------------------------------------------------------------------ the run
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--run-id", default=None, help="name of the run directory (default: a timestamp)")
    ap.add_argument("--output-dir", type=Path, default=SUMMARY_DIR / "runs", help="directory that holds run directories")
    ap.add_argument("--update-summary", action="store_true", help="also replace reports/quality/quality.md and results.json (explicit action)")
    ap.add_argument("--compare", type=Path, default=None, help="results.json of another run: a section with the changes is added")
    ap.add_argument("--no-ablations", action="store_true", help="skip ablations and probe B (faster)")
    return ap.parse_args(argv)


def status_lines(spec: dict, n_all: int, n_pos: int, rules, pre: dict | None = None) -> list[str]:
    reviewed = spec.get("reviewed_by")
    prior = (pre or {}).get("first_run")
    again = [f"**Повторный прогон этого набора** (файл с тем же SHA-256 уже запускался: прогон `{prior['run_id']}`, {prior['at']}): это регрессия, "
             "а не новая слепая попытка, независимо от run-id.\n"] if prior and spec.get("independence") == "independent" else []
    spec = {**spec, "independence": (pre or {}).get("effective_independence", spec.get("independence"))}
    return again + [f"Набор: **{spec.get('set_name', '—')}**; назначение: `{spec.get('usage', '—')}`; независимость: `{spec.get('independence', '—')}`; "
            f"просмотр экспертом (`reviewed_by`): {reviewed if reviewed else 'не указан'}.\n",
            f"{n_all} контрольных запросов ({n_pos} с ответом в базе, {n_all - n_pos} без ответа). Правила: версия {rules.version} ({rules.status}). "
            + ("**Набор использовался при принятии решений об изменениях: это внутренняя регрессия, а не независимая оценка качества.** "
               if spec.get("independence") != "independent" else "") + "Протокол и происхождение запросов: в начале `eval/queries.yaml`.\n"]


def main(argv=None):
    args = parse_args(argv)
    spec = yaml.safe_load(args.queries.read_text(encoding="utf-8"))
    queries = spec["queries"]
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.output_dir / run_id
    if out.exists():
        sys.exit(f"Каталог прогона уже существует: {out}. Прогоны не перезаписываются; укажите другой --run-id.")
    pre = runguard.preflight(spec, args.queries, runguard.snapshot())       # before anything is searched: a violation stops here
    out.mkdir(parents=True)
    snap = runguard.snapshot(out)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    claims = spec.get("independence") == "independent"
    inputs = {"queries": {"file": str(args.queries.name), "sha256": pre["queries_sha256"]},
              "judgments_sha256": runguard.sha256_file(JUDGMENTS_PATH) if JUDGMENTS_PATH.exists() else None,
              "judgments_pairs": len(Judgments.load().pairs)}

    def write_manifest(status: str, **more):
        m = run_manifest(run_id=run_id, queries=str(args.queries.name), retrieval_mode="hybrid", top_k=5, candidates=50, ablations=not args.no_ablations,
                         status=status, started_at=started, ended_at=None if status == "started" else datetime.now().astimezone().isoformat(timespec="seconds"),
                         inputs=inputs, code_snapshot=snap, effective_independence=pre["effective_independence"], **more)
        (out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    write_manifest("started")                                                # on disk before the first search
    if claims:
        runguard.record_usage(pre["queries_sha256"], run_id, spec, snap, "started")
    try:
        _run(args, spec, queries, run_id, out, pre, write_manifest, snap)
    except BaseException as exc:                                              # an interrupted run keeps its answers and says so
        write_manifest("failed", error=f"{type(exc).__name__}: {exc}")
        if claims:
            runguard.record_usage(pre["queries_sha256"], run_id, spec, snap, "failed")
        raise
    write_manifest("completed")
    if claims:
        runguard.record_usage(pre["queries_sha256"], run_id, spec, snap, "completed")


def _run(args, spec, queries, run_id, out, pre, write_manifest, snap):
    log = runguard.AnswerLog(out / "answers.jsonl.gz")
    try:
        _measure(args, spec, queries, run_id, out, pre, log)
    finally:
        log.close()


def _measure(args, spec, queries, run_id, out, pre, log):
    ctx = Context(log=log)
    judg = Judgments.load()
    by_split = split_queries(queries)
    systems, sys_full = make_systems(ctx)
    res = {name: {s: score_system(ctx, fn, by_split[s]) for s in SPLITS} for name, fn in systems.items()}
    names = list(systems)
    n_pos = sum(1 for q in queries if q["answer_rows"])

    L = ["# Проверка качества (часть 3 задания)\n", *status_lines(spec, len(queries), n_pos, ctx.engine.rules, pre),
         "**Как читать.** «Верный в топ-5» и «на 1-м месте» считаются по запросам, у которых ответ в базе есть. Для вариантов 1–3 «топ-5» — первые пять строк "
         "поиска; для полной схемы — до 5 *подтверждённых* строк (без вариантов «к проверке»). Колонка «1-е место с другой важной характеристикой» "
         "считается по характеристикам, которые извлекла наша же система: это **не независимая** оценка (независимая: раздел «Раздельные метрики»). "
         "«Не найдено» у вариантов 1–3 возможно, только если поиск вообще ничего не вернул.\n"]
    for s in SPLITS:
        title = f"Все {len(by_split['all'])} запросов" if s == "all" else f"{SPLIT_NAMES[s]} ({len(by_split[s])}): {spec.get('splits', {}).get(s, 'статус не указан в файле запросов')}"
        L += [f"## {title}\n", *table({n: r[s] for n, r in res.items()}), ""]
    full_all = res[names[3]]["all"]
    L.append(f"Диагностический список (подтверждённые + «не подтверждены», до {3 + 5} строк; НЕ то, что видит пользователь): верный ответ в {pct(full_all['hit5_lenient'], full_all['ans'])} запросов "
             f"(строго подтверждённые: {pct(full_all['hit5'], full_all['ans'])}).\n")

    display = {s: display_stats(ctx, by_split[s]) for s in SPLITS}
    L += ["## Что видит пользователь: основная таблица и пометка «требует проверки»\n",
          "Подтверждённые строки идут первыми, затем строки «требует проверки» (policy.ui). Подтверждение строгое и от этого решения не зависит.\n",
          "| выборка | верная строка подтверждена | верная строка в таблице (подтверждена или к проверке) | без ответа: подтверждён ответ (должно быть 0) | без ответа: пустой экран | без ответа: на экране варианты «к проверке» |",
          "|---|---|---|---|---|---|"]
    for s in SPLITS:
        d = display[s]
        L.append(f"| {SPLIT_NAMES[s]} | {pct(d['confirmed'], d['ans'])} | {pct(d['shown'], d['ans'])} | {d['confirmed_on_noans']} | {pct(d['clean_refusal'], d['noans'])} | {pct(d['review_shown'], d['noans'])} |")
    L.append("")

    per_split, per_split_cand = {}, {}
    for s in SPLITS:
        per_split[s], per_split_cand[s], _ = separate_metrics(ctx, by_split[s], judg)
    n_pairs = len(judg.pairs)
    L += ["## Раздельные метрики : знаменатель у каждого числа\n",
          f"Независимая разметка технических ошибок (`eval/judgments.yaml`): размечено пар {n_pairs}. Пока пара не размечена людьми, её результат «не оценено», а не «ошибки нет».\n",
          *separate_table(per_split), ""]

    e5 = e5_experiment(ctx, queries, judg)
    L += ["## Вклад E5: одна и та же цепочка с E5 и без него\n",
          "Одни и те же пул, словарь, токены, параметры BM25, правила, ранжирование и экран; различается только наличие смыслового списка. "
          "Преимущество всей системы над сырым поиском по словам (раздел выше) **не равно** вкладу E5: оно включает нормализацию, словарь, токены и правила.\n",
          *e5_lines(e5, queries)]
    timing = timing_experiment(ctx, queries)
    L += ["### Скорость\n", *timing_lines(timing)]

    tb = threshold_baseline(ctx, by_split)
    L += ["## Базовый поиск по словам с порогом отказа (ориентир)\n",
          f"Порог по лучшей оценке сырого BM25 выбран на {tb['selected_on']}: наибольший hit@5 при отсутствии ложных ответов на запросах без ответа; порог = {tb['threshold']}. "
          "Дальше применён без изменений. Это экспериментальный ориентир, а не надёжно откалиброванный отказ: dev мал.\n",
          "| выборка | верный в топ-5 (с порогом) | отказ на запросах без ответа |", "|---|---|---|"]
    for s in SPLITS:
        t = tb["per_split"][s]
        L.append(f"| {SPLIT_NAMES[s]} | {pct(t['hit5'], t['positive'])} | {pct(t['refused'], t['negative'])} |")
    L.append("")

    if args.compare and args.compare.exists():
        before = json.loads(args.compare.read_text(encoding="utf-8"))["systems"][names[3]]
        L += [f"## Сравнение с прогоном `{args.compare.parent.name}/{args.compare.name}`\n", "| выборка | подтверждено было | стало | в топ-5 + «не подтверждены» было | стало |", "|---|---|---|---|---|"]
        for s in SPLITS:
            b, a_ = before[s], res[names[3]][s]
            L.append(f"| {SPLIT_NAMES[s]} | {pct(b['hit5'], b['ans'])} | {pct(a_['hit5'], a_['ans'])} | {pct(b['hit5_lenient'], b['ans'])} | {pct(a_['hit5_lenient'], a_['ans'])} |")
        changed = [(qid, before["all"]["per_query"][qid]["rank"], res[names[3]]["all"]["per_query"][qid]["rank"]) for qid in before["all"]["per_query"]
                   if qid in res[names[3]]["all"]["per_query"] and bool(before["all"]["per_query"][qid]["rank"]) != bool(res[names[3]]["all"]["per_query"][qid]["rank"])]
        L += ["", "Запросы, у которых подтверждение изменилось: " + (", ".join(f"{q} ({'—' if b is None else b} → {'—' if a_ is None else a_})" for q, b, a_ in changed) or "нет") + ".\n"]

    L += ["## По каждому запросу\n", "Место верной строки в топ-5 (— нет); для полной схемы: статус и первая выданная строка.\n",
          "| № | сплит | запрос | верные | 1 | 2 | 3 | 4: место | 4: статус | 4: первая строка / причина |", "|---|---|---|---|---|---|---|---|---|---|"]
    for q in queries:
        cells = [str(res[n]["all"]["per_query"][q["id"]]["rank"] or "—") for n in names[:3]]
        pq = res[names[3]]["all"]["per_query"][q["id"]]
        top = ctx.by_id[pq["top1"]]["raw_name"][:48] if pq["top1"] else (("в «не подтверждены»: " + ctx.by_id[pq["extra"][0]]["raw_name"][:36]) if pq["extra"] else "—")
        L.append(f"| {q['id']} | {q['split']} | {q['query']} | {', '.join(q['answer_rows']) or 'нет ответа'} | " + " | ".join(cells) + f" | {pq['rank'] or '—'} | {pq['status']} | {top} |")
    L.append("")

    abl = {}
    if not args.no_ablations:
        probes = [(r["query"], set(r["expected_ids"].split())) for r in csv.DictReader((ROOT / "reports/retrieval/probe_numbers.csv").open(encoding="utf-8-sig"))]
        L += ["## Абляции: что даёт каждое решение схемы\n",
              "Одна и та же полная схема, в которой одно решение отключено или заменено. Слева все контрольные запросы (**1 запрос ≈ 4 п. п., различия в 1–2 запроса лежат в пределах шума**); "
              "справа пробный набор B (200 коротких запросов «тип + характеристики», ожидаемые ответы получены извлекателем: набор не независим и служит индикатором регрессий).\n",
              "| вариант | верный в топ-5 | на 1-м месте | 1-е место с другой характеристикой (по извлекателю) | «не найдено» где ответа нет | выдан ответ где его нет | B: верная подтверждена | B: 1-е место чужое |",
              "|---|---|---|---|---|---|---|---|"]
        for name, (mut, gate) in VARIANTS.items():
            eng = SearchEngine(ctx.records, ctx.engine.retriever, rules_variant(mut), "hybrid", type_gate=gate)
            r = score_system(ctx, lambda t, eng=eng: sys_full(t, eng=eng), by_split["all"])
            pb = probe_b(eng, probes)
            abl[name] = {"hit5": r["hit5"], "hit1": r["hit1"], "wrong_key": r["wrong_key"], "honest": r["honest"], "false_answer": r["false_answer"],
                         "probe_b_confirmed": pb[0], "probe_b_wrong_top1": pb[1]}
            L.append(f"| {name} | {pct(r['hit5'], r['ans'])} | {pct(r['hit1'], r['ans'])} | {r['wrong_key']} | {pct(r['honest'], r['noans'])} | {r['false_answer']} | "
                     f"{100 * pb[0]:.1f} % | {100 * pb[1]:.1f} % |")
        L.append("")

    report = "\n".join(L) + "\n"
    results = {"run_id": run_id, "set": {**{k: spec.get(k) for k in ("set_name", "usage", "independence", "reviewed_by")},
                                         "queries_sha256": pre["queries_sha256"], "effective_independence": pre["effective_independence"]},
               "systems": {n: dict(r.items()) for n, r in res.items()}, "ablations": abl, "display": display, "separate": per_split,
               "e5": {"stages": {k: {"metrics": v["metrics"]} for k, v in e5["stages"].items()}}, "timing": timing, "threshold_baseline": tb}
    (out / "quality.md").write_text(report, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for src, name in ((args.queries, "queries.yaml"), (JUDGMENTS_PATH, "judgments.yaml"), (RULES_PATH, "rules.yaml"), (DICTIONARY_PATH, "dictionaries.yaml")):
        if Path(src).exists():
            shutil.copy(src, out / name)
    if args.update_summary:
        shutil.copy(out / "quality.md", SUMMARY_DIR / "quality.md")
        shutil.copy(out / "results.json", SUMMARY_DIR / "results.json")
    print(report)
    print(f"Прогон сохранён: {out}" + ("; сводка reports/quality/quality.md обновлена" if args.update_summary else "; сводка не менялась (--update-summary)"))


if __name__ == "__main__":
    main()
