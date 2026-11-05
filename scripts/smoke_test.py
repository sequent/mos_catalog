"""Smoke test of a RUNNING service: what a person would check by hand after `docker compose up`, as a script.

Only the standard library is used, so the same file runs on the host and inside the container.
Exit code 0 only if every check passed. The report names what was checked and on which platform.

Checks: /health; the page; a confirmed answer; an allowed substitution (with the rule); a «requires check» answer; an empty refusal;
limit=1 is obeyed; 422 for a wrong request; and (unless --no-expected) 10 control queries compared with a recorded snapshot of the local
engine: ids, order, verdicts, prices and the score with a tolerance; latency and other volatile fields are ignored. The snapshot is bound
to the hash of rules.yaml: if the service runs other rules, the comparison is reported as SKIPPED (not as a failure, not as a pass).

Run:  python3 scripts/smoke_test.py --base-url http://localhost:8765 [--expected eval/smoke_expected.json] [--report out.json]
      python3 scripts/smoke_test.py --write-expected     (host only: records the snapshot from the local engine)
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPECTED = ROOT / "eval" / "smoke_expected.json"
QUERIES = [                         # the seven of the assignment, the two assumptions of the author, a reproduced defect
    "кабель КВВГнг-LS 7*1.5", "камера IP уличная 2 Мп вариофокальная", "ворота откатные 6000х2250", "труба гофра ПВХ 25 мм",
    "ИБП 12В для улицы", "трансформатор силовой сухой 1000 кВА", "огнетушитель порошковый ОП-5",
    "Инфракрасный термометр любого производителя", "Тройник 25х30 мм любой цвет", "светильник светодиодный 100 вт IP66",
]
SCORE_TOLERANCE = 1e-3


def call(base: str, path: str, body: dict | None = None, timeout: float = 30):
    req = urllib.request.Request(base.rstrip("/") + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def fingerprint(answer: dict) -> dict:
    """Everything of an answer that must be the same on every machine; no time, no host."""
    return {"status": answer["status"], "display_status": answer["display_status"], "status_line": answer["status_line"],
            "table": [{"id": x["id"], "verdict": x["verdict"], "price": x["price"], "score": x["score"], "needs_review": x["needs_review"],
                       "match_basis": x["match_basis"]} for x in answer["table"]]}


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool | None, detail: str = ""):
        self.checks.append({"name": name, "result": "skipped" if ok is None else ("pass" if ok else "FAIL"), "detail": detail})
        print(f"  [{self.checks[-1]['result'].upper():7}] {name}" + (f": {detail}" if detail else ""))

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.checks if c["result"] == "FAIL"]


def run_checks(base: str, expected_path: Path | None, rep: Report) -> dict:
    health = {}
    code, health = call(base, "/health")
    rep.add("GET /health", code == 200 and isinstance(health, dict) and health.get("status") == "ok",
            f"HTTP {code}" if code != 200 else f"rows={health.get('rows')}, rules={health.get('rules_version')}, artifacts={health.get('artifacts')}")
    if isinstance(health, dict):
        rep.add("health: 2381 строк, 2277 для поиска, режим hybrid", (health.get("rows"), health.get("searchable"), health.get("mode")) == (2381, 2277, "hybrid"))
        rep.add("health: идентификаторы артефактов (каталог, ревизия модели, правила)", all(k in health.get("artifacts", {}) for k in ("catalog_sha256", "model_revision", "rules_sha256")))
    code, page = call(base, "/")
    rep.add("GET / (страница)", code == 200 and "Поиск аналогов" in str(page) and 'id="form"' in str(page))

    def search(q, **kw):
        c, a = call(base, "/search", {"query": q, **kw})
        return c, a

    c, a = search("ворота откатные 6000х2250")
    top = a["table"][0] if c == 200 and a["table"] else {}
    rep.add("подтверждённая позиция: ворота 6000х2250 = 2250х6000", c == 200 and a["status"] == "found" and top.get("verdict") == "same" and not top.get("needs_review")
            and top.get("match_basis") == "request_constraints" and top.get("verdict_text") == "совпадает по указанным параметрам")
    c, a = search("камера 3 Мп")
    top = a["table"][0] if c == 200 and a["table"] else {}
    rule = next((r for r in top.get("comparison", []) if r["attribute"] == "resolution_mp"), {})
    rep.add("допустимая замена по правилу: камера 3 Мп → 2 Мп", c == 200 and top.get("verdict") == "substitute" and rule.get("rule") == "camera_resolution")
    c, a = search("светильник светодиодный 100 Вт IP66")
    rep.add("«к проверке»: светильник 100 Вт не подтверждается", c == 200 and a["status"] == "not_found" and a["display_status"] == "review" and a["table"]
            and all(x["needs_review"] for x in a["table"]) and len(a["table"]) <= 5)
    c, a = search("огнетушитель порошковый ОП-5")
    rep.add("пустой отказ: огнетушителя в базе нет", c == 200 and a["status"] == "not_found" and a["display_status"] == "not_found" and a["table"] == [])
    c, a = search("светильник светодиодный 100 Вт IP66", limit=1)
    rep.add("limit=1 соблюдается и в ветке «к проверке»", c == 200 and len(a["table"]) == 1)
    c, a = search("светильник светодиодный 100 вт IP66")
    rep.add("защита от потери единицы: «100 вт» не подтверждает светильник на 300 Вт", c == 200 and a["status"] == "not_found" and not any("300 Вт" in x["name"] for x in a["results"]))
    rep.add("limit=6 отклоняется (422)", search("кабель", limit=6)[0] == 422)
    rep.add("пустой запрос отклоняется (422)", search("   ")[0] == 422)
    rep.add("запрос длиннее 500 символов отклоняется (422)", search("а" * 501)[0] == 422)

    if expected_path is None:
        rep.add("сравнение 10 запросов с записанным снимком", None, "не запрашивалось (--no-expected)")
    elif not expected_path.exists():
        rep.add("сравнение 10 запросов с записанным снимком", False, f"нет файла {expected_path}")
    else:
        snap = json.loads(expected_path.read_text(encoding="utf-8"))
        have = (health.get("artifacts") or {}).get("rules_sha256") if isinstance(health, dict) else None
        if have != snap["rules_sha256"][:12]:
            rep.add("сравнение 10 запросов с записанным снимком", None, f"правила в сервисе {have}, снимок записан для {snap['rules_sha256'][:12]}: не сравнивается")
        else:
            for q, want in snap["answers"].items():
                c, a = search(q)
                got = fingerprint(a) if c == 200 else None
                same = got is not None and [x["id"] for x in got["table"]] == [x["id"] for x in want["table"]] \
                    and [(x["verdict"], x["price"], x["needs_review"], x["match_basis"]) for x in got["table"]] == [(x["verdict"], x["price"], x["needs_review"], x["match_basis"]) for x in want["table"]] \
                    and all(abs(x["score"] - y["score"]) <= SCORE_TOLERANCE for x, y in zip(got["table"], want["table"])) \
                    and (got["status"], got["display_status"], got["status_line"]) == (want["status"], want["display_status"], want["status_line"])
                rep.add(f"совпадает со снимком: {q}", same)
    return health if isinstance(health, dict) else {}


def write_expected(path: Path):
    sys.path.insert(0, str(ROOT / "pipeline"))
    from analog_search.engine import SearchEngine
    from analog_search.manifest import RULES_PATH, sha256_file
    eng = SearchEngine.build()
    answers = {q: fingerprint(eng.search(q)) for q in QUERIES}
    path.write_text(json.dumps({"rules_sha256": sha256_file(RULES_PATH), "tolerance": SCORE_TOLERANCE, "queries": QUERIES, "answers": answers}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"снимок записан: {path} ({len(answers)} запросов)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8765")
    ap.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    ap.add_argument("--no-expected", action="store_true")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--write-expected", action="store_true")
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait for /health before checking")
    args = ap.parse_args(argv)
    if args.write_expected:
        write_expected(args.expected)
        return 0
    deadline = time.time() + args.wait
    while args.wait and time.time() < deadline:
        try:
            if call(args.base_url, "/health", timeout=3)[0] == 200:
                break
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    rep = Report()
    print(f"Проверка {args.base_url} на {platform.platform()} ({platform.machine()}), Python {platform.python_version()}")
    t0 = time.time()
    try:
        health = run_checks(args.base_url, None if args.no_expected else args.expected, rep)
    except (OSError, urllib.error.URLError) as e:
        rep.add("доступность сервиса", False, str(e))
        health = {}
    summary = {"base_url": args.base_url, "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
               "seconds": round(time.time() - t0, 1), "health": health, "checks": rep.checks,
               "passed": sum(c["result"] == "pass" for c in rep.checks), "failed": len(rep.failed), "skipped": sum(c["result"] == "skipped" for c in rep.checks)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nИтог: {summary['passed']} прошло, {summary['failed']} провалено, {summary['skipped']} пропущено")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
