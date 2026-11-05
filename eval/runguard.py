"""What must be true BEFORE a measurement starts, and what must be saved WHILE it runs.

* `snapshot()`   the commit, whether tracked files differ from it (the diff is saved as `code.diff`), and every untracked file of the
                 code directories with its SHA-256: one commit does not fix a changed working tree.
* `preflight()`  for a set that claims `independence: independent`: the set must be reviewed (`reviewed_by`), frozen on the commit that
                 is checked out (`frozen_commit`), complete, and the code must be clean. Refuses BEFORE the engine is built.
* the ledger     `reports/quality/set_usage.jsonl`: the first run of a set (by the hash of its file) is its only blind attempt; any later
                 run, under whatever run-id, is a regression run and is reported as such.
* `AnswerLog`    full answers (with `comparison` and `evidence`) written to `answers.jsonl.gz` as they are obtained, so metrics and blind
                 sheets are built from them and nothing has to be searched again.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "reports" / "quality" / "set_usage.jsonl"
CODE_PATHS = ("pipeline", "config", "eval", "scripts", "web", "tests", "Dockerfile", "docker-compose.yml", "requirements.txt",
              "requirements-runtime.txt", "constraints.txt", "pytest.ini")


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str, root: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def snapshot(out: Path | None = None, root: Path = ROOT) -> dict:
    """Commit, tracked changes (saved as code.diff in `out`) and untracked files of the code directories with their hashes."""
    try:
        head = _git("rev-parse", "HEAD", root=root).strip()
        tracked = _git("status", "--porcelain", "--untracked-files=no", root=root).strip()
        untracked = [ln[3:] for ln in _git("status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS, root=root).splitlines()
                     if ln.startswith("??") and "__pycache__" not in ln and not ln.endswith(".pyc")]
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_changes": None, "untracked_files": None}
    snap = {"commit": head, "tracked_changes": bool(tracked),
            "untracked_files": {p: sha256_file(root / p) for p in untracked if (root / p).is_file()}}
    if out is not None and tracked:
        out.mkdir(parents=True, exist_ok=True)
        (out / "code.diff").write_text(_git("diff", "HEAD", root=root), encoding="utf-8")
        snap["diff_file"] = "code.diff"
    return snap


def is_clean(snap: dict) -> bool:
    return snap["commit"] is not None and not snap["tracked_changes"] and not snap["untracked_files"]


def read_ledger(path: Path = LEDGER) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def first_run(queries_sha: str, path: Path = LEDGER) -> dict | None:
    return next((r for r in read_ledger(path) if r["queries_sha256"] == queries_sha), None)


def append_ledger(entry: dict, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def preflight(spec: dict, queries_path: Path, snap: dict, ledger: Path = LEDGER) -> dict:
    """Checks that need no search. Returns {"queries_sha256", "effective_independence", "first_run"}; exits (SystemExit) on a violation
    for a set that claims to be independent. A set that does not claim it is only described honestly."""
    sha = sha256_file(queries_path)
    prior = first_run(sha, ledger)
    claims = spec.get("independence") == "independent"
    if claims:
        problems = []
        if not spec.get("reviewed_by"):
            problems.append("в файле запросов не указан `reviewed_by` (набор не просмотрен специалистом)")
        frozen = str(spec.get("frozen_commit") or "")
        if not frozen:
            problems.append("в файле запросов нет `frozen_commit` (код, на котором набор заморожен)")
        elif snap["commit"] is None or not snap["commit"].startswith(frozen):
            problems.append(f"выбранный коммит {snap['commit']} не совпадает с `frozen_commit` {frozen}")
        if not is_clean(snap):
            problems.append("рабочее дерево не чистое (изменённые или неотслеживаемые файлы кода): независимый прогон только на чистом коммите")
        queries = spec.get("queries") or []
        if not queries:
            problems.append("в файле нет запросов")
        for q in queries:
            miss = [k for k in ("id", "query", "split") if not q.get(k)] + ([] if isinstance(q.get("answer_rows"), list) else ["answer_rows"])
            if miss:
                problems.append(f"{q.get('id', '?')}: нет полей {', '.join(miss)}")
        if problems:
            raise SystemExit("Независимый прогон не начат (поиск не запускался):\n- " + "\n- ".join(problems))
    effective = spec.get("independence")
    if claims and prior:
        effective = "regression_after_first_run"          # a new run-id does not make a new blind attempt
    return {"queries_sha256": sha, "effective_independence": effective, "first_run": prior}


def record_usage(queries_sha: str, run_id: str, spec: dict, snap: dict, status: str, ledger: Path = LEDGER) -> None:
    append_ledger({"queries_sha256": queries_sha, "set_name": spec.get("set_name"), "run_id": run_id, "status": status,
                   "claimed_independence": spec.get("independence"), "commit": snap.get("commit"),
                   "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ledger)


class AnswerLog:
    """Full answers, one JSON line each, written as soon as they are obtained (an interrupted run keeps what it already got)."""

    def __init__(self, path: Path):
        self.path = path
        self._f = gzip.open(path, "wt", encoding="utf-8")

    def write(self, mode: str, query: str, answer: dict) -> None:
        self._f.write(json.dumps({"mode": mode, "query": query, "answer": answer}, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def read_answers(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]
