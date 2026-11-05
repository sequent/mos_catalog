"""the smoke test fails when the service is wrong, not only when it crashes; the recorded snapshot is current."""
import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from analog_search.api import create_app

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smoke_test", ROOT / "scripts" / "smoke_test.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

BUILT = (ROOT / "data" / "processed" / "catalog_v2.jsonl").exists() and (ROOT / "data" / "index" / "dense.npy").exists() \
    and (ROOT / "models" / "multilingual-e5-small" / "onnx" / "model.onnx").exists()
pytestmark = pytest.mark.skipif(not BUILT, reason="catalog, index or model are not built")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Served:
    def __init__(self, app):
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return f"http://127.0.0.1:{self.port}"
            time.sleep(0.05)
        raise RuntimeError("the test server did not start")

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(5)


@pytest.fixture(scope="module")
def engine():
    from analog_search.engine import SearchEngine
    return SearchEngine.build()


def failed_names(report_path):
    return [c["name"] for c in json.loads(Path(report_path).read_text(encoding="utf-8"))["checks"] if c["result"] == "FAIL"]


def run(base, tmp_path, *extra):
    rp = tmp_path / "report.json"
    code = smoke.main(["--base-url", base, "--report", str(rp), *extra])
    return code, json.loads(rp.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- a correct service passes, and the report says what and where
def test_a_correct_service_passes_every_check(engine, tmp_path):
    with Served(create_app(engine)) as base:
        code, rep = run(base, tmp_path)
    assert code == 0 and rep["failed"] == 0 and rep["skipped"] == 0 and rep["passed"] >= 20
    assert rep["platform"] and rep["machine"] and rep["health"]["artifacts"]["model_revision"]          # what was checked and where


# ---------------------------------------------------------------- a wrong service fails with a non-zero code (not only HTTP 500)
class Wrapper:
    """The real engine whose answer is altered: a service that answers 200 everywhere but is wrong."""
    def __init__(self, engine, alter):
        self._e, self._alter = engine, alter
        self.__dict__.update({k: getattr(engine, k) for k in ("records", "rules", "mode", "retriever")})

    def search(self, query, top_k=5, **kw):
        return self._alter(self._e.search(query, top_k=top_k, **kw), query, top_k)


def test_a_service_that_ignores_the_limit_fails(engine, tmp_path):
    def ignore_limit(answer, query, top_k):
        answer["table"] = (answer["table"] + answer["unconfirmed"])[:5] if top_k == 1 else answer["table"]
        if top_k == 1 and len(answer["table"]) < 2:
            answer["table"] = answer["table"] * 2
        return answer
    with Served(create_app(Wrapper(engine, ignore_limit))) as base:
        code, rep = run(base, tmp_path, "--no-expected")
    assert code != 0 and "limit=1 соблюдается и в ветке «к проверке»" in failed_names(tmp_path / "report.json")


def test_a_service_with_a_wrong_status_fails(engine, tmp_path):
    def always_found(answer, query, top_k):
        if "огнетушитель" in query:
            answer["status"], answer["display_status"] = "found", "found"
        return answer
    with Served(create_app(Wrapper(engine, always_found))) as base:
        code, rep = run(base, tmp_path, "--no-expected")
    assert code != 0 and "пустой отказ: огнетушителя в базе нет" in failed_names(tmp_path / "report.json")


def test_a_service_that_confirms_a_wrong_lamp_fails(engine, tmp_path):
    def confirm_300(answer, query, top_k):
        if "100 вт" in query:
            answer["status"] = "found"
            answer["results"] = [{**answer["table"][0], "name": "Светильник Pn=300 Вт", "verdict": "same"}]
        return answer
    with Served(create_app(Wrapper(engine, confirm_300))) as base:
        code, _ = run(base, tmp_path, "--no-expected")
    assert code != 0 and any("100 вт" in n for n in failed_names(tmp_path / "report.json"))


def test_an_unreachable_service_fails(tmp_path):
    code, rep = run(f"http://127.0.0.1:{free_port()}", tmp_path, "--no-expected")
    assert code != 0 and any(c["name"] == "доступность сервиса" for c in rep["checks"])


def test_a_service_that_returns_other_answers_than_the_snapshot_fails(engine, tmp_path):
    def shift_prices(answer, query, top_k):
        for x in answer["table"]:
            x["price"] += 1
        return answer
    with Served(create_app(Wrapper(engine, shift_prices))) as base:
        code, _ = run(base, tmp_path)
    assert code != 0 and any(n.startswith("совпадает со снимком") for n in failed_names(tmp_path / "report.json"))


# ---------------------------------------------------------------- the snapshot is bound to the rules
def test_other_rules_skip_the_snapshot_comparison_instead_of_failing(engine, tmp_path):
    snap = json.loads((ROOT / "eval" / "smoke_expected.json").read_text(encoding="utf-8"))
    snap["rules_sha256"] = "0" * 64
    other = tmp_path / "other.json"
    other.write_text(json.dumps(snap), encoding="utf-8")
    with Served(create_app(engine)) as base:
        code, rep = run(base, tmp_path, "--expected", str(other))
    assert code == 0 and rep["skipped"] == 1 and rep["failed"] == 0


def test_the_recorded_snapshot_is_current(engine):
    """A stale snapshot would make the container check meaningless: it must equal what the local engine says today."""
    snap = json.loads((ROOT / "eval" / "smoke_expected.json").read_text(encoding="utf-8"))
    from analog_search.manifest import RULES_PATH, sha256_file
    assert snap["rules_sha256"] == sha256_file(RULES_PATH)
    assert snap["queries"] == smoke.QUERIES and len(smoke.QUERIES) == 10
    for q in smoke.QUERIES:
        want, got = snap["answers"][q], smoke.fingerprint(engine.search(q))
        assert [x["id"] for x in got["table"]] == [x["id"] for x in want["table"]], q
        assert [(x["verdict"], x["price"]) for x in got["table"]] == [(x["verdict"], x["price"]) for x in want["table"]], q
        assert all(abs(a["score"] - b["score"]) <= smoke.SCORE_TOLERANCE for a, b in zip(got["table"], want["table"])), q
