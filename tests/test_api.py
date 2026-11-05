"""HTTP layer: contract of `POST /search`, `GET /health`, the page, and the examples shown on the page.

Part 1 uses a tiny synthetic catalog without the model. Part 2 uses the real catalog and the real model (skipped when
they are not built) and runs every example of the page through the API.
"""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from analog_search import process
from analog_search.api import MAX_QUERY_LENGTH, PAGE, create_app
from analog_search.engine import SearchEngine
from analog_search.matching import load_rules
from analog_search.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
BUILT = (ROOT / "data" / "processed" / "catalog_v2.jsonl").exists() and (ROOT / "data" / "index" / "dense.npy").exists() \
    and (ROOT / "models" / "multilingual-e5-small" / "onnx" / "model.onnx").exists()
real = pytest.mark.skipif(not BUILT, reason="catalog, index or model are not built")

SMALL = ["Кабель контрольный КВВГнг(A)-LS 7х1,5", "Кабель контрольный КВВГЭнг(A)-LS 7х1,5", "Ворота откатные 2250х6000 мм",
         "Перевозка грузов I класса автомобилями бортовыми на расстояние до 7 км"]


def _small_engine():
    recs = []
    for i, t in enumerate(SMALL):
        r = process(t)
        r.update({"id": f"row-{i:06d}", "raw_name": t, "price": 1000 * (i + 1)})
        recs.append(r)
    return SearchEngine(recs, Retriever(recs), load_rules(), mode="bm25")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(_small_engine())) as c:
        yield c


# ---------------------------------------------------------------- health and the page
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["rows"] == 4 and body["searchable"] == 3
    assert body["rules_status"] == "assumption" and body["mode"] == "bm25"
    assert len(body["artifacts"]["rules_sha256"]) == 12


def test_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "Поиск аналогов" in r.text and 'id="form"' in r.text


def test_page_never_writes_data_as_html():
    """Names come from the database and the query is typed by a person: the page must use textContent only."""
    src = PAGE.read_text(encoding="utf-8")
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\(", src)


def test_page_has_no_external_resources():
    """It must work offline, inside the container."""
    src = PAGE.read_text(encoding="utf-8")
    assert not re.search(r"""(?:src|href)\s*=\s*["']https?://""", src) and "@import" not in src


def test_openapi_describes_both_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/search" in paths and "/health" in paths


# ---------------------------------------------------------------- the contract of /search
def test_search_found(client):
    r = client.post("/search", json={"query": "ворота откатные 6000х2250"})
    assert r.status_code == 200
    a = r.json()
    assert a["status"] == "found" and a["display_status"] == "found" and a["status_line"].startswith("Найдено подходящих позиций: 1")
    top = a["table"][0]
    assert top["name"] == "Ворота откатные 2250х6000 мм" and top["price"] == 3000 and top["verdict"] == "same" and top["needs_review"] is False
    assert {"attribute_coverage", "text_relevance"} <= set(top["score_parts"])
    assert [row["attribute"] for row in top["comparison"]][:2] == ["type", "dims"]
    assert a["score_weights"] == {"attribute_coverage": 0.6, "text_relevance": 0.4}
    assert a["rules_version"] and a["diagnostics"]["candidates"] >= 1


def test_search_review_rows_are_marked(client):
    a = client.post("/search", json={"query": "кабель КВВГнг-LS 7*1.5"}).json()
    assert a["status"] == "found"                                                  # КВВГнг 7х1,5 exists in the small catalog
    a = client.post("/search", json={"query": "кабель КВВГЭнг-LS 7*1.5 экранированный"}).json()
    assert a["status"] in ("found", "not_found")
    for x in a["table"]:
        assert x["needs_review"] == (x["verdict"] == "check")


def test_search_not_found(client):
    a = client.post("/search", json={"query": "огнетушитель порошковый ОП-5"}).json()
    assert a["status"] == "not_found" and a["display_status"] == "not_found" and a["status_line"] == "Не найдено"
    assert a["table"] == [] and a["results"] == []


def test_search_reports_assumptions(client):
    a = client.post("/search", json={"query": "ворота откатные 6000х2250 любой цвет"}).json()
    assert a["query"]["assumptions"][0]["attribute"] == "color"


def test_limit_is_respected(client):
    a = client.post("/search", json={"query": "кабель контрольный КВВГнг", "limit": 1}).json()
    assert len(a["table"]) <= 1


@pytest.mark.parametrize("body", [
    {}, {"query": ""}, {"query": "   "}, {"query": "x" * (MAX_QUERY_LENGTH + 1)}, {"query": 5},
    {"query": "кабель", "limit": 0}, {"query": "кабель", "limit": 6}, {"query": "кабель", "limit": 21},
])
def test_invalid_requests_are_rejected_with_422(client, body):
    assert client.post("/search", json=body).status_code == 422


def test_whitespace_is_normalised(client):
    a = client.post("/search", json={"query": "  ворота   откатные \n 6000х2250  "}).json()
    assert a["status"] == "found"


def test_answer_is_plain_json(client):
    r = client.post("/search", json={"query": "ворота откатные 6000х2250"})
    json.loads(r.text)                                                             # no numpy types, no NaN
    assert "NaN" not in r.text and "Infinity" not in r.text


def test_wrong_method_and_unknown_path(client):
    assert client.get("/search").status_code == 405
    assert client.get("/nothing").status_code == 404


def test_query_text_is_returned_as_data_not_markup(client):
    a = client.post("/search", json={"query": "<img src=x onerror=alert(1)> ворота"}).json()
    assert a["query"]["raw"].startswith("<img")                                    # returned as a string; the page prints it with textContent


# ---------------------------------------------------------------- real catalog: every example of the page
def _examples():
    src = PAGE.read_text(encoding="utf-8")
    block = re.search(r"const EXAMPLES = \[(.*?)\];", src, re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


EXPECTED = {            # status of the answer for each example of the page (the seven of the assignment + two assumptions)
    "кабель КВВГнг-LS 7*1.5": "not_found", "камера IP уличная 2 Мп вариофокальная": "found", "ворота откатные 6000х2250": "found",
    "труба гофра ПВХ 25 мм": "found", "ИБП 12В для улицы": "not_found", "трансформатор силовой сухой 1000 кВА": "not_found",
    "огнетушитель порошковый ОП-5": "not_found", "Инфракрасный термометр любого производителя": "found", "Тройник 25х30 мм любой цвет": "found",
}


def test_page_lists_nine_examples():
    assert _examples() == list(EXPECTED)


@pytest.fixture(scope="module")
def real_client():
    with TestClient(create_app()) as c:
        yield c


@real
@pytest.mark.parametrize("query", list(EXPECTED))
def test_every_example_of_the_page_works(real_client, query):
    r = real_client.post("/search", json={"query": query})
    assert r.status_code == 200
    a = r.json()
    assert a["status"] == EXPECTED[query]
    assert len(a["table"]) <= 5
    for x in a["table"]:
        assert x["name"] and isinstance(x["price"], int) and x["explanation"] and x["comparison"]
    if a["status"] == "found":
        assert a["table"][0]["verdict"] in ("same", "substitute")


@real
def test_real_health(real_client):
    b = real_client.get("/health").json()
    assert b["rows"] == 2381 and b["searchable"] == 2277 and b["mode"] == "hybrid"
    assert b["artifacts"]["model_revision"] == "614241f622f5" and len(b["artifacts"]["catalog_sha256"]) == 12


@real
def test_a_missing_index_stops_the_start_with_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setattr("analog_search.api.SearchEngine.build", classmethod(lambda cls: (_ for _ in ()).throw(FileNotFoundError("dense.npy"))))
    with pytest.raises(RuntimeError, match="download_model"):
        with TestClient(create_app()):
            pass


# ---------------------------------------------------------------- T06, second part: evidence and the candidate's own characteristics in the API and on the page
def test_the_answer_carries_the_extra_attributes_the_remainder_and_the_evidence(client):
    a = client.post("/search", json={"query": "кабель контрольный КВВГнг", "limit": 5}).json()
    assert a["table"]
    for x in a["table"]:
        assert isinstance(x["candidate_extra_attributes"], list) and isinstance(x["candidate_unparsed_constraints"], list)
        assert all("evidence" in row for row in x["comparison"])
        for row in x["comparison"]:
            for v in row["evidence"]:
                assert set(v) >= {"side", "source", "source_text", "fragment", "text", "text_version", "span", "exact"}
                assert v["text"][slice(*v["span"])] == v["fragment"]


def test_the_page_renders_the_new_blocks_through_textcontent_only():
    src = PAGE.read_text(encoding="utf-8")
    for needed in ("candidate_extra_attributes", "candidate_unparsed_constraints", "Дополнительные характеристики позиции", "Не удалось разобрать в названии позиции", "evidenceLines"):
        assert needed in src
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML", src)
