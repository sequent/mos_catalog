"""The batch tool: one result row per input row; a broken row is written with its state and never stops the run."""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("batch_search", ROOT / "scripts" / "batch_search.py")
B = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / "pipeline"))
spec.loader.exec_module(B)


REJECTED = {"name": "Труба ПВХ 32 мм", "price": 7, "verdict_text": "не подходит", "explanation": "Отличается: диаметр (запрос: 25 мм, в базе: 32 мм)."}


def fake_search(text, limit):
    if text == "БОМБА":
        raise RuntimeError("boom")
    found = "кабель" in text
    row = {"name": "Кабель", "price": 100, "verdict_text": "совпадает по указанным параметрам", "explanation": "Совпало: сечение."}
    if text == "камера":                                                          # nothing confirmed, a variant to check is on the screen
        weak = {"name": "Камера 2 Мп", "price": 5, "verdict_text": "проверить", "explanation": "Нет данных: разрешение."}
        return {"status": "not_found", "display_status": "review", "table": [weak], "results": [], "nearest_rejected": REJECTED}
    nearest = None if found or text != "труба" else REJECTED                      # the engine always has this key: None or the closest rejected position
    return {"status": "found" if found else "not_found", "display_status": "found" if found else "not_found", "table": [row] if found else [],
            "results": [row] if found else [], "nearest_rejected": nearest}


def test_classify_the_cells():
    assert B.classify("  кабель   3х1,5 ") == ("ok", "кабель 3х1,5")
    assert B.classify(None) == ("empty", "") and B.classify("   ") == ("empty", "") and B.classify(42) == ("not_text", "42")
    state, text = B.classify("x" * 600)
    assert state == "too_long" and len(text) == 500


def test_every_input_row_gets_a_result_row_and_a_bad_row_never_stops_the_run():
    values = ["кабель 3х1,5", None, "", 42, "x" * 600, "БОМБА", "труба", "кабель 5х6"]
    rows = B.run(values, fake_search)
    assert [r["row"] for r in rows] == list(range(1, len(values) + 1))
    assert [r["status"] for r in rows] == ["found", "not_searched", "not_searched", "not_searched", "not_searched", "error", "not_found", "found"]
    assert rows[5]["error"].startswith("RuntimeError") and "пустая" in rows[1]["error"] and "не текст" in rows[3]["error"] and "длиннее" in rows[4]["error"]
    t = B.totals(rows)
    assert (t["found"], t["not_found"], t["not_searched"], t["error"]) == (2, 1, 4, 1)


def test_reading_txt_csv_and_xlsx(tmp_path):
    (tmp_path / "a.txt").write_text("кабель 3х1,5\n\nтруба\n", encoding="utf-8")
    assert B.read_positions(tmp_path / "a.txt", None) == ["кабель 3х1,5", "", "труба"]
    (tmp_path / "a.csv").write_text("№;Позиция\n1;кабель\n2;\n3;труба\n", encoding="utf-8")
    assert B.read_positions(tmp_path / "a.csv", "Позиция") == ["кабель", "", "труба"] and B.read_positions(tmp_path / "a.csv", "1") == ["1", "2", "3"]
    import openpyxl
    wb = openpyxl.Workbook()
    wb.active.append(["Позиция", "Цена"])
    for r in (["кабель", 1], [None, 2], [123, 3]):
        wb.active.append(r)
    wb.save(tmp_path / "a.xlsx")
    assert B.read_positions(tmp_path / "a.xlsx", None) == ["кабель", None, 123]


def test_an_unknown_column_or_format_is_a_clear_error(tmp_path):
    (tmp_path / "a.csv").write_text("A;B\n1;2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Нет столбца"):
        B.read_positions(tmp_path / "a.csv", "Z")
    (tmp_path / "a.doc").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="Неподдерживаемый"):
        B.read_positions(tmp_path / "a.doc", None)


@pytest.mark.skipif(not ((ROOT / "data" / "index" / "dense.npy").exists() and (ROOT / "models").exists()), reason="the catalog, the index or the model is not built")
def test_the_tool_on_the_real_engine(tmp_path):
    src = tmp_path / "list.txt"
    src.write_text("труба гофра ПВХ 25 мм\n\nворота откатные 6000х2250\n!!!\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    assert B.main([str(src), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig"), delimiter=";"))
    assert [r["status"] for r in rows] == ["found", "not_searched", "found", "not_found"]
    assert rows[0]["top_name"] and rows[1]["error"]


def test_a_not_found_row_says_why_and_a_found_row_does_not():
    """The statement is read without the screen: a row that was not confirmed carries the closest rejected position and the reason."""
    rows = B.run(["кабель 3х1,5", "труба", "огнетушитель", "камера", None], fake_search)
    found, pipe, nothing, review, empty = rows
    assert pipe["nearest_rejected_name"] == "Труба ПВХ 32 мм" and "диаметр" in pipe["nearest_rejected_why"]
    assert review["nearest_rejected_name"] == "Труба ПВХ 32 мм"                       # a variant to check does not hide the closest rejected one
    assert found["nearest_rejected_name"] == "" and found["nearest_rejected_why"] == ""     # a confirmed row needs no reason of refusal
    assert nothing["nearest_rejected_name"] == "" and empty["nearest_rejected_name"] == ""   # nothing close and nothing searched: the cells are empty, not invented


def test_the_summary_counts_the_variants_to_check_apart_from_the_nothing_found():
    rows = B.run(["кабель 3х1,5", "камера", "труба", None], fake_search)
    assert B.review_count(rows) == 1
    assert B.totals(rows)["not_found"] == 2                                           # the strict status stays strict: a variant to check is not a confirmed one


def test_the_csv_and_the_summary_line_of_the_command(tmp_path, monkeypatch, capsys):
    import analog_search.engine as engine

    class Fake:
        def search(self, q, top_k=5):
            return fake_search(q, top_k)

    monkeypatch.setattr(engine.SearchEngine, "build", classmethod(lambda cls: Fake()))
    src = tmp_path / "v.txt"
    src.write_text("кабель 3х1,5\nкамера\nтруба\n\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    assert B.main([str(src), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig"), delimiter=";"))
    assert "nearest_rejected_name" in rows[0] and "nearest_rejected_why" in rows[0]
    assert rows[2]["nearest_rejected_why"].startswith("Отличается: диаметр")
    printed = capsys.readouterr().out
    assert "подтверждено: 1" in printed and "не найдено: 2" in printed and "из них с вариантами для проверки: 1" in printed
