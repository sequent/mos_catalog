"""Batch search: a list of positions (a statement) through the whole scheme, one result row per input row, never stopping on a bad row.

Input: .xlsx, .csv or .txt (one position per line). For .xlsx / .csv the column is chosen by `--column` (a header or a 1-based number; by default the first).
Output: a CSV (`--out`, by default next to the input) with, for every input row: the row number, the text, `input_state` (ok | empty | too_long | not_text),
the search `status` (found | not_found), the screen state, the number of confirmed rows, the first row of the table (name, price, verdict, why), for a row that was not confirmed the closest REJECTED position with the reason of the refusal
(`nearest_rejected_name`, `nearest_rejected_why`) and `error`.
A row that cannot be searched is NOT dropped and NOT silently skipped: it is written with its state, and the totals are printed at the end.

Run:  .venv/bin/python -I scripts/batch_search.py ведомость.xlsx [--column "Позиция"] [--out result.csv] [--limit 5]
Prerequisites: prepare_catalog.py, download_model.py, build_index.py.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

MAX_LEN = 500                                    # the same limit as the API
FIELDS = ["row", "query", "input_state", "status", "display_status", "confirmed", "top_name", "top_price", "top_verdict", "top_why", "nearest_rejected_name", "nearest_rejected_why", "error"]


def read_positions(path: Path, column: str | None) -> list:
    """The raw values of the chosen column, in file order (empty and broken cells included: the caller classifies them)."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8-sig").splitlines()
    if suffix == ".csv":
        text = path.read_text(encoding="utf-8-sig")
        rows = list(csv.reader(io.StringIO(text), delimiter=";" if ";" in text.splitlines()[0] else ","))
    elif suffix == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            rows = [list(r) for r in wb.worksheets[0].values]
        finally:
            wb.close()
    else:
        raise SystemExit(f"Неподдерживаемый формат {suffix}: нужен .xlsx, .csv или .txt")
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if column is None:
        idx = 0
    elif column.isdigit():
        idx = int(column) - 1
    elif column in [str(h) for h in header]:
        idx = [str(h) for h in header].index(column)
    else:
        raise SystemExit(f"Нет столбца «{column}»; есть: {', '.join(str(h) for h in header)}")
    return [r[idx] if idx < len(r) else None for r in body]


def classify(value) -> tuple[str, str]:
    """(input_state, cleaned text). Only a non-empty text of at most MAX_LEN characters is searched."""
    if not isinstance(value, str):
        return ("empty", "") if value is None else ("not_text", str(value))
    text = " ".join(value.split())
    if not text:
        return "empty", ""
    return ("too_long", text[:MAX_LEN]) if len(text) > MAX_LEN else ("ok", text)


def run(values: list, search, limit: int = 5) -> list[dict]:
    out = []
    for n, value in enumerate(values, 1):
        state, text = classify(value)
        row = dict.fromkeys(FIELDS, "")
        row.update(row=n, query=text, input_state=state)
        if state in ("ok",):
            try:
                a = search(text, limit)
                top = a["table"][0] if a["table"] else None
                row.update(status=a["status"], display_status=a["display_status"], confirmed=len(a["results"]))
                if top:
                    row.update(top_name=top["name"], top_price=top["price"], top_verdict=top["verdict_text"], top_why=top["explanation"])
                nearest = a["nearest_rejected"]                                 # the closest position that was refused, with the reason: for a not confirmed row only
                if nearest and a["status"] != "found":
                    row.update(nearest_rejected_name=nearest["name"], nearest_rejected_why=nearest["explanation"])
            except Exception as exc:                                    # one bad row never stops the statement
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        else:
            row.update(status="not_searched", error={"empty": "пустая ячейка", "too_long": f"длиннее {MAX_LEN} знаков", "not_text": "значение не текст"}[state])
        out.append(row)
    return out


def totals(rows: list[dict]) -> Counter:
    return Counter(r["status"] for r in rows)


def review_count(rows: list[dict]) -> int:
    """Rows with nothing confirmed whose screen still shows variants to check (the strict `status` of such a row is not_found)."""
    return sum(r["display_status"] == "review" for r in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("--column", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=5)
    a = ap.parse_args(argv)
    values = read_positions(a.path, a.column)
    from analog_search.engine import SearchEngine
    eng = SearchEngine.build()
    rows = run(values, lambda q, k: eng.search(q, top_k=k), a.limit)
    out = a.out or a.path.with_suffix(".result.csv")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter=";")
        w.writeheader()
        w.writerows(rows)
    t = totals(rows)
    print(f"Строк: {len(rows)}; подтверждено: {t['found']}; не найдено: {t['not_found']} (из них с вариантами для проверки: {review_count(rows)}); не искали (пусто/не текст/длинно): {t['not_searched']}; ошибок: {t['error']}")
    print(f"Результат: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
