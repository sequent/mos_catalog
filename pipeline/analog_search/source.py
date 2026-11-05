"""Reading of the source workbook with a check of its structure.

The catalog is rebuilt from this file, so a wrong sheet, a moved column, an empty name or a fractional price must stop the
preparation with a clear message that names the row — never end as a silently wrong catalog.
"""
from __future__ import annotations

from pathlib import Path

import openpyxl

SHEET_NAME = "База позиций"
HEADERS = ("Позиция (наименование из сметы)", "Цена за ед., руб.")


class SourceError(ValueError):
    """The workbook does not have the expected structure or contains a row that cannot be a catalog position."""


def read_source_rows(path: Path | str, sheet_name: str = SHEET_NAME) -> list[tuple[int, str, int]]:
    """Returns `(Excel row number, name, price)` for every data row.

    The sheet is taken by NAME (the active sheet of another file may be a different one). Completely empty rows are skipped but
    keep the real Excel row number of the rows after them. A row with an empty name, a missing, non-numeric, negative or
    fractional price is rejected (a fractional price is not rounded: kopecks need an agreed change of the data format)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet_name not in wb.sheetnames:
            raise SourceError(f"{Path(path).name}: нет листа «{sheet_name}»; есть: {', '.join(wb.sheetnames) or 'нет листов'}")
        ws = wb[sheet_name]
        it = ws.iter_rows(values_only=True)
        header = next(it, None)
        got = tuple(str(c).strip() if c is not None else "" for c in (header or ())[: len(HEADERS)])
        if got != HEADERS:
            raise SourceError(f"{Path(path).name}, лист «{sheet_name}», строка 1: ожидались заголовки {list(HEADERS)}, получено {list(got)}")
        rows: list[tuple[int, str, int]] = []
        for excel_row, r in enumerate(it, start=2):
            cells = (tuple(r) + (None, None))[:2]
            name, price = cells
            if (name is None or str(name).strip() == "") and price is None:
                continue                                                         # an empty row
            if name is None or str(name).strip() == "":
                raise SourceError(f"строка {excel_row}: пустое название при цене {price!r}")
            rows.append((excel_row, str(name), _price(price, excel_row)))
        if not rows:
            raise SourceError(f"{Path(path).name}: нет ни одной строки данных")
        return rows
    finally:
        wb.close()


def _price(value, excel_row: int) -> int:
    if value is None or str(value).strip() == "":
        raise SourceError(f"строка {excel_row}: нет цены")
    if isinstance(value, bool):
        raise SourceError(f"строка {excel_row}: цена {value!r} не число")
    try:
        number = float(value) if not isinstance(value, (int, float)) else value
    except ValueError:
        raise SourceError(f"строка {excel_row}: цена {value!r} не число") from None
    if number != int(number):
        raise SourceError(f"строка {excel_row}: цена {value!r} дробная; формат данных рассчитан на целые рубли")
    if number < 0:
        raise SourceError(f"строка {excel_row}: цена {value!r} отрицательная")
    return int(number)
