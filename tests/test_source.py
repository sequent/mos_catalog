"""the structure of the source workbook is checked before the catalog is built."""
import hashlib
import subprocess
import sys
from pathlib import Path

import openpyxl
import pytest

from analog_search.source import HEADERS, SHEET_NAME, SourceError, read_source_rows

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "База_позиций_тестовое.xlsx"


def make(path, rows, sheet=SHEET_NAME, header=HEADERS, extra_first_sheet=False):
    wb = openpyxl.Workbook()
    ws = wb.active
    if extra_first_sheet:
        ws.title = "Черновик"
        ws.append(["не", "то"])
        ws = wb.create_sheet(sheet)
    else:
        ws.title = sheet
    ws.append(list(header))
    for r in rows:
        ws.append(list(r))
    wb.save(path)
    return path


def test_the_real_workbook_is_read_completely():
    rows = read_source_rows(XLSX)
    assert len(rows) == 2381 and rows[0][0] == 2 and rows[-1][0] == 2382
    raw = list(openpyxl.load_workbook(XLSX, read_only=True).worksheets[0].iter_rows(min_row=2, values_only=True))
    assert [(r[1], r[2]) for r in rows] == [(str(a), int(b)) for a, b in raw]            # names and prices exactly as in the file


def test_the_sheet_is_taken_by_name_not_as_the_active_one(tmp_path):
    p = make(tmp_path / "a.xlsx", [("Труба", 10)], extra_first_sheet=True)
    assert read_source_rows(p) == [(2, "Труба", 10)]


def test_an_empty_row_is_skipped_but_does_not_shift_row_numbers(tmp_path):
    p = make(tmp_path / "a.xlsx", [("Труба", 10), (None, None), ("Кран", 20)])
    assert read_source_rows(p) == [(2, "Труба", 10), (4, "Кран", 20)]


def test_a_missing_sheet_names_the_available_ones(tmp_path):
    p = make(tmp_path / "a.xlsx", [("Труба", 10)], sheet="Другой")
    with pytest.raises(SourceError, match="Другой"):
        read_source_rows(p)


@pytest.mark.parametrize("header", [("Название", "Цена"), ("Цена за ед., руб.", "Позиция (наименование из сметы)")])
def test_a_wrong_or_swapped_header_is_rejected(tmp_path, header):
    p = make(tmp_path / "a.xlsx", [("Труба", 10)], header=header)
    with pytest.raises(SourceError, match="заголовк"):
        read_source_rows(p)


@pytest.mark.parametrize("row,message", [
    ((None, 10), "строка 3: пустое название"),
    (("   ", 10), "строка 3: пустое название"),
    (("Кран", None), "строка 3: нет цены"),
    (("Кран", "дорого"), "строка 3: цена 'дорого' не число"),
    (("Кран", 10.5), "строка 3: цена 10.5 дробная"),
    (("Кран", -5), "строка 3: цена -5 отрицательная"),
])
def test_a_bad_row_is_rejected_with_its_number(tmp_path, row, message):
    p = make(tmp_path / "a.xlsx", [("Труба", 10), row])
    with pytest.raises(SourceError, match=message):
        read_source_rows(p)


def test_an_integer_price_written_as_float_or_text_is_accepted(tmp_path):
    p = make(tmp_path / "a.xlsx", [("Труба", 10.0), ("Кран", "20")])
    assert read_source_rows(p) == [(2, "Труба", 10), (3, "Кран", 20)]


def test_a_workbook_without_data_is_rejected(tmp_path):
    with pytest.raises(SourceError, match="нет ни одной строки"):
        read_source_rows(make(tmp_path / "a.xlsx", []))


def test_a_failed_preparation_leaves_the_catalog_untouched(tmp_path):
    """The script reads and checks everything first; a bad input must not change the existing catalog."""
    catalog = ROOT / "data" / "processed" / "catalog_v2.jsonl"
    if not catalog.exists():
        pytest.skip("catalog is not built")
    before = hashlib.sha256(catalog.read_bytes()).hexdigest()
    bad = make(tmp_path / "bad.xlsx", [("Труба", 10.5)])
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "prepare_catalog.py")], capture_output=True, text=True,
                       env={"CATALOG_XLSX": str(bad), "PATH": ""}, cwd=tmp_path)
    assert r.returncode != 0 and "дробная" in (r.stderr + r.stdout)
    assert hashlib.sha256(catalog.read_bytes()).hexdigest() == before and not catalog.with_suffix(".jsonl.tmp").exists()
