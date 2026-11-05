"""Offline preparation of the catalog (steps A1–A5): xlsx → data/processed/catalog_v2.jsonl + coverage report.

Run: python3 -I scripts/prepare_catalog.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from analog_search import process  # noqa: E402
from analog_search.manifest import catalog_manifest, write_catalog_manifest  # noqa: E402
from analog_search.source import SHEET_NAME, SourceError, read_source_rows  # noqa: E402

XLSX = Path(os.environ.get("CATALOG_XLSX") or ROOT / "База_позиций_тестовое.xlsx")      # CATALOG_XLSX: ASCII path in the Docker image (the Cyrillic name has two Unicode forms: macOS vs Linux)
OUT_JSONL = ROOT / "data" / "processed" / "catalog_v2.jsonl"
REPORT_DIR = ROOT / "reports" / "extraction"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

try:
    rows = read_source_rows(XLSX)                      # the structure is checked BEFORE anything is written
except SourceError as e:
    sys.exit(f"Ошибка входного файла: {e}")

records = []
for excel_row, name, price in rows:
    rec = process(name)
    rec.update({"id": f"row-{excel_row:06d}", "source_row": excel_row, "raw_name": name, "price": price})
    records.append(rec)

tmp = OUT_JSONL.with_suffix(".jsonl.tmp")               # written aside and moved after success: an error never leaves a half-made catalog
with tmp.open("w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
tmp.replace(OUT_JSONL)
write_catalog_manifest(catalog_manifest(XLSX, SHEET_NAME, len(records), OUT_JSONL), OUT_JSONL)       # what this catalog was built from

N = len(records)
searchable = [r for r in records if not r["is_service"] and not r["is_adjustment"]]
S = len(searchable)


def pct(k: int, n: int = S) -> str:
    return f"{k} ({100 * k / n:.1f}%)"


lines = [f"# Extraction coverage\n\nRows: {N}; searchable (not service/adjustment): {S}\n"]
lines.append(f"- services: {sum(r['is_service'] for r in records)}; adjustments: {sum(r['is_adjustment'] for r in records)}; "
             f"composite: {sum(bool(r['composite']) for r in records)} "
             f"(header {sum(r['composite'] == 'header' for r in records)}, inline {sum(r['composite'] == 'inline' for r in records)})")
lines.append(f"- numbering prefix removed: {sum(bool(r['numbering_prefix']) for r in records)}; "
             f"rows with «(прим.)» fragments: {sum(bool(r['alt_texts']) for r in records)}")
typed = [r for r in searchable if r["type"]]
lines.append(f"- type recognised: {pct(len(typed))}; unknown: {pct(S - len(typed))}")
with_attr = [r for r in searchable if any(a['source'] == 'main' and a['name'] != 'model_codes' for a in r['attributes'])]
lines.append(f"- at least one attribute (excluding model codes): {pct(len(with_attr))}")
clean_numbers = [r for r in searchable if not r["uncovered_numbers"]]
lines.append(f"- every number of the name explained by an attribute or code: {pct(len(clean_numbers))}")

names = Counter(a["name"] for r in searchable for a in r["attributes"] if a["source"] == "main")
rows_by_attr = Counter(n for r in searchable for n in {a["name"] for a in r["attributes"] if a["source"] == "main"})
lines.append("\n## Rows per attribute\n\n| attribute | rows | occurrences |\n|---|---|---|")
for n, c in sorted(rows_by_attr.items(), key=lambda kv: (-kv[1], kv[0])):
    lines.append(f"| {n} | {c} | {names[n]} |")

types = Counter(r["type"] or "(unknown)" for r in searchable)
lines.append("\n## Types (top 40)\n\n| type | rows |\n|---|---|")
for t, c in sorted(types.items(), key=lambda kv: (-kv[1], str(kv[0])))[:40]:
    lines.append(f"| {t} | {c} |")

unknown_heads = Counter(r["head_word"] for r in searchable if not r["type"])
lines.append("\n## Unknown type: top head words (extend dictionaries.yaml)\n")
lines.append(", ".join(f"`{w}`×{c}" for w, c in sorted(unknown_heads.items(), key=lambda kv: (-kv[1], str(kv[0])))[:60]))

unc = Counter(n for r in searchable for n in r["uncovered_numbers"])
lines.append(f"\n## Uncovered numbers: {sum(unc.values())} occurrences in {S - len(clean_numbers)} rows\n")

(REPORT_DIR / "coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump(name, items, header):
    import csv
    with (REPORT_DIR / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(items)


dump("unknown_type.csv", [[r["id"], r["head_word"], r["raw_name"][:160]] for r in searchable if not r["type"]],
     ["id", "head_word", "raw_name"])
dump("uncovered_numbers.csv", [[r["id"], " ".join(r["uncovered_numbers"]), r["type"], r["raw_name"][:200]]
                                for r in searchable if r["uncovered_numbers"]],
     ["id", "numbers", "type", "raw_name"])
dump("attributes_flat.csv", [[r["id"], r["type"], a["name"], json.dumps(a["value"], ensure_ascii=False), a["source"], a.get("kind", ""), a["raw"][:60]]
                              for r in records for a in r["attributes"]],
     ["id", "type", "attribute", "value", "source", "kind", "raw"])

print("\n".join(lines[:16]))
print("written", OUT_JSONL, REPORT_DIR)
