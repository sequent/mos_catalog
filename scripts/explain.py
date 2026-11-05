"""Shows how the system understands a text: cleaned form, type, attributes, flags.

For specialists: use it to check a new synonym or a type entry without reading code.

Run:  python3 -I scripts/explain.py "кабель КВВГнг-LS 7*1.5"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from analog_search import process  # noqa: E402


def main(text: str) -> None:
    r = process(text)
    print(f"Текст:            {text}")
    print(f"После очистки:    {r['clean_name']}")
    if r["numbering_prefix"]:
        print(f"Номер пункта:     {r['numbering_prefix']} (удалён из текста поиска)")
    print(f"Для поиска:       {r['search_text']}")
    kind = "конкретный" if r["type_specific"] else ("общий" if r["type"] else "не определён")
    print(f"Тип изделия:      {r['type'] or '—'}  ({kind}; семейство: {r['family'] or '—'})")
    flags = [n for n, v in (("услуга", r["is_service"]), ("надбавка", r["is_adjustment"]),
                            (f"комплект ({r['composite']})", r["composite"])) if v]
    print(f"Пометки:          {', '.join(flags) or '—'}")
    print("Характеристики:")
    if not r["attributes"]:
        print("  (не найдено)")
    for a in r["attributes"]:
        if a["name"] == "model_codes":
            continue
        unit = f" {a['unit']}" if a.get("unit") else ""
        src = "  [из «прим.», не для строгого сравнения]" if a["source"] == "alt" else ""
        note = f"  <{a['kind']}>" if a.get("kind") else ""
        value = a.get("extra", {}).get("display", a["value"]) if a["name"] == "cable_mark" else a["value"]
        print(f"  {a['name']:<20} {value}{unit}   ← «{a['raw'][:40]}»{note}{src}")
    codes = next((a for a in r["attributes"] if a["name"] == "model_codes"), None)
    if codes:
        print(f"Коды моделей:     {', '.join(codes['extra'].get('display', []))}")
    if r["uncovered_numbers"]:
        print(f"Числа, не объяснённые ни одной характеристикой: {', '.join(r['uncovered_numbers'])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Использование: python3 -I scripts/explain.py "текст позиции или запроса"')
    main(" ".join(sys.argv[1:]))
