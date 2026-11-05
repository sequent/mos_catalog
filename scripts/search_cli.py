"""Command-line search: prints the answer of the whole scheme for a query (for the author and the specialist).

Run:  .venv/bin/python -I scripts/search_cli.py "кабель КВВГнг-LS 7*1.5" [--debug]
Prerequisites: prepare_catalog.py, download_model.py, build_index.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from analog_search.engine import SearchEngine  # noqa: E402


def show(item: dict, debug: bool) -> None:
    price = f"{item['price']:,}".replace(",", " ")
    print(f"  • [{item['verdict_text']}{' — требует проверки' if item['needs_review'] else ''}] {item['name'][:110]}")
    print(f"    цена: {price} руб.   оценка: {item['score']}  ({item['score_parts']['attribute_coverage']:.2f} совпадение характеристик, "
          f"{item['score_parts']['text_relevance']:.2f} близость текста)")
    if item.get("match_note"):
        print(f"    ({item['match_note']})")
    print(f"    {item['explanation']}")
    if debug:
        for r in item["comparison"]:
            print(f"      - {r['label']:<28} запрос: {r['query']:<22} база: {r['candidate']:<22} {r['status_text']}")
        print(f"      retrieval: {item['retrieval']}")


def main(query: str, debug: bool) -> None:
    eng = SearchEngine.build()
    ans = eng.search(query)
    q = ans["query"]
    print(f"Запрос: {q['raw']}")
    print(f"  понято: тип = {q['type'] or '—'} ({'конкретный' if q['type_specific'] else 'общий/не определён'}); "
          + "; ".join(f"{a['label']} = {a['value']}" for a in q["attributes"]) + (f"; модель {', '.join(q['model_codes'])}" if q["model_codes"] else ""))
    if q["assumptions"]:
        print("  принято допущение: " + "; ".join(f"{a['label']} не важен («{a['phrase']}»)" for a in q["assumptions"]))
    blocking = [u["raw"] for u in q["unparsed_constraints"] if u["severity"] == "blocking"]
    if blocking:
        print("  не удалось разобрать: " + "; ".join(blocking) + " (ответ не выше «проверить»)")
    print(f"Статус: {ans['status_line'].upper()}   (правила {ans['rules_version']}, {ans['rules_status']})")
    for it in ans["table"]:
        show(it, debug)
    if ans["nearest_rejected"]:
        print("Ближайшее, но не подходит:")
        show(ans["nearest_rejected"], debug)
    d = ans["diagnostics"]
    print(f"[{d['candidates']} кандидатов; классы: {d['classes']}; поиск {d['retrieval_ms']} мс, сравнение {d['matching_ms']} мс]")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit('Использование: python -I scripts/search_cli.py "запрос" [--debug]')
    main(" ".join(args), "--debug" in sys.argv)
