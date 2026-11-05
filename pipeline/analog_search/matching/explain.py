"""B7: a template explanation (no AI): what matched, what differs, what is not stated, and why a candidate is only «проверить»."""
from __future__ import annotations

from .compat import AttrComparison, Compatibility

LABELS = {
    "cores_section": "жилы × сечение", "dims": "размеры", "diameter_mm": "диаметр", "nominal_bore_mm": "условный проход",
    "thickness_mm": "толщина", "voltage_v": "напряжение", "current_a": "ток", "power_kw": "мощность",
    "apparent_power_kva": "полная мощность", "ip": "степень защиты", "resolution_mp": "разрешение",
    "pressure_mpa": "давление", "pn": "PN", "climate": "климатическое исполнение", "standard": "стандарт",
    "poles": "полюса", "cable_mark": "марка кабеля", "cable_screen": "экран кабеля", "fire_class": "класс пожарной безопасности",
    "rated_voltage_v": "номинальное напряжение кабеля", "length_mm": "длина", "perimeter_mm": "периметр", "height_mm": "высота",
    "side_mm": "сторона", "mass_kg": "масса", "area_m2": "площадь", "volume_m3": "объём", "volume_l": "объём",
    "temperature_c": "температура", "range_mm": "диапазон", "numeric_code": "серия / код", "model_codes": "модель",
    "placement": "исполнение", "fire_resistance_ei": "предел огнестойкости EI", "bare_mm": "размер", "type": "тип изделия",
    "name": "название", "material": "материал", "color": "цвет", "switch_gang_count": "число клавиш", "series": "производитель / серия",
    "port_count": "число портов", "quantity_pcs": "количество, шт", "storage_duration_days": "срок хранения", "print_technology": "технология печати",
    "print_color_mode": "цвет печати", "paper_format": "формат бумаги", "connector_type": "тип разъёма", "mounting": "исполнение (монтаж)",
    "mount_surface": "крепление к", "class_roman": "класс", "kit_components": "состав комплекта",
}
STATUS_TEXT = {
    "match": "совпало", "match_reordered": "совпало с точностью до порядка сторон", "substitute": "допустимая замена",
    "conflict": "отличается", "flag": "отличается (не блокирует, проверить)", "no_data": "нет данных",
    "alt_only": "только в скобке «прим.»", "ambiguous": "значение без смысла совпало", "info": "справочно", "info_diff": "другая модель",
}
_UNITS = {"diameter_mm": "мм", "nominal_bore_mm": "мм", "thickness_mm": "мм", "length_mm": "мм", "perimeter_mm": "мм",
          "height_mm": "мм", "side_mm": "мм", "bare_mm": "мм", "current_a": "А", "power_kw": "кВт", "apparent_power_kva": "кВА",
          "resolution_mp": "Мп", "pressure_mpa": "МПа", "pn": "бар", "mass_kg": "кг", "area_m2": "м²", "volume_m3": "м³",
          "volume_l": "л", "temperature_c": "°C", "rated_voltage_v": "В", "storage_duration_days": "сут", "quantity_pcs": "шт"}


def _num(x) -> str:
    return f"{x:g}".replace(".", ",") if isinstance(x, float) else str(x)


def fmt(name: str, v) -> str:
    if name == "cores_section":
        return "×".join(_num(x) for x in v) + (" мм²" if len(v) == 2 else "")
    if name in ("dims", "range_mm"):
        return ("–" if name == "range_mm" else "×").join(_num(x) for x in v) + " мм"
    if name == "voltage_v":
        return " / ".join(f"{_num(x / 1000)} кВ" if x >= 1000 and x % 1000 == 0 else f"{_num(x)} В" for x in v)
    if name == "cable_screen":
        return "есть" if v else "нет"
    if name == "placement":
        return {"outdoor": "уличное", "indoor": "внутреннее"}.get(v, str(v))
    if name in ("model_codes", "material", "color"):
        return ", ".join(v)
    if isinstance(v, list):
        return " / ".join(_num(x) for x in v) + (f" {_UNITS[name]}" if name in _UNITS else "")
    return f"{_num(v)} {_UNITS[name]}" if name in _UNITS else str(v)


def _shown(name: str, display) -> str:
    return ", ".join(display) if isinstance(display, list) else str(display)


def _row(c: AttrComparison) -> dict:
    if c.query_display is not None:                       # codes are shown as written, not in the matching form
        cand = "; ".join(_shown(c.name, d) for d in c.candidate_display) if c.candidate_display else "—"
        return {"attribute": c.name, "label": LABELS.get(c.name, c.name), "query": _shown(c.name, c.query_display),
                "candidate": cand, "status": c.status, "status_text": STATUS_TEXT.get(c.status, c.status),
                "rule": c.rule_id, "rule_scope": c.rule_scope, "note": c.note}
    cand = "; ".join(fmt(c.name, v) for v in c.candidate_values) if c.candidate_values else "—"
    return {"attribute": c.name, "label": LABELS.get(c.name, c.name), "query": fmt(c.name, c.query_value), "candidate": cand,
            "status": c.status, "status_text": STATUS_TEXT.get(c.status, c.status), "rule": c.rule_id, "rule_scope": c.rule_scope, "note": c.note}


def explain(comp: Compatibility, q: dict, c: dict) -> dict:
    """Structured comparison rows plus a one-paragraph text. Every row carries `evidence`: where each side read its value from."""
    from .evidence import evidence_by_attribute
    rows = [_row(x) for x in comp.comparisons]
    ev = evidence_by_attribute(comp, q, c)
    for r in rows:
        r["evidence"] = ev.get(r["attribute"], [])
    type_row = {"attribute": "type", "label": LABELS["type"], "query": q["type"] or "не определён",
                "candidate": c["type"] or "не определён", "status": comp.type_status,
                "status_text": {"match": "совпало", "check": "не подтверждено", "unknown": "нет данных", "conflict": "отличается"}[comp.type_status],
                "rule": None, "note": comp.type_relation, "evidence": []}
    parts = []
    same = [r["label"] for r in rows if r["status"] == "match" and r["attribute"] != "model_codes"]       # an allowed difference is NOT a match
    if any(r["attribute"] == "model_codes" and r["status"] == "match" for r in rows):
        same.append("модель")
    if same:
        parts.append("Совпало: " + ", ".join(dict.fromkeys(same)) + ".")
    named = [r for r in rows if r["attribute"] == "series" and r["status"] == "info"]
    if named:
        parts.append(f"Названо и в позиции (само по себе не подтверждает): {named[0]['label']} {named[0]['query']}.")
    required = [r for r in rows if r["status"] not in ("info", "info_diff")]
    if required and comp.klass != "incompatible":
        ok = sum(r["status"] in ("match", "match_reordered", "substitute") for r in required)
        parts.append(f"Требований запроса подтверждено {ok} из {len(required)}.")
    if comp.type_confirmed and comp.klass != "unverified":
        parts.append(f"Совпал тип изделия ({c['type']}). Подобрано по типу; диапазон и точность не проверялись; действует допущение автора.")
    diff = [f"{r['label']} (запрос: {r['query']}, в базе: {r['candidate']})" for r in rows if r["status"] in ("conflict", "flag")]
    if diff:
        parts.append("Отличается: " + "; ".join(diff) + ".")
    missing = [r["label"] for r in rows if r["status"] == "no_data"]
    if missing:
        parts.append("Нет данных в названии из базы: " + ", ".join(missing) + ".")
    subs = [f"{r['label']}: {r['query']} → {r['candidate']} (правило {r['rule']})" for r in rows if r["status"] == "substitute"]
    if subs:
        parts.append("Допустимо по правилу: " + "; ".join(subs) + ".")
    if comp.text_matched and comp.klass != "unverified":
        parts.append("Не разобрано как параметр, но то же записано в названии позиции (сравнение по написанию): "
                     + ", ".join(comp.text_matched) + ".")
    if comp.klass == "unverified" and comp.reasons:
        parts.append("Не подтверждено: " + "; ".join(comp.reasons) + ".")
    if comp.type_status == "conflict":
        parts.append(f"Другой тип изделия: {c['type']}.")
    if not parts:
        parts.append("Подтверждённых совпадений нет.")
    return {"rows": [type_row] + rows, "text": " ".join(parts)}
