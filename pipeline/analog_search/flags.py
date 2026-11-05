"""Structural flags of a name: service, price adjustment, composite item."""
from __future__ import annotations

import re

_SERVICE_RE = re.compile(r"^\s*(?:(?:перевозк|погрузк|разгрузк|транспортировк)\w*|(?:перевоз|погр|разгр|трансп)\.)", re.I)      # also the abbreviations «Погр.», «Разгр.»
_ADJUSTMENT_RE = re.compile(r"^\s*(?:надбавк|доплат)\w*|^\s*ФССЦ\b|^\s*[cс]выше\s+\d", re.I)   # «Cвыше 200 км …» (латинская C)
_COMPOSITE_RE = re.compile(r"\bв\s+(?:составе|комплекте)\s*:", re.I)


def is_service(clean_text: str) -> bool:
    """Transport/loading lines are services, not equipment: they never are analogs."""
    return bool(_SERVICE_RE.match(clean_text))


def is_adjustment(clean_text: str) -> bool:
    """«Надбавка…», «Доплата…»: a price modifier of another item, not an item."""
    return bool(_ADJUSTMENT_RE.match(clean_text))


def composite_kind(clean_text: str) -> str | None:
    """`header` — «… в составе:» with the list in following rows (or nothing); `inline` — list in the same name."""
    m = _COMPOSITE_RE.search(clean_text)
    if not m:
        return None
    return "header" if not clean_text[m.end():].strip() else "inline"
