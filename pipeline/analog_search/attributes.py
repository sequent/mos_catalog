"""Extraction of typed technical attributes from a name.

Input is text already passed through `normalize.normalize_text` (mult-sign → `x`,
decimal dot, canonical units, case preserved). Output values are in canonical units:
mm, V, A, kW, kVA, MPa, Mп — so «4Мп», «4 Мпикс» and «4MP» give the same value.

Every attribute keeps `raw` (evidence in the text) and `source`:
`main` for the name itself, `alt` for «(прим. …)» / «(или аналог)» fragments (not for strict comparison).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import load_dictionaries
from .normalize import code_pairs, code_skeleton, code_tokens, compact_code

NUM = r"\d+(?:\.\d+)?"
_L = r"A-Za-zА-Яа-яЁё"          # letters (both scripts)


def _num(s: str) -> int | float:
    v = float(s)
    return int(v) if v.is_integer() else v


@dataclass
class Attr:
    name: str
    value: Any
    raw: str
    unit: str | None = None
    source: str = "main"
    kind: str | None = None
    span: tuple[int, int] = (0, 0)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["span"] = list(self.span)
        return {k: v for k, v in d.items() if v not in (None, {}, [])}


# ---------------------------------------------------------------- patterns
CHAIN_RE = re.compile(rf"(?<![\d.x±])(?P<chain>{NUM}(?:x{NUM}){{1,3}})(?!\d|\.\d|x\d)")
CABLE_SUFFIX_RE = re.compile(r"(?:\s*(?P<mm2>мм2))?(?P<form>ок|мк|мс|ож|мж)?(?P<neutral>\([^)]*\))?(?:-(?P<volt>\d{3,4})(?!\d))?")
DIM_UNIT_RE = re.compile(rf"\s*(?P<u>мм|mm|см|м)(?![{_L}])")
MARK_RE = re.compile(
    rf"(?<![{_L}0-9])(?P<mark>[А-ЯЁ]{{2,}}[А-ЯЁа-яё]*(?:-[А-ЯЁ]+[А-ЯЁа-яё]*)?)(?P<cat>\((?:А|A)\))?"
    r"(?P<fire>(?:-(?:FRHF|FRLS|LSLTx|LSZH|LS|HF|LTx))+)?"
)
VOLT_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM}(?:/{NUM})?)(?P<sp>\s*)(?P<u>кВ|В|V)(?![{_L}0-9])")
VOLT_ATTACHED_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM}(?:/{NUM})?)(?P<u>в|кв)(?![{_L}0-9])")
CURRENT_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM})\s*(?P<u>кА|А|A)(?![{_L}0-9])")
POWER_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM})\s*(?P<u>МВт|кВт|Вт)(?![{_L}0-9])")
APPARENT_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM})\s*(?P<u>МВА|кВА|ВА)(?![{_L}0-9])")
IP_RE = re.compile(r"\b(?i:IP)\s?(?P<v>\d{2})\b")
EI_RE = re.compile(r"\bEI\s?(?P<v>\d{2,3})\b")
RES_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM})\s*Мп(?![{_L}])")
PRESSURE_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM})\s*(?P<u>МПа|кПа|бар)(?![{_L}])")
PN_RE = re.compile(rf"\b(?:PN|Ру)\s?(?P<v>{NUM})(?![0-9])")
CLIMATE_RE = re.compile(rf"(?<![{_L}0-9])(?P<c>УХЛ|ХЛ)\s?(?P<n>[1-5](?:\.\d)?)(?!\d)|(?<=-)(?P<c2>У|Т)(?P<n2>[1-5])(?![0-9{_L}])")
GOST_RE = re.compile(r"\bГОСТ(?:\s*Р)?\s*(?P<n>\d[\d.]*(?:-\d{2,4})?)")
TU_RE = re.compile(r"\bТУ\s?(?P<n>\d[\d.\-]*\d)")
POLES_RE = re.compile(r"(?<![A-Za-zА-Яа-я0-9])(?P<n>\d)[PР](?P<rest>(?:\+[NE]){0,2})(?![A-Za-zА-Яа-я0-9])")   # «3Р» may be Cyrillic
DIAM_WORD_RE = re.compile(
    rf"(?:(?P<kind>наружн\w*|внутренн\w*|номинальн\w*)\s+)?диаметр\w*\s*(?:(?P<what>резьбы|патрубка|стенки|отверстия)\s*)?"
    rf"[:=]?\s*(?P<q>до\s+|от\s+)?(?P<v>{NUM})(?:\s*(?P<u>мм|см|м)(?![{_L}]))?"
)
DIAM_SHORT_RE = re.compile(
    rf"(?<![{_L}0-9])(?P<sym>[dDдД]|Ø|ø)\s*(?P<sep>[=.:]?)\s*(?P<v>{NUM})(?!\s*/)(?:\s*(?P<u>мм|см|м)(?![{_L}]))?"
)
BORE_RE = re.compile(rf"(?<![{_L}])(?:Ду|ДУ|DN|Dy)\s*[=:]?\s*(?P<v>{NUM})(?![0-9])")
THICK_RE = re.compile(rf"толщин\w*\s*(?:стенки\s*)?[:=]?\s*(?:до\s+|от\s+)?(?P<v>{NUM})\s*(?P<u>мм|см)?")
# «L=3000 мм», «L 3000 мм», «L: 3 м». Without «=»/«:» the unit is required («L 3000» alone may be a code). The whole construct is captured
# first and the unit decides the scale; an explicit unit that is not supported leaves the number unexplained (see extract_attributes).
# number of keys of a switch: «одноклавишный», «двухклавишный», «1-кл.», «2 клавиши»; «1 кл.» (no hyphen) only for a switch, where «кл» is not «класс»
GANG_WORD_RE = re.compile(rf"(?<![{_L}])(?P<w>одно|двух|тр[её]х|четыр[её]х|пяти|шести)клавишн\w*", re.I)
_GANG_WORDS = {"одно": 1, "двух": 2, "трех": 3, "трёх": 3, "четырех": 4, "четырёх": 4, "пяти": 5, "шести": 6}
GANG_HYPHEN_RE = re.compile(rf"(?<![{_L}0-9])(?P<n>[1-6])\s*-\s*кл(?![{_L}])\.?", re.I)
GANG_SPACED_RE = re.compile(rf"(?<![{_L}0-9])(?P<n>[1-6])\s*(?:кл(?![{_L}])\.?|клавиш\w*)", re.I)
# storage time («хранение 5 сут.», «срок хранения газа - 5 дней», «5 суток хранения»): only with the word of storage, a bare «5 дней» may be anything
_DAYS = r"(?P<u>сут\w*|дн\w*|дней|день|нед\w*)(?![{L}])".replace("{L}", _L)
STORAGE_RE = re.compile(rf"хранени\w*[^,;.0-9]{{0,25}}(?P<v>\d+(?:\.\d+)?)\s*{_DAYS}", re.I)
STORAGE_REV_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>\d+(?:\.\d+)?)\s*{_DAYS}\s+хранени", re.I)
PORTS_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>\d{{1,4}})\s*-?\s*(?:портов\w*|порта|порт|port\w*)(?![{_L}])", re.I)
QTY_RE = re.compile(rf"(?<![{_L}0-9.x])(?P<v>\d{{1,4}})\s*шт(?![{_L}])\.?", re.I)
MOUNT_SURFACE_RE = re.compile(rf"(?<![{_L}])к\s+(?P<s>крыш\w*|стен\w*|потолк\w*|пол[уе](?![{_L}])|земл\w*)", re.I)
_SURFACE = {"крыш": "крыша", "стен": "стена", "потолк": "потолок", "пол": "пол", "земл": "земля"}
# a class written with a Roman numeral («грузов II класса», «песок II класс», «класс I»): a requirement; «I» and «II» are different things
CLASS_ROMAN_RE = re.compile(rf"(?<![{_L}0-9])(?P<n>[IVX]{{1,4}})(?![{_L}0-9])\s+класс\w*|(?<![{_L}])класс\w*\s+(?P<n2>[IVX]{{1,4}})(?![{_L}0-9])")
# the purpose of the thing is not its own property: «толщиномер для контроля толщины стальных труб» is not made of steel
PURPOSE_RE = re.compile(rf"(?<![{_L}])для\s+(?:контрол|измерен|проверк|определен|диагностик|испытан)\w*[^,;.]*", re.I)
LENGTH_RE = re.compile(rf"(?<![{_L}0-9])L\s*(?P<sep>[=:])?\s*(?P<v>{NUM})(?:\s*(?P<u>мм|см|м)(?![{_L}0-9]))?")
MASS_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM})\s*(?P<u>кг|т)(?![{_L}0-9])")
AREA_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM})\s*м2(?![{_L}0-9])")
VOLUME_M3_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM})\s*м3(?![{_L}0-9])")
VOLUME_L_RE = re.compile(rf"(?<![{_L}0-9.])(?P<v>{NUM})\s*(?P<u>мл|л)(?![{_L}0-9])")
LENGTH_M_RE = re.compile(rf"(?<![{_L}0-9.x])(?P<v>{NUM})\s*(?:м|метр\w*)(?![{_L}0-9=/])")
TEMP_RE = re.compile(rf"(?P<a>[+-]?{NUM})(?:\s*(?:до|\.\.\.|…|–)\s*(?P<b>[+-]?{NUM}))?\s*°\s*[СC]")
NUMCODE_RE = re.compile(rf"(?<![{_L}0-9./-])(?P<c>\d+(?:\.\d+)?(?:[/-]\d+(?:\.\d+)?)+)(?![{_L}0-9])")
RANGE_MM_RE = re.compile(rf"(?<![{_L}0-9.])(?P<a>{NUM})\s*[–—-]\s*(?P<b>{NUM})\s*(?P<u>мм|м)(?![{_L}])")
BARE_MM_RE = re.compile(rf"(?<![{_L}0-9.x])(?P<v>{NUM})\s*мм(?![{_L}0-9])")
OUTDOOR_RE = re.compile(r"\b(?:уличн\w*|наружн(?!\w*\s+диаметр)\w*)|для улицы", re.I)
INDOOR_RE = re.compile(r"\bвнутренн(?!\w*\s+(?:диаметр|угол|вертикальн|горизонтальн|резьб|стенк|блок))\w*|для помещени\w*", re.I)

_TO_MM = {"мм": 1, "mm": 1, "см": 10, "м": 1000}
_DN_SERIES = {6, 8, 10, 15, 20, 25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800,
              900, 1000, 1200, 1400, 1600, 1800, 2000}


def _measure(words: str) -> re.Pattern[str]:
    return re.compile(rf"(?:{words})\s*[:=]?\s*(?:до\s+|от\s+)?(?P<v>{NUM})\s*(?:(?P<u>мм|см|м)(?![{_L}0-9]))?")


_WORD_MEASURES = [
    ("length_mm", re.compile(rf"длин\w*\s*[:=]?\s*(?:до\s+|от\s+)?(?P<v>{NUM})\s*(?P<u>мм|см|м)(?![{_L}0-9])")),
    ("perimeter_mm", _measure(r"периметр\w*")),
    ("height_mm", re.compile(rf"(?:высот\w*(?:\s+подъема)?|(?<![{_L}])[HН]\s*=)\s*(?:до\s+)?(?P<v>{NUM})\s*(?:(?P<u>мм|см|м)(?![{_L}0-9]))?")),
    ("side_mm", _measure(r"сторон\w*")),
]
_CABLE_FAMILIES = ("кабель", "провод")


def is_cable_family(family: str | None) -> bool:
    return bool(family) and family in _CABLE_FAMILIES


# ---------------------------------------------------------------- extraction
def extract_attributes(text: str, family: str | None = None, source: str = "main", query: bool = False) -> list[Attr]:
    out: list[Attr] = []

    def add(name, value, m_or_span, *, unit=None, kind=None, extra=None, raw=None):
        span = m_or_span.span() if hasattr(m_or_span, "span") else m_or_span
        out.append(Attr(name, value, raw if raw is not None else text[span[0]:span[1]], unit, source, kind, span, extra or {}))

    # --- sizes: cable «cores×section» or generic dimensions
    cable_taken = False
    for m in CHAIN_RE.finditer(text):
        parts = [_num(x) for x in m.group("chain").split("x")]
        if is_cable_family(family) and not cable_taken:
            cable_taken = True
            sm = CABLE_SUFFIX_RE.match(text, m.end())
            end = sm.end() if sm else m.end()
            extra = {}
            if sm and sm.group("form"):
                extra["conductor_form"] = sm.group("form")
            add("cores_section", parts, (m.start(), end), unit="mm2" if len(parts) == 2 else "pairs", extra=extra)
            if sm and sm.group("volt"):
                add("rated_voltage_v", int(sm.group("volt")), sm.span("volt"))
            continue
        um = DIM_UNIT_RE.match(text, m.end())
        if um:
            k = _TO_MM[um.group("u")]
            parts = [_num(str(p * k)) for p in parts]
            add("dims", parts, (m.start(), um.end()), unit="mm", extra={"sorted": sorted(parts)})
        elif not re.match(rf"[{_L}]", text[m.end():m.end() + 1] or " "):
            add("dims", parts, m, unit="mm", extra={"sorted": sorted(parts)})

    # --- cable marking: base mark, screen letter, fire class
    if is_cable_family(family):
        for m in MARK_RE.finditer(text):
            mark = m.group("mark")
            if not ("нг" in mark or m.group("cat") or m.group("fire")):
                continue
            screen = "Э" in mark[1:]
            base = mark[0] + mark[1:].replace("Э", "")
            add("cable_mark", code_skeleton(base), m, kind="base", extra={"display": base})
            add("cable_screen", screen, m)
            if m.group("fire"):
                add("fire_class", m.group("fire").strip("-"), m)
            break

    # --- voltage
    for m in VOLT_RE.finditer(text):
        if m.group("u") == "В" and m.group("sp") and re.match(r"\s+[а-яё]", text[m.end():m.end() + 3]):
            continue                                   # «5 В комплекте»
        k = 1000 if m.group("u") == "кВ" else 1
        vals = [_num(str(float(x) * k)) for x in m.group("v").split("/")]
        add("voltage_v", vals, m, unit="V")
    for m in VOLT_ATTACHED_RE.finditer(text):
        k = 1000 if m.group("u") == "кв" else 1
        vals = [_num(str(float(x) * k)) for x in m.group("v").split("/")]
        add("voltage_v", vals, m, unit="V")

    for m in CURRENT_RE.finditer(text):
        k = 1000 if m.group("u") == "кА" else 1
        add("current_a", _num(str(float(m.group("v")) * k)), m, unit="A")
    for m in POWER_RE.finditer(text):
        k = {"МВт": 1000, "кВт": 1, "Вт": 0.001}[m.group("u")]
        add("power_kw", _num(str(round(float(m.group("v")) * k, 6))), m, unit="kW")
    for m in APPARENT_RE.finditer(text):
        k = {"МВА": 1000, "кВА": 1, "ВА": 0.001}[m.group("u")]
        add("apparent_power_kva", _num(str(round(float(m.group("v")) * k, 6))), m, unit="kVA")

    for m in IP_RE.finditer(text):
        add("ip", f"IP{m.group('v')}", m)
    for m in EI_RE.finditer(text):
        add("fire_resistance_ei", int(m.group("v")), m)
    for m in RES_RE.finditer(text):
        add("resolution_mp", _num(m.group("v")), m, unit="Мп")
    for m in PRESSURE_RE.finditer(text):
        k = {"МПа": 1, "кПа": 0.001, "бар": 0.1}[m.group("u")]
        add("pressure_mpa", _num(str(round(float(m.group("v")) * k, 6))), m, unit="MPa")
    for m in PN_RE.finditer(text):
        add("pn", _num(m.group("v")), m, unit="bar")
    for m in CLIMATE_RE.finditer(text):
        c = m.group("c") or m.group("c2")
        n = m.group("n") or m.group("n2")
        add("climate", f"{c}{n}", m)
    for m in GOST_RE.finditer(text):
        add("standard", f"ГОСТ {m.group('n')}", m)
    for m in TU_RE.finditer(text):
        add("standard", f"ТУ {m.group('n')}", m)
    for m in POLES_RE.finditer(text):
        rest = m.group("rest")
        add("poles", f"{m.group('n')}P" + ("+N" if "N" in rest else "") + ("+E" if "E" in rest else ""), m)

    # --- diameters, bore, thickness, length
    diam_spans: list[tuple[int, int]] = []
    for m in DIAM_WORD_RE.finditer(text):
        k = _TO_MM.get(m.group("u") or "мм", 1)
        kind = (m.group("kind") or m.group("what") or "").lower()[:6] or None
        extra = {"qualifier": m.group("q").strip()} if m.group("q") else {}
        add("diameter_mm", _num(str(float(m.group("v")) * k)), m, unit="mm", kind=kind, extra=extra)
        diam_spans.append(m.span())
    for m in DIAM_SHORT_RE.finditer(text):
        if any(a <= m.start() < b for a, b in diam_spans):
            continue
        if not m.group("u") and not m.group("sep") and m.group("sym") not in ("Ø", "ø"):
            continue                                   # «D400» (load class), not a diameter
        k = _TO_MM.get(m.group("u") or "мм", 1)
        add("diameter_mm", _num(str(float(m.group("v")) * k)), m, unit="mm", kind="short")
    for m in BORE_RE.finditer(text):
        if float(m.group("v")) not in _DN_SERIES:        # «DN1201» is a product code, not a nominal bore
            continue
        add("nominal_bore_mm", _num(m.group("v")), m, unit="mm")
    for name, rx in _WORD_MEASURES:                      # «длина 3 м», «периметром 2000 мм», «высота 12 м», «со стороной до 600 мм»
        for m in rx.finditer(text):
            k = _TO_MM[m.group("u") or "мм"]
            add(name, _num(str(float(m.group("v")) * k)), m, unit="mm")
    for m in THICK_RE.finditer(text):
        k = _TO_MM.get(m.group("u") or "мм", 1)
        add("thickness_mm", _num(str(float(m.group("v")) * k)), m, unit="mm")
    for m in LENGTH_RE.finditer(text):
        u = m.group("u")
        if not u and (not m.group("sep") or re.match(rf"\s*[{_L}°%]{{1,4}}(?![{_L}])", text[m.end():])):
            continue                                   # no «=» / «:» and no unit, or an explicit unit we do not know («L=3000 дм»): never a silent default
        add("length_mm", _num(str(float(m.group("v")) * _TO_MM[u or "мм"])), m, unit="mm", extra=({} if u else {"assumed_unit": "мм"}))

    # --- other measures: mass, area, volume, length in metres, temperature
    for m in MASS_RE.finditer(text):
        add("mass_kg", _num(str(float(m.group("v")) * (1000 if m.group("u") == "т" else 1))), m, unit="kg")
    for m in AREA_RE.finditer(text):
        add("area_m2", _num(m.group("v")), m, unit="m2")
    for m in VOLUME_M3_RE.finditer(text):
        add("volume_m3", _num(m.group("v")), m, unit="m3")
    for m in VOLUME_L_RE.finditer(text):
        add("volume_l", _num(str(float(m.group("v")) * (0.001 if m.group("u") == "мл" else 1))), m, unit="l")
    for m in LENGTH_M_RE.finditer(text):
        add("length_mm", _num(str(float(m.group("v")) * 1000)), m, unit="mm", kind="metres")
    for m in TEMP_RE.finditer(text):
        vals = [_num(x) for x in (m.group("a"), m.group("b")) if x] if m.group("b") else [_num(m.group("a"))]
        add("temperature_c", vals, m, unit="C")

    # --- ranges «2.8 – 12 мм» and bare millimetres «25 мм» (meaning depends on the type, see config)
    for m in RANGE_MM_RE.finditer(text):
        k = _TO_MM[m.group("u")]
        add("range_mm", [_num(str(float(m.group("a")) * k)), _num(str(float(m.group("b")) * k))], m, unit="mm")
    taken = [a.span for a in out]
    code_spans = [(s, e) for _, _, s, e in code_tokens(text)]
    for m in NUMCODE_RE.finditer(text):               # series like «221.04-1», «50/125», «41/21»
        if any(a <= m.start() < b for a, b in taken + code_spans):
            continue
        add("numeric_code", m.group("c"), m)
    taken = [a.span for a in out]
    for m in BARE_MM_RE.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        add("bare_mm", _num(m.group("v")), m, unit="mm")

    for m in CLASS_ROMAN_RE.finditer(text):
        add("class_roman", m.group("n") or m.group("n2"), m)

    # --- the number of keys of a switch (a separate characteristic: it is not the number of poles and not a model code)
    for m in GANG_WORD_RE.finditer(text):
        add("switch_gang_count", _GANG_WORDS[m.group("w").lower()], m)
    for m in GANG_HYPHEN_RE.finditer(text):
        add("switch_gang_count", int(m.group("n")), m)
    if family == "выключатель":
        for m in GANG_SPACED_RE.finditer(text):
            if not any(a.name == "switch_gang_count" and a.span[0] <= m.start() < a.span[1] for a in out):
                add("switch_gang_count", int(m.group("n")), m)

    # --- counts and times that a person writes with a counting word: ports, pieces, storage time; the surface a thing is fixed to
    for rx in (STORAGE_RE, STORAGE_REV_RE):
        for m in rx.finditer(text):
            k = 7 if m.group("u").lower().startswith("нед") else 1
            add("storage_duration_days", _num(str(float(m.group("v")) * k)), m, unit="day")
    for m in PORTS_RE.finditer(text):
        add("port_count", int(m.group("v")), m)
    for m in QTY_RE.finditer(text):
        add("quantity_pcs", int(m.group("v")), m)
    for m in MOUNT_SURFACE_RE.finditer(text):
        add("mount_surface", next(v for k, v in _SURFACE.items() if m.group("s").lower().startswith(k)), m)

    # --- categorical characteristics and kit components from the dictionary
    d = load_dictionaries()
    for attr, _label, contexts, values in d.features:
        if contexts and family not in contexts:
            continue
        for value, rx in values:
            m = rx.search(text)
            if m:
                add(attr, value, m)
                break
    comps = {}
    for comp, rx in d.kit_components:
        for m in rx.finditer(text):
            if not text[:m.start()].rstrip().lower().endswith("без"):              # «без гильз» is not a component
                comps.setdefault(comp, m.span())
                break
    if comps:
        add("kit_components", sorted(comps), min(comps.values()), raw=", ".join(sorted(comps)))

    # --- material (dictionary-driven): {ПВХ, полиэтилен, сталь…}; a material named inside «для контроля …» belongs to the object of the purpose
    found = []
    first_span = None
    purpose = [pm.span() for pm in PURPOSE_RE.finditer(text)]
    for name, rx in load_dictionaries().materials:
        m = next((x for x in rx.finditer(text) if not any(a <= x.start() < b for a, b in purpose)), None)
        if m:
            found.append(name)
            first_span = min(first_span, m.span()) if first_span else m.span()
    if found:
        add("material", sorted(found), first_span, raw=text[first_span[0]:first_span[1]])

    # --- color (dictionary-driven)
    found, first_span = [], None
    for name, rx in load_dictionaries().colors:
        m = rx.search(text)
        if m:
            found.append(name)
            first_span = min(first_span, m.span()) if first_span else m.span()
    if found:
        add("color", sorted(found), first_span, raw=text[first_span[0]:first_span[1]])

    # --- placement
    if OUTDOOR_RE.search(text):
        add("placement", "outdoor", OUTDOOR_RE.search(text))
    elif INDOOR_RE.search(text):
        add("placement", "indoor", INDOOR_RE.search(text))

    # --- model codes
    codes = code_tokens(text)
    pairs = code_pairs(text, query)
    if codes or pairs:
        allc = sorted([(s, e, t, compact_code(sk)) for t, sk, s, e in codes] + [(s, e, t, k) for t, k, s, e in pairs])
        keys = list(dict.fromkeys(k for _, _, _, k in allc))
        add("model_codes", keys, (allc[0][0], allc[-1][1]), raw=" ".join(t for _, _, t, _ in allc),
            extra={"display": list(dict.fromkeys(t for _, _, t, _ in allc))})
        for a in out:                               # «скат 1200м»: «1200м» is a code, not a length of 1200 metres
            if a.name == "length_mm" and a.kind == "metres" and any(s < a.span[1] and a.span[0] < e for _, _, s, e in pairs):
                a.extra["ambiguous_code"] = True

    out.sort(key=lambda a: a.span)
    return out


def uncovered_numbers(text: str, attrs: list[Attr]) -> list[str]:
    """Numbers of the name that no attribute explains: shows where extraction may be incomplete."""
    spans = [a.span for a in attrs if a.name != "model_codes"]
    code_spans = [(s, e) for _, _, s, e in code_tokens(text)]
    left = []
    for m in re.finditer(NUM, text):
        s = m.start()
        if any(a <= s < b for a, b in spans) or any(a <= s < b for a, b in code_spans):
            continue
        if re.match(rf"[{_L}0-9]", text[s - 1:s] or " ") and text[s - 1:s] not in ("=", "(", ":"):
            continue                                   # part of a word/code
        left.append(m.group(0))
    return left


# ---------------------------------------------------------------- the remainder: numbers that no attribute explains
# Technical units a person writes after a requirement. A number followed by one of them that the extractor did not turn into
# an attribute is a requirement the system cannot compare: it must not silently vanish from the decision.
_UNIT_WORDS = {
    "вт", "квт", "мвт", "ва", "ква", "мва", "квар", "ка", "а", "в", "кв", "мв", "гц", "кгц", "мгц", "ггц", "ом", "ком", "мом",
    "кг", "мм", "см", "км", "мл", "бар", "мпа", "кпа", "па", "атм", "дб", "лм", "лк", "кд", "мкф", "пф",
    "гб", "мб", "тб", "мбит", "гбит", "мин", "ккал", "гкал", "дж", "кдж", "мдж", "вб", "тл", "гн", "мгн",
    "w", "kw", "va", "kva", "hz", "khz", "mhz", "ghz", "v", "a", "ma", "ohm", "kg", "mm", "cm", "db", "lm", "lx", "cd",
}
# Only the units whose single letter is certain enough («16 А», «220 В»). «м», «т», «л», «с», «ф» after a number are far more often
# a fragment of a code («31Т», «36М», «300 L») or an ordinary letter than a unit; real metres, tonnes and litres are extracted anyway.
_SINGLE_LETTER_UNITS = {"а", "в", "a", "v"}
_UNIT_AFTER_RE = re.compile(rf"(?P<sp>\s*)(?P<u>[{_L}°%²³]{{1,5}})(?:/(?P<d>[{_L}]{{1,4}}))?(?![{_L}0-9])")
_LABEL_BEFORE_RE = re.compile(
    rf"(?<![{_L}])(?:мощност\w*|ток[аеу]?|напряжени\w*|давлени\w*|производительност\w*|[её]мкост\w*|масс\w*|вес|"
    rf"объ[её]м\w*|расход\w*|частот\w*|сечени\w*|ширин\w*|глубин\w*|номинал\w*|длин\w*)\s*[:=]?\s*(?:до\s+|от\s+)?$"
    rf"|(?<![{_L}0-9])L\s*[=:]?\s*$", re.I)
_REFERENCE_BEFORE_RE = re.compile(rf"(?:№|(?<![{_L}])(?:п|пп|исп|вар|тип|ряд|серия|рис|табл|поз)\.?)\s*$", re.I)


def find_unparsed_constraints(text: str, attrs: list[Attr], source: str = "main") -> list[dict]:
    """Numbers of `text` that no attribute or model code explains, each with a reason and a severity.

    reason: `number_with_unit` (a number followed by a technical unit), `measurement_label` («мощность 100»: the name of a
    quantity without its unit), `code_or_reference` («№5», «п.3», «исп.56»), `bare_number` (nothing says what it is).
    severity: `blocking` for the first two (the rules decide, see policy.unparsed_constraints), otherwise `informational`.
    Coordinates (`span`) refer to `text`, not to the original name. A fragment from «(прим. …)» is never blocking."""
    spans = [a.span for a in attrs if a.name != "model_codes"]
    code_spans = [(s, e) for _, _, s, e in code_tokens(text)]
    out = []
    for m in re.finditer(NUM, text):
        s, e = m.span()
        if any(a <= s < b for a, b in spans) or any(a <= s < b for a, b in code_spans):
            continue
        if re.match(rf"[{_L}0-9]", text[s - 1:s] or " ") and text[s - 1:s] not in ("=", "(", ":"):
            continue                                                       # part of a word or a code
        um = _UNIT_AFTER_RE.match(text, e)
        unit, end = None, e
        if um:
            u = um.group("u").lower().replace("ё", "е")
            single = u in _SINGLE_LETTER_UNITS
            if u in _UNIT_WORDS and (not single or not um.group("sp") or um.group("u").isupper()):
                unit, end = u + ("/" + um.group("d").lower() if um.group("d") else ""), um.end()
        before = text[max(0, s - 40):s]
        if unit:
            reason = "number_with_unit"
        elif _REFERENCE_BEFORE_RE.search(before):
            reason = "code_or_reference"
        elif _LABEL_BEFORE_RE.search(before):
            reason = "measurement_label"
        else:
            reason = "bare_number"
        severity = "blocking" if reason in ("number_with_unit", "measurement_label") and source == "main" else "informational"
        out.append({"raw": text[s:end].strip(), "span": [s, end], "source": source, "reason": reason, "severity": severity,
                    "value": _num(m.group(0)), "unit": unit})
    return out
