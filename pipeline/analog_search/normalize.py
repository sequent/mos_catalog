"""Text normalization shared by the catalog (offline) and the query (online).

The same functions must be used for both sides: otherwise «3х1,5» in the catalog
and «3*1.5» in the query will never be recognised as the same value.

Case is preserved by `normalize_text` (units such as В / кВ / А are case-sensitive);
`search_text` lowercases and applies the dictionary synonyms.
"""
from __future__ import annotations

import re
import unicodedata

from .config import load_dictionaries

# ---------------------------------------------------------------- cleaning
NUMBERING_RE = re.compile(r"^\s*(?:п\.?\s*)?\d{1,3}(?:\.\d{1,3})*\s*[.)]\s*(?=[^\W\d_])", re.I)
PLACEHOLDER_RE = re.compile(r"\b(?:Объект[-–][А-ЯA-Z0-9]+(?:/[а-яa-z]+)?|Г?ЭС[-–]\d+|ГРЭС[-–]\d+)", re.I)
ALT_RE = re.compile(r"\(\s*(?:прим(?:\.|ерно|енительно)?|применительно)[^()]*\)|\(\s*или\s+аналог[^()]*\)", re.I)


def clean(raw: str) -> tuple[str, str | None]:
    """Whitespace/Unicode cleanup, numbering prefix and anonymisation placeholders removed.

    Returns (clean_text, numbering_prefix). The original name is never modified elsewhere.
    """
    s = unicodedata.normalize("NFKC", raw).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    prefix = None
    m = NUMBERING_RE.match(s)
    if m:
        prefix = m.group(0).strip()
        s = s[m.end():]
    s = PLACEHOLDER_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, prefix


def split_alt(text: str) -> tuple[str, list[str]]:
    """Split «(прим. …)» / «(или аналог)» fragments from the main text.

    Numbers inside such fragments belong to an alternative item and must not take part
    in strict comparison (e.g. «100х10 мм (прим. 60х10)»).
    """
    alts = [m.group(0).strip("() ").strip() for m in ALT_RE.finditer(text)]
    main = re.sub(r"\s+", " ", ALT_RE.sub(" ", text)).strip()
    return main, alts


# ---------------------------------------------------------------- numbers and units
_MULT_RE = re.compile(r"(?<=\d)\s*[хxХX×*]\s*(?=\d)")
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_MM2_RE = re.compile(r"(?<=\d)\s*(?:мм\s*[²2](?!\d)|кв\.?\s*мм|мм\^2)", re.I)
_MPIX_RE = re.compile(r"(?i:(?<=\d)\s*(?:мп(?:икс\w*)?|mp(?:ix(?:els?)?)?|мегапикс\w*))(?![A-Za-zА-Яа-я])")


_LET = "A-Za-zА-Яа-яЁё"
# Spellings of a unit written after a number → the canonical one the extractor knows. Only combinations that cannot be read
# another way: «квт» is a kilowatt, but «мвт» may be a milliwatt or a megawatt, so it is NOT flattened (it stays unparsed and the
# query is sent to a check instead of getting a wrong value). Lowercase «кв» is ambiguous too and is left alone; «в», «а» after a number and a space are read as units unless a Cyrillic word follows.
_UNIT_SPELLINGS = [
    ("Вт", r"[вВ][тТ]|[wW]"),
    ("кВт", r"[кК][вВ][тТ]|kW|KW|kw|Kw"),
    ("кВА", r"[кК][вВ][аА]|kVA|KVA|kva|Kva"),
    ("ВА", r"[вВ][аА]|VA|va"),
    ("кА", r"[кК][аА]|kA|KA"),
    ("МПа", r"[мМ][пП][аА]|MPa|MPA|mpa"),
    ("кПа", r"[кК][пП][аА]|kPa|KPa|kpa"),
    ("кг", r"[кК][гГ]|kg|KG|Kg"),
    ("мм", r"ММ|Мм|мМ|MM|Mm|mM|mm"),                 # Latin «mm» too; «mm2» / «mm²» (cross-section) are excluded by the guards below
    ("см", r"[сС][мМ]|cm|Cm|CM"),
    ("мл", r"[мМ][лЛ]|ml|Ml|ML"),
    ("км", r"km|Km|KM"),
    ("т", r"t"),                                      # «до 3 t»: a tonne after a number and a space (never inside «10W-40» or a code)
    ("м", r"m"),
    ("кВ", r"КВ|Кв|kV|KV"),
]
# Abbreviated names of a measured quantity («диам. 20 мм», «толщ.: 0,7 мм»): the abbreviation sign is required, the label then reads as the full word.
_LABEL_ABBREVIATIONS = [(re.compile(rf"(?<![{_LET}0-9]){a}\."), full) for a, full in (
    ("диам", "диаметр"), ("толщ", "толщина"), ("дл", "длина"), ("выс", "высота"), ("шир", "ширина"), ("глуб", "глубина"))]
_UNIT_AFTER_NUM_RES = [
    (re.compile(rf"(?<![{_LET}0-9.]){'(?<!-)' if canon in ('см', 'м', 'км', 'т') else ''}(?P<n>\d+(?:\.\d+)?)(?P<sp>\s*)(?:{alts})(?![{_LET}0-9²³]|-\d)"), canon)      # «10W-40» is an oil grade, not 10 W; «SKAT-5m» is a code
    for canon, alts in _UNIT_SPELLINGS
]
# Lowercase «в» / «а» after a number and a space («380 в IP66», «10 а, IP20»): a volt and an ampere when nothing Cyrillic follows;
# before a Cyrillic word it is a preposition or a conjunction («100 в комплекте», «2 а также»). Without this the requirement
# would turn into a bare number and silently drop out of the decision (re-check R02, case «380 в»).
_SPACED_VA_RE = re.compile(rf"(?<![{_LET}0-9.])(?P<n>\d+(?:\.\d+)?)(?P<sp>\s+)(?P<u>[вВаА])(?![{_LET}0-9-])(?!\s*[А-Яа-яЁё])")
_BEFORE_NUM_RES = [
    (re.compile(rf"(?<![{_LET}0-9])(?:ei|Ei|eI)(?P<sp>\s?)(?=\d)"), "EI"),
    (re.compile(rf"(?<![{_LET}0-9])(?:ду|ДУ|дУ)(?P<sp>\s?)(?=\d)"), "Ду"),
    (re.compile(rf"(?<![{_LET}0-9])(?:dn|Dn|dN)(?P<sp>\s?)(?=\d)"), "DN"),
]


def normalize_unit_notation(text: str) -> str:
    """One spelling per unit: «100 вт», «100 ВТ», «100W» → «100 Вт»; «ei60» → «EI60»; «ду80» → «Ду80» (value and spacing kept)."""
    for rx, canon in _UNIT_AFTER_NUM_RES:
        text = rx.sub(lambda m, c=canon: f"{m.group('n')}{m.group('sp')}{c}", text)
    text = _SPACED_VA_RE.sub(lambda m: f"{m.group('n')}{m.group('sp')}{m.group('u').upper()}", text)
    for rx, canon in _BEFORE_NUM_RES:
        text = rx.sub(lambda m, c=canon: f"{c}{m.group('sp')}", text)
    return text


def normalize_text(text: str) -> str:
    """Canonical notation for numbers and units (case preserved).

    - multiplication sign between digits (х x × * Х X) → `x`;  «3 х 1,5» → «3x1.5»
    - decimal comma between digits → dot
    - cross-section units (мм2, мм², кв.мм) → `мм2`;  resolution units (Мп, Мпикс, MP) → `Мп`
    - unit spelling (вт, ВТ, W → Вт; ква → кВА; ei → EI; ду → Ду), see `normalize_unit_notation`
    """
    for rx, full in _LABEL_ABBREVIATIONS:
        text = rx.sub(full, text)
    s = _MULT_RE.sub("x", text)
    s = _DECIMAL_COMMA_RE.sub(".", s)
    s = _MM2_RE.sub("мм2", s)
    s = _MPIX_RE.sub("Мп", s)
    return normalize_unit_notation(s)


# ---------------------------------------------------------------- codes
# Cyrillic letters that look like Latin ones, mapped for matching only (display keeps the original).
_HOMOGLYPHS = str.maketrans("АВЕКМНОРСТХУавекмнорстху", "ABEKMHOPCTXYabekmhopctxy")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-./+_]*")
_UNIT_LIKE_RE = re.compile(
    r"^(?:\d+(?:\.\d+)?[a-zа-я°%²³]{0,5}\d?"      # 220в, 10мм, 2мп, 25mm2
    r"|\d+(?:\.\d+)?x\d.*"                         # 25x17, 3x1.5, 600x600
    r"|ip\d{2}|ei\d+|ухл\d|[мm]\d{1,2}|d\d+|[wf]\d{1,3}|b\d{2}|\d+p(?:\+[ne])*"
    r"|гост.*|ту\d.*)$"
)


def code_skeleton(token: str) -> str:
    """Matching form of a model code: lowercase, look-alike Cyrillic → Latin, l/I → i."""
    s = token.translate(_HOMOGLYPHS).lower().replace("ё", "е")
    return s.replace("l", "i")


_SEP_RE = re.compile(r"[-./\s_+]")
_WORD_ONLY_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]{2,8}$")
_NUMLIKE_RE = re.compile(r"^\d+(?:[./-]\d+)*[A-Za-zА-Яа-яЁё]{0,2}$")


def compact_code(skeleton: str) -> str:
    """Matching key of a code: separators removed, so «DVN-800-6D», «DVN 800-6D» and «dvn800/6d» meet."""
    return _SEP_RE.sub("", skeleton)


def code_pairs(text: str, query: bool = False) -> list[tuple[str, str, int, int]]:
    """Codes that are written as «word NUMBER» («KTCS 200», «АС 240/32», «скат 1200м»): (original, compact key, start, end).

    The catalog joins only Latin words and UPPERCASE Cyrillic words (series names); a query is typed by a person, often in
    lowercase, so there a short (≤ 5 letters) Cyrillic word is joined too, but only before a number that carries a letter
    suffix or a separator («скат 1200м»). A bare quantity never joins («труба 50»)."""
    toks = []
    for m in _TOKEN_RE.finditer(text):
        t = m.group(0).strip("-./+_")
        if t:
            toks.append((t, m.start(), m.start() + len(t)))
    out = []
    for (w, ws, we), (t, ts, te) in zip(toks, toks[1:]):
        if text[we:ts] not in (" ", "-") or not _WORD_ONLY_RE.match(w) or not _NUMLIKE_RE.match(t):
            continue
        latin = re.fullmatch(r"[A-Za-z]+", w) is not None
        cyr = re.fullmatch(r"[А-Яа-яЁё]+", w) is not None
        # a lowercase Cyrillic word typed by a person joins only when the number looks like part of a code
        # («скат 1200м»: letter suffix), never for a bare quantity («труба 50», «кабель 100»)
        typed_code = query and cyr and len(w) <= 5 and re.search(r"\d[A-Za-zА-Яа-яЁё]|[./-]", t) is not None
        if not (latin or (cyr and w.isupper()) or typed_code):
            continue
        out.append((text[ws:te], compact_code(code_skeleton(w + t)), ws, te))
    return out


def is_code_like(token: str) -> bool:
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    return has_digit and has_alpha and len(token) >= 3


def code_tokens(text: str) -> list[tuple[str, str, int, int]]:
    """Model-code candidates: (original, skeleton, start, end). Unit-like tokens are excluded."""
    out = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).strip("-./+_")
        if not is_code_like(tok):
            continue
        sk = code_skeleton(tok)
        if _UNIT_LIKE_RE.match(sk):
            continue
        out.append((tok, sk, m.start(), m.end()))
    return out


# ---------------------------------------------------------------- search text
_WORD_RE = re.compile(r"[a-zа-я0-9]+(?:[-./+][a-zа-я0-9]+)*")


def apply_synonyms(text_lower: str) -> tuple[str, list[str]]:
    """Apply dictionary `replace` (abbreviation → expansion) and collect `expand` additions."""
    d = load_dictionaries()
    s = text_lower
    for pat, to in d.replace:
        s = pat.sub(to, s)
    added = []
    for pat, add in d.expand:
        if pat.search(s):
            added.append(add)
    return s, added


def search_text(text: str) -> tuple[str, list[str]]:
    """Lowercased text with synonyms and code skeletons; returns (text, tokens).

    `text` must already be passed through `normalize_text`.
    """
    s = text.replace("ё", "е").replace("Ё", "Е")
    lowered, added = apply_synonyms(s.lower())
    tokens: list[str] = []
    for m in _WORD_RE.finditer(lowered + " " + " ".join(added)):
        tok = m.group(0)
        tokens.append(code_skeleton(tok) if is_code_like(tok) and not _UNIT_LIKE_RE.match(code_skeleton(tok)) else tok)
    return lowered, tokens
