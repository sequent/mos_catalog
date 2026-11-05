"""B3 + B4: type as a signal and technical compatibility of a candidate with a query.

Only FACTS are produced here (statuses of characteristics and a compatibility class); no scores, no ordering.
Ranking is B5, the user-facing verdict is B6.

Statuses of one characteristic:
  match            equal after normalization
  match_reordered  sizes equal only after swapping sides, and the type has no rule on order → «проверить»
  substitute       differs, but a rule allows it (ladder / min_ge)
  conflict         differs and nothing allows it
  flag             differs; the rule only marks it («проверить»)
  no_data          the candidate does not state the characteristic
  alt_only         the value is present only inside «(прим. …)» of the candidate → «проверить»
  ambiguous        «25 мм» without a meaning equals some value of the candidate → «проверить»
  info / info_diff informational (model code): equal confirms, different blocks nothing
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import load_dictionaries
from ..normalize import _WORD_RE, apply_synonyms, code_skeleton
from ..retrieval.tokens import stem
from .rules import Rules, TypeContext

PRIORITY = {"match": 0, "substitute": 1, "match_reordered": 2, "flag": 3, "ambiguous": 4, "alt_only": 5,
            "no_data": 6, "info_diff": 7, "conflict": 8}
CONFIRMING = {"match", "substitute"}
UNVERIFYING = {"match_reordered", "flag", "ambiguous", "alt_only", "no_data"}
MM_LIKE = ("diameter_mm", "thickness_mm", "length_mm", "perimeter_mm", "height_mm", "side_mm", "nominal_bore_mm", "bare_mm")
_WORD_STOP = frozenset("для при без под над или что как его тип гост мпа кгс бар мкм ква квт".split())

REASON_TEXT = {
    "match_reordered": "размеры совпали только после перестановки сторон",
    "flag": "отличие, которое правило только помечает",
    "ambiguous": "значение названо без смысла",
    "alt_only": "значение есть только в скобке «прим.»",
    "no_data": "в названии нет характеристики",
}


@dataclass
class AttrComparison:
    name: str
    status: str
    query_value: object
    candidate_values: list
    rule_id: str | None = None
    note: str | None = None
    rule_scope: str | None = None                   # why this rule: «тип запроса», «семейство запроса», …
    query_display: object = None                    # original spelling (matching uses the skeleton form of codes)
    candidate_display: list = field(default_factory=list)


@dataclass
class Compatibility:
    klass: str                                  # compatible | compatible_with_substitution | unverified | incompatible
    type_relation: str                          # key of policy.type.relation
    type_status: str                            # match | check | unknown | conflict
    comparisons: list[AttrComparison]
    reasons: list[str] = field(default_factory=list)          # why the class is «unverified»
    conflicts: list[str] = field(default_factory=list)        # human-readable conflicts
    confirmed: int = 0
    confirmed_strong: int = 0               # confirmations that count as evidence (weak attributes excluded)
    compared: int = 0
    weak_evidence: bool = False
    word_coverage: float = 1.0
    missing_words: list[str] = field(default_factory=list)
    type_confirmed: bool = False            # confirmed by the type alone (policy.evidence.confirm_by_type)
    unparsed: list[str] = field(default_factory=list)       # requirements of the query that could not be parsed and compared
    match_basis: str | None = None          # what a «compatible» answer rests on: request_constraints | normalized_name | type_assumption
    text_matched: list[str] = field(default_factory=list)   # blocking requirements not parsed as an attribute, resolved only by the same written value

    @property
    def attribute_coverage(self) -> float:
        """Share of the compared characteristics that are confirmed (1.0 for a verbatim name match)."""
        if not self.compared:
            return 1.0 if self.confirmed else 0.0
        return min(1.0, self.confirmed / self.compared)


# ---------------------------------------------------------------- helpers
def _canon(v):
    if isinstance(v, (list, tuple)):
        return tuple(_canon(x) for x in v)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 9)
    return v


def _alias(a: dict) -> tuple[str, object]:
    """The same quantity written two ways is compared as one:
    «Ду80» = «номинальный диаметр 80 мм» (nominal bore), «Ру10» (10 bar) = «давление 1,0 МПа»."""
    name, value = a["name"], a["value"]
    if name == "diameter_mm" and str(a.get("kind") or "").startswith("номин"):
        return "nominal_bore_mm", value
    if name == "pn":
        return "pressure_mpa", round(value / 10, 6)
    return name, value


def _values(rec: dict, source: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for a in rec["attributes"]:
        if a["source"] == source:
            name, value = _alias(a)
            out.setdefault(name, []).append(value)
    return out


def _displays(rec: dict, source: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for a in rec["attributes"]:
        d = a.get("extra", {}).get("display")
        if a["source"] == source and d is not None:
            out.setdefault(a["name"], []).append(d)
    return out


def _alias_voltages(vals: dict[str, list]) -> None:
    """A cable's «-660» (rated voltage) and a plain «660 В» are the same quantity: let either spelling answer the other.

    Found on the dev control queries: «кабель … 0.66кВ» did not meet «ВВГнг … 5х6-660»."""
    if "voltage_v" not in vals and "rated_voltage_v" in vals:
        vals["voltage_v"] = [[v] for v in vals["rated_voltage_v"]]
    if "rated_voltage_v" not in vals and "voltage_v" in vals:
        vals["rated_voltage_v"] = [v[0] for v in vals["voltage_v"] if len(v) == 1]
        if not vals["rated_voltage_v"]:
            del vals["rated_voltage_v"]


def _best(statuses: list[str]) -> str:
    return min(statuses, key=lambda s: PRIORITY[s])


# ---------------------------------------------------------------- B3: type as a signal
def type_relation(q: dict, c: dict, rules: Rules) -> tuple[str, str]:
    """(relation key, status). Never removes a candidate: the status only feeds the compatibility class."""
    qt, ct = q["type"], c["type"]
    if not qt or not ct:
        key = "unknown_either_side"
    elif qt == ct:
        key = "same"
    elif q["family"] == c["family"]:
        # «ворота» is the beginning of «ворота откатные»: the query is just less specific, which is not a difference
        if not q["type_specific"] or ct.startswith(qt + " "):
            key = "same_family_q_generic"
        elif not c["type_specific"] or qt.startswith(ct + " "):
            key = "same_family_c_generic"
        else:
            key = "same_family_both_specific"
    else:
        key = "different_both_specific" if q["type_specific"] and c["type_specific"] else "different_one_generic"
    return key, rules.policy["type"]["relation"][key]


# ---------------------------------------------------------------- B4: one characteristic
def _compare_dims(qv, cvals, rule, ctx: TypeContext, rules: Rules) -> tuple[str, str | None]:
    fallback = rules.defaults.get("dims_order_unlisted", "check")
    order = rule.dims if rule and rule.dims else (rules.dims_order_for(ctx)[0] or (None if fallback == "check" else fallback))
    found = []
    for cv in cvals:
        if len(cv) != len(qv):
            found.append("conflict")
        elif _canon(qv) == _canon(cv):
            found.append("match")
        elif _canon(sorted(qv)) == _canon(sorted(cv)):
            found.append("match" if order == "sorted" else "conflict" if order == "ordered" else "match_reordered")
        else:
            found.append("conflict")
    return _best(found), order


_CODE_NOISE = tuple(code_skeleton(x) for x in ("исп", "ту", "гост", "iec", "din"))      # compared in the skeleton form of codes


_SHORT_NOISE_RE = re.compile(r"^(?:ip|ei|dn|ду|pn|cat|rs|usb|rj|ст|ту|кл|[0-9])")


def strong_codes(codes, short_ok: bool = False) -> set[str]:
    """Codes that can identify a model: long, with letters AND digits, not a service token.

    «исп.56», «л-1», «4-х» are not model codes: sharing one of them must never confirm a candidate («РИП-12» ≠ «РИП-24»).
    `short_ok` (both sides name the same type): a four-character code that starts with letters («МО-05», «TOR-1») also counts, but never a
    unit, degree of protection, standard or connector token («IP20», «DN50», «RJ45»): a short code alone is not a model in any context."""
    def ok(c: str) -> bool:
        if not (re.search(r"\d", c) and re.search(r"[a-zа-я]", c)) or c.startswith(_CODE_NOISE):
            return False
        return len(c) >= 5 or (short_ok and len(c) == 4 and not _SHORT_NOISE_RE.match(c))
    return {c for c in codes if ok(c)}


def _series(display: str) -> tuple[str, int] | None:
    """The series of a hyphenated model written as `BA10-001B`, `МО-05`, `SKAT-V.12DC-18`: (skeleton of the first part, number of parts).
    Service tokens («1-кл», «исп.56», «IP 20», «ГОСТ 8639-82») have no series."""
    parts = display.replace(" ", "").split("-")
    sk = code_skeleton(parts[0])
    if len(parts) < 2 or len(parts[0]) < 2 or not re.search(r"[a-zа-я]", sk) or sk.startswith(_CODE_NOISE):
        return None
    return sk, len(parts)


def _other_model_of_the_same_series(qdisplays: list[str], cdisplays: list[str]) -> tuple[str, str] | None:
    """A model that is explicitly requested and a model of the SAME series that differs (`BA10-001B` ≠ `BA10-006B`, `МО-05` ≠ `МО-06`).
    A candidate that states the requested code, or the same code with a suffix, is not in conflict."""
    for q in qdisplays:
        sq, kq = code_skeleton(q), _series(q)
        if kq is None:
            continue
        same_series = [c for c in cdisplays if _series(c) == kq]
        if same_series and not any(code_skeleton(c).startswith(sq) or sq.startswith(code_skeleton(c)) for c in cdisplays):
            return q, same_series[0]
    return None


# A Latin word of the query («Fluke», «JKL», «Noirot») is a manufacturer or a series the user asked for. Common technical abbreviations are not.
_IDENT_RE = re.compile(r"[A-Za-z]{2,}")
_WORD_IDENT_RE = re.compile(r"(?<![A-Za-z0-9-])[A-Za-z]{3,}(?![A-Za-z0-9-])")      # a whole word, not a part of a hyphenated code («XNV-6081RE»)
_NOT_AN_IDENTIFIER = frozenset("""led utp ftp stp sftp usb hdmi pvc gsm wifi lan vga dvi gps sim sfp cat ip pn dn rj din iec dc ac lcd oled ups nvr dvr poe
    lc sc fc st rs db mm cm kv kw ma hz vac vdc mp ir ptz rgb sms ac dc um mm2""".split())


def _identifiers(text: str, min_len: int, whole_words: bool = False) -> set[str]:
    out = set()
    for w in (_WORD_IDENT_RE if whole_words else _IDENT_RE).findall(text):
        w = w.lower()
        if len(w) >= min_len and w not in _NOT_AN_IDENTIFIER:
            out.add(w)
    return out


def _series_requirement(q: dict, c: dict) -> AttrComparison | None:
    """The manufacturer or series named in the query as a Latin word. The candidate that states it: `info` (a match of a brand alone never
    confirms a thing). A candidate that states OTHER Latin identifiers: `conflict` («JKL» asked, «BN-Z» given). A candidate that states none:
    `no_data` (unknown is not a match)."""
    if any(a["attribute"] == "manufacturer" for a in q.get("assumptions", [])):
        return None                                              # «любого производителя»: the user withdrew the requirement
    want = _identifiers(q["main_text"], 3, whole_words=True)
    if not want:
        return None
    have = _identifiers(c["main_text"], 2)
    shown = sorted(w.upper() if len(w) <= 4 else w.capitalize() for w in want)
    if want & have:
        return AttrComparison("series", "info", shown, [], None, "производитель/серия названы и в позиции")
    if have:
        return AttrComparison("series", "conflict", shown, sorted(w.upper() for w in have), None, "названа другая серия/производитель")
    return AttrComparison("series", "no_data", shown, [], None, "производитель/серия в названии позиции не названы")


def _compare_one(name, qv, cmain, calt, rule, mode, ctx: TypeContext, rules: Rules, c: dict, qdisplay=None) -> AttrComparison:
    cvals = cmain.get(name, [])
    rid = rule.id if rule else None
    if name == "model_codes":
        cdisp = [d for ds in _displays(c, "main").get("model_codes", []) for d in (ds if isinstance(ds, list) else [ds])]
        other = _other_model_of_the_same_series(qdisplay or [], cdisp)
        if other:                                    # the model is named in the query: another model of the series is a conflict, not «info»
            return AttrComparison(name, "conflict", qv, cvals, rid, f"другая модель той же серии: {other[1]} вместо {other[0]}")
        short = bool(c["type"]) and c["type"] == ctx.query_type
        qs, cs = strong_codes(qv, short), strong_codes([x for v in cvals for x in v], short)
        if not qs:
            return AttrComparison(name, "info_diff", qv, cvals, rid, "в запросе нет надёжного кода модели")
        if not cs:
            return AttrComparison(name, "info_diff", qv, cvals, rid, "у кандидата нет надёжного кода модели")
        return AttrComparison(name, "match" if qs & cs else "info_diff", qv, cvals, rid)
    if name == "bare_mm":
        if any(_canon(qv) == _canon(x) for x in cvals):              # the candidate states it the same way: a real match
            return AttrComparison(name, "match", qv, cvals, rid)
        like = [x for n in MM_LIKE if n != "bare_mm" for x in cmain.get(n, [])]
        return AttrComparison(name, "ambiguous" if any(_canon(qv) == _canon(x) for x in like) else "no_data", qv, like, rid)
    if name == "kit_components":                      # every component of the query must be NAMED by the candidate; unknown is not a match
        have = {x for v in cvals for x in v}
        missing = [x for x in qv if x not in have]
        if not missing:
            return AttrComparison(name, "match", qv, cvals, rid)
        return AttrComparison(name, "no_data", qv, cvals, rid, "в названии не названы: " + ", ".join(missing))
    if not cvals:
        alt = calt.get(name, [])
        st = "alt_only" if any(_canon(qv) == _canon(x) for x in alt) else "no_data"
        return AttrComparison(name, st, qv, [], rid)
    if name == "numeric_code":                       # «22/6» is the beginning of «22/6-2»: a series written without its suffix
        def _same(a, b):
            return a == b or a.startswith((b + "-", b + "/")) or b.startswith((a + "-", a + "/"))
        return AttrComparison(name, "match" if any(_same(str(qv), str(x)) for x in cvals) else "conflict", qv, cvals, rid)
    if name == "material":                           # any common material confirms; only disjoint materials conflict
        have, want, gen = {m for v in cvals for m in v}, set(qv), load_dictionaries().generic_materials

        def expand(s):                                  # «металл» also counts as what it is the same as («сталь»)
            return s | {t for g in s if g in gen for t in gen[g]["same_as"]}
        if expand(want) & expand(have):
            return AttrComparison(name, "match", qv, cvals, rid)
        for a, b in ((want, have), (have, want)):       # a generic word does not exclude the other metals of its group
            if any(b & gen[g]["any_of"] for g in a if g in gen):
                return AttrComparison(name, "no_data", qv, cvals, rid, "общее слово «металлическая» не исключает этот металл")
        return AttrComparison(name, "conflict", qv, cvals, rid)
    if name == "color":                              # any common colour confirms; a different one follows the mode of the attribute
        have = {m for v in cvals for m in v}
        if set(qv) & have:
            return AttrComparison(name, "match", qv, cvals, rid)
        return AttrComparison(name, {"exact": "conflict", "flag": "flag"}.get(mode, "info_diff"), qv, cvals, rid)
    if name == "dims":
        st, order = _compare_dims(qv, cvals, rule, ctx, rules)
        return AttrComparison(name, st, qv, cvals, rid, f"порядок сторон: {order or 'не задан'}" if st in ("match_reordered", "conflict") else None)
    eq = [_canon(qv) == _canon(x) for x in cvals]
    if mode in ("exact", "ambiguous"):
        return AttrComparison(name, "match" if any(eq) else "conflict", qv, cvals, rid)
    if mode == "flag":
        return AttrComparison(name, "match" if any(eq) else "flag", qv, cvals, rid)
    if mode == "info":
        return AttrComparison(name, "match" if any(eq) else "info_diff", qv, cvals, rid)
    if mode == "ladder":
        res = []
        for cv in cvals:
            if qv in rule.ladder and cv in rule.ladder:
                d = abs(rule.ladder.index(qv) - rule.ladder.index(cv))
                res.append("match" if d == 0 else "substitute" if d <= rule.max_steps else "conflict")
            else:
                res.append("match" if _canon(qv) == _canon(cv) else "conflict")
        return AttrComparison(name, _best(res), qv, cvals, rid)
    if mode == "min_ge":
        res = ["match" if _canon(cv) == _canon(qv) else "substitute" if cv > qv else "conflict" for cv in cvals]
        return AttrComparison(name, _best(res), qv, cvals, rid)
    raise ValueError(f"unexpected mode {mode}")


# ---------------------------------------------------------------- words of the query not explained by characteristics
def content_words(rec: dict) -> set[str]:
    """Stems of the plain words of a record (hyphenated codes and numbers are not words)."""
    out = set()
    for t in rec["tokens"]:
        if t.isalpha() and len(t) >= 3 and t not in _WORD_STOP:
            out.add(stem(t))
    return out


def _query_word_groups(q: dict) -> list[tuple[str, set[str]]]:
    """Every plain word of the query with the stems that count as covering it: the word itself and its dictionary synonyms
    («камера» is covered by «видеокамера»)."""
    groups, seen = [], set()
    abbreviated = {m.group(1).lower().replace("ё", "е") for m in re.finditer(r"(?<![а-яёa-z])([а-яё]{3,})\.", q.get("main_text", ""), re.I)}
    for w in _WORD_RE.findall(q["search_text"]):
        if not w.isalpha() or len(w) < 3 or w in _WORD_STOP or w in seen:
            continue
        seen.add(w)
        eq = {stem(w)}
        if w in abbreviated:
            eq.add("~" + w)                                                  # «соед.» is covered by any word that begins with «соед»
        _, added = apply_synonyms(w)
        for phrase in added:
            eq |= {stem(t) for t in phrase.split()}
        groups.append((w, eq))
    return groups


def _covered(eq: set[str], cw: set[str]) -> bool:
    """A query word is covered if one of its stems equals a stem of the candidate or is its prefix/extension (≥ 5 letters):
    «комплект» ↔ «комплектн(ое)». Short stems must match exactly."""
    if eq & cw:
        return True
    if any(e.startswith("~") and any(w.startswith(e[1:]) for w in cw) for e in eq):
        return True
    return any(len(e) >= 5 and len(w) >= 5 and (w.startswith(e) or e.startswith(w)) for e in eq for w in cw if not e.startswith("~"))


_CONTAINER_WORDS = frozenset({"комплект", "набор", "комплекты", "наборы"})
_ADJECTIVE_END_RE = re.compile(r"(?:ый|ой|ий|ая|яя|ую|юю|ые|ого|его|ых|ым|ими|ыми)$")


def _covered_strict(eq: set[str], cw: set[str]) -> bool:
    """Coverage of the SUBJECT of a query: the same stem, or an inflection of it (a short difference of at most three letters).
    The loose rule of `_covered` would let «толщиномер» be covered by «толщина»."""
    if eq & cw:
        return True
    if any(e.startswith("~") and any(w.startswith(e[1:]) for w in cw) for e in eq):
        return True
    return any(min(len(e), len(w)) >= 5 and abs(len(e) - len(w)) <= 3 and (w.startswith(e) or e.startswith(w)) for e in eq for w in cw if not e.startswith("~"))


def query_subject(q: dict) -> tuple[str, set[str]] | None:
    """The word the query names its thing with: the first plain word that is not an adjective («ультразвуковой толщиномер» → толщиномер,
    «цифровой манометр» → манометр, «выкл.» → выключатель by the dictionary). None when the query has no such word."""
    groups = dict(_query_word_groups(q))
    for w in _WORD_RE.findall(q["search_text"]):
        if "-" in w and w.replace("-", "").isalpha():
            return None                              # a compound noun («патч-корд»): the plain-word check cannot judge it
        if w in _CONTAINER_WORDS and any(x not in _CONTAINER_WORDS and x != w for x in groups):
            continue                                 # «комплект кронштейнов»: the thing is the bracket, the kit is only its form of supply
        if w in groups and not (len(w) >= 6 and _ADJECTIVE_END_RE.search(w)):
            return w, groups[w]
    return None


def word_coverage(q: dict, c: dict) -> tuple[float, list[str], int]:
    """Share of the query's words found in the candidate (a word may be covered by its synonym), the missing words and
    the number of words the query has."""
    groups = _query_word_groups(q)
    if not groups:
        return 1.0, [], 0
    cw = content_words(c)
    missing = [w for w, eq in groups if not _covered(eq, cw)]
    return 1 - len(missing) / len(groups), missing, len(groups)


def _split_constraints(q: dict, c: dict, rules: Rules) -> tuple[list[dict], list[dict]]:
    """Blocking unparsed constraints of the query (policy.unparsed_constraints.blocking_reasons), split into
    (unresolved, resolved by the same written value). Policy: the same number with the same unit written in the candidate's main
    name resolves the requirement («50 Гц» on both sides), but it is only a match of the notation, not a checked parameter: the
    answer says so (`text_matched`). A different value, no value, or a value only in «прим.» leaves it open."""
    blocking = set((rules.policy.get("unparsed_constraints") or {}).get("blocking_reasons") or [])
    if not blocking:
        return [], []
    have = {(x["value"], x["unit"]) for x in c.get("unparsed_constraints", []) if x.get("unit") and x["source"] == "main"}
    open_, done = [], []
    for u in q.get("unparsed_constraints", []):
        if u["source"] != "main" or u["reason"] not in blocking:
            continue
        (done if u.get("unit") and (u["value"], u["unit"]) in have else open_).append(u)
    return open_, done


def _unresolved_constraints(q: dict, c: dict, rules: Rules) -> list[dict]:
    return _split_constraints(q, c, rules)[0]


_DECORATIVE_PUNCT_RE = re.compile(r"[,;](?!\d)|(?<=\d)[,;](?=\s)|[«»\"“”„]|:(?!\d)")        # commas, semicolons, quotes and the colon after a label


def _name_key(rec: dict) -> str:
    """One key for «the whole name is the same» (B4 and the ground of the match). Only decoration is removed: case, repeated spaces and commas
    or semicolons between words (a decimal comma has already become a dot in the technical normalisation). Order, numbers, units, signs,
    class marks («II»), hyphens, slashes and every word stay: no stemming, synonyms, sorting or stop words."""
    return " ".join(_DECORATIVE_PUNCT_RE.sub(" ", rec["main_text"].lower()).split())


_ALPHA_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")


def _abbreviation_of(x: str, y: str) -> bool:
    """`x` is an abbreviation of the word `y`: «соед.» of «соединительный», «приемно-контр.» of «приемно-контрольный»."""
    return x.endswith(".") and len(x) >= 5 and len(x) - 1 < len(y) and y.startswith(x[:-1]) and re.fullmatch(r"[\w-]+", y) is not None


def _same_name_written_differently(q: dict, c: dict) -> bool:
    """The whole name is the same, but written with abbreviated words («Компл. соед. KTY» = «Комплект соединительный KTY») or with two neighbouring
    words swapped («Соединительный зажим ПС-2-1» = «Зажим соединительный ПС-2-1»). Numbers, codes and every other token must stay in place, and the
    swap is allowed only inside a run of plain words: «2x16» never equals «16x2»."""
    a, b = _name_key(q).split(), _name_key(c).split()
    if len(a) != len(b) or a == b:
        return a == b
    plain = lambda t: _ALPHA_TOKEN_RE.fullmatch(t) is not None                          # noqa: E731
    a2, b2 = list(a), list(b)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y and (_abbreviation_of(x, y) or _abbreviation_of(y, x)):
            a2[i] = b2[i] = y if _abbreviation_of(x, y) else x                         # the same word after the expansion
    i = 0
    while i < len(a2):
        if a2[i] == b2[i] and not (plain(a2[i]) and plain(b2[i])):
            i += 1
            continue
        j = i
        while j < len(a2) and plain(a2[j]) and plain(b2[j]):
            j += 1
        if j == i:                                                                      # a token that differs and is not a plain word on both sides
            return False
        if sorted(a2[i:j]) != sorted(b2[i:j]):
            return False
        i = j
    return True


def _match_basis(klass: str, q: dict, c: dict, type_confirmed: bool) -> str | None:
    """The ground of a positive answer, decided from checked facts and never from a score.
    `normalized_name`: the whole main name equals after technical normalisation (not after meaning-synonyms); not for kits;
    `type_assumption`: confirmed by the type alone (policy.evidence.confirm_by_type); otherwise `request_constraints`."""
    if klass not in ("compatible", "compatible_with_substitution"):
        return None
    if type_confirmed:
        return "type_assumption"
    if klass == "compatible" and not c["composite"] and _name_key(q) == _name_key(c):
        return "normalized_name"
    return "request_constraints"


# ---------------------------------------------------------------- B4: whole candidate
def compare(q: dict, c: dict, rules: Rules) -> Compatibility:
    ctx = TypeContext(q["type"], q["family"], c["type"], c["family"])
    cmain, calt = _values(c, "main"), _values(c, "alt")
    _alias_voltages(cmain)
    _alias_voltages(calt)
    cdisp = _displays(c, "main")
    per_attr = rules.defaults.get("per_attribute", {})
    unlisted = rules.defaults["unlisted_numeric_attribute"]

    comps: list[AttrComparison] = []
    for a in q["attributes"]:
        if a["source"] != "main":
            continue
        name, qvalue = _alias(a)
        rule, scope = rules.rule_for(name, ctx)
        mode = rule.mode if rule else per_attr.get(name, unlisted)
        if mode == "ignore":
            continue
        if a.get("extra", {}).get("ambiguous_code") and not cmain.get(name):
            one = AttrComparison(name, "info_diff", qvalue, [], rule.id if rule else None, "«число+м» могло быть частью кода модели")
        else:
            one = _compare_one(name, qvalue, cmain, calt, rule, mode, ctx, rules, c, a.get("extra", {}).get("display"))
        one.rule_scope = scope
        one.query_display = a.get("extra", {}).get("display")
        one.candidate_display = cdisp.get(name, [])
        comps.append(one)

    series = _series_requirement(q, c)
    if series is not None:
        comps.append(series)
    identical = _name_key(q) == _name_key(c) or q["search_text"].strip() == c["search_text"].strip() or _same_name_written_differently(q, c)
    weak_attrs = set(rules.policy["evidence"].get("weak_attributes", []))
    rel_key, rel_status = type_relation(q, c, rules)
    cov, missing, n_words = word_coverage(q, c)

    decisive = [x for x in comps if x.status not in ("info", "info_diff")]
    confirmed = sum(1 for x in comps if x.status in CONFIRMING) + (1 if identical else 0)
    strong = sum(1 for x in comps if x.status in CONFIRMING and x.name not in weak_attrs) + (1 if identical else 0)
    conflicts = [x for x in comps if x.status == "conflict"]
    # допущение: для перечисленных типов (где производитель и модель не важны) совпадения самого типа достаточно
    by_type = set(rules.policy["evidence"].get("confirm_by_type") or [])
    type_confirmed = rel_key == "same" and q["type_specific"] and q["type"] in by_type and not conflicts and strong == 0
    if type_confirmed:
        strong += 1
        confirmed += 1

    reasons: list[str] = []
    for st in sorted({x.status for x in comps if x.status in UNVERIFYING}, key=lambda s: PRIORITY[s]):
        reasons.append(REASON_TEXT[st])
    if rel_status == "check":
        reasons.append("тип изделия иной или не подтверждён")
    # a kit whose list stands IN the name (`inline`) is checked for what the query named: when every component the user asked for is named, the
    # kit is not held back for its composition; a header «в составе:» without a list stays unverified
    kit = next((x for x in comps if x.name == "kit_components"), None)
    composition_checked = c["composite"] == "inline" and kit is not None and kit.status == "match"
    if c["composite"] and rules.policy["composite"].get("max_compatibility") and not composition_checked:          # null in rules.yaml = no cap (ablation)
        reasons.append(f"позиция — {rules.policy['composite'].get('label', 'комплект')}, состав не сверяется")
    if rules.policy["evidence"].get("require_confirmed_attribute") and strong == 0:
        reasons.append("нет подтверждённых характеристик")
    min_cov = rules.policy.get("word_coverage", {}).get("min_for_confirmed")
    if min_cov is not None and cov < min_cov:
        reasons.append("в названии нет слов запроса: " + ", ".join(missing))
    # a requirement the extractor could not turn into an attribute («50 Гц», «мощность 100») must not silently vanish from
    # the decision: the answer cannot be «compatible» while it is unknown. The same number with the same unit on the candidate
    # resolves it (both sides say «50 Гц»); a different value or no value leaves it open.
    unresolved, text_resolved = _split_constraints(q, c, rules)
    if unresolved:
        reasons.append("в запросе есть ограничение, которое не удалось разобрать и сравнить: " + ", ".join(u["raw"] for u in unresolved))
    # the noun the query names its thing with («задвижка») must be in the candidate, whatever the share of other words found:
    # otherwise a two-word query at coverage exactly 0.5 is confirmed by a different thing that shares the number
    type_word_missing = []
    # (not when the candidate is of the same family by the dictionary: a cable named by its mark «ВВГнг 5х6» has no word «кабель»)
    if rules.policy.get("word_coverage", {}).get("type_word_required") and q["family"] and q["family"] != c["family"]:
        fam = {stem(q["family"])}
        type_word_missing = [w for w in missing if _covered(fam, {stem(w)})]
        if type_word_missing and not (min_cov is not None and cov < min_cov):
            reasons.append("в названии нет слова типа из запроса: " + ", ".join(type_word_missing))

    # the subject of the query must be named by the candidate, whatever else matches: other numbers, a material or a code never stand for the thing
    # itself («розетка на 250 В / 16 А» is not a switch, «стальная труба» is not a gauge). Not needed when both have the same known type.
    subject_missing = None
    subj = query_subject(q)
    if (subj and rules.policy.get("word_coverage", {}).get("type_word_required") and rel_key not in ("same", "same_family_q_generic", "same_family_c_generic", "same_family_both_specific") and not type_word_missing
            and not _covered_strict(subj[1], content_words(c))):
        subject_missing = subj[0]
        reasons.append("в названии нет слова предмета запроса: " + subj[0])

    conflict_texts = [x.name for x in conflicts] + (["тип изделия"] if rel_status == "conflict" else [])
    if conflicts or rel_status == "conflict":
        klass = "incompatible"
    elif reasons:
        klass = "unverified"
    elif any(x.status == "substitute" for x in comps):
        klass = "compatible_with_substitution"
    else:
        klass = "compatible"

    # weak evidence decides only whether a non-confirmed candidate may be SHOWN under «не найдено» (never the class)
    type_ok = rel_key in ("same", "same_family_q_generic", "same_family_both_specific", "same_family_c_generic", "unknown_either_side")
    nf = rules.policy["not_found"]
    partial = nf.get("partial_confirmations_count", False) and any(
        x.status in ("match_reordered", "alt_only", "ambiguous") for x in comps)
    te = nf.get("text_evidence") or {}
    text_strong = bool(te) and n_words >= te.get("min_words", 3) and cov >= te.get("min_coverage", 0.9)
    has_evidence = strong > 0 or partial or text_strong or (rel_key == "same" and q["type_specific"])
    min_list = nf.get("min_word_coverage_for_listing", 0.0)
    weak = (type_ok and has_evidence and (cov >= min_list or rel_key == "same") and not (type_word_missing and rel_key != "same")
            and subject_missing is None)
    return Compatibility(klass, rel_key, rel_status, comps, reasons, conflict_texts, confirmed, strong, len(decisive), weak, cov, missing,
                         type_confirmed, [u["raw"] for u in unresolved], _match_basis(klass, q, c, type_confirmed),
                         [u["raw"] for u in text_resolved])
