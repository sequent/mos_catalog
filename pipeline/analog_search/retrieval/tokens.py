"""Tokens for word search (step B2).

Two token streams exist on purpose:

* `raw_tokens`      — the «simple word search» baseline of the assignment: lowercase, split on non-alphanumerics,
                      nothing else. It is frozen: do not improve it, it is the yardstick for the other steps.
* `lexical_tokens`  — the search of the scheme: normalized text (numbers/units/codes), light stemming,
                      synonyms, plus canonical *attribute tokens* («cs:7x1.5», «dims:2250x6000»), so that
                      BM25 matches «3*1.5» with «3х1,5» and «6000×2250» with «2250х6000» without any model.
"""
from __future__ import annotations

import re

_RAW_RE = re.compile(r"[a-zа-яё0-9]+")
_STOP = frozenset("и в во на с со из для по при до от к а или не под над без у о об за".split())
_CYR_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)*")
_SUFFIXES = sorted(
    "ами ями ого его ому ему ыми ими ать ять ить ают яют ают ует ают ов ев ей ом ем ам ям ах ях ую юю ая яя ое ее ые ие "
    "ый ий ой ым им ых их ет ут ют ит ат ят а я ы и е у ю о ь й".split(),
    key=len, reverse=True,
)


def raw_tokens(text: str) -> list[str]:
    """Baseline tokens: «3х1,5» → «3», «1», «5» (that is exactly why this baseline is weak)."""
    return _RAW_RE.findall(text.lower())


def stem(token: str) -> str:
    """Light suffix stripping for Cyrillic words (no morphology library available)."""
    if len(token) < 5 or not _CYR_RE.fullmatch(token):
        return token
    if token.endswith("ец") and len(token) >= 6:         # «фланец» / «фланцы», «колодец» / «колодцы»: the vowel drops out
        return token[:-2] + "ц"
    for s in _SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            return token[: -len(s)]
    return token


def _fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).lower().replace(" ", "")


def attribute_tokens(record: dict, ordered_dims: bool = True, type_tokens: bool | str = True) -> list[str]:
    """Canonical tokens of the extracted attributes (main text only; «прим.» values are not matched strictly).

    `ordered_dims=False` drops the order-sensitive size token: order is decided later by the type rule (B4).
    `type_tokens`: True = `type:` and `fam:`; "type" = only `type:`; "family" = only `fam:`; False = none.
    Type tokens are a soft signal (never a filter) and slightly favour rows that have a type."""
    out: list[str] = []
    typ, fam = record.get("type"), record.get("family")
    if typ and type_tokens in (True, "type"):
        out.append("type:" + typ.replace(" ", "_"))
    if fam and type_tokens in (True, "family"):
        out.append("fam:" + fam)
    for a in record["attributes"]:
        if a["source"] != "main":
            continue
        n, v = a["name"], a["value"]
        if n == "dims":
            out.append("dims:" + "x".join(_fmt(x) for x in sorted(v)))
            if ordered_dims:
                out.append("dimo:" + "x".join(_fmt(x) for x in v))
            out += ["dim:" + _fmt(x) for x in v]
        elif n == "cores_section":
            out.append("cs:" + "x".join(_fmt(x) for x in v))
            out += ["cores:" + _fmt(v[0]), "sec:" + _fmt(v[-1])]
        elif n == "voltage_v":
            out += ["v:" + _fmt(x) for x in v]
        elif n == "model_codes":
            continue                                    # already in `tokens` as skeletons
        elif n == "cable_screen":
            out.append("scr:" + ("1" if v else "0"))
        elif n == "cable_mark":
            out.append("mk:" + str(v))
        elif n == "fire_class":
            out.append("fire:" + _fmt(v))
        elif n == "ip":
            out.append("ip:" + _fmt(v))
        elif n == "placement":
            out.append("pl:" + str(v))
        elif n == "standard":
            out.append("std:" + _fmt(v))
        elif isinstance(v, list):
            out += [f"{n}:{_fmt(x)}" for x in v]
        else:
            out.append(f"{n}:{_fmt(v)}")
    return out


def lexical_tokens(record: dict, *, use_stem: bool = True, use_attrs: bool = True, ordered_dims: bool = True,
                   type_tokens: bool | str = True) -> list[str]:
    words = [t for t in record["tokens"] if t not in _STOP]
    if use_stem:
        words = [stem(t) for t in words]
    return words + (attribute_tokens(record, ordered_dims, type_tokens) if use_attrs else [])
