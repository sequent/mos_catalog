"""Where a value comes from, and what else the candidate says.

For every compared characteristic the answer carries `evidence`: the fragment of the name that the value was read from, on the side of the
query and on the side of the candidate, with the SOURCE (main name or the «(прим. …)» note). The candidate's characteristics that the
query did not name are listed as `candidate_extra_attributes`; numbers of the candidate that nothing explains as
`candidate_unparsed_constraints`.

Spans are counted in the NORMALISED text (`text`), not in the original name: the normaliser changes characters, so a span of the
normalised text must never be used to highlight `raw_name`. That is why every entry carries the text it refers to.
"""
from __future__ import annotations

from .compat import _alias
from .explain import LABELS, fmt

SOURCE_TEXT = {"main": "основное название", "alt": "примечание «прим.»"}
ALT_NOTE = "альтернатива, не подтверждает основное название"
UNPARSED_REASON_TEXT = {"number_with_unit": "число с единицей, не стало характеристикой", "measurement_label": "число после названия величины без единицы"}


def _text_of(rec: dict, attr: dict) -> tuple[str, str]:
    """(the text the span of this attribute refers to, its version name)."""
    if attr["source"] == "alt":
        alts = rec.get("alt_norm_texts") or []
        i = attr.get("extra", {}).get("alt_index", 0)
        return (alts[i] if i < len(alts) else ""), "normalized_alt"
    return rec["main_text"], "normalized_main"


def _entry(side: str, rec: dict, a: dict) -> dict:
    text, version = _text_of(rec, a)
    s, e = a.get("span", [0, 0])
    fragment = a.get("raw", text[s:e])
    e_ = {"side": side, "source": a["source"], "source_text": SOURCE_TEXT[a["source"]], "fragment": fragment,
          "text": text, "text_version": version, "span": [s, e], "exact": text[s:e] == fragment}
    if a["source"] == "alt":
        e_["note"] = ALT_NOTE
    return e_


def evidence_by_attribute(comp, q: dict, c: dict) -> dict[str, list[dict]]:
    """name of a compared characteristic -> where each side read its value from."""
    names = {x.name for x in comp.comparisons}
    out: dict[str, list[dict]] = {n: [] for n in names}
    for side, rec in (("query", q), ("candidate", c)):
        seen = set()
        for a in rec["attributes"]:
            name, _ = _alias(a)
            if name not in names or (side == "query" and a["source"] != "main"):
                continue
            key = (side, name, a["source"], tuple(a.get("span", ())), a.get("raw"))
            if key in seen:
                continue
            seen.add(key)
            out[name].append(_entry(side, rec, a))
    return out


def extra_attributes(comp, c: dict) -> list[dict]:
    """The candidate's characteristics that the query did not ask for (so they are not in the comparison): the material of «труба 25 мм»,
    the colour, the model. A value from the «(прим. …)» note is marked: it does not confirm the main name."""
    compared = {x.name for x in comp.comparisons}
    out, seen = [], set()
    for a in c["attributes"]:
        name, value = _alias(a)
        if name in compared:
            continue
        shown = a.get("extra", {}).get("display") if name == "model_codes" else None
        value_text = ", ".join(shown) if shown else fmt(name, value)
        key = (name, value_text, a["source"])
        if key in seen:
            continue
        seen.add(key)
        label = "обозначение (код)" if name == "model_codes" else LABELS.get(name, name)        # «ПЭ 100» is a grade, not always a model
        out.append({"attribute": name, "label": label, "value": value_text, **_entry("candidate", c, a)})
    out.sort(key=lambda x: (x["source"] == "alt", x["span"][0]))                      # main name first, then notes; by position
    return out


def candidate_unparsed(c: dict) -> list[dict]:
    """Numbers of the candidate's name that nothing explains and that look like requirements (a number with a unit, a number after a quantity)."""
    return [{"raw": u["raw"], "reason": u["reason"], "reason_text": UNPARSED_REASON_TEXT.get(u["reason"], u["reason"]),
             "text": c["main_text"], "text_version": "normalized_main", "span": u["span"]}
            for u in c.get("unparsed_constraints", []) if u["source"] == "main" and u["reason"] in UNPARSED_REASON_TEXT]
