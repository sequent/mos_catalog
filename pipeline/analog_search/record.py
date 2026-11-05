"""One entry point for both sides of the search: a catalog row and a user query.

`process(text)` runs the same chain A1→A5 (= B1): clean → split «прим.» → normalize →
type → attributes → flags → search text. The catalog is processed offline, the query online.
"""
from __future__ import annotations

from typing import Any

from . import flags
from .attributes import extract_attributes, find_unparsed_constraints, uncovered_numbers
from .config import load_dictionaries
from .normalize import apply_synonyms, clean, normalize_text, search_text, split_alt
from .product_type import detect_type


RECORD_SCHEMA_VERSION = 8          # 8: class with a Roman numeral; 7: label abbreviations, Latin t and km, adjective forms are not types, service abbreviations; 6: counts, storage time, categorical features, kit components (stage 3); 5: units of length, number of keys, purpose clause (fixes); 4: alt_index of «прим.» attributes and alt_norm_texts (T06); 3: unparsed_constraints, assumptions (T02); raise when the fields of a record change


def process(raw: str, query: bool = False) -> dict[str, Any]:
    cleaned, numbering = clean(raw)
    main, alts = split_alt(cleaned)
    assumptions: list[dict] = []
    if query:                                   # «любого производителя», «любой цвет»: характеристика не важна, фраза вырезается
        for attr, label, rx in load_dictionaries().any_phrases:
            m = rx.search(main)
            if m:
                assumptions.append({"attribute": attr, "label": label, "phrase": m.group(0)})
                main = " ".join((main[:m.start()] + " " + main[m.end():]).split())
    norm_main = normalize_text(main)
    norm_alts = [normalize_text(a) for a in alts]

    expanded, _ = apply_synonyms(norm_main.replace("ё", "е").replace("Ё", "Е").lower())
    ptype = detect_type(expanded, shift=query)           # only a user's query may put adjectives before the noun

    attrs = extract_attributes(norm_main, ptype.family, "main", query)
    main_attrs = list(attrs)
    unparsed = []
    for i, a in enumerate(norm_alts):
        alt_attrs = extract_attributes(a, ptype.family, "alt", query)
        for x in alt_attrs:
            x.extra["alt_index"] = i                      # the spans of an alt attribute are counted in THIS alt text (evidence, T06)
        attrs += alt_attrs
        unparsed += find_unparsed_constraints(a, alt_attrs, "alt")
    seen: set[tuple] = set()                         # the same value written twice in one name counts once
    unique = []
    for a in attrs:
        key = (a.name, repr(a.value), a.source)
        if key not in seen:
            seen.add(key)
            unique.append(a)
    attrs = unique
    meaning = load_dictionaries().bare_mm_means.get(ptype.family or "")
    if meaning:                                     # «труба … 25 мм» → диаметр 25 мм
        for a in attrs:
            if a.name == "bare_mm":
                a.name, a.kind = meaning, "bare"
    not_explained = uncovered_numbers(norm_main, [a for a in attrs if a.source == "main"])
    # positions BEFORE the removal of repeated values: «220 кВ … 220 кВ» is explained twice, not once
    unparsed = find_unparsed_constraints(norm_main, main_attrs, "main") + unparsed

    lowered, tokens = search_text(norm_main)
    return {
        "clean_name": cleaned,
        "numbering_prefix": numbering,
        "main_text": norm_main,
        "alt_texts": alts,
        "alt_norm_texts": norm_alts,
        "search_text": lowered,
        "tokens": tokens,
        "type": ptype.name,
        "family": ptype.family,
        "type_specific": ptype.specific,
        "head_word": ptype.head_word,
        "attributes": [a.to_dict() for a in attrs],
        "uncovered_numbers": not_explained,
        "unparsed_constraints": unparsed,
        "is_service": flags.is_service(cleaned),
        "is_adjustment": flags.is_adjustment(cleaned),
        "composite": flags.composite_kind(cleaned),
        "assumptions": assumptions,
    }
