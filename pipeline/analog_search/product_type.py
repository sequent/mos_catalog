"""Product type (кабель контрольный, видеокамера, ворота…) from the start of the name.

The type is a SIGNAL for the search, not a source of truth: it is coarse (one word often covers many
different items), a query may name the same thing with another noun, and part of the catalog gets no
type at all. Therefore every type carries a `specific` flag, and a type that is missing or not specific
must never remove a candidate on its own (see config/rules.yaml → policy.type).

The dictionary lives in config/dictionaries.yaml and is extended by a specialist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import load_dictionaries

_ADJECTIVE_FORM_RE = re.compile(r"(?:ый|ий|ой|ая|яя|ое|ее|ые|ие)$")      # «соединительный» is not the type «соединитель»

_HEAD_RE = re.compile(r"[а-яa-z][а-яa-z\-]*")


@dataclass(frozen=True)
class ProductType:
    name: str | None
    family: str | None
    head_word: str | None
    specific: bool = False       # True: multiword type or a concrete single-word type; False: broad/unknown


def head_word(text_lower: str) -> str | None:
    m = _HEAD_RE.match(text_lower.strip())
    return m.group(0) if m else None


def detect_type(text_lower: str, shift: bool = False) -> ProductType:
    """`text_lower` — lowercase, ё→е, abbreviations expanded (see normalize.apply_synonyms).

    The catalog names the thing first («Лоток перфорированный»). A USER often puts adjectives first («перфорированный лоток»,
    «крышный вентилятор»): with `shift=True` (queries only) the type is also looked for after up to three leading words."""
    s = text_lower.strip()
    d = load_dictionaries()
    starts = [s]
    if shift:
        words = s.split()
        starts += [" ".join(words[k:]) for k in range(1, min(4, len(words)))]
    for start in starts:
        for t in d.types:
            m = t.regex.match(start)
            if m:
                last = m.group(0).split()[-1] if m.group(0).split() else ""
                if " " not in t.name and len(last) >= 6 and _ADJECTIVE_FORM_RE.search(last):
                    continue                                          # a one-word type that matched only the stem of an adjective
                specific = " " in t.name or t.name in d.specific_single
                return ProductType(t.name, t.family, head_word(s), specific)
    return ProductType(None, None, head_word(s), False)
