"""Loading of the specialist-maintained dictionaries (config/dictionaries.yaml)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DICTIONARIES_PATH = ROOT / "config" / "dictionaries.yaml"


@dataclass(frozen=True)
class TypeEntry:
    name: str
    family: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class Dictionaries:
    replace: list[tuple[re.Pattern[str], str]]
    expand: list[tuple[re.Pattern[str], str]]
    types: list[TypeEntry]
    bare_mm_means: dict[str, str]
    specific_single: frozenset[str]
    materials: list[tuple[str, re.Pattern[str]]]
    generic_materials: dict[str, dict[str, frozenset[str]]]
    colors: list[tuple[str, re.Pattern[str]]]
    any_phrases: list[tuple[str, str, re.Pattern[str]]]     # (attribute, label, pattern)
    version: str
    features: list = field(default_factory=list)            # categorical characteristics: (attribute, label, contexts, [(value, pattern)])
    kit_components: list = field(default_factory=list)      # what a kit may contain: (component, pattern)


@lru_cache(maxsize=4)
def load_dictionaries(path: str | None = None) -> Dictionaries:
    p = Path(path) if path else DICTIONARIES_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    syn = raw.get("synonyms", {})
    replace = [(re.compile(x["pattern"]), x["to"]) for x in syn.get("replace", [])]
    expand = [(re.compile(x["pattern"]), x["add"]) for x in syn.get("expand", [])]
    adj = "|".join(raw.get("leading_adjectives", []))
    prefix = rf"(?:(?:{adj})\w*\s+)*" if adj else ""
    types = []
    for t in raw.get("types", []):
        name = t["name"]
        types.append(TypeEntry(name=name, family=name.split()[0], regex=re.compile(prefix + "(?:" + t["pattern"] + ")")))
    return Dictionaries(replace=replace, expand=expand, types=types,
                        bare_mm_means=dict(raw.get("bare_mm_means", {})),
                        specific_single=frozenset(raw.get("specific_single_word_types", [])),
                        materials=[(m["name"], re.compile(r"(?<![а-яёa-z0-9])(?:" + "|".join(m["patterns"]) + r")(?![а-яёa-z0-9])", re.I))
                                   for m in raw.get("materials", [])],
                        generic_materials={k: {"same_as": frozenset(v.get("same_as", [])), "any_of": frozenset(v.get("any_of", []))}
                                           for k, v in (raw.get("generic_materials") or {}).items()},
                        colors=[(m["name"], re.compile(r"(?<![а-яёa-z0-9])(?:" + "|".join(m["patterns"]) + r")(?![а-яёa-z0-9])", re.I))
                                for m in raw.get("colors", [])],
                        any_phrases=[(m["attribute"], m["label"], re.compile(r"(?<![а-яёa-z0-9])(?:" + "|".join(m["patterns"]) + r")(?![а-яёa-z0-9])", re.I))
                                     for m in raw.get("any_phrases", [])],
                        version=str(p.stat().st_mtime_ns),
                        features=[(f["attribute"], f.get("label", f["attribute"]), frozenset(f.get("contexts") or []),
                                   [(val, re.compile("|".join(pats), re.I)) for val, pats in f["values"].items()]) for f in raw.get("features", [])],
                        kit_components=[(name, re.compile("|".join(pats), re.I)) for name, pats in (raw.get("kit_components") or {}).items()])
