"""Loading and validation of config/rules.yaml (rules of substitution and the policy of the answer).

The file is edited by a specialist, so every mistake must be loud: an unknown mode, a missing ladder or a
duplicated id raises ValueError with a readable message instead of silently changing the behaviour of the search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
RULES_PATH = ROOT / "config" / "rules.yaml"
ASSUMPTIONS_PATH = ROOT / "config" / "assumptions.yaml"
ASSUMPTION_STATUSES = {"author_assumption", "specialist_approved", "rejected"}

MODES = {"exact", "ladder", "min_ge", "flag", "ignore"}                     # modes of a rule in `rules:`
DEFAULT_MODES = {"exact", "info", "flag", "ambiguous", "ignore"}             # modes of `defaults.per_attribute`
TYPE_RELATIONS = {"same", "same_family_q_generic", "same_family_both_specific", "same_family_c_generic",
                  "unknown_either_side", "different_both_specific", "different_one_generic"}
RELATION_RESULTS = {"match", "check", "unknown", "conflict"}
DIMS_MODES = {"sorted", "ordered"}
CLASSES = ("compatible", "compatible_with_substitution", "unverified", "incompatible")
VERDICTS = {"same", "substitute", "check", "reject"}
UNPARSED_REASONS = {"number_with_unit", "measurement_label", "code_or_reference", "bare_number"}

# The registry of what the comparison code really supports. A mode that the code of an attribute does not honour
# would be silently ignored, so it is rejected at load time with the list of the modes that work.
_SCALAR = ("exact", "flag", "ignore", "min_ge", "ladder", "info")
_PLAIN = ("exact", "flag", "ignore", "info")
ATTRIBUTE_MODES: dict[str, tuple[str, ...]] = {
    **{a: _SCALAR for a in ("current_a", "power_kw", "apparent_power_kva", "resolution_mp", "pressure_mpa", "pn", "diameter_mm",
                            "nominal_bore_mm", "thickness_mm", "length_mm", "perimeter_mm", "height_mm", "side_mm", "mass_kg",
                            "area_m2", "volume_m3", "volume_l", "fire_resistance_ei", "rated_voltage_v", "distance_km")},
    **{a: _PLAIN for a in ("voltage_v", "temperature_c", "range_mm", "ip", "climate", "standard", "poles", "cable_mark",
                           "cable_screen", "fire_class", "placement", "conductor_form", "switch_gang_count", "port_count", "quantity_pcs",
                           "storage_duration_days", "print_technology", "print_color_mode", "paper_format", "connector_type", "mounting",
                           "mount_surface", "class_roman")},
    "kit_components": ("exact", "ignore"),                 # a set: the query's components must all be named by the candidate
    "cores_section": ("exact", "flag", "ignore"),          # a list [cores, section]: no scale and no «not less than»
    "dims": ("exact", "ignore"),                           # the order of the sides is decided by `dims`/`dims_order`
    "material": ("exact", "ignore"),                       # disjoint materials always conflict
    "color": ("exact", "flag", "info", "ignore"),
    "model_codes": ("info", "ignore"),
    "numeric_code": ("exact", "ignore"),
    "bare_mm": ("ambiguous", "ignore"),
}


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    applies_to: frozenset[str]
    attribute: str
    mode: str
    ladder: tuple = ()
    max_steps: int = 0
    dims: str | None = None
    examples: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Rules:
    version: str
    status: str
    rules: tuple[Rule, ...]
    by_attribute: dict[str, tuple[Rule, ...]]
    dims_sorted: frozenset[str]
    dims_ordered: frozenset[str]
    defaults: dict[str, Any]
    policy: dict[str, Any]
    raw: dict = field(repr=False, default_factory=dict)

    def rule_for(self, attribute: str, ctx: "TypeContext") -> tuple[Rule | None, str | None]:
        """The rule of the attribute for this pair and the scope it was chosen by.

        The steps are ordered from the most to the least specific (see TypeContext.steps); the first step that has a rule wins.
        A candidate of ANOTHER known family never softens the rule of the query. At most one rule exists per attribute and name
        (checked at load time), so the result does not depend on the order of the file."""
        for scope, name in ctx.steps():
            hit = [r for r in self.by_attribute.get(attribute, ()) if name in r.applies_to]
            if len(hit) > 1:                                  # impossible after load_rules; kept as a loud guard
                raise RuntimeError(f"два правила для «{attribute}» и «{name}»: {', '.join(r.id for r in hit)}")
            if hit:
                return hit[0], scope
        return None, None

    def dims_order_for(self, ctx: "TypeContext") -> tuple[str | None, str | None]:
        """`sorted` / `ordered` / None for the pair, with the scope, by the same steps as the rules."""
        for scope, name in ctx.steps():
            if name in self.dims_sorted:
                return "sorted", scope
            if name in self.dims_ordered:
                return "ordered", scope
        return None, None


@dataclass(frozen=True)
class TypeContext:
    """Types of the two sides of a comparison. Selection order: type of the query → family of the query → (only when the query
    has NO type at all) type of the candidate → family of the candidate."""
    query_type: str | None
    query_family: str | None
    candidate_type: str | None
    candidate_family: str | None

    def steps(self):
        if self.query_type:
            yield "тип запроса", self.query_type
        if self.query_family:
            yield "семейство запроса", self.query_family
        if not self.query_type and not self.query_family:
            if self.candidate_type:
                yield "тип кандидата (тип запроса неизвестен)", self.candidate_type
            if self.candidate_family:
                yield "семейство кандидата (тип запроса неизвестен)", self.candidate_family


def load_assumptions(path: Path | str = ASSUMPTIONS_PATH) -> dict[str, dict]:
    """The register of assumptions. «specialist_approved» requires the name and the date of the approval: nobody is credited without a record."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for a in raw["assumptions"]:
        if a["id"] in out:
            raise ValueError(f"config/assumptions.yaml: id «{a['id']}» повторяется")
        if a["status"] not in ASSUMPTION_STATUSES:
            raise ValueError(f"config/assumptions.yaml: {a['id']}: status должен быть один из {sorted(ASSUMPTION_STATUSES)}")
        if a["status"] == "specialist_approved" and not (a.get("approved_by") and a.get("approved_on")):
            raise ValueError(f"config/assumptions.yaml: {a['id']}: для specialist_approved нужны approved_by и approved_on")
        out[a["id"]] = a
    return out


def _fail(msg: str) -> None:
    raise ValueError(f"config/rules.yaml: {msg}")


# ---------------------------------------------------------------- small checkers that always name the path of the field
def _dict(node, path: str, allowed: set[str], required: tuple[str, ...] = ()) -> dict:
    if not isinstance(node, dict):
        _fail(f"{path}: нужен словарь")
    extra = sorted(set(node) - allowed)
    if extra:
        _fail(f"{path}: неизвестные поля {extra}; допустимо: {sorted(allowed)}")
    for k in required:
        if k not in node:
            _fail(f"{path}: нет обязательного поля «{k}»")
    return node


def _bool(node: dict, key: str, path: str) -> None:
    if key in node and not isinstance(node[key], bool):
        _fail(f"{path}.{key}: нужно true или false, получено {node[key]!r}")


def _number(v, path: str, lo: float = 0.0, hi: float | None = None) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")) or v < lo or (hi is not None and v > hi):
        _fail(f"{path}: нужно конечное число" + (f" от {lo:g}" if lo != float("-inf") else "") + (f" до {hi:g}" if hi is not None else "")
              + f", получено {v!r}")


def _count(v, path: str) -> None:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        _fail(f"{path}: нужно целое число ≥ 0, получено {v!r}")


def _names(node, path: str, allowed: set[str], what: str) -> list:
    if not isinstance(node, list) or not all(isinstance(x, str) for x in node):
        _fail(f"{path}: нужен список названий")
    bad = [x for x in node if x not in allowed]
    if bad:
        _fail(f"{path}: неизвестные {what}: {', '.join(bad)}")
    return node


def _type_names() -> set[str]:
    from ..config import load_dictionaries
    return {t.name for t in load_dictionaries().types}


def _known_types() -> set[str]:
    names = _type_names()
    return names | {n.split()[0] for n in names}                # a type name or its family (the first word)


def _validate_policy(policy: dict, types: set[str]) -> None:
    _dict(policy, "policy", {"type", "composite", "evidence", "unparsed_constraints", "word_coverage", "ranking", "verdict",
                             "not_found", "ui"}, ("type", "composite", "evidence", "ranking", "verdict", "not_found"))
    # --- type
    t = _dict(policy["type"], "policy.type", {"relation"}, ("relation",))
    rel = _dict(t["relation"], "policy.type.relation", TYPE_RELATIONS)
    if TYPE_RELATIONS - set(rel):
        _fail(f"policy.type.relation: нет записей для {', '.join(sorted(TYPE_RELATIONS - set(rel)))}")
    for k, v in rel.items():
        if v not in RELATION_RESULTS:
            _fail(f"policy.type.relation.{k}: недопустимый результат «{v}»; допустимо: {', '.join(sorted(RELATION_RESULTS))}")
    # --- composite: only `unverified` or null (null = no cap, the experimental ablation)
    if "composite" in policy:
        c = _dict(policy["composite"], "policy.composite", {"max_compatibility", "label"})
        if c.get("max_compatibility") not in (None, "unverified"):
            _fail(f"policy.composite.max_compatibility: допустимо unverified или null, получено {c['max_compatibility']!r}")
        if "label" in c and not isinstance(c["label"], str):
            _fail("policy.composite.label: нужна строка")
    # --- evidence
    if "evidence" in policy:
        e = _dict(policy["evidence"], "policy.evidence", {"require_confirmed_attribute", "weak_attributes", "confirm_by_type"})
        _bool(e, "require_confirmed_attribute", "policy.evidence")
        if "weak_attributes" in e:
            _names(e["weak_attributes"], "policy.evidence.weak_attributes", set(ATTRIBUTE_MODES), "характеристики")
        if "confirm_by_type" in e:
            _names(e["confirm_by_type"] or [], "policy.evidence.confirm_by_type", _type_names(), "типы (их нет в config/dictionaries.yaml)")
    # --- unparsed constraints
    if "unparsed_constraints" in policy:
        u = _dict(policy["unparsed_constraints"], "policy.unparsed_constraints", {"blocking_reasons"})
        _names(u.get("blocking_reasons") or [], "policy.unparsed_constraints.blocking_reasons", UNPARSED_REASONS, "причины")
    # --- word coverage
    if "word_coverage" in policy:
        w = _dict(policy["word_coverage"], "policy.word_coverage", {"type_word_required", "min_for_confirmed"})
        _bool(w, "type_word_required", "policy.word_coverage")
        if w.get("min_for_confirmed") is not None:
            _number(w["min_for_confirmed"], "policy.word_coverage.min_for_confirmed", 0.0, 1.0)
    # --- ranking
    r = _dict(policy["ranking"], "policy.ranking", {"order", "score", "class_first"}, ("order", "score"))
    _names(r["order"], "policy.ranking.order", set(CLASSES), "классы совместимости")
    _bool(r, "class_first", "policy.ranking")
    sc = _dict(r["score"], "policy.ranking.score", {"weights", "text_mix", "pair_boost"}, ("weights", "text_mix"))
    for key, names in (("weights", {"attribute_coverage", "text_relevance"}), ("text_mix", {"bm25", "dense"})):
        block = _dict(sc[key], f"policy.ranking.score.{key}", names, tuple(sorted(names)))
        for k, v in block.items():
            _number(v, f"policy.ranking.score.{key}.{k}")
    if "pair_boost" in sc:
        _number(sc["pair_boost"], "policy.ranking.score.pair_boost")
    # --- verdict
    vm = _dict(_dict(policy["verdict"], "policy.verdict", {"map"}, ("map",))["map"], "policy.verdict.map", set(CLASSES), CLASSES)
    for k, v in vm.items():
        if v not in VERDICTS:
            _fail(f"policy.verdict.map.{k}: недопустимый вердикт «{v}»; допустимо: {', '.join(sorted(VERDICTS))}")
    # --- not_found
    nf = _dict(policy["not_found"], "policy.not_found",
               {"confirmed_verdicts", "show_unconfirmed", "unconfirmed_limit", "show_nearest_rejected", "require_weak_evidence",
                "min_word_coverage_for_listing", "partial_confirmations_count", "text_evidence"}, ("confirmed_verdicts", "unconfirmed_limit"))
    _names(nf["confirmed_verdicts"], "policy.not_found.confirmed_verdicts", VERDICTS, "вердикты")
    _count(nf["unconfirmed_limit"], "policy.not_found.unconfirmed_limit")
    for key in ("show_unconfirmed", "show_nearest_rejected", "require_weak_evidence", "partial_confirmations_count"):
        _bool(nf, key, "policy.not_found")
    if "min_word_coverage_for_listing" in nf:
        _number(nf["min_word_coverage_for_listing"], "policy.not_found.min_word_coverage_for_listing", 0.0, 1.0)
    if "text_evidence" in nf:
        te = _dict(nf["text_evidence"], "policy.not_found.text_evidence", {"min_coverage", "min_words"})
        if "min_coverage" in te:
            _number(te["min_coverage"], "policy.not_found.text_evidence.min_coverage", 0.0, 1.0)
        if "min_words" in te:
            _count(te["min_words"], "policy.not_found.text_evidence.min_words")
    # --- ui
    if "ui" in policy:
        ui = _dict(policy["ui"], "policy.ui", {"review_in_main_table", "review_rows_when_found", "review_table_limit", "status_lines"})
        _bool(ui, "review_in_main_table", "policy.ui")
        for key in ("review_rows_when_found", "review_table_limit"):
            if key in ui:
                _count(ui[key], f"policy.ui.{key}")
        if "status_lines" in ui:
            sl = _dict(ui["status_lines"], "policy.ui.status_lines", {"found", "found_with_review", "review", "not_found"})
            for k, v in sl.items():
                if not isinstance(v, str):
                    _fail(f"policy.ui.status_lines.{k}: нужна строка")
            for k in ("found", "found_with_review"):
                if k in sl and "{n}" not in sl[k]:
                    _fail(f"policy.ui.status_lines.{k}: в тексте должно быть {{n}} (число подтверждённых позиций)")


def load_rules(path: Path | str | None = None) -> Rules:
    p = Path(path) if path else RULES_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    _dict(raw, "файл", {"version", "status", "rules", "dims_order", "defaults", "policy"}, ("version", "rules", "defaults", "policy"))
    types = _known_types()
    rules, seen = [], set()
    if not isinstance(raw["rules"], list):
        _fail("rules: нужен список правил")
    for i, r in enumerate(raw["rules"], 1):
        rid = (r or {}).get("id") or _fail(f"правило №{i} без id")
        if rid in seen:
            _fail(f"id «{rid}» повторяется")
        seen.add(rid)
        _dict(r, f"правило «{rid}»", {"id", "title", "applies_to", "attribute", "mode", "ladder", "max_steps", "dims", "examples", "assumption"},
              ("applies_to", "attribute", "mode"))
        if "assumption" in r and r["assumption"] not in load_assumptions():
            _fail(f"правило «{rid}»: допущения «{r['assumption']}» нет в config/assumptions.yaml")
        attr = r["attribute"]
        if attr not in ATTRIBUTE_MODES:
            _fail(f"правило «{rid}»: неизвестная характеристика «{attr}»; допустимо: {', '.join(sorted(ATTRIBUTE_MODES))}")
        if r["mode"] not in MODES:
            _fail(f"правило «{rid}»: неизвестный режим «{r['mode']}» (допустимо: {', '.join(sorted(MODES))})")
        works = [m for m in ATTRIBUTE_MODES[attr] if m in MODES]
        if r["mode"] not in works:
            _fail(f"правило «{rid}»: для «{attr}» режим «{r['mode']}» не поддержан; работают: {', '.join(works)}")
        _names(r["applies_to"], f"правило «{rid}».applies_to", types, "типы (их нет в config/dictionaries.yaml)")
        if r["mode"] == "ladder":
            lad = r.get("ladder")
            if not isinstance(lad, list) or not lad or len(set(lad)) != len(lad):
                _fail(f"правило «{rid}».ladder: нужен непустой список ступеней без повторов")
            for j, step in enumerate(lad):
                _number(step, f"правило «{rid}».ladder[{j}]", lo=float("-inf"))      # a finite number, not a flag or a text
            if "max_steps" not in r:
                _fail(f"правило «{rid}»: для режима ladder нужен max_steps")
        if "max_steps" in r:
            _count(r["max_steps"], f"правило «{rid}».max_steps")                       # 1.9 and true are not silently turned into 1
        if r.get("dims") is not None:
            if r["dims"] not in DIMS_MODES:
                _fail(f"правило «{rid}»: dims должен быть sorted или ordered")
            if attr != "dims":
                _fail(f"правило «{rid}»: поле dims относится только к характеристике dims")
        rules.append(Rule(rid, r.get("title", rid), frozenset(r["applies_to"]), attr, r["mode"],
                          tuple(r.get("ladder", ())), int(r.get("max_steps", 0)), r.get("dims"),
                          tuple(r.get("examples", ()))))
    by_attr: dict[str, list[Rule]] = {}
    for r in rules:
        by_attr.setdefault(r.attribute, []).append(r)
    for attr, group in by_attr.items():                       # one rule per attribute and type name: the order of the file must not decide
        owner: dict[str, str] = {}
        for r in group:
            for name in sorted(r.applies_to):
                if name in owner:
                    _fail(f"правила «{owner[name]}» и «{r.id}» оба относятся к «{name}» для характеристики «{attr}»: выбор был бы "
                          "случайным; оставьте тип только в одном правиле")
                owner[name] = r.id
    order = _dict(raw.get("dims_order", {}), "dims_order", {"sorted", "ordered"})
    ds = frozenset(_names(order.get("sorted", []), "dims_order.sorted", types, "типы (их нет в config/dictionaries.yaml)"))
    do = frozenset(_names(order.get("ordered", []), "dims_order.ordered", types, "типы (их нет в config/dictionaries.yaml)"))
    if ds & do:
        _fail(f"dims_order: типы в обоих списках: {', '.join(sorted(ds & do))}")
    defaults = _dict(raw["defaults"], "defaults", {"dims_order_unlisted", "unlisted_numeric_attribute", "per_attribute"},
                     ("unlisted_numeric_attribute",))
    if defaults.get("dims_order_unlisted", "check") not in ("check", "sorted", "ordered"):
        _fail("defaults.dims_order_unlisted: допустимо check, sorted или ordered")
    if defaults["unlisted_numeric_attribute"] not in DEFAULT_MODES:
        _fail(f"defaults.unlisted_numeric_attribute: недопустимый режим; допустимо: {', '.join(sorted(DEFAULT_MODES))}")
    for attr, mode in (defaults.get("per_attribute") or {}).items():
        if attr not in ATTRIBUTE_MODES:
            _fail(f"defaults.per_attribute: неизвестная характеристика «{attr}»; допустимо: {', '.join(sorted(ATTRIBUTE_MODES))}")
        works = [m for m in ATTRIBUTE_MODES[attr] if m in DEFAULT_MODES]
        if mode not in works:
            _fail(f"defaults.per_attribute.{attr}: режим «{mode}» не поддержан; работают: {', '.join(works)}")
    _validate_policy(raw["policy"], types)
    return Rules(str(raw["version"]), str(raw.get("status", "")), tuple(rules),
                 {k: tuple(v) for k, v in by_attr.items()}, ds, do, defaults, raw["policy"], raw)
