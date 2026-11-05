"""the answer says where every value comes from and what else the position states.

`evidence`: fragment + source + span, counted in the NORMALISED text that the entry carries (never in the original name);
`candidate_extra_attributes`: what the position states that the query did not ask for; `candidate_unparsed_constraints`: what the
position states that nothing explains.
"""
import re
from pathlib import Path

import pytest

from analog_search import process
from analog_search.matching import compare, load_rules
from analog_search.matching.evidence import ALT_NOTE, candidate_unparsed, evidence_by_attribute, extra_attributes
from analog_search.matching.explain import explain

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules()
BUILT = (ROOT / "data" / "processed" / "catalog_v2.jsonl").exists() and (ROOT / "data" / "index" / "dense.npy").exists() \
    and (ROOT / "models" / "multilingual-e5-small" / "onnx" / "model.onnx").exists()
real = pytest.mark.skipif(not BUILT, reason="catalog, index or model are not built")


def parts(q, c):
    qr, cr = process(q, query=True), process(c)
    return qr, cr, compare(qr, cr, RULES)


# ---------------------------------------------------------------- the extra attributes: what the query did not ask for
def test_the_material_of_a_pipe_is_shown_when_the_query_did_not_name_it():
    q, c, comp = parts("труба 25 мм", "Гофрированная труба из ПВХ с протяжкой d=25 мм")
    extra = extra_attributes(comp, c)
    assert [(x["label"], x["value"], x["source"]) for x in extra] == [("материал", "ПВХ", "main")]
    assert "diameter_mm" not in {x["attribute"] for x in extra}                    # the compared one is not repeated


def test_a_code_is_called_a_designation_not_a_model():
    """«ПЭ 100» is a grade of polyethylene: calling every code «модель» would mislead."""
    q, c, comp = parts("труба", "Труба ПНД ПЭ 100 SDR 11 d=110 мм")
    codes = [x for x in extra_attributes(comp, c) if x["attribute"] == "model_codes"]
    assert codes and all(x["label"] == "обозначение (код)" for x in codes)


def test_a_characteristic_the_query_named_is_not_an_extra_one():
    q, c, comp = parts("труба ПВХ 25 мм", "Труба ПВХ 25 мм")
    assert extra_attributes(comp, c) == []


def test_a_value_from_the_note_is_marked_and_does_not_count_as_the_main_one():
    q, c, comp = parts("сталь полосовая", "Сталь полосовая: 100х10 мм, марка Ст3сп (прим. 60х10)")
    extra = extra_attributes(comp, c)
    main = [x for x in extra if x["source"] == "main"]
    alt = [x for x in extra if x["source"] == "alt"]
    assert any(x["attribute"] == "dims" and x["value"].startswith("100×10") for x in main)
    assert alt and all(x["note"] == ALT_NOTE and x["source_text"] == "примечание «прим.»" for x in alt)
    assert extra.index(main[0]) < extra.index(alt[0])                              # the main name first, then the notes


def test_an_alt_value_is_read_from_its_own_text_not_from_the_main_name():
    q, c, comp = parts("сталь полосовая", "Сталь полосовая: 100х10 мм (прим. 60х10)")
    alt = next(x for x in extra_attributes(comp, c) if x["source"] == "alt")
    assert alt["text_version"] == "normalized_alt" and alt["text"] == "прим. 60x10"                       # the whole note, as normalised
    assert alt["text"][slice(*alt["span"])] == alt["fragment"] == "60x10" and alt["exact"]


def test_two_notes_keep_their_own_texts():
    c = process("Лента (прим. 20х0,5) или (прим. 30х0,5)")
    q, _, comp = process("лента", query=True), c, None
    comp = compare(q, c, RULES)
    alts = [x for x in extra_attributes(comp, c) if x["source"] == "alt"]
    assert {x["text"] for x in alts} == set(c["alt_norm_texts"]) and len(c["alt_norm_texts"]) == 2


# ---------------------------------------------------------------- the evidence of a compared value
def test_the_evidence_names_both_sides_and_the_fragment():
    q, c, comp = parts("труба 25 мм", "Гофрированная труба из ПВХ (серия 9) с протяжкой d=25 мм")
    ev = evidence_by_attribute(comp, q, c)["diameter_mm"]
    assert [(e["side"], e["source"], e["fragment"]) for e in ev] == [("query", "main", "25 мм"), ("candidate", "main", "d=25 мм")]


def test_the_evidence_of_a_note_is_not_mixed_with_the_main_one():
    q, c, comp = parts("сталь полосовая 100х10 мм", "Сталь полосовая: 100х10 мм (прим. 60х10)")
    ev = evidence_by_attribute(comp, q, c)["dims"]
    assert [(e["side"], e["source"]) for e in ev] == [("query", "main"), ("candidate", "main"), ("candidate", "alt")]
    assert [e for e in ev if e["source"] == "alt"][0]["note"] == ALT_NOTE


def test_a_note_of_the_query_is_never_evidence():
    q = process("сталь 100х10 (прим. 60х10)", query=True)
    c = process("Сталь 100х10 мм")
    ev = evidence_by_attribute(compare(q, c, RULES), q, c)["dims"]
    assert all(e["side"] != "query" or e["source"] == "main" for e in ev)


def test_two_names_of_one_quantity_are_one_evidence_row():
    """Ду80 and «номинальный диаметр 80 мм» are compared as one quantity (alias): the evidence shows both fragments."""
    q, c, comp = parts("фланец Ду80", "Фланец, номинальный диаметр 80 мм")
    ev = evidence_by_attribute(comp, q, c)["nominal_bore_mm"]
    assert {e["side"] for e in ev} == {"query", "candidate"}


def test_the_rows_of_the_answer_carry_the_evidence_and_the_type_row_an_empty_one():
    q, c, comp = parts("труба 25 мм", "Труба ПВХ d=25 мм")
    rows = explain(comp, q, c)["rows"]
    assert rows[0]["attribute"] == "type" and rows[0]["evidence"] == []
    assert any(r["evidence"] for r in rows[1:])


# ---------------------------------------------------------------- the span is counted in the normalised text
def test_the_span_refers_to_the_normalised_text_not_to_the_original_name():
    q, c, comp = parts("ворота 6000х2250", "Ворота откатные 2250х6000 мм")
    e = next(e for e in evidence_by_attribute(comp, q, c)["dims"] if e["side"] == "candidate")
    assert e["text_version"] == "normalized_main" and e["text"] == c["main_text"] != c["clean_name"]
    assert e["text"][slice(*e["span"])] == e["fragment"] == "2250x6000 мм"             # «х» is already «x» here
    assert "2250х6000" in c["clean_name"] and e["fragment"] not in c["clean_name"]      # the original uses another letter: do not highlight it by this span


# ---------------------------------------------------------------- the candidate's own remainder
def test_the_unparsed_requirements_of_the_candidate_are_listed():
    c = process("Ноутбук с характеристиками не хуже (15.6'' 1920х1080, 8Гб, 2.4ГГц)")
    got = candidate_unparsed(c)
    assert {u["raw"] for u in got} == {"8Гб", "2.4ГГц"}
    assert all(u["reason"] == "number_with_unit" and u["text"][slice(*u["span"])] == u["raw"] for u in got)


def test_a_plain_name_has_no_unparsed_requirements():
    assert candidate_unparsed(process("Труба ПВХ 25 мм")) == [] and candidate_unparsed(process("Ворота откатные 2250х6000 мм")) == []


def test_the_remainder_of_a_note_is_not_listed_as_a_requirement_of_the_candidate():
    assert candidate_unparsed(process("Труба ПВХ 25 мм (прим. 60 Гц)")) == []


# ---------------------------------------------------------------- on the real catalog: every span says what it claims
@real
def test_every_span_of_every_evidence_entry_points_at_its_fragment():
    from analog_search.engine import SearchEngine
    e = SearchEngine.build()
    queries = ["кабель КВВГнг-LS 7*1.5", "камера IP уличная 2 Мп вариофокальная", "ворота откатные 6000х2250", "труба гофра ПВХ 25 мм", "ИБП 12В для улицы",
               "трансформатор силовой сухой 1000 кВА", "Инфракрасный термометр любого производителя", "Тройник 25х30 мм любой цвет", "фланец приварной Ду80 Ру10",
               "сталь полосовая 100х10 мм", "клапан огнезадерживающий EI60 200х200", "ноутбук 8 Гб", "шкаф управления IP66"]
    checked = 0
    for q in queries:
        a = e.search(q)
        for item in a["table"] + a["unconfirmed"] + ([a["nearest_rejected"]] if a["nearest_rejected"] else []):
            entries = [(r["attribute"], v) for r in item["comparison"] for v in r["evidence"]] + [(v["attribute"], v) for v in item["candidate_extra_attributes"]]
            for attribute, v in entries:
                if attribute == "model_codes":
                    continue                                                       # several codes are shown joined; their span is a range
                checked += 1
                assert v["text"][slice(*v["span"])] == v["fragment"] and v["exact"], (q, attribute, v)
            for u in item["candidate_unparsed_constraints"]:
                checked += 1
                assert u["text"][slice(*u["span"])] == u["raw"], (q, u)
    assert checked > 100


@real
def test_the_material_of_a_pipe_reaches_the_answer_of_the_engine():
    from analog_search.engine import SearchEngine
    a = SearchEngine.build().search("труба гофра ПВХ 25 мм")
    assert a["table"] and all("candidate_extra_attributes" in x and "candidate_unparsed_constraints" in x for x in a["table"])
    t = SearchEngine.build().search("труба 25 мм")["table"][0]
    assert any(x["label"] == "материал" for x in t["candidate_extra_attributes"])
