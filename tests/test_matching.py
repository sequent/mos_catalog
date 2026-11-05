import pytest

from analog_search import process
from analog_search.matching import compare, load_rules, type_relation
from analog_search.matching.ranking import rank
from analog_search.matching.verdict import decide

RULES = load_rules()


def st(query, candidate, attribute):
    comp = compare(process(query), process(candidate), RULES)
    return {c.name: c.status for c in comp.comparisons}.get(attribute)


def cls(query, candidate):
    return compare(process(query), process(candidate), RULES)


# ---------------------------------------------------------------- B3: type is a signal
@pytest.mark.parametrize("q,c,key,status", [
    ("кабель контрольный 7х1,5", "Кабель контрольный КВВГнг 7х1,5", "same", "match"),
    ("кабель 7х1,5", "Кабель контрольный КВВГнг 7х1,5", "same_family_q_generic", "match"),
    ("кабель контрольный 7х1,5", "Кабель силовой ВВГнг 3х6", "same_family_both_specific", "check"),
    ("кабель контрольный 7х1,5", "Кабель ВВГ 3х6", "same_family_c_generic", "check"),
    ("камера 2 Мп", "Фиксация разрывов транзитов", "unknown_either_side", "unknown"),
    ("камера 2 Мп", "Воздуховоды из оцинкованной стали", "different_both_specific", "conflict"),
    ("блок бесперебойного питания 12В", "Источник бесперебойного питания СКАТ", "different_one_generic", "check"),
])
def test_type_relation(q, c, key, status):
    got = type_relation(process(q), process(c), RULES)
    assert got == (key, status)


def test_type_never_removes_a_candidate():
    """Even a type conflict is only a status: the candidate is still compared, never silently dropped."""
    comp = cls("камера 2 Мп", "Воздуховоды из оцинкованной стали толщиной 2 Мп")
    assert comp.type_status == "conflict" and comp.klass == "incompatible" and comp.comparisons


def test_unknown_type_is_neutral():
    comp = cls("ворота откатные 6000х2250", "Створка 2250х6000 мм (без типа)")
    assert comp.type_status == "unknown"
    assert comp.klass != "incompatible"


# ---------------------------------------------------------------- B4: statuses of characteristics
def test_exact_and_conflict():
    assert st("кабель КВВГнг-LS 7*1.5", "Кабель контрольный КВВГнг(A)-LS 7х1,5", "cores_section") == "match"
    assert st("кабель КВВГнг-LS 7*1.5", "Кабель контрольный КВВГнг(A)-LS 7х2,5", "cores_section") == "conflict"


def test_ladder_substitute_and_conflict():
    assert st("камера 3 Мп", "видеокамера 2Мп", "resolution_mp") == "substitute"
    assert st("камера 2 Мп", "видеокамера 2Мп", "resolution_mp") == "match"
    assert st("камера 5 Мп", "видеокамера 2Мп", "resolution_mp") == "conflict"


def test_flag_mode_marks_but_does_not_reject():
    comp = cls("кабель КВВГнг-LS 7*1.5", "Кабель контрольный КВВГЭнг(A)-LS 7х1,5")
    assert {c.name: c.status for c in comp.comparisons}["cable_screen"] == "flag"
    assert comp.klass == "unverified"                       # not «compatible», not «incompatible»


def test_sizes_sorted_ordered_and_unlisted():
    assert st("ворота откатные 6000х2250", "Ворота откатные 2250х6000 мм", "dims") == "match"            # sorted: side order is free
    assert st("лоток 100х200", "Лоток перфорированный 200х100", "dims") == "conflict"                    # ordered: width×height
    assert st("лоток 100х200", "Лоток перфорированный 100х200", "dims") == "match"
    assert st("плита 600х400 мм", "Плита перекрытия 400х600 мм", "dims") == "match_reordered"           # no rule → check
    assert st("плита 600х400 мм", "Плита перекрытия 400х650 мм", "dims") == "conflict"


def test_no_data_when_candidate_does_not_state_it():
    assert st("ИБП 12В для улицы", "Источник бесперебойного питания СКАТ-1200М", "voltage_v") == "no_data"


def test_alt_only_value_is_not_a_match():
    assert st("сталь полосовая 60х10 мм", "Сталь полосовая: 100х10 мм, марка Ст3сп (прим. 60х10)", "dims") == "conflict"
    assert st("сталь 60х10 мм", "Сталь полосовая марка Ст3сп (прим. 60х10)", "dims") == "alt_only"


def test_bare_mm_is_ambiguous_not_a_match():
    assert st("профиль 25 мм", "Профиль крепежный толщиной 25 мм", "bare_mm") == "ambiguous"
    assert st("профиль 25 мм", "Профиль крепежный 40х20 мм", "bare_mm") == "no_data"


def test_same_bare_value_on_both_sides_is_a_match():
    """«сверло 14 мм» searched by its own text must confirm itself, not stay «значение без смысла»."""
    assert st("Саморез 5,5х240 со сверлом 14 мм", "Саморез 5,5х240 со сверлом 14 мм", "bare_mm") == "match"


def test_different_model_code_is_not_a_conflict():
    """An analog may be another model: a different code blocks nothing, an equal code confirms."""
    assert st("камера XNV-6081RE", "Видеокамера XNV-6081RE 2Мп", "model_codes") == "match"
    assert st("камера XNV-6081RE", "Видеокамера XNP-6400RW 2Мп", "model_codes") == "info_diff"
    assert cls("камера XNV-6081RE", "Видеокамера XNP-6400RW 2Мп").klass != "incompatible"


def test_standard_differences_are_flags():
    assert st("труба ГОСТ 8639-82 100х100", "Трубы стальные квадратные (ГОСТ 8732-78) 100х100 мм", "standard") == "flag"


def test_cable_mark_difference_is_a_conflict():
    assert st("кабель КВВГнг-LS 3х1,5", "Кабель ВВГнг(A)-LS 3х1,5", "cable_mark") == "conflict"


def test_placement_conflict():
    assert st("камера уличная 2 Мп", "видеокамера внутренняя 2Мп", "placement") == "conflict"


# ---------------------------------------------------------------- B4: compatibility classes
def test_class_compatible_substitution_unverified_incompatible():
    assert cls("ворота откатные 6000х2250", "Ворота откатные 2250х6000 мм").klass == "compatible"
    assert cls("камера 3 Мп", "видеокамера 2Мп").klass == "compatible_with_substitution"
    assert cls("ИБП 12В для улицы", "Уличный источник бесперебойного питания SKAT-V.12DC-18").klass == "unverified"
    assert cls("трансформатор силовой сухой 1000 кВА", "Трансформатор силовой сухой ТСЗ-1250/11-У1, мощностью 1250 кВА").klass == "incompatible"


def test_composite_can_never_be_compatible():
    comp = cls("шкаф управления IP66", "Шкаф управления навесной IP66 в составе:")
    assert comp.klass == "unverified" and any("комплект" in r for r in comp.reasons)


def test_a_query_without_confirmable_characteristics_is_never_compatible():
    """«камера» alone: every camera has the same type, which confirms nothing."""
    comp = cls("видеокамера", "Видеокамера уличная корпусная 2Мп")
    assert comp.klass == "unverified" and "нет подтверждённых характеристик" in comp.reasons


def test_placement_alone_is_not_evidence():
    comp = cls("уличный светильник", "Светильник уличный консольный")
    assert comp.klass == "unverified"


def test_identical_name_is_evidence():
    comp = cls("Шкаф ШК-FA.4", "Шкаф ШК-FA.4")
    assert comp.confirmed_strong >= 1


def test_different_material_is_a_conflict_not_just_a_missing_word():
    """«ПВХ» vs «ПНД»: material is a characteristic, so the difference is a conflict (it used to be only a missing word)."""
    good = cls("труба гофра ПВХ 25 мм", "Гофрированная труба из ПВХ (серия 9) d=25 мм")
    bad = cls("труба гофра ПВХ 25 мм", "Труба ПНД гибкая гофр. д.25мм, тяжёлая")
    assert good.klass == "compatible"
    assert bad.klass == "incompatible" and "material" in bad.conflicts


def test_material_statuses():
    assert st("труба ПВХ 25 мм", "Труба гофрированная из ПВХ d=25 мм", "material") == "match"
    assert st("труба ПНД 25 мм", "Труба ПЭ 100 SDR 11 d=25 мм", "material") == "match"          # ПНД and ПЭ are one material
    assert st("труба ПВХ 25 мм", "Труба гофрированная d=25 мм", "material") == "no_data"
    assert st("кабель медный ВВГнг 3х1,5", "Кабель с алюминиевыми жилами ВВГнг 3х1,5", "material") == "conflict"


def test_many_missing_words_still_cap_the_class():
    """The word guard stays for what is not a characteristic: three of five words are missing → not confirmed."""
    comp = cls("труба гофрированная 25 мм оранжевая ударопрочная морозостойкая", "Гофрированная труба d=25 мм")
    assert comp.klass == "unverified" and any("оранжев" in r for r in comp.reasons)


def test_descriptive_word_does_not_block_a_confirmed_row():
    """Found on the dev queries: one descriptive word («теплоизоляция») must not bury an otherwise confirmed row."""
    comp = cls("конвектор Noirot 1500 Вт теплый пол", "Конвектор электрический Noirot CNX-4 N=1,5 кВт")
    assert comp.klass in ("compatible", "unverified") and comp.word_coverage >= 0.5


def test_series_code_written_without_suffix_matches():
    """Found on the dev queries: «2ST 22/6» must meet «2ST 22/6-2»; a different size must still conflict."""
    assert st("трубка K-FLEX 2ST 22/6", "Трубки из вспененного каучука, диаметр 22 мм (K-FLEX - 2ST 22/6-2)", "numeric_code") == "match"
    assert st("трубка K-FLEX 2ST 22/6", "Трубки из вспененного каучука (K-FLEX - 2ST 35/6-2)", "numeric_code") == "conflict"


def test_conflict_wins_over_everything():
    comp = cls("кабель КВВГнг-LS 7*1.5", "Кабель контрольный КВВГЭнг(A)-FRLS 7х1,5")
    assert comp.klass == "incompatible" and "fire_class" in comp.conflicts


def test_weak_evidence_rules_for_listing():
    """Under «not found» only candidates with real grounds may be shown; a 12 V lock is not a possible UPS."""
    lock = cls("ИБП 12В для улицы", "Замок электромагнитный AL-350FB-MG 12V")
    ups = cls("ИБП 12В для улицы", "Уличный источник бесперебойного питания SKAT-V.12DC-18")
    assert not lock.weak_evidence
    assert ups.weak_evidence


# ---------------------------------------------------------------- B5 / B6
def _item(i, klass_text_pair, bm25, dense=0.8, boost_id=None):
    q, c = klass_text_pair
    return {"index": i, "id": f"row-{i}", "compat": compare(process(q), process(c), RULES), "bm25_score": bm25, "dense_score": dense}


def test_score_never_lifts_a_candidate_across_classes():
    same = _item(1, ("ворота 6000х2250", "Ворота откатные 2250х6000 мм"), bm25=1.0)
    weak = _item(2, ("ворота 6000х2250", "Ворота 6000х2250 мм в составе:"), bm25=100.0)          # huge text score, but «комплект»
    ranked = rank([weak, same], RULES)
    assert [it["index"] for it in ranked] == [1, 2]
    assert ranked[0]["compat"].klass == "compatible" and ranked[1]["compat"].klass == "unverified"


def test_pair_boost_is_local_and_cannot_change_class():
    a = _item(1, ("кабель КВВГнг-LS 7*1.5", "Кабель КВВГЭнг(A)-FRLS 7х1,5"), bm25=5.0)          # incompatible
    b = _item(2, ("кабель КВВГнг-LS 7*1.5", "Кабель контрольный КВВГнг(A)-LS 7х1,5"), bm25=1.0)   # compatible
    ranked = rank([a, b], RULES, pair_boosts={"row-1": 1.0})
    assert ranked[0]["index"] == 2
    decide(ranked, RULES)
    assert ranked[1]["verdict"] == "reject"


def test_not_found_when_only_unconfirmed_exist():
    items = [_item(1, ("ИБП 12В для улицы", "Уличный источник бесперебойного питания SKAT-V.12DC-18"), 3.0)]
    out = decide(rank(items, RULES), RULES)
    assert out["status"] == "not_found" and not out["results"] and len(out["unconfirmed"]) == 1


def test_found_lists_only_confirmed_and_limits():
    items = [_item(i, ("ворота 6000х2250", "Ворота откатные 2250х6000 мм"), 1.0 + i / 10) for i in range(8)]
    out = decide(rank(items, RULES), RULES, top_k=5)
    assert out["status"] == "found" and len(out["results"]) == 5
    assert all(it["verdict"] in ("same", "substitute") for it in out["results"])


def test_nearest_rejected_is_the_one_with_fewest_conflicts():
    two = _item(1, ("кабель КВВГнг-LS 7*1.5", "Кабель ВВГнг(A)-FRLS 5х2,5"), bm25=9.0)                 # several conflicts
    one = _item(2, ("кабель КВВГнг-LS 7*1.5", "Кабель контрольный КВВГнг(A)-LS 24х1,5"), bm25=1.0)   # one conflict
    out = decide(rank([two, one], RULES), RULES)
    assert out["status"] == "not_found" and out["nearest_rejected"]["index"] == 2


# ---------------------------------------------------------------- configurable order of sides for types without a rule
def test_dims_order_default_is_configurable(tmp_path):
    import copy
    import yaml
    from analog_search.matching.rules import RULES_PATH
    q, c = process("плита 600х400 мм"), process("Плита перекрытия 400х600 мм")
    assert {x.name: x.status for x in compare(q, c, RULES).comparisons}["dims"] == "match_reordered"
    for value, expected in (("sorted", "match"), ("ordered", "conflict"), ("check", "match_reordered")):
        raw = copy.deepcopy(yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")))
        raw["defaults"]["dims_order_unlisted"] = value
        p = tmp_path / f"{value}.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        assert {x.name: x.status for x in compare(q, c, load_rules(p)).comparisons}["dims"] == expected
    raw["defaults"]["dims_order_unlisted"] = "maybe"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="dims_order_unlisted"):
        load_rules(bad)


def test_word_stem_prefix_counts_as_covered():
    cov, missing, n = __import__("analog_search.matching.compat", fromlist=["word_coverage"]).word_coverage(
        process("комплект 220 кВ"), process("Комплектное распределительное устройство 220 кВ"))
    assert "комплект" not in missing


# ---------------------------------------------------------------- model codes: only strong codes may confirm
def test_service_tokens_are_not_model_codes():
    """Found on probe A: «РИП-12 исп.56» was «confirmed» by «РИП-24 исп.56» through the shared token «исп.56»."""
    # контрольной проверки changed the policy: a model of the SAME series that differs is a conflict («РИП-12» ≠ «РИП-24»), not just «other model»;
    # the shared service token «исп.56» still confirms nothing
    assert st("резервированный источник питания РИП-12 исп.56", "Резервированный источник питания РИП-24 ИСП.56", "model_codes") == "conflict"
    assert st("резервированный источник питания РИП-12 исп.56", "Резервированный источник питания РИП-12 ИСП.56", "model_codes") == "match"
    assert st("4-х поточный блок BM-SGN45R4/N1", "4-х поточный блок BM-SGN90R4/N1", "model_codes") == "conflict"


def test_strong_code_definition():
    from analog_search.matching.compat import strong_codes
    from analog_search.normalize import code_skeleton as sk
    good = {sk("XNV-6081RE"), sk("ШК-FA.4")}
    assert strong_codes(good) == good                                     # codes are stored in the skeleton form
    assert strong_codes({sk(x) for x in ("исп.56", "4-х", "Л-1", "ТУ3449-003", "IP66")}) == set()


def test_rated_voltage_of_a_cable_answers_a_plain_voltage():
    """Found on the dev control queries: «0.66кВ» in the query, «-660» in the cable name."""
    assert st("силовой кабель медный ВВГнг LS 5x6 0.66кВ", "Кабель силовой с медными жилами ВВГнг(A)-LS 5х6-660", "voltage_v") == "match"
    assert st("силовой кабель ВВГнг LS 5x6 1 кВ", "Кабель силовой с медными жилами ВВГнг(A)-LS 5х6-660", "voltage_v") == "conflict"
    assert st("кабель ВВГнг 5х6-660", "Кабель силовой ВВГнг(A)-LS 5х6 0,66 кВ", "rated_voltage_v") == "match"


# ---------------------------------------------------------------- quick fixes found on test/holdout queries (stage 6)
def stq(query, candidate, attribute):
    """Like `st` but the query goes through the query-side chain (as in the real search)."""
    comp = compare(process(query, query=True), process(candidate), RULES)
    return {c.name: c.status for c in comp.comparisons}.get(attribute)


@pytest.mark.parametrize("query,candidate", [
    ("крышный вентилятор DVN-800-6D с автоматикой", "Вентилятор крышный в комплекте с автоматикой DVN 800-6D"),
    ("крышка на Т-отвод Стандарт KTCS-200", "Крышка Т-образного отвода «Стандарт KTCS 200»"),
    ("ибп скат 1200м", "Источник бесперебойного питания: СКАТ-1200М"),
    ("провод АС-240/32 неизолированный для ВЛ", "Провод неизолированный для воздушных линий электропередачи АС 240/32"),
])
def test_code_written_with_other_separators_matches(query, candidate):
    """Found on the test/holdout queries: the same code written with a hyphen, a space or a slash must be one code."""
    assert stq(query, candidate, "model_codes") == "match"


@pytest.mark.parametrize("query,candidate", [
    ("крышка на Т-отвод Стандарт KTCS-200", "Крышка Т-образного отвода «Стандарт KTCS 300»"),
    ("провод АС-240/32 неизолированный", "Провод неизолированный АС 300/39"),
])
def test_different_codes_still_differ(query, candidate):
    """A code written with a space on one side has no hyphenated series: it stays «другая модель» (info), not a conflict."""
    assert stq(query, candidate, "model_codes") == "info_diff"


def test_another_model_of_the_same_hyphenated_series_is_a_conflict():
    assert stq("резервированный источник питания РИП-12 исп.56", "Резервированный источник питания РИП-24 ИСП.56", "model_codes") == "conflict"


def test_common_word_plus_number_is_not_a_code():
    """«труба 50» must not become a code that confirms every «труба 50 мм»."""
    r = process("труба 50 мм", query=True)
    assert not [a for a in r["attributes"] if a["name"] == "model_codes"]
    r = process("кабель 100м", query=True)
    assert not [a for a in r["attributes"] if a["name"] == "model_codes"]


def test_number_plus_metre_inside_a_code_is_not_a_length_conflict():
    assert stq("ибп скат 1200м", "Источник бесперебойного питания: СКАТ-1200М", "length_mm") == "info_diff"      # neutral, not «нет данных»
    assert stq("кабель 100м", "Кабель контрольный 4х1,5 длиной 50 м", "length_mm") == "conflict"                # a real length still conflicts


def test_nominal_bore_and_nominal_diameter_are_one_quantity():
    cand = "Фланцы стальные приварные, номинальное давление 1,0 МПа, номинальный диаметр 80 мм"
    assert stq("фланец приварной Ду80", cand, "nominal_bore_mm") == "match"
    assert stq("фланец Ду100", cand, "nominal_bore_mm") == "conflict"
    assert stq("фланец номинальный диаметр 80 мм", "Фланец Ду80", "nominal_bore_mm") == "match"


def test_pn_in_bar_and_pressure_in_mpa_are_one_quantity():
    cand = "Фланцы стальные приварные, номинальное давление 1,0 МПа, номинальный диаметр 80 мм"
    assert stq("фланец Ру10", cand, "pressure_mpa") == "match"                  # 10 bar = 1.0 MPa
    assert stq("фланец Ру16", cand, "pressure_mpa") == "conflict"
    assert stq("фланец PN10", cand, "pressure_mpa") == "match"


def test_word_that_changes_its_stem_is_covered():
    from analog_search.matching.compat import word_coverage
    from analog_search.retrieval.tokens import stem
    assert stem("фланец") == stem("фланцы") and stem("колодец") == stem("колодцы")
    cov, missing, _ = word_coverage(process("фланец приварной", query=True), process("Фланцы стальные приварные"))
    assert "фланец" not in missing
