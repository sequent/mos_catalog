"""Counts and times, categorical characteristics, kit components and the composite policy.
Synthetic names and other values than the ones that failed; the expected result is written by hand."""
import pytest

from analog_search import process
from analog_search.matching import compare, load_rules
from analog_search.matching.compat import query_subject

RULES = load_rules()


def attrs(text, query=True):
    return {a["name"]: a["value"] for a in process(text, query=query)["attributes"]}


def comp(q, c):
    return compare(process(q, query=True), process(c), RULES)


def status(q, c, attribute):
    return {x.name: x.status for x in comp(q, c).comparisons}.get(attribute)


# ---------------------------------------------------------------- volume and storage time
@pytest.mark.parametrize("text", ["ёмкость 100 мл", "ёмкость 100 МЛ", "ёмкость 100мл", "ёмкость 0,1 л", "ёмкость 0.1 л", "объём 100 ml"])
def test_one_volume_in_every_spelling(text):
    assert attrs(text)["volume_l"] == pytest.approx(0.1)


def test_two_volumes_conflict_and_the_same_volume_matches_in_another_case():
    assert status("пробоотборник 200 МЛ", "Пробоотборник ёмкость 100 мл", "volume_l") == "conflict"
    assert status("пробоотборник 100 МЛ", "Пробоотборник ёмкость 0,1 л", "volume_l") == "match"


@pytest.mark.parametrize("text,days", [("хранение 5 сут.", 5), ("срок хранения газа - 5 дней", 5), ("5 суток хранения", 5), ("хранение 2 недели", 14), ("хранение до 3 дней", 3)])
def test_storage_time_in_days(text, days):
    assert attrs(text)["storage_duration_days"] == days


@pytest.mark.parametrize("text", ["гарантия 5 дней", "срок поставки 5 дней", "работа 5 суток"])
def test_a_number_of_days_without_storage_is_not_a_storage_time(text):
    assert "storage_duration_days" not in attrs(text)


def test_a_different_storage_time_conflicts_and_an_unknown_one_is_no_data():
    base = "пробоотборник хранение 5 сут."
    assert status(base, "Пробоотборник, срок хранения 3 дня", "storage_duration_days") == "conflict"
    assert status(base, "Пробоотборник, срок хранения 5 дней", "storage_duration_days") == "match"
    assert status(base, "Пробоотборник", "storage_duration_days") == "no_data"
    assert comp(base, "Пробоотборник").klass != "compatible"


def test_the_numbers_are_explained_not_left_as_a_remainder():
    p = process("пробоотборник 100 МЛ, хранение 5 сут., 16 портов, 4 шт.", query=True)
    assert p["unparsed_constraints"] == []


# ---------------------------------------------------------------- counts and categorical characteristics
def test_ports_pieces_and_the_surface():
    a = attrs("кросс на 24 порта, 6 шт., крепление к стене")
    assert a["port_count"] == 24 and a["quantity_pcs"] == 6 and a["mount_surface"] == "стена"


def test_eight_ports_are_not_sixteen_and_one_connector_is_not_another():
    q = "оптический кросс на 16 портов LC"
    assert status(q, "Оптический кросс 8 портов LC", "port_count") == "conflict"
    assert status(q, "Оптический кросс 16 портов SC", "connector_type") == "conflict"
    assert comp(q, "Оптический кросс 16 портов LC").klass == "compatible"


def test_a_switch_is_not_an_optical_cross_with_the_same_ports():
    assert comp(q := "оптический кросс 16 портов", "Коммутатор 16 портов").klass != "compatible"


def test_four_pieces_and_eight_pieces_conflict_and_a_surface_conflicts():
    q = "комплект кронштейнов для лестницы к крыше, 4 шт. и метизы"
    assert status(q, "Комплект кронштейнов для лестницы к крыше 8 шт и метизы", "quantity_pcs") == "conflict"
    assert status(q, "Комплект кронштейнов для лестницы к стене 4 шт и метизы", "mount_surface") == "conflict"
    assert comp(q, "Комплект кронштейнов для лестницы к стене 4 шт и метизы").klass == "incompatible"


def test_the_number_of_pieces_alone_never_confirms_anything():
    assert comp("кабель 4 шт.", "Кабель 4 шт.").klass == "compatible"                  # the whole name is the same
    assert comp("труба 4 шт.", "Резистор 4 шт.").klass != "compatible"


def test_printer_characteristics_are_compared_one_by_one():
    q = "лазерный принтер A4, цветной"
    ok = "Принтер лазерный цветная печать A4"
    assert comp(q, ok).klass == "compatible"
    assert status(q, "Принтер лазерный монохромный A4", "print_color_mode") == "conflict"
    assert status(q, "Принтер струйный цветная печать A4", "print_technology") == "conflict"
    assert status(q, "Принтер лазерный цветная печать A3", "paper_format") == "conflict"


def test_unknown_technology_and_format_are_a_check_and_a_full_colour_cartridge_is_not_colour_printing():
    c = comp("лазерный принтер A3, цветной", "Принтер Zebra 123 (в комплекте с полноцветным картриджем)")
    assert c.klass == "unverified" and "print_color_mode" not in {x.name for x in c.comparisons if x.status == "match"}
    assert "print_color_mode" not in attrs("Принтер с полноцветным картриджем", query=False)


def test_a_colour_word_outside_printers_is_not_a_print_mode():
    assert "print_color_mode" not in attrs("Шкаф цветной серый") and "paper_format" not in attrs("Кабель A3")


# ---------------------------------------------------------------- kit components and the composite policy
def test_components_are_named_and_a_negated_one_is_not_a_component():
    assert attrs("кросс с пигтейлами и гильзами")["kit_components"] == ["гильза", "пигтейл"]
    assert attrs("кросс без гильз, пигтейлы")["kit_components"] == ["пигтейл"]
    assert "kit_components" not in attrs("кросс без гильз")


def test_every_component_of_the_query_must_be_named_unknown_is_not_a_match():
    q = "оптический кросс 16 портов с пигтейлами и гильзами"
    assert status(q, "Оптический кросс 16 портов. В комплекте: пигтейлы, гильзы", "kit_components") == "match"
    assert status(q, "Оптический кросс 16 портов. В комплекте: пигтейлы", "kit_components") == "no_data"
    assert status(q, "Оптический кросс 16 портов", "kit_components") == "no_data"


def test_a_kit_with_its_list_in_the_name_is_confirmed_when_everything_the_user_asked_for_is_named():
    q = "оптический кросс 16 портов с пигтейлами и гильзами"
    assert comp(q, "Оптический кросс 16 портов. В комплекте: пигтейлы, гильзы").klass == "compatible"


@pytest.mark.parametrize("candidate", ["Оптический кросс 16 портов. В комплекте: пигтейлы", "Оптический кросс 16 портов в составе:", "Оптический кросс 16 портов"])
def test_a_kit_is_held_back_when_what_the_user_asked_for_is_not_named(candidate):
    assert comp("оптический кросс 16 портов с пигтейлами и гильзами", candidate).klass == "unverified"


def test_a_composite_without_a_kit_requirement_is_still_held_back():
    c = comp("оптический кросс 16 портов", "Оптический кросс 16 портов. В комплекте: пигтейлы, гильзы")
    assert c.klass == "unverified" and any("комплект" in r for r in c.reasons)


def test_a_header_of_a_kit_never_counts_as_a_verified_list():
    c = comp("оптический кросс с пигтейлами", "Оптический кросс, в составе:")
    assert c.klass == "unverified"


# ---------------------------------------------------------------- synonyms and the subject of a kit
def test_the_hydropress_is_the_same_type_as_a_hydraulic_press_and_the_kit_is_not_the_subject():
    assert process("гидропресс ПРГ-1 с набором матриц", query=True)["type"] == "пресс гидравлический"
    assert process("Пресс гидравлический с комплектом матриц. ПРГ-1")["type"] == "пресс гидравлический"
    assert comp("гидропресс ПРГ-1 с набором матриц", "Пресс гидравлический с комплектом матриц. ПРГ-1").klass == "compatible"
    assert comp("гидропресс ПРГ-1 с набором матриц", "Пресс гидравлический ПРГ-2 с комплектом матриц").klass == "incompatible"


def test_a_kit_form_is_not_the_subject_the_thing_inside_is():
    assert query_subject(process("компл. кронштейнов для лестницы", query=True))[0] == "кронштейнов"
    assert query_subject(process("набор инструментов", query=True))[0] == "инструментов"


def test_the_abbreviations_are_expanded_only_in_their_context():
    assert process("компл. кронштейнов", query=True)["type"] == "комплект"
    assert "оптический" in process("настенный опт. кросс", query=True)["search_text"]
    assert "оптический" not in process("опт. 5 мм", query=True)["search_text"]


def test_the_model_of_a_kit_is_still_a_requirement():
    assert comp("гидропресс ПРГ-1", "Пресс гидравлический с комплектом матриц. ПРГ-2").klass == "incompatible"


def test_an_optical_cross_on_the_catalog_side_has_a_type_and_a_mounting():
    p = process("Оптический кросс настенный 16 портов, LC разъемы")
    assert p["type"] == "кросс" and {a["name"]: a["value"] for a in p["attributes"]}["mounting"] == "настенное"


# ---------------------------------------------------------------- the explanation says what was required and what was not named
def explain_text(q, c):
    from analog_search.matching.explain import explain
    pq, pc = process(q, query=True), process(c)
    return explain(compare(pq, pc, RULES), pq, pc)["text"]


def test_the_explanation_counts_the_requirements_and_names_the_series_that_is_only_named():
    t = explain_text("воздушный клапан JKL 700х400", "Воздушный клапан JKL-700x400")
    assert "Требований запроса подтверждено 1 из 1" in t and "JKL" in t and "само по себе не подтверждает" in t


def test_the_explanation_of_an_unverified_pair_names_the_missing_component_and_the_count():
    t = explain_text("оптический кросс 16 портов с пигтейлами и гильзами", "Оптический кросс 16 портов")
    assert "Требований запроса подтверждено 1 из 2" in t and "состав комплекта" in t and "не названы" not in t.split("Не подтверждено")[0]


def test_a_rejected_pair_does_not_claim_a_count_of_confirmed_requirements():
    assert "Требований запроса подтверждено" not in explain_text("клапан JKL 700х400", "Клапан BN-Z-200")
