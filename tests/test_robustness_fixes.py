"""Defects of normalisation and robustness found with a set of rewritten names (diagnostic run before the fixes: 22 of 224 single
operations lost the source row). Synthetic names and other values than the set's own."""
import pytest

from analog_search import flags, process
from analog_search.matching import compare, load_rules
from analog_search.matching.compat import _name_key, _same_name_written_differently
from analog_search.normalize import normalize_text

RULES = load_rules()


def attrs(text, query=True):
    return {a["name"]: a["value"] for a in process(text, query=query)["attributes"]}


def comp(q, c):
    return compare(process(q, query=True), process(c), RULES)


# ---------------------------------------------------------------- abbreviated names of a quantity
@pytest.mark.parametrize("short,full", [("диам. 25 мм", "диаметр 25 мм"), ("толщ.: 3 мм", "толщина: 3 мм"), ("выс. 40 мм", "высота 40 мм"), ("шир. 100 мм", "ширина 100 мм"),
                                       ("дл. 2 м", "длина 2 м"), ("глуб. 5 мм", "глубина 5 мм")])
def test_an_abbreviated_label_reads_as_the_full_word(short, full):
    assert normalize_text(short) == normalize_text(full)


def test_the_abbreviated_label_gives_the_same_characteristic_as_the_full_word():
    assert attrs("труба диам. 25 мм") == attrs("труба диаметр 25 мм") and attrs("лист толщ. 3 мм")["thickness_mm"] == 3


@pytest.mark.parametrize("text", ["диаметр 25", "дл20", "задел 5", "индикатор диаметра"])
def test_a_word_without_the_abbreviation_sign_is_left_alone(text):
    assert normalize_text(text) == text


# ---------------------------------------------------------------- Latin tonne and kilometre
def test_latin_t_and_km_after_a_number():
    assert attrs("груз до 5 t")["mass_kg"] == 5000 and normalize_text("до 2 km") == "до 2 км"
    for safe in ("10W-40 масло", "SKAT-5t", "модель X5t", "5 tool"):
        assert normalize_text(safe) == safe


# ---------------------------------------------------------------- services written in short
@pytest.mark.parametrize("name", ["Погр. при перевозках материалов", "Разгр. материалов", "Перевоз. грузов I класса", "Трансп. грузов"])
def test_an_abbreviated_service_is_a_service(name):
    assert flags.is_service(name) and process(name, query=True)["is_service"]


@pytest.mark.parametrize("name", ["Погрешность измерения", "Погрузчик", "Перевозчик ООО"])
def test_other_words_that_begin_alike_are_not_services(name):
    assert not flags.is_service(name)


def test_a_short_service_is_covered_by_the_full_service_words():
    c = comp("Погр. материалов в ящиках", "Погрузка материалов в ящиках")
    assert c.klass == "compatible"


# ---------------------------------------------------------------- an adjective is not a type
@pytest.mark.parametrize("text,type_", [("Соединительный зажим плашечный", "зажим"), ("Измерительная установка", "установка"), ("Установка измерительная", "установка"),
                                        ("Соединитель кабельный", "соединитель"), ("Измеритель сопротивления", "измеритель"), ("Контрольный кабель 7х1,5", "кабель контрольный")])
def test_the_type_of_a_name_that_starts_with_an_adjective(text, type_):
    assert process(text, query=True)["type"] == type_


def test_a_catalog_name_that_starts_with_an_adjective_gets_no_wrong_type_and_the_query_side_finds_the_noun():
    assert process("Прижимная скоба СР", query=True)["type"] == "скоба" and process("Скоба прижимная СР")["type"] == "скоба"
    assert process("Соединительный зажим плашечный")["type"] is None                       # unknown, not «соединитель»


# ---------------------------------------------------------------- the same name written differently
@pytest.mark.parametrize("q,c", [("Компл. соед. KTY", "Комплект соединительный KTY"), ("Прижимная скоба СР", "Скоба прижимная СР"),
                                 ("Прибор приемно-контр. управления Альфа", "Прибор приемно-контрольный управления Альфа"),
                                 ('Мембрана "Flagon BSL" тоннельная', "Мембрана Flagon BSL тоннельная"), ("Лист: оцинкованный, 3 мм", "Лист оцинкованный 3 мм")])
def test_a_name_with_abbreviations_swapped_words_quotes_or_a_colon_is_the_same_name(q, c):
    assert comp(q, c).klass == "compatible"


@pytest.mark.parametrize("q,c", [("Скоба прижимная СР", "Скоба прижимная СТ"), ("Кабель 5 медный", "Медный 5 кабель"), ("Кабель 2x16 медный", "Кабель 16x2 медный"),
                                 ("Скоба прижимная СР", "Скоба прижимная СР 2"), ("Компл. соед. KTY", "Комплект соединительный KTX")])
def test_a_different_name_is_not_made_equal_by_the_tolerance(q, c):
    assert not _same_name_written_differently(process(q, query=True), process(c))


@pytest.mark.parametrize("q,c", [("Скоба прижимная СР", "Скоба прижимная СТ"), ("Скоба прижимная СР", "Скоба прижимная СР 2"), ("Компл. соед. KTY", "Комплект соединительный KTX")])
def test_and_a_pair_with_nothing_else_to_confirm_it_stays_unconfirmed(q, c):
    assert comp(q, c).klass != "compatible"


def test_the_swap_does_not_cross_a_number_or_a_code():
    a, b = process("Труба стальная 25 мм ПВХ"), process("Труба 25 мм стальная ПВХ")
    assert not _same_name_written_differently(a, b)


def test_the_strict_key_still_separates_what_it_separated():
    assert _name_key(process("Реле 10±1 А")) != _name_key(process("Реле 10 А"))
    assert _name_key(process('Мембрана "Flagon"')) == _name_key(process("Мембрана Flagon"))


def test_an_abbreviation_with_a_dot_is_covered_by_a_longer_word_of_the_candidate():
    c = comp("Шкаф распред. оперативного тока 220 В", "Шкаф распределения оперативного тока 220 В")
    assert c.klass == "compatible" and not c.missing_words


# ---------------------------------------------------------------- the class with a Roman numeral
def test_a_roman_class_is_a_requirement_and_i_is_not_ii():
    assert attrs("перевозка грузов II класса")["class_roman"] == "II" and attrs("песок II класс")["class_roman"] == "II" and attrs("класс I")["class_roman"] == "I"
    q = "перевозка грузов II класса автомобилями бортовыми до 5 т"
    assert comp(q, "Перевозка грузов I класса автомобилями бортовыми до 5 т").klass == "incompatible"
    assert comp(q, "Перевозка грузов II класса автомобилями бортовыми до 5 т").klass == "compatible"
    assert comp(q, "Перевозка грузов автомобилями бортовыми до 5 т").klass == "unverified"


def test_an_accuracy_class_in_arabic_digits_is_not_a_roman_class():
    assert "class_roman" not in attrs("прибор класс точности 0,5") and "class_roman" not in attrs("трансформатор класса 1")
