"""«Разные записи — одно значение»: the core promise of the normalizer + extractor.

Every group lists spellings that a specialist considers the same; they must yield the same
canonical value. The `different` tests guard against over-normalization (3×1,5 ≠ 3×2,5).
"""
import pytest

from helpers import attr, attrs, sorted_dims


def same(extractor, variants):
    values = [extractor(v) for v in variants]
    assert values[0] is not None, f"nothing extracted from {variants[0]!r}"
    assert all(v == values[0] for v in values), dict(zip(variants, values))


# ---------------------------------------------------------------- cable cores × section
def test_cable_cores_section_notations():
    same(lambda t: attr(t, "cores_section"), [
        "Кабель контрольный КВВГнг-LS 3х1,5",
        "кабель КВВГнг-LS 3*1.5",
        "Кабель контрольный КВВГнг-LS 3×1,5 мм2",
        "кабель КВВГнг-LS 3 х 1,5",
        "Кабель КВВГнг-LS 3Х1,5",
        "Кабель КВВГнг-LS 3x1.5 мм²",
        "Кабель КВВГнг-LS 3х1,5 кв.мм",
    ])


@pytest.mark.parametrize("a,b", [
    ("Кабель КВВГнг-LS 3х1,5", "Кабель КВВГнг-LS 3х2,5"),
    ("Кабель КВВГнг-LS 7х1,5", "Кабель КВВГнг-LS 5х1,5"),
    ("Кабель ВВГнг(A)-LS 5х6-660", "Кабель ВВГнг(A)-LS 5х10-660"),
])
def test_cable_cores_section_differ(a, b):
    assert attr(a, "cores_section") != attr(b, "cores_section")


def test_cable_glued_suffix_and_rated_voltage():
    t = "Кабель силовой с медными жилами ВВГнг(A)-FRLS 5х70мс(N, РЕ)-1000"
    assert attr(t, "cores_section") == [5, 70]
    assert attr(t, "rated_voltage_v") == 1000
    assert attr(t, "fire_class") == "FRLS"


def test_cable_pairs_notation():
    assert attr("Кабель КПСЭнг(A)-FRLS 1х2х0,75", "cores_section") == [1, 2, 0.75]


def test_cable_screen_letter_and_mark():
    plain = "кабель КВВГнг-LS 7*1.5"
    screened = "Кабель контрольный КВВГЭнг(А)-LS 7х1,5"
    assert attr(plain, "cable_screen") is False
    assert attr(screened, "cable_screen") is True
    assert attr(plain, "cable_mark") == attr(screened, "cable_mark")      # same base mark: screen is a separate attribute
    assert attr(plain, "fire_class") == attr(screened, "fire_class") == "LS"


def test_cable_fire_class_differs():
    assert attr("Кабель ВВГнг(A)-LS 3х6-660", "fire_class") != attr("Кабель ВВГнг(A)-FRLS 3х6-660", "fire_class")


# ---------------------------------------------------------------- resolution
def test_resolution_notations():
    same(lambda t: attr(t, "resolution_mp"), [
        "камера IP 4Мп", "камера IP 4 Мп", "камера IP 4 Мпикс", "камера IP 4MP", "камера IP 4 мп", "камера 4 мегапикселя",
    ])


def test_resolution_not_confused_with_pressure():
    assert attr("Фланцы, номинальное давление 1,0 МПа", "resolution_mp") is None
    assert attr("Фланцы, номинальное давление 1,0 МПа", "pressure_mpa") == 1


def test_resolution_differs():
    assert attr("камера 2Мп", "resolution_mp") != attr("камера 4Мп", "resolution_mp")


# ---------------------------------------------------------------- dimensions
def test_dimensions_order_and_sign():
    same(sorted_dims, [
        "Ворота откатные 2250х6000 мм", "ворота откатные 6000х2250", "ворота откатные 6000×2250",
        "ворота откатные 6000 x 2250", "Ворота откатные 6000*2250 мм",
    ])


def test_dimensions_differ():
    assert sorted_dims("ворота 2250х6000") != sorted_dims("ворота 2250х6500")


def test_dimensions_three_parts_and_units():
    assert attr("Лоток 800x510x250 мм", "dims") == [800, 510, 250]
    assert attr("Лоток 80х51 см", "dims") == [800, 510]


def test_not_a_dimension_when_glued_to_current():
    assert attr("выключатель 2х25А", "dims") is None


# ---------------------------------------------------------------- voltage, current, power
def test_voltage_notations():
    same(lambda t: attr(t, "voltage_v"), [
        "источник питания 12В", "источник питания 12 В", "источник питания 12V", "источник питания 12в", "источник питания 12 V",
    ])


def test_voltage_kilovolts_and_pair():
    assert attr("КВЛ 220 кВ", "voltage_v") == [220000]
    assert attr("напряжение ~380/220В", "voltage_v") == [380, 220]
    assert attr("муфта 0,4 кВ", "voltage_v") == attr("муфта 400 В", "voltage_v")


def test_preposition_v_is_not_voltage():
    assert attr("Шкаф 2 В комплекте с автоматикой", "voltage_v") is None


def test_current_latin_and_cyrillic_letter():
    same(lambda t: attr(t, "current_a"), ["вилка 125A", "вилка 125А", "вилка 125 А"])


def test_power_units():
    assert attr("обогреватель N=1,5 кВт", "power_kw") == attr("обогреватель 1500 Вт", "power_kw") == 1.5


def test_apparent_power():
    assert attr("трансформатор силовой сухой 1000 кВА", "apparent_power_kva") == 1000
    assert attr("мощностью 1250 кВА", "apparent_power_kva") != attr("мощностью 1000 кВА", "apparent_power_kva")


# ---------------------------------------------------------------- diameter, thickness, ip, poles
def test_diameter_notations():
    same(lambda t: attr(t, "diameter_mm"), [
        "Труба гофрированная d=25 мм", "Труба гофрированная диаметр 25 мм", "Труба гофрированная диаметром 25 мм",
        "Труба гофрированная Ø25", "Труба гофр. д.25мм", "Труба гофрированная D 25 мм", "труба гофра ПВХ 25 мм",
    ])


def test_diameter_load_class_is_not_a_diameter():
    assert attr("Дождеприемник ДК-60.80.10-ВЧ-Ф5.000 D400 32158", "diameter_mm") is None


def test_nominal_bore_is_separate_from_diameter():
    assert attr("Труба Ду 50", "nominal_bore_mm") == 50
    assert attr("Труба Ду 50", "diameter_mm") is None


def test_thickness():
    assert attr("Воздуховоды из оцинкованной стали толщиной: 0,7 мм", "thickness_mm") == 0.7


def test_ip_notations():
    same(lambda t: attr(t, "ip"), ["Шкаф IP66", "Шкаф IP 66", "Шкаф ip66"])
    assert attr("Шкаф IP66", "ip") != attr("Шкаф IP65", "ip")


def test_poles_notations():
    same(lambda t: attr(t, "poles"), ["Вилка 3P+E+N", "Вилка 3P+N+E", "Вилка 3Р+N+E"])
    assert attr("Вилка 3P+N+E", "poles") != attr("Вилка 3P+N", "poles")


def test_pressure_units():
    assert attr("номинальное давление 1,0 МПа", "pressure_mpa") == attr("давление 10 бар", "pressure_mpa")


# ---------------------------------------------------------------- codes
def test_homoglyph_codes_are_the_same_code():
    a = attr("Клапан противопожарный (М-Ik/220)-К-СН", "model_codes")
    b = attr("Клапан противопожарный (М-lk/220)-К-CH", "model_codes")        # l vs I, Cyrillic С/Н vs Latin C/H
    assert a is not None and set(a) & set(b)


def test_cyrillic_latin_letters_in_code():
    assert attr("Шкаф ШК-FA.4", "model_codes") == attr("Шкаф ШК-FА.4", "model_codes")   # Latin A vs Cyrillic А


# ---------------------------------------------------------------- alt fragments
def test_alt_numbers_are_kept_apart():
    t = "Сталь полосовая: 100х10 мм, марка Ст3сп(прим. 60х10)"
    assert attr(t, "dims") == [100, 10]
    assert attr(t, "dims", source="alt") == [60, 10]
    assert attrs(t, "dims") == [[100, 10]]


def test_or_analog_fragment_is_not_main_text():
    t = "Колодец по типу Rainpark WL (или аналог), D1600мм, H=4000мм"
    assert attr(t, "diameter_mm") == 1600


# ---------------------------------------------------------------- regressions found by manual review of the catalog
def test_dn_must_be_a_standard_series_value():
    assert attr("Пена двухкомпонентная огнестойкая DN1201", "nominal_bore_mm") is None
    assert attr("Металлорукав DN20мм в герметичной изоляции", "nominal_bore_mm") == 20


def test_pipe_length_is_not_the_diameter():
    t = "Труба металлическая для водосточных систем, покрытие полиэстер, диаметр 150 мм, длина 3000 мм"
    assert attrs(t, "diameter_mm") == [150]
    assert attr(t, "length_mm") == 3000


def test_duct_perimeter_and_side():
    t = "Воздуховоды из оцинкованной стали с шиной и уголками толщиной: 0,7 мм, периметром 2000 мм (со стороной до 600 мм) (500х500 мм)"
    assert attr(t, "perimeter_mm") == 2000
    assert attr(t, "side_mm") == 600
    assert attr(t, "thickness_mm") == 0.7
    assert attr(t, "dims") == [500, 500]
    assert attr(t, "bare_mm") is None


def test_same_value_written_twice_counts_once():
    t = "Клапан размером 700x700 мм (ОЗ-60-НО-700х700 мм)"
    assert attrs(t, "dims") == [[700, 700]]


# ---------------------------------------------------------------- units and dimension order (A4)
def test_dims_keep_written_order_and_expose_sorted_form():
    r = __import__("analog_search").process("Ворота откатные 6000х2250")
    d = next(a for a in r["attributes"] if a["name"] == "dims")
    assert d["value"] == [6000, 2250]                # as written: the rule decides whether order matters
    assert d["extra"]["sorted"] == [2250, 6000]
    assert d["unit"] == "mm"


def test_units_are_canonical_and_reported():
    r = __import__("analog_search").process("Опора 6,3т, высота 12 м, напряжение 0,4 кВ, сечение 25 см2 не считать")
    by = {a["name"]: a for a in r["attributes"]}
    assert by["mass_kg"]["value"] == 6300 and by["mass_kg"]["unit"] == "kg"
    assert by["voltage_v"]["value"] == [400] and by["voltage_v"]["unit"] == "V"
    assert by["height_mm"]["value"] == 12000 and by["height_mm"]["unit"] == "mm"
