import pytest

from analog_search import process
from analog_search.normalize import clean, code_skeleton, normalize_text, split_alt


# ---------------------------------------------------------------- cleaning
@pytest.mark.parametrize("raw,prefix,rest", [
    ("1.7.2)Шкаф оборудования", "1.7.2)", "Шкаф оборудования"),
    ("2.Шкаф оборудования ЛВС в составе:", "2.", "Шкаф оборудования ЛВС в составе:"),
    ("1)Рабочая станция HP", "1)", "Рабочая станция HP"),
])
def test_numbering_prefix_removed(raw, prefix, rest):
    text, p = clean(raw)
    assert p == prefix and text == rest


def test_number_with_unit_is_not_a_prefix():
    assert clean("5.2 мм прокладка")[1] is None


def test_whitespace_and_placeholders():
    text, _ = clean("УПАСК  ВЧ КВЛ 220 кВ Объект-А – Объект-Б/т  ПС Объект-Г")
    assert "Объект" not in text and "  " not in text


def test_original_name_is_not_lost():
    raw = "1.7.2)Кабель КВВГнг-LS 3х1,5 Объект-А"
    assert process(raw)["clean_name"] != raw       # cleaned for search…
    # …but the catalog row keeps `raw_name` (prepare_catalog.py), see test below
    from pathlib import Path
    assert "raw_name" in (Path(__file__).parent.parent / "scripts" / "prepare_catalog.py").read_text(encoding="utf-8")


def test_split_alt():
    main, alts = split_alt("Тройники, диаметр 63 мм (прим. 60,3 х 5,6 исп.1 )")
    assert main == "Тройники, диаметр 63 мм" and alts == ["прим. 60,3 х 5,6 исп.1"]


def test_normalize_text_keeps_codes_and_lists():
    assert normalize_text("Н-1, Н-2, сечение 13x44 мм") == "Н-1, Н-2, сечение 13x44 мм"
    assert normalize_text("221.04-1") == "221.04-1"
    assert normalize_text("0,66") == "0.66"


def test_skeleton_ignores_script_and_case():
    assert code_skeleton("ШК-FA.4") == code_skeleton("ШК-FА.4")
    assert code_skeleton("М-Ik/220") == code_skeleton("М-lk/220")


# ---------------------------------------------------------------- types
@pytest.mark.parametrize("text,expected", [
    # example queries from the assignment
    ("кабель КВВГнг-LS 7*1.5", "кабель"),
    ("камера IP уличная 2 Мп вариофокальная", "видеокамера"),
    ("ворота откатные 6000х2250", "ворота откатные"),
    ("труба гофра ПВХ 25 мм", "труба гофрированная"),
    ("ИБП 12В для улицы", "источник бесперебойного питания"),
    ("трансформатор силовой сухой 1000 кВА", "трансформатор силовой"),
    ("огнетушитель порошковый ОП-5", "огнетушитель"),
    # catalog rows
    ("Кабель контрольный КВВГЭнг(А)-LS 4x6", "кабель контрольный"),
    ("Кабель силовой с медными жилами ВВГнг(A)-LS 5х6-660", "кабель силовой"),
    ("Видеокамера уличная корпусная 2Мп, моторизированный объектив 2,8 – 12 мм. XNV-6081RE", "видеокамера"),
    ("IP-камера PTZ, поворотная с ик подсветкой. XNP-6400RW", "видеокамера"),
    ("Ворота откатные 2250х6000 мм", "ворота откатные"),
    ("Гофрированная труба из ПВХ (серия 9) с протяжкой d=25 мм", "труба гофрированная"),
    ("Труба легкая гофр. с протяжкой, без галогенов. 81820", "труба гофрированная"),
    ("Трубы гибкие гофрированные из самозатухающего ПВХ легкие с протяжкой, диаметр 20 мм", "труба гофрированная"),
    ("Уличный источник бесперебойного питания SKAT-V.12DC-18 исп.5М", "источник бесперебойного питания"),
    ("Источник бесперебойного питания: СКАТ-1200М", "источник бесперебойного питания"),
    ("Трансформатор силовой сухой трехфазный двухобмоточный ТСЗ-1250/11-У1", "трансформатор силовой"),
    ("Оптоволоконный патч-корд многомодовый 50/125 мкм", "патч-корд оптический"),
    ("Кабель-канал с крышкой 60х40 мм", "кабель-канал"),
    ("Кабельный хомут СЭ-1 85/105", "хомут кабельный"),
    ("Шкаф управления навесной ШУ-ТС-3-50-2000", "шкаф управления"),
    ("Воздуховоды из оцинкованной стали с шиной и уголками", "воздуховод"),
    ("Решетки жалюзийные регулируемые из алюминиевого профиля", "решетка жалюзийная"),
    ("Внутренний вертикальный угол «Стандарт LVI-90» 200х100", "угол"),
])
def test_product_type(text, expected):
    assert process(text)["type"] == expected


def test_family_groups_related_types():
    assert process("кабель КВВГнг-LS 7*1.5")["family"] == process("Кабель контрольный КВВГнг(A)-LS 7х1,5")["family"] == "кабель"
    assert process("Кабель-канал 16х16 мм")["family"] != "кабель"


def test_cable_channel_dims_are_dimensions_not_cores():
    r = process("Кабель-канал (короб) 16х16 мм")
    names = {a["name"] for a in r["attributes"]}
    assert "dims" in names and "cores_section" not in names


def test_unknown_type_is_none_not_a_guess():
    r = process("Фиксация разрывов транзитов 220 кВ")
    assert r["type"] is None and r["head_word"] == "фиксация"


def test_leading_adjective_does_not_change_type():
    assert process("Настенный внутренний блок BM-SGN22W/MU1")["type"] == "блок кондиционера"
    assert process("Уличный источник бесперебойного питания X")["type"] == "источник бесперебойного питания"


def test_placement():
    from helpers import attr
    assert attr("ИБП 12В для улицы", "placement") == "outdoor"
    assert attr("Уличный источник бесперебойного питания", "placement") == "outdoor"
    assert attr("Труба, наружный диаметр 25 мм", "placement") is None          # «наружный диаметр» ≠ outdoor


def test_bare_mm_means_diameter_only_for_pipes():
    from helpers import attr
    assert attr("труба гофра ПВХ 25 мм", "diameter_mm") == 25
    assert attr("Профиль 25 мм", "diameter_mm") is None
    assert attr("Профиль 25 мм", "bare_mm") == 25


# ---------------------------------------------------------------- flags
@pytest.mark.parametrize("text", [
    "Перевозка длинномерных грузов I класса трубоплетевозом грузоподъемностью 12 т на расстояние до 7 км",
    "Погрузка при автомобильных перевозках прочих материалов, деталей",
    "Разгрузка при автомобильных перевозках изделий",
])
def test_services(text):
    assert process(text)["is_service"] is True


def test_material_mentioning_transport_is_not_a_service():
    assert process("Песок природный II класс, средний (перевозка на расстояние 10 км)")["is_service"] is False


def test_adjustments():
    assert process("Надбавка на водонепроницаемость W6 к смеси бетонной")["is_adjustment"] is True
    assert process("Cвыше 200 км добавлять на каждый последующий 1 км")["is_adjustment"] is True
    assert process("Кабель КВВГнг-LS 3х1,5")["is_adjustment"] is False


def test_composite_header_and_inline():
    assert process("Шкаф оборудования ЛВС в составе:")["composite"] == "header"
    assert process("Набор инструмента в составе: 1)- зачистка проводов")["composite"] == "inline"
    assert process("Шинная опора ШОП-35 комплектно с: опорной стойкой")["composite"] is None
    assert process("Кабель КВВГнг-LS 3х1,5")["composite"] is None


# ---------------------------------------------------------------- type is a signal: specific vs broad vs unknown
def test_type_specificity_flag():
    assert process("Кабель контрольный КВВГнг-LS 3х1,5")["type_specific"] is True        # multiword
    assert process("камера IP уличная 2 Мп")["type_specific"] is True                     # concrete single word
    assert process("Шкаф ШК-FA.4")["type_specific"] is False                              # broad family «шкаф»
    assert process("Комплект монтажа MK AL-350FB")["type_specific"] is False
    assert process("Фиксация разрывов транзитов 220 кВ")["type_specific"] is False        # unknown type


def test_type_unknown_for_a_large_share_is_expected_not_an_error():
    # the search must work with type=None: it is a normal state, not a failure
    r = process("Крепежное изделие МС3 для подвески панелей 200 мм")
    assert r["type"] is None and r["attributes"]


# ---------------------------------------------------------------- found on the dev control queries (stage 6)
def test_user_may_put_the_adjective_before_the_noun():
    """The catalog says «Кабель контрольный …»; a user writes «контрольный кабель …». Only QUERIES are searched past the first words."""
    q = process("контрольный кабель КВВГЭнг LS 4*2.5", query=True)
    assert q["type"] == "кабель контрольный" and q["family"] == "кабель"
    names = {a["name"] for a in q["attributes"]}
    assert "cores_section" in names and "dims" not in names                  # 4*2.5 is cores×section, not a size
    assert process("перфорированный лоток 100*300", query=True)["type"] == "лоток"
    assert process("крышный вентилятор DVN", query=True)["type"] == "вентилятор"


def test_catalog_rows_are_never_typed_by_the_shifted_search():
    """The shift is for queries only: a catalog name that starts with an adjective stays untyped instead of being mistyped."""
    row = process("Магистральный лоток", query=False)
    assert row["type"] is None


def test_synonyms_found_on_dev_queries():
    assert process("патчкорд оптический 50/125", query=True)["type"] == "патч-корд"       # general type: the candidate is more specific
    assert process("оптический патчкорд 50/125", query=True)["type"] == "патч-корд оптический"
    assert process("электроконвектор Noirot 1500 Вт", query=True)["type"] == "обогреватель"
    toks = process("оптический патч-корд")["tokens"]
    assert "оптоволоконный" in toks


def test_material_extraction():
    from helpers import attr
    assert attr("Труба гофрированная из ПВХ d=25 мм", "material") == ["ПВХ"]
    assert attr("Труба ПНД гибкая", "material") == ["полиэтилен"]
    assert attr("Воздуховоды из нержавеющей стали", "material") == ["нержавеющая сталь", "сталь"]
    assert attr("Трубы с шиной", "material") is None
