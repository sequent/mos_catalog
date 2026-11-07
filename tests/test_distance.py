"""Distance in kilometres («до 12 км») is a characteristic: found on the unseen sheet, where the whole phrase stayed
«не удалось разобрать» and the position with «до 2 км» was never compared with the query's «до 12 км»."""
import pytest

from analog_search import process
from analog_search.matching import compare, load_rules
from analog_search.normalize import normalize_text

RULES = load_rules()
BASE = "Перевозка длинномерных грузов I класса трубоплетевозом грузоподъемностью 12 т на расстояние до "


def attrs(text, query=True):
    return {a["name"]: a["value"] for a in process(text, query=query)["attributes"]}


def comp(q, c):
    return compare(process(q, query=True), process(c), RULES)


@pytest.mark.parametrize("text,km", [("перевозка грузов до 12 км", 12), ("расстояние 1,5 км", 1.5), ("на расстояние до 2 km", 2), ("перевозка 7км", 7)])
def test_kilometres_give_a_distance(text, km):
    assert attrs(text)["distance_km"] == km


@pytest.mark.parametrize("text", ["скорость 60 км/ч", "знак 5 кмч", "поселок Ростокино", "5 км/ч"])
def test_a_speed_or_a_word_is_not_a_distance(text):
    assert "distance_km" not in attrs(normalize_text(text))


def test_a_distance_is_no_longer_an_unparsed_constraint():
    q = process("перевозка грузов до 12 км", query=True)
    assert not [u for u in q["unparsed_constraints"] if "км" in u["raw"]]


def test_the_same_distance_matches_and_another_conflicts():
    q = BASE + "12 км"
    same = comp(q, BASE + "12 км")
    other = comp(q, BASE + "2 км")
    assert [c for c in same.comparisons if c.name == "distance_km"][0].status == "match"
    assert [c for c in other.comparisons if c.name == "distance_km"][0].status == "conflict"
    assert other.klass == "incompatible"


def test_the_distance_alone_does_not_confirm_a_position():
    """A matched distance is a weak sign: «до 12 км» is written by any transport service."""
    c = comp("услуга до 12 км", "Перевозка щебня до 12 км")
    assert c.klass != "compatible"


def test_the_refusal_names_the_distance():
    from analog_search.matching.explain import LABELS, _UNITS
    assert LABELS["distance_km"] == "расстояние" and _UNITS["distance_km"] == "км"
