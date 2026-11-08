"""Case H05: a pair of diameters with the stiffness class after each number.

The query writes «250/400 SN8», the catalog writes the same row of diameters «250 SN8/400 SN8» (SN8 is the ring-stiffness
class). The query got `numeric_code` = «250/400» while the catalog name got none, so the series stayed «unknown» and the position
was only «проверить». The reading is narrow on purpose: «<n> SN<n>/<n> SN<n>» only — a single «SN8», «250 SN8» without the second
number and a bare «1/2» are not turned into a series, and an unknown characteristic still blocks the confirmation.
"""
import pytest

from analog_search import process
from analog_search.engine import SearchEngine
from analog_search.matching import load_rules
from analog_search.retrieval import Retriever

from helpers import attr, attrs
from test_engine import CATALOG_PATH, INDEX_PATH, DEFAULT_MODEL_DIR, real

QUERY = "Изокорсис У 250/400 SN8 труба подгоночная 1,5 м"
TARGET = "Труба подгоночная ИЗОКОРСИС У 250 SN8/400 SN8 L= 1500"
OTHER_PAIR = "Труба подгоночная ИЗОКОРСИС У 400/630 SN8 L= 1500"
NO_SERIES = "Труба подгоночная ИЗОКОРСИС L= 1500"


# ---------------------------------------------------------------- the real catalog (the case as the specialist reported it)
@pytest.fixture(scope="module")
def engine():
    return SearchEngine.build()


@real
def test_h05_the_pipe_is_confirmed_by_the_real_catalog(engine):
    a = engine.search(QUERY)
    confirmed = [x["name"] for x in a["results"]]
    assert any("250 SN8/400 SN8" in n and "1500" in n for n in confirmed), (confirmed, [x["name"] for x in a["unconfirmed"]])
    assert not any("400/630" in n for n in confirmed)


@real
def test_h05_a_different_second_diameter_is_not_confirmed_by_the_real_catalog(engine):
    a = engine.search("Изокорсис У 250/315 SN8 труба подгоночная 1,5 м")
    assert not any("250 SN8/400 SN8" in x["name"] for x in a["results"])


# ---------------------------------------------------------------- extraction
def test_the_pair_with_class_tails_is_the_same_series_as_the_plain_pair():
    assert attr(QUERY, "numeric_code") == "250/400"
    assert attr(TARGET, "numeric_code") == "250/400"
    assert attr("Труба ИЗОКОРСИС У 160 SN8/315 SN8", "numeric_code") == attr("Труба ИЗОКОРСИС У 160/315", "numeric_code") == "160/315"


def test_the_pair_with_class_tails_differs_when_a_number_differs():
    assert attr(TARGET, "numeric_code") != attr("Труба ИЗОКОРСИС У 250 SN8/315 SN8 L= 1500", "numeric_code")


@pytest.mark.parametrize("text", [
    "Труба ИЗОКОРСИС У SN8",                      # the class alone
    "Труба ИЗОКОРСИС У 250 SN8",                  # one diameter with its class, no pair
    "Труба ИЗОКОРСИС У 250 SN8/400",              # the second number has no class of its own: outside the narrow pattern
    "Труба ИЗОКОРСИС У 160 SN8/315",              # the catalog row without the second class
    "Труба ИЗОКОРСИС У SN8/SN10",                 # two classes, no diameters
])
def test_these_spellings_are_not_a_series_made_by_the_pair_rule(text):
    assert attr(text, "numeric_code") is None


def test_a_plain_slash_pair_still_reads_as_before():
    assert attr("Насос 50/125", "numeric_code") == "50/125"
    assert attr("Реле 1/2", "numeric_code") == "1/2"                      # unchanged: the pair rule does not touch a bare fraction
    assert attr("Реле SN8", "numeric_code") is None
    assert attr("Труба ИЗОКОРСИС У 400/630 SN8 L= 1500", "numeric_code") == "400/630"


def test_the_pair_is_one_reading_not_two():
    assert attrs(TARGET, "numeric_code") == ["250/400"]


# ---------------------------------------------------------------- the engine on a small catalog: nothing else moved
def _rec(text, i, price=100):
    r = process(text)
    r.update({"id": f"row-{i:06d}", "raw_name": text, "price": price})
    return r


@pytest.fixture(scope="module")
def small():
    recs = [_rec(t, i) for i, t in enumerate([TARGET, OTHER_PAIR, NO_SERIES, "Труба подгоночная ИЗОКОРСИС У 160 SN8/315"])]
    return SearchEngine(recs, Retriever(recs), load_rules(), mode="bm25")


def _names(items):
    return [x["name"] for x in items]


def test_the_query_confirms_the_row_with_class_tails(small):
    a = small.search(QUERY)
    assert TARGET in _names(a["results"]), (_names(a["results"]), _names(a["unconfirmed"]))
    assert OTHER_PAIR not in _names(a["results"])


def test_another_second_diameter_is_not_confirmed(small):
    a = small.search("Изокорсис У 250/315 SN8 труба подгоночная 1,5 м")
    assert TARGET not in _names(a["results"])


def test_an_unknown_series_still_blocks_the_confirmation(small):
    a = small.search(QUERY)
    assert NO_SERIES not in _names(a["results"])                 # the row names no series: unknown is not a match
    assert NO_SERIES in _names(a["unconfirmed"])                 # it stays a variant «to check»
