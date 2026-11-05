"""the scheme and its previews come from one description.

The checks read the SAME description that draws the pictures (scripts/build_scheme.py) and the generated files, so a node that is
renamed, an arrow that is moved or a claim that stops being true is noticed here, not on the defence.
"""
import hashlib
import importlib.util
import re
import subprocess
import sys
import xml.dom.minidom as minidom
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
spec = importlib.util.spec_from_file_location("build_scheme", ROOT / "scripts" / "build_scheme.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)
MAIN, DECISION, FAILURES = bs.main_scheme(), bs.decision_scheme(), bs.failure_scheme()
PAGES = [("p1", MAIN, "scheme.svg"), ("p2", DECISION, "scheme_decision.svg"), ("p3", FAILURES, "scheme_failures.svg")]
GENERATED_FILES = ("scheme.drawio", *(filename for _, _, filename in PAGES))


def edges(page):
    return {(d["src"], d["dst"]): d for k, d in page.items if k == "edge"}


E = edges(MAIN)


# ---------------------------------------------------------------- the files are consistent with the description
def test_the_files_are_well_formed_and_have_three_sheets():
    doc = minidom.parse(str(DOCS / "scheme.drawio"))
    assert [d.getAttribute("name") for d in doc.getElementsByTagName("diagram")] == [page.name for _, page, _ in PAGES]
    for _, _, f in PAGES:
        minidom.parse(str(DOCS / f))


@pytest.mark.parametrize("pid,page", [(pid, page) for pid, page, _ in PAGES])
def test_every_arrow_of_the_drawio_connects_existing_nodes_and_ids_are_unique(pid, page):
    doc = minidom.parse(str(DOCS / "scheme.drawio"))
    diagram = next(d for d in doc.getElementsByTagName("diagram") if d.getAttribute("id") == pid)
    cells = diagram.getElementsByTagName("mxCell")
    ids = [c.getAttribute("id") for c in cells]
    assert len(ids) == len(set(ids))
    known = set(ids)
    arrows = [c for c in cells if c.getAttribute("edge") == "1"]
    assert len(arrows) == sum(1 for k, _ in page.items if k == "edge")
    for c in arrows:
        assert c.getAttribute("source") in known and c.getAttribute("target") in known


@pytest.mark.parametrize("pid,page,filename", PAGES)
def test_the_svg_and_the_drawio_carry_the_same_node_titles(pid, page, filename):
    svg = (DOCS / filename).read_text(encoding="utf-8")
    drawio = (DOCS / "scheme.drawio").read_text(encoding="utf-8")
    for kind, n in page.items:
        if kind == "node" and n["title"]:
            t = bs.escape(n["title"])
            assert t in svg and t in drawio, n["title"]


def test_the_checked_in_files_equal_what_the_description_renders():
    for _, page, filename in PAGES:
        assert (DOCS / filename).read_text(encoding="utf-8") == bs.render_svg(page)
    drawio = minidom.parse(str(DOCS / "scheme.drawio"))
    for pid, page, _ in PAGES:
        expected = minidom.parseString(bs.render_drawio_page(page, pid)).documentElement
        actual = next(d for d in drawio.getElementsByTagName("diagram") if d.getAttribute("id") == pid)
        assert actual.toxml() == expected.toxml()


def test_the_generator_refuses_to_overwrite_without_the_flag():
    before = {f: hashlib.sha256((DOCS / f).read_bytes()).hexdigest() for f in GENERATED_FILES}
    for script in ("build_scheme.py",):
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / script)], capture_output=True, text=True)
        assert r.returncode != 0 and "--overwrite" in (r.stdout + r.stderr), script
    assert {f: hashlib.sha256((DOCS / f).read_bytes()).hexdigest() for f in before} == before


# ---------------------------------------------------------------- the content of the main sheet
def test_the_type_is_decided_before_the_characteristics():
    assert ("A3", "A5") in E and ("A5", "A4") in E and ("A4", "A6") in E
    assert ("A3", "A4") not in E and ("A4", "A5") not in E and ("A5", "A6") not in E          # the old, wrong order


def test_the_unparsed_guard_sits_between_the_parsing_and_the_comparison():
    assert ("B1", "PG") in E and ("PG", "B4") in E and ("B1", "B2") in E                      # retrieval is not replaced


def test_the_failure_branch_does_not_bypass_the_explanation():
    assert ("B6", "OUT") not in E
    for a, b in [("B6", "UIP"), ("UIP", "B7"), ("B7", "OUT")]:
        assert (a, b) in E
    assert not any(src == "B6" and dst in ("OUT", "B7") for (src, dst) in E)


def test_the_service_start_is_a_separate_stage_with_the_pools():
    for key in ("S1", "S2", "S3"):
        assert key in MAIN.nodes
    assert ("S3", "B2") in E and ("S3", "B3") in E and ("S3", "B4") in E
    assert "пулы" in " ".join(MAIN.nodes["S3"]["lines"])


def test_mixed_roles_are_marked_as_mixed():
    assert MAIN.nodes["B2"]["cls"] == "mixed" and MAIN.nodes["A6"]["cls"] == "mixed"
    assert "RRF" in " ".join(MAIN.nodes["B2"]["lines"])


def test_the_planned_parts_are_dashed_and_the_implemented_ones_are_not():
    planned = {"A4p", "APPR", "B8", "FB", "EVAL", "PROP", "APPR2"}
    for key in planned:
        assert bs.COLORS[MAIN.nodes[key]["cls"]][3] == 1, key
        assert "[не реализовано]" in " ".join(MAIN.nodes[key]["lines"]) or key == "APPR"
    for key in ("A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "UIP", "PG", "S1", "S2", "S3"):
        assert bs.COLORS[MAIN.nodes[key]["cls"]][3] == 0, key
    feedback = [(s, d) for (s, d), e in E.items() if s in planned | {"OUT"} and d in planned | {"B5", "RULES", "DICT"}]
    assert feedback and all(E[k]["dashed"] for k in feedback)                                   # nothing planned is drawn as working


def test_the_feedback_loop_reaches_the_ranking_the_queue_the_eval_set_and_the_human():
    for pair in [("OUT", "B8"), ("B8", "FB"), ("FB", "B5"), ("FB", "PROP"), ("FB", "EVAL"), ("PROP", "APPR2"), ("APPR2", "RULES"), ("APPR2", "DICT")]:
        assert pair in E and E[pair]["dashed"]


def test_the_consumers_of_the_two_config_files_are_marked_on_the_blocks():
    tagged = {n["key"]: set(n["tags"]) for k, n in MAIN.items if k == "node" and n["tags"]}
    assert {k for k, t in tagged.items() if "словарь" in t} == {"A3", "A5", "A4", "B1", "B4"}
    assert {k for k, t in tagged.items() if "правила" in t} == {"S2", "B3", "B4", "B5", "B6", "UIP"}
    assert "A3 A4 A5 B1 B4" in " ".join(MAIN.nodes["DICT"]["lines"])


# ---------------------------------------------------------------- the claims on the sheet are the facts of the code
def test_the_limit_on_the_sheet_is_the_limit_of_the_code():
    from analog_search.engine import MAX_TOP_K
    assert f"limit от 1 до {MAX_TOP_K}" in " ".join(MAIN.nodes["Q"]["lines"]) and f"limit ≤ {MAX_TOP_K}" in " ".join(MAIN.nodes["UIP"]["lines"])
    from analog_search.api import MAX_QUERY_LENGTH
    assert str(MAX_QUERY_LENGTH) in " ".join(MAIN.nodes["Q"]["lines"])


def test_the_weights_on_the_sheet_are_the_weights_of_the_rules():
    from analog_search.matching import load_rules
    w = load_rules().policy["ranking"]["score"]["weights"]
    text = " ".join(MAIN.nodes["B5"]["lines"])
    assert f"{w['attribute_coverage']:g}".replace(".", ",") in text and f"{w['text_relevance']:g}".replace(".", ",") in text


def test_the_pool_sizes_on_the_sheet_are_the_real_ones():
    cat = ROOT / "data" / "processed" / "catalog_v2.jsonl"
    if not cat.exists():
        pytest.skip("catalog is not built")
    import json
    rows = [json.loads(x) for x in cat.read_text(encoding="utf-8").splitlines()]
    service = sum(1 for r in rows if r["is_service"] or r["is_adjustment"])
    assert f"main {len(rows) - service} · service {service}" == MAIN.nodes["A2"]["count"]
    assert f"{len(rows)} строка" == MAIN.nodes["A1"]["count"]


def test_the_candidate_numbers_are_the_real_ones():
    from analog_search.engine import SearchEngine
    import inspect
    src = inspect.getsource(SearchEngine.search)
    assert "candidates: int = 50" in src
    assert "до 100" in " ".join(MAIN.nodes["B2"]["lines"])


def test_the_labels_of_the_result_are_the_labels_of_the_code():
    from analog_search.matching.verdict import SAME_BY_NAME_LABEL, VERDICT_LABEL
    out = " ".join(MAIN.nodes["OUT"]["lines"])
    assert VERDICT_LABEL["same"] in out and SAME_BY_NAME_LABEL in out and VERDICT_LABEL["substitute"].split()[0] in out
    from analog_search.matching import load_rules
    assert load_rules().policy["ui"]["status_lines"]["review"] in out and load_rules().policy["ui"]["status_lines"]["not_found"].strip(".") in out


# ---------------------------------------------------------------- the decision sheet
def test_the_decision_sheet_has_the_four_verdicts_and_the_identity_ground():
    from analog_search.matching.verdict import SAME_BY_NAME_LABEL, VERDICT_LABEL
    titles = {n["key"]: n["title"].lower() for k, n in DECISION.items if k == "node"}
    assert VERDICT_LABEL["same"] in titles["SAME"] and SAME_BY_NAME_LABEL in titles["SAMEN"]
    assert "reject" in titles["REJ"] and "check" in titles["CHK"] and "substitute" in titles["SUB"]


def test_the_decision_order_matches_the_code():
    e = edges(DECISION)
    assert [(s, d) for (s, d) in e] == [("START", "D1"), ("D1", "REJ"), ("D1", "D2"), ("D2", "CHK"), ("D2", "D3"), ("D3", "CHK"),
                                         ("D3", "D4"), ("D4", "SUB"), ("D4", "D5"), ("D5", "SAMEN"), ("D5", "SAME")]
    assert {(s, d): e[(s, d)]["label"].split(":")[0] for (s, d) in e if e[(s, d)]["label"]}[("D5", "SAMEN")] == "да"


# ---------------------------------------------------------------- the text fits into its box
@pytest.mark.parametrize("page", [MAIN, DECISION, FAILURES], ids=["main", "decision", "failures"])
def test_no_line_is_wider_than_its_box(page):
    """A rough width estimate (≈ 0.52 em per character); a longer line would be drawn over the border of its node."""
    for kind, n in page.items:
        if kind != "node":
            continue
        for _, line in bs._label_lines(n):
            est = len(line) * n["size"] * 0.52
            assert est <= n["w"] - 8, f"{n['key']}: «{line}» ≈ {est:.0f} px in a box of {n['w']} px"
