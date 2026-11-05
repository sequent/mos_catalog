"""Blind sheets for the independent judgment of technical errors.

For every query of a query file the sheet lists the candidates that a person has to judge: the expected rows and the rows that the compared
systems put first. The sheet hides WHICH system returned a row, its score and its verdict (the reviewer must judge the position, not the
system), and the order is shuffled with a fixed seed. The reviewer fills `relevance`, `critical_values`, `evidence_text`, `reason`, `author`;
the filled rows are copied to eval/judgments.yaml (format: eval/judgments.py).

The sheet is built from the SAVED results of a finished run (`--from-run reports/quality/runs/<id>`: its results.json); no system is run again,
so exporting a sheet after the final measurement does not touch the measurement.

Run: .venv/bin/python -I eval/make_blind_sheets.py --from-run RUN_DIR [--queries PATH] [--out PATH] [--seed N]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

COLUMNS = ["query_id", "query", "candidate_id", "source_row", "original_name", "price",
           "relevance (acceptable/unacceptable/unknown)", "critical_values", "evidence_text", "reason", "author"]


def build_rows(spec: dict, systems_first_rows: dict[str, list[str]], by_id: dict, seed: int = 20260604) -> list[list]:
    rnd = random.Random(seed)
    rows = []
    for q in spec["queries"]:
        ids = list(dict.fromkeys(list(q["answer_rows"]) + systems_first_rows.get(q["id"], [])))
        rnd.shuffle(ids)                                                      # the order says nothing about the system
        for cid in ids:
            r = by_id[cid]
            rows.append([q["id"], q["query"], cid, r["source_row"], r["raw_name"], r["price"], "", "", "", "", ""])
    return rows


def first_rows_from_results(results: dict, spec: dict) -> dict[str, list[str]]:
    """The first row of every compared system and the first three rows of the screen, read from a finished run (nothing is searched)."""
    first: dict[str, list[str]] = {}
    for q in spec["queries"]:
        got = []
        for system in results["systems"].values():
            row = system["all"]["per_query"].get(q["id"])
            if row and row.get("top5"):
                got.append(row["top5"][0])
        got += results["display"]["all"]["per_query"].get(q["id"], {}).get("table", [])[:3]
        first[q["id"]] = got
    return first


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, default=ROOT / "eval" / "queries.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "blind_sheets" / "sheet.csv")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--from-run", type=Path, required=True, help="run directory whose results.json supplies the rows each system put first")
    args = ap.parse_args(argv)
    from analog_search.retrieval import load_records
    spec = yaml.safe_load(args.queries.read_text(encoding="utf-8"))
    results = json.loads((args.from_run / "results.json").read_text(encoding="utf-8"))
    first = first_rows_from_results(results, spec)
    rows = build_rows(spec, first, {r["id"]: r for r in load_records()}, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"{len(rows)} строк для слепой разметки: {args.out}")


if __name__ == "__main__":
    main()
