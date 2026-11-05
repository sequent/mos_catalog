"""The whole search of the scheme in one object: B1 (query) → B2 (candidates) → B3/B4 (compatibility) → B5 (ranking) → B6 (verdict) → B7 (explanation)."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from .matching import compare, load_rules
from .matching.evidence import candidate_unparsed, extra_attributes
from .matching.explain import LABELS, explain, fmt
from .matching.ranking import rank
from .matching.rules import Rules
from .matching.verdict import decide, match_note, verdict_label
from .record import process
from .retrieval import Retriever, embed_catalog, load_records
from .retrieval.dense import E5Encoder


MAX_TOP_K = 5                       # the assignment: «до 5 позиций»; one limit for the engine, the CLI and the API


class SearchEngine:
    def __init__(self, records: list[dict], retriever: Retriever, rules: Rules | None = None, mode: str = "hybrid",
                 type_gate: str = "soft"):
        self.type_gate = type_gate            # «soft» is the design; «hard» exists only for the ablation (drops candidates of another/unknown type)
        self.mode = mode                      # «bm25» works without the model (tests); «hybrid» is the real configuration
        self.records = records
        self.retriever = retriever
        self.rules = rules or load_rules()

    @classmethod
    def build(cls, catalog_path: Path | str | None = None, rules_path: Path | str | None = None) -> "SearchEngine":
        """Loads the catalog, the local model and the cached embeddings (run scripts/build_index.py first)."""
        from .manifest import CATALOG_PATH, require_fresh_catalog
        require_fresh_catalog(catalog_path or CATALOG_PATH)         # a catalog built from another dictionary or code is refused, with the command
        records = load_records(catalog_path)
        encoder = E5Encoder()
        vectors = embed_catalog(records, encoder)
        return cls(records, Retriever(records, vectors, encoder), load_rules(rules_path))

    # ------------------------------------------------------------ helpers
    def _query_summary(self, raw: str, q: dict, pool: str) -> dict:
        attrs = [{"attribute": a["name"], "label": LABELS.get(a["name"], a["name"]),
                  "value": a.get("extra", {}).get("display", fmt(a["name"], a["value"]))}
                 for a in q["attributes"] if a["source"] == "main" and a["name"] != "model_codes"]
        codes = next((a for a in q["attributes"] if a["name"] == "model_codes" and a["source"] == "main"), None)
        return {"raw": raw, "search_text": q["search_text"], "type": q["type"], "type_specific": q["type_specific"],
                "family": q["family"], "attributes": attrs, "model_codes": codes["extra"].get("display", []) if codes else [],
                "pool": pool, "assumptions": q.get("assumptions", []),
                "unparsed_constraints": [{k: u[k] for k in ("raw", "reason", "severity")} for u in q.get("unparsed_constraints", []) if u["source"] == "main"]}

    def _item(self, it: dict, q: dict) -> dict:
        r = self.records[it["index"]]
        comp = it["compat"]
        ex = explain(comp, q, r)
        return {
            "id": r["id"], "name": r["raw_name"], "price": r["price"],
            "verdict": it["verdict"], "verdict_text": verdict_label(it["verdict"], comp.match_basis), "compatibility": comp.klass,
            "match_basis": comp.match_basis, "text_matched_constraints": comp.text_matched, "match_note": match_note(it["verdict"], comp.match_basis),
            "score": it["score"], "score_parts": it["score_parts"],
            "explanation": ex["text"], "comparison": ex["rows"], "reasons": comp.reasons, "conflicts": comp.conflicts,
            "candidate_extra_attributes": extra_attributes(comp, r), "candidate_unparsed_constraints": candidate_unparsed(r),
            "type": r["type"], "composite": r["composite"], "needs_review": it["verdict"] == "check",
            "retrieval": {k: it.get(k) for k in ("bm25_rank", "bm25_score", "dense_rank", "dense_score")},
        }

    def _table(self, out: dict, top_k: int) -> tuple[list[dict], str, str]:
        """The main table of the screen. Decision of the author: positions that need checking are shown IN the table,
        marked, under an honest status line («точных совпадений нет, ниже варианты для проверки»)."""
        ui = self.rules.policy.get("ui", {})
        lines = ui.get("status_lines", {})
        show = ui.get("review_in_main_table", True)
        confirmed, weak = out["results"], out["unconfirmed_all"] if show else []
        if confirmed:
            extra = weak[: max(0, min(top_k - len(confirmed), ui.get("review_rows_when_found", 2)))]
            key = "found_with_review" if extra else "found"
            default = "Найдено подходящих позиций: {n}" + (". Ниже варианты, которые требуют проверки" if extra else "")
            return confirmed + extra, "found", lines.get(key, default).format(n=len(confirmed))
        if weak:
            limit = min(top_k, ui.get("review_table_limit", top_k))          # never more rows than the limit asked for
            return weak[:limit], "review", lines.get("review", "Подтверждённых подходящих позиций не найдено. Ниже варианты для проверки")
        return [], "not_found", lines.get("not_found", "Не найдено")

    # ------------------------------------------------------------ search
    def search(self, query: str, top_k: int = 5, candidates: int = 50, pair_boosts: dict[str, float] | None = None, trace: bool = False) -> dict:
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k должен быть от 1 до {MAX_TOP_K}, получено {top_k}")
        t0 = time.perf_counter()
        q = process(query, query=True)                                                # B1
        pool = "service" if q["is_service"] or q["is_adjustment"] else "main"
        cand = self.retriever.search(query, top_k=candidates, mode=self.mode, pool=pool, q=q)    # B2
        t1 = time.perf_counter()
        items = []
        for c in cand:
            rec = self.records[c["index"]]
            c = dict(c)
            c["compat"] = compare(q, rec, self.rules)                                 # B3 + B4
            if self.type_gate == "hard" and q["type"] and c["compat"].type_status != "match":
                continue
            items.append(c)
        ranked = rank(items, self.rules, pair_boosts)                                 # B5
        out = decide(ranked, self.rules, top_k)                                       # B6
        table, display_status, status_line = self._table(out, top_k)
        t2 = time.perf_counter()
        answer = {
            "query": self._query_summary(query, q, pool),
            "status": out["status"],                                  # found = at least one CONFIRMED position
            "display_status": display_status,                          # found | review | not_found: what the screen says
            "status_line": status_line,
            "table": [self._item(it, q) for it in table],              # the main table: confirmed first, then «требует проверки»
            "status_text": "найдено" if out["status"] == "found" else "не найдено подтверждённо подходящих позиций",
            "results": [self._item(it, q) for it in out["results"]],                  # B7
            "unconfirmed": [self._item(it, q) for it in out["unconfirmed"]],
            "nearest_rejected": self._item(out["nearest_rejected"], q) if out["nearest_rejected"] else None,
            "rules_version": self.rules.version,
            "rules_status": self.rules.status,
            "diagnostics": {"candidates": len(cand), "classes": dict(Counter(it["compat"].klass for it in ranked)),
                            "retrieval_ms": round((t1 - t0) * 1000, 1), "matching_ms": round((t2 - t1) * 1000, 1)},
        }
        if trace:
            answer["trace"] = self._trace(pool, cand, ranked, out, table)         # diagnostics only: nothing above depends on it
        return answer

    def _trace(self, pool: str, cand: list[dict], ranked: list[dict], out: dict, table: list[dict]) -> dict:
        """What happened to EVERY candidate of B2 (position in B2, class and reasons of B4, verdict of B6, whether it reached the screen).
        An evaluator maps the ids of its reference rows onto this AFTER the search: the engine knows nothing about them."""
        pos = {self.records[c["index"]]["id"]: i for i, c in enumerate(cand, 1)}
        shown = {it["index"] for it in table}
        confirmed = {it["index"] for it in out["results"]}
        rows = []
        for it in ranked:
            r, comp = self.records[it["index"]], it["compat"]
            rows.append({"id": r["id"], "b2_position": pos.get(r["id"]), "bm25_rank": it.get("bm25_rank"), "dense_rank": it.get("dense_rank"),
                         "class": comp.klass, "verdict": it["verdict"], "reasons": list(comp.reasons), "conflicts": list(comp.conflicts),
                         "composite": r["composite"], "type_relation": comp.type_relation, "score": it["score"],
                         "in_results": it["index"] in confirmed, "on_screen": it["index"] in shown,
                         "comparisons": [{"attribute": c.name, "status": c.status} for c in comp.comparisons]})
        return {"pool": pool, "b2_candidates": [self.records[c["index"]]["id"] for c in cand], "ranked": rows}
