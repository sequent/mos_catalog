"""B6: the final verdict and the status of the answer («found» / «not_found»).

It only translates compatibility classes into verdicts and cuts the lists; it creates no new facts.
If there is no confirmed candidate («same» or «substitute») the status is «not_found», whatever the weak candidates are.
"""
from __future__ import annotations

from .rules import Rules

# The machine verdict `same` is a historical name of the API: it means «all requirements of the QUERY that could be parsed are met»,
# not «the same product». The user-facing text says exactly that; «та же позиция по данным названия» is allowed only
# when the whole normalised name of the query equals the name of the candidate (match_basis = normalized_name).
VERDICT_LABEL = {"same": "совпадает по указанным параметрам", "substitute": "допустимая замена", "check": "проверить", "reject": "не подходит"}
SAME_BY_NAME_LABEL = "та же позиция по данным названия"
MATCH_NOTE = {
    "request_constraints": "Совпадение по параметрам, указанным в запросе; остальные свойства позиции не проверялись.",
    "type_assumption": "Подобрано по типу; диапазон и точность не проверялись; действует допущение автора.",
}


def verdict_label(verdict: str, match_basis: str | None = None) -> str:
    if verdict == "same" and match_basis == "normalized_name":
        return SAME_BY_NAME_LABEL
    return VERDICT_LABEL[verdict]


def match_note(verdict: str, match_basis: str | None) -> str | None:
    """What the match does NOT prove; shown next to the verdict."""
    return MATCH_NOTE.get(match_basis) if verdict in ("same", "substitute") else None


def decide(ranked: list[dict], rules: Rules, top_k: int = 5) -> dict:
    """`ranked` — output of ranking.rank. Returns status and three lists."""
    vmap = rules.policy["verdict"]["map"]
    nf = rules.policy["not_found"]
    confirmed_set = set(nf["confirmed_verdicts"])
    need_weak = nf.get("require_weak_evidence", True)
    for it in ranked:
        it["verdict"] = vmap[it["compat"].klass]

    confirmed = [it for it in ranked if it["verdict"] in confirmed_set]
    unconfirmed = [it for it in ranked if it["verdict"] == "check" and (it["compat"].weak_evidence or not need_weak)]
    rejected = [it for it in ranked if it["verdict"] == "reject" and (it["compat"].weak_evidence or not need_weak)]
    # the nearest rejected item is the one with the FEWEST conflicts (the most useful «почти подходит»), then by score
    rejected.sort(key=lambda it: (len(it["compat"].conflicts), -it["score"], it["index"]))

    found = bool(confirmed)
    return {
        "status": "found" if found else "not_found",
        "results": confirmed[:top_k],
        "unconfirmed": unconfirmed[: nf["unconfirmed_limit"]] if nf.get("show_unconfirmed", True) else [],
        "unconfirmed_all": unconfirmed,
        "nearest_rejected": (rejected[0] if rejected and not found and nf.get("show_nearest_rejected", True) else None),
    }
