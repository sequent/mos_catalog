from __future__ import annotations

from analog_search import process


def attr(text: str, name: str, source: str = "main"):
    """Value of the first attribute `name` extracted from `text` (None if absent)."""
    for a in process(text)["attributes"]:
        if a["name"] == name and a["source"] == source:
            return a["value"]
    return None


def attrs(text: str, name: str, source: str = "main") -> list:
    return [a["value"] for a in process(text)["attributes"] if a["name"] == name and a["source"] == source]


def sorted_dims(text: str):
    v = attr(text, "dims")
    return sorted(v) if v else None
