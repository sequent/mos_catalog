"""Okapi BM25 over token lists (≈50 lines, no dependency, parameters visible)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                postings[term].append((i, tf))
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, items in postings.items():
            df = len(items)
            self.idf[term] = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            ids = np.fromiter((i for i, _ in items), dtype=np.int64, count=df)
            tfs = np.fromiter((tf for _, tf in items), dtype=np.float64, count=df)
            self.postings[term] = (ids, tfs)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float64)
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term in set(query_tokens):
            p = self.postings.get(term)
            if p is None:
                continue
            ids, tfs = p
            s[ids] += self.idf[term] * tfs * (self.k1 + 1.0) / (tfs + norm[ids])
        return s

    def top(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        return self.top_from_scores(self.scores(query_tokens), k)

    def top_from_scores(self, s: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self.n)
        idx = np.argpartition(-s, k - 1)[:k] if k < self.n else np.arange(self.n)
        idx = idx[np.lexsort((idx, -s[idx]))]              # score desc, index asc: deterministic ties
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]
