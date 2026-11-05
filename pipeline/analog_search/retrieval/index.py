"""Candidate generation (step B2): words + meaning, merged by RRF; and the frozen «simple word search» baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..record import process
from .bm25 import BM25
from .dense import E5Encoder, dense_text
from .fusion import rrf
from .tokens import lexical_tokens, raw_tokens

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_DIR = ROOT / "data" / "index"


# ---------------------------------------------------------------- embeddings cache
INDEX_FORMAT = 2


def _texts_hash(records: list[dict], texts: list[str]) -> str:
    """Ordered hash of (id, text): a reordered or edited catalog gives another hash."""
    h = hashlib.sha256()
    for r, t in zip(records, texts):
        h.update(r["id"].encode("utf-8") + b"\x00" + t.encode("utf-8") + b"\x01")
    return h.hexdigest()


def _matrix_hash(vectors: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()


def _valid(meta: dict, vectors: np.ndarray, spec: dict | None, texts_sha: str, rows: int, dim: int | None = None) -> list[str]:
    problems = []
    if vectors.dtype != np.float32 or vectors.ndim != 2:
        problems.append("матрица не float32 или не двумерная")
    elif dim is not None and vectors.shape[1] != dim:
        problems.append(f"размерность {vectors.shape[1]} вместо {dim}")
    if meta.get("format") != INDEX_FORMAT:
        problems.append("формат индекса другой")
    if spec is not None and meta.get("spec") != spec:
        problems.append("модель или параметры эмбеддингов другие")
    if meta.get("texts_sha256") != texts_sha:
        problems.append("тексты или порядок каталога другие")
    if vectors.shape != (rows, meta.get("dim")) or meta.get("rows") != rows:
        problems.append("размер матрицы не совпадает с каталогом")
    elif meta.get("matrix_sha256") != _matrix_hash(vectors):
        problems.append("матрица изменена или повреждена")
    return problems


def embed_catalog(records: list[dict], encoder: E5Encoder | None, index_dir: Path = DEFAULT_INDEX_DIR,
                  force: bool = False, rebuild: bool = False) -> np.ndarray:
    """Embeddings of every catalog row (record order). The cache is accepted only if the embedding spec (model revision, weights,
    tokenizer, max_length, prefixes, pooling), the ordered texts, the dtype/size and the matrix itself all match.
    A missing or stale index is recomputed only when `rebuild` or `force` is set (scripts/build_index.py); a server start never
    embeds the catalog: it stops with the command to run."""
    texts = [dense_text(r) for r in records]
    texts_sha = _texts_hash(records, texts)
    spec = encoder.spec() if encoder is not None else None
    vec_path, meta_path = index_dir / "dense.npy", index_dir / "dense_meta.json"
    if not force and vec_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(vec_path)
        problems = _valid(meta, vectors, spec, texts_sha, len(texts), getattr(encoder, "dim", None))
        if not problems:
            return vectors
        if encoder is None or not (rebuild or force):
            raise RuntimeError("Dense-индекс устарел: " + "; ".join(problems) + ". Выполните scripts/build_index.py")
    if encoder is None or not (rebuild or force):
        raise RuntimeError("Dense-индекс отсутствует; выполните scripts/build_index.py (сервер сам индекс не пересчитывает)")
    vectors = encoder.encode_passages(texts)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vectors)
    meta_path.write_text(json.dumps({"format": INDEX_FORMAT, "spec": spec, "texts_sha256": texts_sha, "rows": len(texts),
                                     "dim": int(vectors.shape[1]), "dtype": str(vectors.dtype), "matrix_sha256": _matrix_hash(vectors)},
                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return vectors


# ---------------------------------------------------------------- retriever
class Retriever:
    """Two pools: `main` (equipment, materials, composites) and `service` (transport, loading, adjustments).

    Defaults are the configuration chosen on the stage-4 probes (reports/retrieval/stage4.md):
    * one `type:` token (no `fam:`): a soft signal that lifts rows of the requested type on number queries. It slightly
      favours rows that HAVE a type; that cost is not measured (see reports/retrieval/conclusions.md). Never a filter;
    * no order-sensitive size token — whether side order matters is decided by the type rule in B4;
    * BM25 b=0.3 (softer length normalization: long names with many parameters must not be buried; 0.2–0.4 is a plateau);
    * RRF weight of the meaning list 0.5: it makes the CANDIDATE SET more complete (synonyms outside the dictionary) but it
      worsens the top of the list on number queries. The order inside the 50 is therefore not final: B5 must rank by its own
      features (use `bm25_score` for numbers), not by this RRF rank.
    """

    def __init__(self, records: list[dict], vectors: np.ndarray | None = None, encoder: E5Encoder | None = None,
                 *, use_stem: bool = True, use_attrs: bool = True, ordered_dims: bool = False,
                 type_tokens: bool | str = "type", k1: float = 1.5, b: float = 0.3):
        self.records = records
        self.encoder = encoder
        self.use_stem, self.use_attrs, self.ordered_dims = use_stem, use_attrs, ordered_dims
        self.type_tokens = type_tokens
        self.pools: dict[str, dict] = {}
        for name, pred in (("main", lambda r: not (r["is_service"] or r["is_adjustment"])),
                           ("service", lambda r: r["is_service"] or r["is_adjustment"])):
            ids = [i for i, r in enumerate(records) if pred(r)]
            docs = [lexical_tokens(records[i], use_stem=use_stem, use_attrs=use_attrs, ordered_dims=ordered_dims,
                                 type_tokens=type_tokens) for i in ids]
            self.pools[name] = {"ids": ids, "bm25": BM25(docs, k1, b),
                                "vec": vectors[ids] if vectors is not None else None}

    def query_record(self, query: str) -> dict:
        return process(query, query=True)

    def search(self, query: str, top_k: int = 50, depth: int = 100, mode: str = "hybrid",
               pool: str | None = None, rrf_k: int = 60, dense_weight: float = 0.5, q: dict | None = None) -> list[dict]:
        q = q or self.query_record(query)
        pool = pool or ("service" if q["is_service"] or q["is_adjustment"] else "main")
        P = self.pools[pool]
        ids = P["ids"]
        lex: list[tuple[int, float]] = []
        den: list[tuple[int, float]] = []
        bm_all = sims = None                      # scores of EVERY row of the pool: B5 needs them for all candidates
        if mode in ("hybrid", "bm25"):
            bm_all = P["bm25"].scores(lexical_tokens(q, use_stem=self.use_stem, use_attrs=self.use_attrs,
                                                      ordered_dims=self.ordered_dims, type_tokens=self.type_tokens))
            lex = P["bm25"].top_from_scores(bm_all, depth)
        if mode in ("hybrid", "dense"):
            if P["vec"] is None or self.encoder is None:
                raise RuntimeError("Dense search needs embeddings and an encoder")
            qv = self.encoder.encode_queries([dense_text(q)])[0]
            sims = P["vec"] @ qv
            order = np.lexsort((np.arange(len(sims)), -sims))[:depth]
            den = [(int(i), float(sims[i])) for i in order]
        lex_rank = {i: r for r, (i, _) in enumerate(lex, 1)}
        den_rank = {i: r for r, (i, _) in enumerate(den, 1)}
        if mode == "hybrid":
            fused = rrf([[i for i, _ in lex], [i for i, _ in den]], k=rrf_k, weights=[1.0, dense_weight])[:top_k]
        elif mode == "bm25":
            fused = [(i, 1.0 / (rrf_k + r)) for i, r in ((i, lex_rank[i]) for i, _ in lex[:top_k])]
        else:
            fused = [(i, 1.0 / (rrf_k + r)) for i, r in ((i, den_rank[i]) for i, _ in den[:top_k])]
        out = []
        for local, score in fused:
            rec_i = ids[local]
            out.append({"index": rec_i, "id": self.records[rec_i]["id"], "score": score,
                        "bm25_rank": lex_rank.get(local), "bm25_score": float(bm_all[local]) if bm_all is not None else None,
                        "dense_rank": den_rank.get(local), "dense_score": float(sims[local]) if sims is not None else None})
        return out


class RawBM25Baseline:
    """«Simple word search» of the assignment: BM25 over raw lowercase words of ALL rows, nothing else.

    FROZEN on purpose: no normalization, stemming, synonyms, attribute tokens, types or pools.
    It is the yardstick that shows what the other steps of the scheme add.
    """

    def __init__(self, records: list[dict]):
        self.records = records
        self.bm25 = BM25([raw_tokens(r["raw_name"]) for r in records], k1=1.5, b=0.75)

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        return [{"index": i, "id": self.records[i]["id"], "score": s} for i, s in self.bm25.top(raw_tokens(query), top_k)]


def load_records(path: Path | str | None = None) -> list[dict]:
    p = Path(path) if path else ROOT / "data" / "processed" / "catalog_v2.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
