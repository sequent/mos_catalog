import json

import numpy as np
import pytest

from analog_search import process
from analog_search.retrieval import RawBM25Baseline, Retriever, embed_catalog
from analog_search.retrieval.bm25 import BM25
from analog_search.retrieval.dense import DEFAULT_MODEL_DIR
from analog_search.retrieval.fusion import rrf
from analog_search.retrieval.tokens import attribute_tokens, lexical_tokens, raw_tokens, stem


def rec(text, i=0):
    r = process(text)
    r.update({"id": f"row-{i:06d}", "raw_name": text, "price": 1})
    return r


# ---------------------------------------------------------------- RRF
def test_rrf_scores_and_order():
    out = rrf([[1, 2, 3], [3, 1]], k=60)
    d = dict(out)
    assert d[1] == pytest.approx(1 / 61 + 1 / 62)
    assert d[3] == pytest.approx(1 / 63 + 1 / 61)
    assert [i for i, _ in out][0] == 1                     # best in both lists
    assert d[2] == pytest.approx(1 / 62)


def test_rrf_weights_and_tie_break():
    out = rrf([[5], [7]], k=10, weights=[1.0, 0.5])
    assert [i for i, _ in out] == [5, 7]
    tie = rrf([[9], [4]], k=10)
    assert [i for i, _ in tie] == [4, 9]                    # equal score → smaller id first (deterministic)


# ---------------------------------------------------------------- BM25
def test_bm25_ranks_rare_terms_and_is_deterministic():
    docs = [["кабель", "7x1.5"], ["кабель", "5x1.5"], ["кабель", "кабель", "кабель"], ["труба"]]
    bm = BM25(docs)
    top = bm.top(["кабель", "7x1.5"], 4)
    assert top[0][0] == 0                                   # the rare term decides
    assert 3 not in [i for i, _ in top]                     # nothing matched → not returned
    assert bm.top(["кабель", "7x1.5"], 4) == top


def test_bm25_empty_query_returns_nothing():
    assert BM25([["a"], ["b"]]).top([], 5) == []


# ---------------------------------------------------------------- tokens
def test_stem_is_light_and_safe():
    assert stem("гофрированная") == stem("гофрированные") == stem("гофрированной")
    assert stem("ip66") == "ip66" and stem("7x1.5") == "7x1.5"      # codes and numbers are never stemmed
    assert stem("кабель") == "кабел" or stem("кабель") == "кабель"   # short words stay readable either way


def test_baseline_tokens_do_not_normalize():
    """The frozen baseline must keep failing on notation: it is the yardstick."""
    assert raw_tokens("кабель 3*1.5") != raw_tokens("кабель 3х1,5")
    assert raw_tokens("Ворота 6000х2250") != raw_tokens("Ворота 2250х6000")


def test_attribute_tokens_unify_notations():
    a = lexical_tokens(process("кабель КВВГнг-LS 7*1.5"))
    b = lexical_tokens(process("Кабель контрольный КВВГнг(A)-LS 7х1,5"))
    assert "cs:7x1.5" in a and "cs:7x1.5" in b


def test_size_token_is_order_insensitive():
    a = attribute_tokens(process("ворота откатные 6000х2250"))
    b = attribute_tokens(process("Ворота откатные 2250х6000 мм"))
    assert "dims:2250x6000" in a and "dims:2250x6000" in b


def test_type_tokens_are_optional():
    r = process("Кабель контрольный КВВГнг-LS 7х1,5")
    assert any(t.startswith("type:") for t in attribute_tokens(r, type_tokens=True))
    assert not any(t.startswith(("type:", "fam:")) for t in attribute_tokens(r, type_tokens=False))


def test_query_and_catalog_use_the_same_normalization():
    """Same string through the same chain → identical tokens (otherwise «3*1.5» would never meet «3х1,5»)."""
    assert lexical_tokens(process("Кабель КВВГнг-LS 3*1.5")) == lexical_tokens(process("Кабель КВВГнг-LS 3×1,5"))


# ---------------------------------------------------------------- retriever (words only, no model)
CATALOG = [
    "Кабель контрольный КВВГнг(A)-LS 7х1,5", "Кабель контрольный КВВГнг(A)-LS 5х1,5", "Кабель контрольный КВВГнг(A)-LS 7x2,5",
    "Ворота откатные 2250х6000 мм", "Гофрированная труба из ПВХ с протяжкой d=25 мм", "Шкаф управления IP66",
    "Перевозка грузов I класса автомобилями бортовыми на расстояние до 7 км",
    "Погрузка при автомобильных перевозках прочих материалов",
]


@pytest.fixture(scope="module")
def retriever():
    return Retriever([rec(t, i) for i, t in enumerate(CATALOG)])


def test_numbers_decide_between_template_siblings(retriever):
    top = retriever.search("кабель 7*1.5", top_k=3, mode="bm25")
    assert CATALOG[top[0]["index"]].endswith("7х1,5")
    top = retriever.search("кабель 5х1,5", top_k=3, mode="bm25")
    assert CATALOG[top[0]["index"]].endswith("5х1,5")


def test_side_order_does_not_matter(retriever):
    top = retriever.search("ворота откатные 6000х2250", top_k=2, mode="bm25")
    assert CATALOG[top[0]["index"]].startswith("Ворота")


def test_pools_are_separate(retriever):
    equip = retriever.search("кабель 7х1,5", top_k=10, mode="bm25")
    assert all(not retriever.records[c["index"]]["is_service"] for c in equip)
    service = retriever.search("перевозка грузов", top_k=10, mode="bm25")
    assert service and all(retriever.records[c["index"]]["is_service"] for c in service)


def test_nothing_matches_returns_empty(retriever):
    assert retriever.search("огнетушитель порошковый", top_k=5, mode="bm25") == []


def test_dense_mode_requires_embeddings(retriever):
    with pytest.raises(RuntimeError):
        retriever.search("кабель", mode="dense")


def test_baseline_covers_all_rows_without_pools():
    b = RawBM25Baseline([rec(t, i) for i, t in enumerate(CATALOG)])
    ids = [c["index"] for c in b.search("перевозка грузов", 5)]
    assert 6 in ids                                          # the baseline does not separate services


# ---------------------------------------------------------------- embedding cache
class FakeEncoder:
    def __init__(self, spec=None):
        self.calls = 0
        self._spec = spec or {"model": "intfloat/multilingual-e5-small", "revision": "r1", "max_length": 256}

    def spec(self):
        return dict(self._spec)

    def encode_passages(self, texts):
        self.calls += 1
        return np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3)


def test_embedding_cache_hit_and_invalidation(tmp_path):
    records = [rec(t, i) for i, t in enumerate(CATALOG)]
    enc = FakeEncoder()
    v1 = embed_catalog(records, enc, index_dir=tmp_path, rebuild=True)
    v2 = embed_catalog(records, enc, index_dir=tmp_path, rebuild=True)           # served from cache: the encoder is asked for its spec only
    assert enc.calls == 1 and np.array_equal(v1, v2)
    changed = [dict(r) for r in records]
    changed[0]["main_text"] = changed[0]["main_text"] + " изменено"
    with pytest.raises(RuntimeError):                                # texts changed → cache is stale
        embed_catalog(changed, None, index_dir=tmp_path)
    meta = json.loads((tmp_path / "dense_meta.json").read_text(encoding="utf-8"))
    assert meta["rows"] == len(records) and "intfloat/multilingual-e5-small" in meta["spec"]["model"]


# ---------------------------------------------------------------- real model (skipped when it is not downloaded)
@pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "onnx" / "model.onnx").exists(), reason="local e5 model is not downloaded")
def test_e5_encoder_shapes_and_meaning():
    from analog_search.retrieval.dense import E5Encoder
    enc = E5Encoder()
    q = enc.encode_queries(["ноутбук"])
    p = enc.encode_passages(["Ноутбук HP ZBook Firefly 14", "Песок природный II класс"])
    assert q.shape == (1, 384) and p.shape == (2, 384)
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-4)
    sims = (p @ q[0])
    assert sims[0] > sims[1]
    again = enc.encode_queries(["ноутбук"])
    assert np.allclose(q, again, atol=1e-5)                          # deterministic
