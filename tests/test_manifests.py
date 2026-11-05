"""artifacts know what they were built from; a stale one is refused, a change of rules alone invalidates nothing."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from analog_search import process
from analog_search.manifest import (CATALOG_CODE, ArtifactError, catalog_manifest, check_catalog, code_hashes, require_fresh_catalog,
                                    run_manifest, sha256_file, write_catalog_manifest)
from analog_search.retrieval import embed_catalog
from analog_search.retrieval.dense import embedding_spec

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "multilingual-e5-small"


# ---------------------------------------------------------------- the catalog manifest
@pytest.fixture()
def workdir(tmp_path):
    """A tiny catalog with its manifest, built from a copy of the dictionary and of the code files that decide a record."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for name in CATALOG_CODE:
        shutil.copy(ROOT / "pipeline" / "analog_search" / name, pkg / name)
    dictionary = tmp_path / "dictionaries.yaml"
    shutil.copy(ROOT / "config" / "dictionaries.yaml", dictionary)
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"pretend workbook")
    catalog = tmp_path / "catalog_v2.jsonl"
    catalog.write_text(json.dumps({"id": "row-000002"}) + "\n", encoding="utf-8")
    write_catalog_manifest(catalog_manifest(source, "База позиций", 1, catalog, dictionary=dictionary, package=pkg), catalog)
    return {"catalog": catalog, "dictionary": dictionary, "package": pkg, "source": source}


def problems(w, **over):
    kw = {"dictionary": w["dictionary"], "package": w["package"], "source": w["source"]}
    kw.update(over)
    return check_catalog(w["catalog"], **kw)


def test_a_fresh_catalog_has_no_problems(workdir):
    assert problems(workdir) == []


def test_the_manifest_records_the_inputs(workdir):
    m = json.loads(workdir["catalog"].with_name("catalog_manifest.json").read_text(encoding="utf-8"))
    assert m["source"]["sha256"] == sha256_file(workdir["source"]) and m["source"]["rows"] == 1 and m["source"]["sheet"] == "База позиций"
    assert m["dictionary_sha256"] == sha256_file(workdir["dictionary"]) and set(m["code"]) == set(CATALOG_CODE)
    assert m["catalog_sha256"] == sha256_file(workdir["catalog"])


def test_an_edited_dictionary_makes_the_catalog_stale(workdir):
    workdir["dictionary"].write_text(workdir["dictionary"].read_text(encoding="utf-8") + "\n# an edit\n", encoding="utf-8")
    assert any("словарь" in p for p in problems(workdir))


def test_an_edited_extractor_makes_the_catalog_stale(workdir):
    f = workdir["package"] / "attributes.py"
    f.write_text(f.read_text(encoding="utf-8") + "\n# an edit\n", encoding="utf-8")
    assert any("attributes.py" in p for p in problems(workdir))


def test_a_changed_or_damaged_catalog_file_is_found(workdir):
    workdir["catalog"].write_text(json.dumps({"id": "row-000003"}) + "\n", encoding="utf-8")
    assert any("каталога изменён" in p for p in problems(workdir))


def test_a_changed_source_is_found_when_the_source_is_present(workdir):
    workdir["source"].write_bytes(b"another workbook")
    assert any("xlsx изменён" in p for p in problems(workdir))
    workdir["source"].unlink()
    assert not any("xlsx" in p for p in problems(workdir))                       # a runtime without the workbook still works


def test_a_catalog_without_a_manifest_is_refused(workdir):
    workdir["catalog"].with_name("catalog_manifest.json").unlink()
    assert any("без манифеста" in p for p in problems(workdir))


def test_the_refusal_names_the_command_that_fixes_it(workdir):
    workdir["dictionary"].write_text("synonyms: {}\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="prepare_catalog.py.*build_index.py|docker compose up --build"):
        require_fresh_catalog(workdir["catalog"], dictionary=workdir["dictionary"], package=workdir["package"])


def test_the_engine_refuses_a_stale_catalog_before_loading_anything(workdir):
    from analog_search.engine import SearchEngine
    workdir["catalog"].write_text("", encoding="utf-8")
    with pytest.raises(ArtifactError, match="Каталог устарел"):
        SearchEngine.build(catalog_path=workdir["catalog"])


def test_the_shipped_catalog_matches_the_current_files():
    cat = ROOT / "data" / "processed" / "catalog_v2.jsonl"
    if not cat.exists():
        pytest.skip("catalog is not built")
    assert check_catalog(cat, source=ROOT / "База_позиций_тестовое.xlsx") == []


# ---------------------------------------------------------------- the dense index
class Enc:
    def __init__(self, **spec):
        self.calls = 0
        self.s = {"model": "m", "revision": "r1", "weights_sha256": "w", "tokenizer_sha256": "t", "max_length": 256, "pooling": "mean", **spec}

    def spec(self):
        return dict(self.s)

    def encode_passages(self, texts):
        self.calls += 1
        rng = np.random.default_rng(len(texts))
        v = rng.normal(size=(len(texts), 4)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def recs(*names):
    out = []
    for i, n in enumerate(names):
        r = process(n)
        r["id"] = f"row-{i:06d}"
        out.append(r)
    return out


R = recs("Труба ПВХ 25 мм", "Кабель КВВГнг 5х1,5", "Ворота откатные 6000х2250")


def test_the_cache_is_reused_for_an_identical_spec(tmp_path):
    e = Enc()
    a = embed_catalog(R, e, tmp_path, rebuild=True)
    b = embed_catalog(R, e, tmp_path, rebuild=True)
    assert e.calls == 1 and np.array_equal(a, b)


@pytest.mark.parametrize("change", [{"revision": "r2"}, {"weights_sha256": "w2"}, {"tokenizer_sha256": "t2"}, {"max_length": 128}, {"pooling": "cls"}])
def test_a_change_of_the_model_or_its_parameters_invalidates_the_cache(tmp_path, change):
    embed_catalog(R, Enc(), tmp_path, rebuild=True)
    e2 = Enc(**change)
    embed_catalog(R, e2, tmp_path, rebuild=True)
    assert e2.calls == 1


def test_a_changed_or_reordered_catalog_invalidates_the_cache(tmp_path):
    embed_catalog(R, Enc(), tmp_path, rebuild=True)
    e = Enc()
    embed_catalog(list(reversed(R)), e, tmp_path, rebuild=True)
    assert e.calls == 1


def test_a_damaged_or_swapped_matrix_is_not_accepted(tmp_path):
    embed_catalog(R, Enc(), tmp_path, rebuild=True)
    m = np.load(tmp_path / "dense.npy")
    np.save(tmp_path / "dense.npy", m[::-1].copy())                                # rows swapped, shape intact
    e = Enc()
    embed_catalog(R, e, tmp_path, rebuild=True)
    assert e.calls == 1                                                            # recomputed, not trusted
    np.save(tmp_path / "dense.npy", m[::-1].copy())
    with pytest.raises(RuntimeError, match="матрица"):
        embed_catalog(R, None, tmp_path)                                           # no encoder: an explicit error, not wrong answers


def test_changing_the_rules_never_touches_the_embeddings(tmp_path):
    """Rules are not part of the spec: editing rules.yaml must not trigger a re-embedding."""
    e = Enc()
    embed_catalog(R, e, tmp_path, rebuild=True)
    rules = ROOT / "config" / "rules.yaml"
    assert "rules" not in json.dumps(json.loads((tmp_path / "dense_meta.json").read_text(encoding="utf-8")))
    assert e.calls == 1 and rules.exists()


def test_the_meta_describes_the_matrix(tmp_path):
    embed_catalog(R, Enc(), tmp_path, rebuild=True)
    meta = json.loads((tmp_path / "dense_meta.json").read_text(encoding="utf-8"))
    assert meta["rows"] == 3 and meta["dim"] == 4 and meta["dtype"] == "float32" and len(meta["texts_sha256"]) == 64 and len(meta["matrix_sha256"]) == 64


# ---------------------------------------------------------------- the real model
real_model = pytest.mark.skipif(not (MODEL_DIR / "model_manifest.json").exists(), reason="model manifest is missing: run scripts/download_model.py")


@real_model
def test_the_spec_pins_the_revision_and_the_files():
    s = embedding_spec(MODEL_DIR, 256)
    assert s["revision"] == "614241f622f53c4eeff9890bdc4f31cfecc418b3" and len(s["weights_sha256"]) == 64 and s["max_length"] == 256
    assert embedding_spec(MODEL_DIR, 128) != s


@real_model
def test_the_download_script_pins_the_same_revision():
    src = (ROOT / "scripts" / "download_model.py").read_text(encoding="utf-8")
    assert '"614241f622f53c4eeff9890bdc4f31cfecc418b3"' in src and "revision=REVISION" in src


@real_model
def test_same_inputs_give_the_same_vectors(tmp_path):
    from analog_search.retrieval.dense import E5Encoder
    e = E5Encoder()
    a = e.encode_passages(["Труба ПВХ 25 мм", "Кабель КВВГнг 5х1,5"])
    b = e.encode_passages(["Труба ПВХ 25 мм", "Кабель КВВГнг 5х1,5"])
    assert np.allclose(a, b, atol=1e-6)


def test_the_run_manifest_names_the_configuration_and_the_environment():
    m = run_manifest(extra_note="x")
    assert len(m["rules_sha256"]) == 64 and m["libraries"]["numpy"] and m["python"] and m["extra_note"] == "x"
    assert set(code_hashes()) == set(CATALOG_CODE)
