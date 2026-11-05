"""Manifests of the artifacts the search depends on: what the catalog was built from, what the index was embedded with.

Nothing here is a service: a manifest is a small JSON next to the artifact. At start the engine compares the hashes and refuses to run
on a stale artifact with the command that rebuilds it, instead of answering from an old catalog (the failure we had after a dictionary edit).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "processed" / "catalog_v2.jsonl"
CATALOG_MANIFEST_NAME = "catalog_manifest.json"
DICTIONARY_PATH = ROOT / "config" / "dictionaries.yaml"
RULES_PATH = ROOT / "config" / "rules.yaml"
# the modules whose behaviour decides what a catalog row contains: a change in any of them makes an old catalog stale
CATALOG_CODE = ("normalize.py", "attributes.py", "product_type.py", "flags.py", "record.py", "config.py", "source.py")
MANIFEST_FORMAT = 1


class ArtifactError(RuntimeError):
    """An artifact is missing or was built from something else than the current files."""


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hashes(files: tuple[str, ...] = CATALOG_CODE, package: Path = PACKAGE) -> dict[str, str]:
    return {name: sha256_file(package / name) for name in files}


def catalog_manifest(source: Path | str, sheet: str, rows: int, catalog: Path | str, *, dictionary: Path = DICTIONARY_PATH,
                     package: Path = PACKAGE) -> dict:
    """What a catalog was built from. `source.file` is only a name (the Cyrillic path differs between systems); the hash identifies it."""
    from .record import RECORD_SCHEMA_VERSION
    code = code_hashes(package=package)
    return {
        "format": MANIFEST_FORMAT,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "source": {"file": Path(source).name, "sha256": sha256_file(source), "sheet": sheet, "rows": rows},
        "dictionary_sha256": sha256_file(dictionary),
        "code": code,
        "code_sha256": sha256_text(json.dumps(code, sort_keys=True)),
        "catalog_sha256": sha256_file(catalog),
    }


def write_catalog_manifest(manifest: dict, catalog: Path | str) -> Path:
    path = Path(catalog).with_name(CATALOG_MANIFEST_NAME)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def check_catalog(catalog: Path | str = CATALOG_PATH, *, dictionary: Path = DICTIONARY_PATH, package: Path = PACKAGE,
                  source: Path | str | None = None) -> list[str]:
    """Reasons why the catalog may not be used; empty when it matches the current dictionary, code and (if present) source file."""
    catalog = Path(catalog)
    path = catalog.with_name(CATALOG_MANIFEST_NAME)
    if not path.exists():
        return [f"нет {path.name}: каталог создан без манифеста"]
    m = json.loads(path.read_text(encoding="utf-8"))
    from .record import RECORD_SCHEMA_VERSION
    problems = []
    if m.get("format") != MANIFEST_FORMAT:
        problems.append("формат манифеста другой")
    if m.get("record_schema_version") != RECORD_SCHEMA_VERSION:
        problems.append(f"версия схемы записи {m.get('record_schema_version')} вместо {RECORD_SCHEMA_VERSION}")
    if m.get("catalog_sha256") != sha256_file(catalog):
        problems.append("файл каталога изменён или повреждён после подготовки")
    if m.get("dictionary_sha256") != sha256_file(dictionary):
        problems.append("словарь config/dictionaries.yaml изменён после подготовки каталога")
    now = code_hashes(package=package)
    changed = sorted(k for k in now if m.get("code", {}).get(k) != now[k])
    if changed:
        problems.append("изменён код подготовки: " + ", ".join(changed))
    if source is not None and Path(source).exists() and m["source"]["sha256"] != sha256_file(source):
        problems.append("исходный xlsx изменён после подготовки каталога")
    return problems


def require_fresh_catalog(catalog: Path | str = CATALOG_PATH, **kw) -> None:
    problems = check_catalog(catalog, **kw)
    if problems:
        raise ArtifactError("Каталог устарел: " + "; ".join(problems) + ". Выполните: .venv/bin/python -I scripts/prepare_catalog.py "
                            "и .venv/bin/python -I scripts/build_index.py (в Docker: docker compose up --build)")


def git_state(root: Path = ROOT) -> dict:
    """The commit a measurement was made on and whether the working tree had uncommitted changes (then the commit alone is not the code)."""
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": head, "uncommitted_changes": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def run_manifest(rules: Path | str = RULES_PATH, **extra) -> dict:
    """What produced a particular measurement or answer: the configuration and the environment, to be stored beside the result."""
    libs = {}
    for name in ("numpy", "onnxruntime", "tokenizers", "PyYAML", "openpyxl", "fastapi"):
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libs[name] = None
    cat = CATALOG_PATH.with_name(CATALOG_MANIFEST_NAME)
    idx = ROOT / "data" / "index" / "dense_meta.json"
    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "libraries": libs,
        "rules_sha256": sha256_file(rules), "dictionary_sha256": sha256_file(DICTIONARY_PATH),
        "catalog_manifest": json.loads(cat.read_text(encoding="utf-8")) if cat.exists() else None,
        "dense_meta": json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else None,
        "env": {k: os.environ[k] for k in ("CATALOG_XLSX",) if k in os.environ}, "git": git_state(), **extra,
    }
