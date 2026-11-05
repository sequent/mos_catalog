"""Downloads the local embedding model (multilingual-e5-small, ONNX, ≈470 MB) into models/ at a FIXED revision.

The model is not stored in the repository. After the download the prototype works without any external service.
The script also writes models/multilingual-e5-small/model_manifest.json (revision and SHA-256 of every file): the index of the
catalog is bound to it, so a different model or tokenizer can never be mixed in silently.
Run: .venv/bin/python -I scripts/download_model.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "intfloat/multilingual-e5-small"
# the commit of the snapshot the prototype was developed and measured with (taken from the metadata of the downloaded files)
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
TARGET = Path(__file__).resolve().parent.parent / "models" / "multilingual-e5-small"
FILES = ["onnx/model.onnx", "onnx/tokenizer.json", "onnx/config.json", "onnx/tokenizer_config.json",
         "onnx/special_tokens_map.json", "1_Pooling/config.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


files = {}
for name in FILES:
    path = Path(hf_hub_download(repo_id=REPO, filename=name, revision=REVISION, local_dir=TARGET))
    files[name] = {"sha256": sha256(path), "size": path.stat().st_size}
    print("ok", name)
(TARGET / "model_manifest.json").write_text(json.dumps({"repo": REPO, "revision": REVISION, "files": files}, indent=2) + "\n", encoding="utf-8")
print("model in", TARGET, "revision", REVISION)
