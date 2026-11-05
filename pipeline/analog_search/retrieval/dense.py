"""Search by meaning: multilingual-e5-small run through ONNX Runtime on CPU (no torch).

e5 models expect the prefixes «query: » and «passage: » and mean pooling with L2 normalization.
The model is local (models/multilingual-e5-small); the prototype works without external services.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT / "models" / "multilingual-e5-small"
MODEL_NAME = "intfloat/multilingual-e5-small"
QUERY_PREFIX, PASSAGE_PREFIX = "query: ", "passage: "
MODEL_MANIFEST = "model_manifest.json"


_HASHED: dict[tuple, str] = {}       # (path, size, mtime) → sha256: a file is read once per process, not on every call


def _file_sha256(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _HASHED:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _HASHED[key] = h.hexdigest()
    return _HASHED[key]


def embedding_spec(model_dir: Path | str, max_length: int) -> dict:
    """Everything that changes a vector: model and its exact revision, the files that were hashed when it was downloaded, the
    truncation length, prefixes, pooling, normalisation, dtype. The cached matrix is valid only for an identical spec; rules and
    the comparison code are NOT part of it (changing them must not trigger embedding)."""
    manifest = Path(model_dir) / MODEL_MANIFEST
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest}: запустите scripts/download_model.py (он фиксирует ревизию модели и хеши файлов)")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    files = m["files"]
    for name, info in files.items():
        p = Path(model_dir) / name
        if not p.exists() or p.stat().st_size != info["size"]:              # fast check first: the size
            raise FileNotFoundError(f"{p}: файл модели отсутствует или другого размера; перезапустите scripts/download_model.py")
        if _file_sha256(p) != info["sha256"]:                               # then the content: a same-size edit is caught too
            raise ValueError(f"{p}: содержимое файла не совпадает с model_manifest.json (SHA-256); перезапустите scripts/download_model.py")
    return {"model": m["repo"], "revision": m["revision"], "weights_sha256": files["onnx/model.onnx"]["sha256"],
            "tokenizer_sha256": files["onnx/tokenizer.json"]["sha256"], "max_length": max_length,
            "query_prefix": QUERY_PREFIX, "passage_prefix": PASSAGE_PREFIX, "pooling": "mean", "normalize": "l2", "dtype": "float32"}


class E5Encoder:
    dim = 384                                                          # the contract of the vector size (multilingual-e5-small)

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR, max_length: int = 256, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.model_dir, self.max_length = model_dir, max_length
        self.tokenizer = Tokenizer.from_file(str(model_dir / "onnx" / "tokenizer.json"))
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / "onnx" / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str], prefix: str, batch_size: int = 32) -> np.ndarray:
        """Returns float32 array (n, dim), rows L2-normalized. Texts are batched by length for speed."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        chunks: list[np.ndarray] = []
        for s in range(0, len(order), batch_size):
            ids = order[s:s + batch_size]
            enc = self.tokenizer.encode_batch([prefix + texts[i] for i in ids])
            input_ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": input_ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(input_ids)
            hidden = self.session.run(None, feed)[0]                       # (b, seq, dim)
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
            chunks.append(pooled.astype(np.float32))
        sorted_vecs = np.vstack(chunks)
        out = np.zeros_like(sorted_vecs)
        out[order] = sorted_vecs
        return out

    def spec(self) -> dict:
        return embedding_spec(self.model_dir, self.max_length)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts, QUERY_PREFIX)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts, PASSAGE_PREFIX)


def dense_text(record: dict) -> str:
    """What is embedded: the name with canonical numbers (but case and wording kept)."""
    return record["main_text"]
