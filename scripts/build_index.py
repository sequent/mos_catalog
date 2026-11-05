"""Offline step A6: embeddings of the whole catalog → data/index/dense.npy (+ meta with a hash of texts and model name).

Run: .venv/bin/python -I scripts/build_index.py [--force]
Prerequisites: scripts/prepare_catalog.py (catalog_v2.jsonl) and the local model in models/multilingual-e5-small.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from analog_search.retrieval import embed_catalog, load_records  # noqa: E402
from analog_search.retrieval.dense import E5Encoder  # noqa: E402

force = "--force" in sys.argv
records = load_records()
t0 = time.time()
vectors = embed_catalog(records, E5Encoder(), force=force, rebuild=True)
print(f"embeddings: {vectors.shape}, {time.time() - t0:.1f} s  → data/index/dense.npy")
