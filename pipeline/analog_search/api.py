"""HTTP face of the prototype: `POST /search`, `GET /health` and the single page `GET /`.

The engine is built ONCE at start (catalog, local model, cached embeddings) and shared by all requests.
Run:  .venv/bin/python -I -m uvicorn --app-dir pipeline analog_search.api:app --port 8000
"""
from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .engine import MAX_TOP_K, SearchEngine

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "web" / "index.html"
MAX_QUERY_LENGTH = 500


class SearchRequest(BaseModel):
    query: str = Field(description="Запрос на естественном языке, например «кабель КВВГнг-LS 7*1.5»")
    limit: int = Field(MAX_TOP_K, ge=1, le=MAX_TOP_K, description=f"Сколько позиций показать в основной таблице (1–{MAX_TOP_K})")

    @field_validator("query")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("запрос пуст")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"запрос длиннее {MAX_QUERY_LENGTH} символов")
        return v


def _plain(o: Any) -> Any:
    """JSON-safe copy: numpy scalars (scores from the model) become ordinary numbers."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, "item") and callable(o.item) and type(o).__module__ == "numpy":
        return o.item()
    return o


def _artifacts(eng: SearchEngine) -> dict:
    """Short identifiers of what this process runs on (hashes are computed once at start, never per request)."""
    from .manifest import CATALOG_PATH, RULES_PATH, sha256_file
    out = {"rules_sha256": sha256_file(RULES_PATH)[:12]}
    manifest = CATALOG_PATH.with_name("catalog_manifest.json")
    if manifest.exists():
        out["catalog_sha256"] = json.loads(manifest.read_text(encoding="utf-8"))["catalog_sha256"][:12]
    meta = CATALOG_PATH.parent.parent / "index" / "dense_meta.json"
    if meta.exists():
        out["model_revision"] = json.loads(meta.read_text(encoding="utf-8"))["spec"]["revision"][:12]
    return out


def create_app(engine: SearchEngine | None = None) -> FastAPI:
    """`engine` is injected by the tests; in production it is built at start."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
        else:
            try:
                app.state.engine = SearchEngine.build()
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Нет данных поиска ({e}). Выполните: scripts/download_model.py, scripts/prepare_catalog.py, "
                    "scripts/build_index.py (или запустите через docker compose up)") from e
        app.state.lock = threading.Lock()
        app.state.artifacts = _artifacts(app.state.engine)
        yield

    app = FastAPI(title="Поиск аналогов", version="1.0",
                  description="Поиск аналогов оборудования и материалов по базе позиций. Правила допустимых замен: config/rules.yaml.",
                  lifespan=lifespan)

    @app.get("/health", summary="Состояние сервиса")
    def health(request: Request) -> dict:
        eng: SearchEngine = request.app.state.engine
        return {"status": "ok", "rows": len(eng.records), "searchable": sum(1 for r in eng.records if not r["is_service"] and not r["is_adjustment"]),
                "rules_version": eng.rules.version, "rules_status": eng.rules.status, "mode": eng.mode,
                "artifacts": request.app.state.artifacts}

    @app.post("/search", summary="Найти аналоги")
    def search(body: SearchRequest, request: Request) -> dict:
        eng: SearchEngine = request.app.state.engine
        try:
            with request.app.state.lock:
                answer = eng.search(body.query, top_k=body.limit)
        except Exception as e:                       # the engine never raises on a text query; a defect must not leak internals
            raise HTTPException(status_code=500, detail="Ошибка поиска") from e
        answer["score_weights"] = eng.rules.policy["ranking"]["score"]["weights"]      # for the «from what the score is made» line of the page
        return _plain(answer)

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(PAGE, media_type="text/html; charset=utf-8")

    return app


app = create_app()
