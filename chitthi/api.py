"""HTTP API + static UI.   uvicorn chitthi.api:app --reload"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer import answer
from .config import get_settings
from .llm import get_llm
from .retrieve import search
from .store import Store

WEB = Path(__file__).resolve().parent.parent / "web"


class AskIn(BaseModel):
    collection: str
    question: str = Field(min_length=2, max_length=500)
    mode: str = Field("archivist", pattern="^(archivist|seance)$")
    lang: str = Field("en", pattern="^(en|hi|ur)$")


def create_app() -> FastAPI:
    s = get_settings()
    store, llm = Store(s.db_path), get_llm(s)
    s.pages_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Chitthi")
    app.mount("/pages", StaticFiles(directory=s.pages_dir), name="pages")

    @app.get("/api/health")
    def health():
        return {"ok": True, "backend": s.llm_backend, "chat_model": s.chat_model, "embed_model": s.embed_model}

    @app.get("/api/collections")
    def collections():
        return [c | {"documents": store.documents(c["slug"])} for c in store.collections()]

    @app.post("/api/ask")
    def ask(body: AskIn):
        col = store.collection(body.collection)
        if not col:
            raise HTTPException(404, "unknown collection")
        passages = search(store, llm, body.collection, body.question, k=s.top_k)
        return answer(llm, col, body.question, passages, mode=body.mode, answer_lang=body.lang).to_dict()

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


app = create_app()
