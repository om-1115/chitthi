"""SQLite storage: collections (one per ancestor/author), documents, passages, FTS5 + embeddings."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    years TEXT,
    persona TEXT,           -- short voice notes used in séance mode
    meta TEXT               -- JSON: anything else
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    collection TEXT NOT NULL REFERENCES collections(slug),
    title TEXT NOT NULL,
    lang TEXT NOT NULL,
    year TEXT,
    source_url TEXT,
    rights TEXT,
    path TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS pages (
    doc_id INTEGER NOT NULL REFERENCES documents(id),
    page INTEGER NOT NULL,
    method TEXT,
    text TEXT,
    image TEXT,             -- relative path of the rendered scan
    PRIMARY KEY (doc_id, page)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id),
    page INTEGER NOT NULL,
    ord INTEGER NOT NULL,
    script TEXT,
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 0'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


@dataclass
class Chunk:
    id: int
    doc_id: int
    page: int
    text: str
    script: str
    title: str
    lang: str
    year: str | None
    source_url: str | None
    image: str | None
    score: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._vec_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # --- writes ---------------------------------------------------------------
    def upsert_collection(self, slug: str, name: str, years: str = "", persona: str = "", **meta) -> None:
        self.db.execute(
            "INSERT INTO collections(slug,name,years,persona,meta) VALUES(?,?,?,?,?) "
            "ON CONFLICT(slug) DO UPDATE SET name=excluded.name, years=excluded.years, "
            "persona=excluded.persona, meta=excluded.meta",
            (slug, name, years, persona, json.dumps(meta, ensure_ascii=False)),
        )
        self.db.commit()

    def add_document(self, collection: str, title: str, lang: str, path: str, **kw) -> int:
        old = self.db.execute("SELECT id FROM documents WHERE path=?", (path,)).fetchone()
        if old:  # re-ingest replaces the previous version
            self.delete_document(old["id"])
        cur = self.db.execute(
            "INSERT INTO documents(collection,title,lang,year,source_url,rights,path) VALUES(?,?,?,?,?,?,?)",
            (collection, title, lang, kw.get("year"), kw.get("source_url"), kw.get("rights"), path),
        )
        self.db.commit()
        self._vec_cache.pop(collection, None)
        return int(cur.lastrowid)

    def delete_document(self, doc_id: int) -> None:
        self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.db.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.db.commit()
        self._vec_cache.clear()

    def add_page(self, doc_id: int, page: int, method: str, text: str, image: str | None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO pages(doc_id,page,method,text,image) VALUES(?,?,?,?,?)",
            (doc_id, page, method, text, image),
        )

    def add_chunks(self, doc_id: int, rows: list[tuple[int, str, str, list[float]]]) -> None:
        """rows: (page, script, text, embedding)"""
        for i, (page, script, text, emb) in enumerate(rows):
            v = np.asarray(emb, dtype=np.float32)
            v /= np.linalg.norm(v) or 1.0
            self.db.execute(
                "INSERT INTO chunks(doc_id,page,ord,script,text,embedding) VALUES(?,?,?,?,?,?)",
                (doc_id, page, i, script, text, v.tobytes()),
            )
        self.db.commit()
        self._vec_cache.clear()

    # --- reads ----------------------------------------------------------------
    def collections(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT c.*, COUNT(DISTINCT d.id) AS docs, COUNT(k.id) AS passages FROM collections c "
            "LEFT JOIN documents d ON d.collection=c.slug LEFT JOIN chunks k ON k.doc_id=d.id "
            "GROUP BY c.slug ORDER BY c.name"
        ).fetchall()
        return [dict(r) | {"meta": json.loads(r["meta"] or "{}")} for r in rows]

    def collection(self, slug: str) -> dict | None:
        r = self.db.execute("SELECT * FROM collections WHERE slug=?", (slug,)).fetchone()
        return dict(r) if r else None

    def documents(self, collection: str) -> list[dict]:
        rows = self.db.execute("SELECT * FROM documents WHERE collection=? ORDER BY year, title", (collection,))
        return [dict(r) for r in rows]

    _CHUNK_SELECT = (
        "SELECT k.id, k.doc_id, k.page, k.text, k.script, d.title, d.lang, d.year, d.source_url, p.image "
        "FROM chunks k JOIN documents d ON d.id=k.doc_id "
        "LEFT JOIN pages p ON p.doc_id=k.doc_id AND p.page=k.page "
    )

    def chunks_by_id(self, ids: list[int]) -> dict[int, Chunk]:
        if not ids:
            return {}
        q = self._CHUNK_SELECT + f"WHERE k.id IN ({','.join('?' * len(ids))})"
        return {r["id"]: Chunk(**dict(r)) for r in self.db.execute(q, ids)}

    def fts(self, collection: str, query: str, limit: int) -> list[tuple[int, float]]:
        rows = self.db.execute(
            "SELECT k.id, bm25(chunks_fts) AS s FROM chunks_fts JOIN chunks k ON k.id=chunks_fts.rowid "
            "JOIN documents d ON d.id=k.doc_id WHERE chunks_fts MATCH ? AND d.collection=? "
            "ORDER BY s LIMIT ?",
            (query, collection, limit),
        ).fetchall()
        return [(r["id"], -r["s"]) for r in rows]

    def vectors(self, collection: str) -> tuple[np.ndarray, np.ndarray]:
        if collection not in self._vec_cache:
            rows = self.db.execute(
                "SELECT k.id, k.embedding FROM chunks k JOIN documents d ON d.id=k.doc_id "
                "WHERE d.collection=? AND k.embedding IS NOT NULL",
                (collection,),
            ).fetchall()
            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            mat = (
                np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                if rows
                else np.zeros((0, 1), dtype=np.float32)
            )
            self._vec_cache[collection] = (ids, mat)
        return self._vec_cache[collection]
