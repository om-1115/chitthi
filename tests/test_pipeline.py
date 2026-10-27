"""End-to-end tests with the offline FakeLLM (no Ollama needed).   pytest -q"""

from __future__ import annotations

import importlib

import pytest
from PIL import Image

from chitthi.answer import check
from chitthi.chunk import split_pages
from chitthi.config import Settings
from chitthi.ingest import ingest_file
from chitthi.llm import FakeLLM
from chitthi.ocr import detect_script, script_quality
from chitthi.retrieve import search
from chitthi.store import Store

# Invented sample letters (not real historical text) in three scripts, pages split by form feed.
SAMPLE = (
    "My dear Chhoti,\n\nThe monsoon reached Allahabad today and the mango tree by the well "
    "is heavy with fruit. Your mother sends her love.\n\nI have planted the jasmine you asked for."
    "\f"
    "प्रिय बेटी,\n\nआज गाँव में मेला लगा था। मैंने तुम्हारे लिए एक लाल चूड़ी ख़रीदी है। "
    "दादी कहती हैं कि तुम्हें पढ़ाई पूरी करनी चाहिए।"
    "\f"
    "عزیز بھائی،\n\nدلی کی گلیوں میں آج بہت سردی تھی۔ میں نے تمہارا خط تین بار پڑھا۔"
)


@pytest.fixture()
def env(tmp_path):
    s = Settings(data_dir=tmp_path, llm_backend="fake")
    store, llm = Store(s.db_path), FakeLLM()
    src = tmp_path / "letters.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    store.upsert_collection("family", "Nana-ji", "1901–1980", "Gentle and practical.")
    ingest_file(store, llm, s, src, collection="family", title="Family letters", lang="en", year="1950")
    return s, store, llm


def test_scripts():
    assert detect_script("प्रिय बेटी") == "deva"
    assert detect_script("عزیز بھائی") == "arab"
    assert detect_script("My dear") == "latn"
    assert script_quality("ab" * 30, "hi") == 0.0
    assert script_quality("नमस्ते " * 20, "hi") > 0.9


def test_chunking_keeps_pages():
    ps = split_pages([(1, "a " * 600), (2, "short tail")])
    assert {p.page for p in ps} == {1, 2}
    assert all(len(p.text) <= 900 for p in ps)


def test_ingest_and_search(env):
    s, store, llm = env
    cols = store.collections()
    assert cols[0]["passages"] == 3
    hits = search(store, llm, "family", "mango tree monsoon", k=3)
    assert hits[0].page == 1
    hits = search(store, llm, "family", "लाल चूड़ी मेला", k=3)
    assert hits[0].page == 2 and hits[0].script == "deva"
    hits = search(store, llm, "family", "دلی سردی خط", k=3)
    assert hits[0].page == 3 and hits[0].script == "arab"


def test_reingest_replaces(env):
    s, store, llm = env
    src = s.data_dir / "letters.txt"
    ingest_file(store, llm, s, src, collection="family", title="Family letters", lang="en")
    assert store.collections()[0]["passages"] == 3


def test_quote_and_citation_check(env):
    s, store, llm = env
    passages = search(store, llm, "family", "mango tree", k=2)
    text, cited, quotes = check(
        'He wrote "the mango tree by the well is heavy with fruit" [1] and "I bought you a bicycle" [1] [9]',
        passages,
    )
    assert "[9]" not in text and cited == [1]
    assert [q.verified for q in quotes] == [True, False]


def test_image_pdf_goes_through_vision(tmp_path):
    s = Settings(data_dir=tmp_path, llm_backend="fake")
    store, llm = Store(s.db_path), FakeLLM()
    pdf = tmp_path / "scan.pdf"
    Image.new("RGB", (400, 600), "white").save(pdf, "PDF")
    store.upsert_collection("x", "X")
    doc = ingest_file(store, llm, s, pdf, collection="x", title="Scan", lang="hi")
    row = store.db.execute("SELECT method, image FROM pages WHERE doc_id=?", (doc,)).fetchone()
    assert row["method"] == "gemma" and (s.pages_dir / row["image"]).exists()


def test_api(env, monkeypatch):
    s, *_ = env
    monkeypatch.setenv("CHITTHI_DATA", str(s.data_dir))
    monkeypatch.setenv("LLM_BACKEND", "fake")
    from fastapi.testclient import TestClient

    import chitthi.api as api

    client = TestClient(importlib.reload(api).app)
    assert client.get("/api/collections").json()[0]["slug"] == "family"
    r = client.post("/api/ask", json={"collection": "family", "question": "What about the mango tree?"})
    body = r.json()
    assert r.status_code == 200 and body["cited"] == [1] and body["passages"]
    assert client.post("/api/ask", json={"collection": "nope", "question": "hi there"}).status_code == 404
    assert client.get("/").status_code == 200
