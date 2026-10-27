# Chitthi · चिट्ठी

**Ask old letters questions. They answer only in their own words.**

Chitthi reads scanned letters and old books in English, Hindi and Urdu, indexes them on your own
machine, and lets you ask questions of the people who wrote them. Every answer cites the page it
came from, shows you the original scan, and flags any quote it could not find word-for-word.
No cloud, no API keys: Gemma runs locally through Ollama.

Built for the DEV Hacktoberfest 2026 "Ghost in the Machine" challenge.

**Demo:** [docs/chitthi-demo.mp4](docs/chitthi-demo.mp4) (97 s): a séance answer in English, the same
letters in Hindi, and how the quote check works. Recorded from the app running against local Gemma 4.

## How it works

```
scan (PDF / image / txt)
  │  pypdfium2 renders each page
  ├─ text layer good enough for the expected script?  → use it
  └─ otherwise → Gemma 4 (vision) transcribes in the original script   (Tesseract fallback)
  │
  ├─ split into passages, page numbers kept
  ├─ EmbeddingGemma (multilingual) vectors  +  SQLite FTS5 keywords
  │
question ─► hybrid search (RRF) ─► Gemma 4 answers from the passages only
                                     │
                                     └─► checks: citation numbers real? quotes found verbatim?
```

Two voices:
- **Archivist**: third person, plain.
- **Séance**: first person, in the writer's voice, still bound to what the letters say. If the
  letters are silent, the ghost says so.

## Setup (macOS, Apple Silicon)

```bash
brew install ollama            # or the app from ollama.com
ollama serve &                 # if the app isn't running
ollama pull gemma4:e4b         # answers + page transcription (~10 GB)
ollama pull embeddinggemma     # multilingual embeddings

# needs Python 3.11+ (tomllib); macOS's built-in python3 is too old -> brew install python@3.12
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # optional; defaults work
```

Optional Tesseract fallback: `brew install tesseract tesseract-lang && pip install pytesseract`.

## Run

```bash
python scripts/fetch_archive.py                    # download the scans in sources.toml
python -m chitthi.ingest sources sources.toml       # OCR + index (first runs use small page ranges)
uvicorn chitthi.api:app --reload                    # open http://localhost:8000
```

Add your own scan:

```bash
python -m chitthi.ingest add ~/Scans/nana-letters.pdf --collection nana --name "Nana-ji" \
  --title "Letters to Amma, 1962-70" --lang hi --pages 1-12
```

Work on the UI without Ollama: `LLM_BACKEND=fake uvicorn chitthi.api:app --reload`.

## Tests

```bash
pip install pytest && pytest -q      # uses the offline fake model
```

## Sources and rights

`sources.toml` lists the starter collections (archive.org scans): Nehru's *Letters from a Father
to His Daughter* (1929) and Premchand's Hindi translation, and Ghalib's Urdu letters. Each entry
has a rights note. Check each scan's edition before you publish screenshots or text.

## Project layout

```
chitthi/
  config.py    env settings
  llm.py       Ollama client (chat, vision, embeddings) + offline FakeLLM
  ocr.py       page rendering, text-layer check, Gemma/Tesseract transcription
  chunk.py     passage splitting (English, Hindi danda, Urdu full stop)
  store.py     SQLite schema, FTS5, embedding storage
  retrieve.py  hybrid keyword + vector search
  answer.py    grounded prompts, citation + quote verification
  ingest.py    CLI
  api.py       FastAPI + static UI
web/index.html single-file UI on Pico CSS (vendored in web/vendor/, light mode)
scripts/fetch_archive.py
```

## Challenge note

Started 27 Oct 2026, inside the challenge window. Any commits after 31 Oct 2026 11:59 PM PDT will
be listed here.

License: MIT
