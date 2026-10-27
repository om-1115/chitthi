# Chitthi: context for Claude Code

Entry for the DEV Hacktoberfest 2026 "Ghost in the Machine" challenge.
**Submission deadline: Sun 1 Nov 2026, 12:29 PM IST** (31 Oct, 11:59 PM PDT). See `PLAN.md`.

Ask questions of old scanned letters (English, Hindi, Urdu) and get answers grounded **only** in
the letters' own words, with page citations and the original scan. Fully local: Gemma 4 via Ollama.

## Commands

Python 3.11+ is required (`tomllib`). Create the venv with `python3.12 -m venv .venv`.

```bash
source .venv/bin/activate
pytest -q                                         # offline tests (FakeLLM, no Ollama needed)
python scripts/fetch_archive.py [--only nehru]    # download scans listed in sources.toml
python -m chitthi.ingest sources sources.toml [--only nehru] [--pages 1-10] [--ocr auto|text|gemma|tesseract]
uvicorn chitthi.api:app --reload                  # http://localhost:8000
LLM_BACKEND=fake uvicorn chitthi.api:app --reload # UI work without Ollama
```

Models (env, see `.env.example`): `CHAT_MODEL` / `VISION_MODEL` = `gemma4:e4b`, `EMBED_MODEL` = `embeddinggemma`.

## Architecture

- `chitthi/ocr.py`: per page, use the PDF text layer if `script_quality()` passes for the expected
  language, else Gemma vision transcription, else Tesseract.
- `chitthi/chunk.py`: passages of ≤900 chars, never crossing pages (citations are per page).
- `chitthi/store.py`: SQLite, FTS5 (`unicode61 remove_diacritics 0`) + float32 embeddings in a BLOB;
  brute-force cosine in numpy (the corpus is small).
- `chitthi/retrieve.py`: FTS5 + vector results fused with RRF (k=60).
- `chitthi/answer.py`: prompts (archivist / séance) + `check()`, which drops citations that point at
  passages that don't exist and verifies each quoted span verbatim against the cited passages.
- `chitthi/api.py`: FastAPI; `web/index.html` is a single-file UI (no build step).

## Rules for this project

- **Grounding is the product.** Never let answers use knowledge outside the retrieved passages.
  Keep the quote check and the silent answer working; add tests when you touch `answer.py`.
- **Offline-first.** No cloud APIs, no web fonts, no CDN scripts in `web/index.html`.
- Keep text in its original script; never transliterate stored text. Hindi chunks split on `।`,
  Urdu on `۔`.
- Add or update tests in `tests/` for pipeline changes; `pytest -q` must pass before commits.
- Rights: the `sources.toml` rights notes are a starting point, not legal advice; quote letters, not modern editorial notes.

## Status (27 Oct)

Scaffold done, and 7 tests pass with FakeLLM. **Not yet run against real Ollama/Gemma or live archive.org.**
Next: run the full pipeline on Nehru (English + Hindi), compare OCR methods on Hindi pages, test Urdu
OCR on Ghalib, and decide by Wed night whether Urdu stays in.
