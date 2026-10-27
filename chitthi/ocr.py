"""Turn scans into text, page by page.

Strategy per page (``mode="auto"``):
  1. Use the PDF's embedded text layer if it is clean enough for the expected script.
     (Archive.org/DLI scans often have a text layer, but Hindi/Urdu ones are frequently garbage.)
  2. Otherwise render the page and ask Gemma (vision) to transcribe it in its original script.
  3. If Ollama is unavailable and Tesseract is installed, fall back to Tesseract.
"""

from __future__ import annotations

import difflib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .llm import LLM

SCRIPT_RANGES = {
    "deva": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "arab": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "latn": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
}
LANG_SCRIPT = {"hi": "deva", "mr": "deva", "ne": "deva", "ur": "arab", "fa": "arab", "en": "latn"}
LANG_NAME = {"hi": "Hindi (Devanagari)", "ur": "Urdu (Nastaliq/Arabic script)", "en": "English"}
TESS_LANG = {"hi": "hin", "ur": "urd", "en": "eng"}

# Some PDF producers break Devanagari extraction: a syllable with ु/ू is emitted twice
# ("पुस्पुतक" for पुस्तक, "दुनिदु या" for दुनिया). The script is right, so script_quality()
# can't see it; count these echoes instead. Reduplication like कुछ-कुछ is excluded by the hyphen.
DEVA_ECHO = re.compile(r"([क-ह]़?[ुू]ं?)[^\s\-।]{0,3}\s?\1")
DEVA_WORD = re.compile(r"[\u0900-\u097F]+")

# Tesseract's hin model misses the second stroke of ै/ौ/ऐ/औ (है→हे, कैसे→केसे, और→ओर). A broken
# text layer still spells those right, so fix_matras() takes the layer's spelling wherever the two
# agree word for word up to that stroke. Such layers also drop the space after ं ("बातेंपूछा").
MATRA_FOLD = str.maketrans({"ै": "े", "ौ": "ो", "ऐ": "ए", "औ": "ओ"})
DEVA_JOINED = re.compile(r"(?<=[ंँ])(?=[ऄ-ह])")
TOKEN = re.compile(r"\S+")

TRANSCRIBE_PROMPT = """You are transcribing one page of an old scanned book or letter.
Expected language: {lang}.
Rules:
- Write out exactly the text you see, in its ORIGINAL script. Do not translate.
- Keep paragraph breaks. Skip running headers, page numbers and library stamps.
- If a word is unreadable, write [illegible]. Never guess or invent text.
- Output only the transcription, nothing else."""


@dataclass
class Page:
    number: int  # 1-based
    text: str
    image_png: bytes | None
    method: str  # "text-layer" | "gemma" | "tesseract" | "tesseract+layer" | "plain"


def detect_script(text: str) -> str:
    counts = {k: 0 for k in SCRIPT_RANGES}
    for ch in text:
        o = ord(ch)
        for k, ranges in SCRIPT_RANGES.items():
            if any(a <= o <= b for a, b in ranges):
                counts[k] += 1
                break
    best = max(counts, key=counts.get)
    return best if counts[best] else "latn"


def script_quality(text: str, lang: str) -> float:
    """Share of letter characters that belong to the expected script (0..1)."""
    want = LANG_SCRIPT.get(lang, "latn")
    letters = [c for c in text if c.isalpha() or 0x0900 <= ord(c) <= 0x097F]
    if len(letters) < 40:
        return 0.0
    ranges = SCRIPT_RANGES[want]
    ok = sum(1 for c in letters if any(a <= ord(c) <= b for a, b in ranges))
    return ok / len(letters)


def echo_rate(text: str) -> float:
    """Duplicated-syllable artifacts per Devanagari word (clean text: ~0, broken layers: 0.03+)."""
    words = len(DEVA_WORD.findall(text))
    return len(DEVA_ECHO.findall(text)) / words if words else 0.0


def layer_ok(text: str, lang: str, min_quality: float = 0.8, max_echo: float = 0.02) -> bool:
    if script_quality(text, lang) < min_quality:
        return False
    return LANG_SCRIPT.get(lang) != "deva" or echo_rate(text) < max_echo


def fix_matras(ocr: str, layer: str) -> str:
    """Restore ै/ौ in OCR text from a (possibly garbled) text layer of the same page."""
    toks = [(m.start(), m.end(), m.group()) for m in TOKEN.finditer(ocr)]
    ref = TOKEN.findall(DEVA_JOINED.sub(" ", layer))
    sm = difflib.SequenceMatcher(
        None, [t.translate(MATRA_FOLD) for *_, t in toks], [w.translate(MATRA_FOLD) for w in ref], autojunk=False
    )
    out, pos = [], 0
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            start, end, word = toks[a + k]
            if ref[b + k] != word:
                out.append(ocr[pos:start] + ref[b + k])
                pos = end
    return "".join(out) + ocr[pos:]


def _parse_range(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(1, n + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b or a)
        pages.update(range(max(1, lo), min(n, hi) + 1))
    return sorted(pages)


def _tesseract(png: bytes, lang: str) -> str | None:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(png)), lang=TESS_LANG.get(lang, "eng"))
    except Exception:
        return None


def transcribe(llm: LLM, png: bytes, lang: str) -> tuple[str, str]:
    try:
        system = TRANSCRIBE_PROMPT.format(lang=LANG_NAME.get(lang, lang))
        return llm.chat(system, "Transcribe this page.", images=[png]), "gemma"
    except Exception:
        text = _tesseract(png, lang)
        if text is None:
            raise
        return text, "tesseract"


def read_pages(
    path: Path,
    llm: LLM,
    *,
    lang: str,
    mode: str = "auto",
    pages: str | None = None,
    min_quality: float = 0.8,
) -> Iterator[Page]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        # Plain text: split on form feeds (djvu.txt) or treat as one page.
        chunks = path.read_text(encoding="utf-8", errors="replace").split("\f")
        for i in _parse_range(pages, len(chunks)):
            yield Page(i, chunks[i - 1], None, "plain")
        return
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        from PIL import Image

        buf = io.BytesIO()
        Image.open(path).convert("RGB").save(buf, format="PNG")
        png = buf.getvalue()
        text, method = transcribe(llm, png, lang) if mode != "tesseract" else (_tesseract(png, lang) or "", "tesseract")
        yield Page(1, text, png, method)
        return

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    for num in _parse_range(pages, len(pdf)):
        page = pdf[num - 1]
        layer = page.get_textpage().get_text_range()
        img = page.render(scale=2.0).to_pil()
        img.thumbnail((1600, 1600))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()

        if mode == "text" or (mode == "auto" and layer_ok(layer, lang, min_quality)):
            yield Page(num, layer, png, "text-layer")
        elif mode == "tesseract":
            text = _tesseract(png, lang) or ""
            if LANG_SCRIPT.get(lang) == "deva" and script_quality(layer, lang) >= min_quality:
                yield Page(num, fix_matras(text, layer), png, "tesseract+layer")
            else:
                yield Page(num, text, png, "tesseract")
        else:
            text, method = transcribe(llm, png, lang)
            yield Page(num, text, png, method)
