"""Split page text into passages small enough to cite, keeping their page number."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Passage:
    page: int
    text: str


_PARA = re.compile(r"\n\s*\n")
# Sentence ends for English, Hindi (danda ।), and Urdu (۔ ؟).
_SENT = re.compile(r"(?<=[.!?।॥۔؟])\s+")


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"-\n(?=\w)", "", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _pieces(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    for para in _PARA.split(text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= max_chars:
            out.append(para)
            continue
        buf = ""
        sents: list[str] = []
        for sent in _SENT.split(para):
            # OCR text often has no punctuation at all: hard-wrap on words as a last resort.
            while len(sent) > max_chars:
                cut = sent.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                sents.append(sent[:cut])
                sent = sent[cut:].strip()
            sents.append(sent)
        for sent in sents:
            if buf and len(buf) + len(sent) + 1 > max_chars:
                out.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            out.append(buf)
    return out


def split_pages(pages: list[tuple[int, str]], max_chars: int = 900, min_chars: int = 120) -> list[Passage]:
    """Pack paragraphs into passages of up to ``max_chars``; never cross a page boundary."""
    passages: list[Passage] = []
    for num, raw in pages:
        buf = ""
        for piece in _pieces(_clean(raw), max_chars):
            if buf and len(buf) + len(piece) + 2 > max_chars:
                passages.append(Passage(num, buf))
                buf = piece
            else:
                buf = f"{buf}\n\n{piece}".strip()
        if buf:
            if len(buf) < min_chars and passages and passages[-1].page == num:
                passages[-1].text += "\n\n" + buf
            else:
                passages.append(Passage(num, buf))
    return passages
