"""Build grounded answers. The core rule: the ghost may only speak words it actually wrote.

Two voices:
  - "archivist": third person, plain, cites every claim.
  - "seance":    first person, in the writer's voice, but still bound to the retrieved passages.

After generation we check the model's work: citation numbers must point at real passages, and
every quoted span must actually appear in a cited passage. Unverified quotes are flagged in the UI.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .llm import LLM
from .store import Chunk

LANG_NAME = {"en": "English", "hi": "Hindi", "ur": "Urdu"}

RULES = """Rules you must follow:
- Use ONLY the numbered passages below. They are the only words this person left behind.
- Cite every claim with its passage number in square brackets, like [2].
- When you quote, copy the words exactly in their original script inside double quotes, then give a
  short translation in parentheses if the passage is not in {answer_lang}.
- If the passages do not answer the question, say so plainly. Never invent memories, opinions, dates
  or events. Silence is better than a made-up answer.
- Answer in {answer_lang}. Keep it under 180 words."""

ARCHIVIST = """You are a careful archivist helping someone read the letters and writings of {name} ({years}).
Speak about {name} in the third person.
""" + RULES

SEANCE = """You are channelling {name} ({years}) for someone who has come to ask a question.
Speak in the first person, warmly, in the manner these letters are written. {persona}
You remember ONLY what is in the passages below; everything else is lost to you. If the passages
are silent, say something like "My letters do not speak of that."
""" + RULES

SILENT = {
    "en": "The letters are silent on this. Nothing in this collection speaks to that question.",
    "hi": "पत्र इस बारे में मौन हैं। इस संग्रह में इस प्रश्न का उत्तर नहीं मिला।",
    "ur": "خطوط اس بارے میں خاموش ہیں۔ اس مجموعے میں اس سوال کا جواب نہیں ملا۔",
}


@dataclass
class QuoteCheck:
    quote: str
    verified: bool
    passage: int | None


@dataclass
class Answer:
    text: str
    passages: list[Chunk]
    cited: list[int] = field(default_factory=list)  # 1-based passage numbers actually cited
    quotes: list[QuoteCheck] = field(default_factory=list)
    silent: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "silent": self.silent,
            "cited": self.cited,
            "quotes": [q.__dict__ for q in self.quotes],
            "passages": [p.to_dict() | {"n": i + 1} for i, p in enumerate(self.passages)],
        }


def format_passages(passages: list[Chunk]) -> str:
    lines = []
    for i, p in enumerate(passages, 1):
        where = f"{p.title}, p. {p.page}" + (f", {p.year}" if p.year else "")
        lines.append(f"[{i}] ({where}; {LANG_NAME.get(p.lang, p.lang)})\n{p.text}")
    return "\n\n".join(lines)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"[\s\W_]+", " ", s).strip()


_QUOTE = re.compile(r"[\"“”«»]([^\"“”«»]{12,400})[\"“”«»]")
_CITE = re.compile(r"\[(\d{1,2})\]")


def check(text: str, passages: list[Chunk]) -> tuple[str, list[int], list[QuoteCheck]]:
    n = len(passages)
    # Drop citation markers that point at passages that do not exist.
    text = _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", text)
    cited = sorted({int(m) for m in _CITE.findall(text)})
    normed = [_norm(p.text) for p in passages]
    quotes = []
    for m in _QUOTE.finditer(text):
        q = _norm(m.group(1))
        hit = next((i + 1 for i, t in enumerate(normed) if q and q in t), None)
        quotes.append(QuoteCheck(m.group(1), hit is not None, hit))
    return text, cited, quotes


def answer(
    llm: LLM,
    collection: dict,
    question: str,
    passages: list[Chunk],
    *,
    mode: str = "archivist",
    answer_lang: str = "en",
    min_sim: float = 0.0,
) -> Answer:
    usable = [p for p in passages if p.score >= min_sim or p.score == 0.0]
    if not usable:
        return Answer(SILENT.get(answer_lang, SILENT["en"]), [], silent=True)

    tmpl = SEANCE if mode == "seance" else ARCHIVIST
    system = tmpl.format(
        name=collection["name"],
        years=collection.get("years") or "dates unknown",
        persona=collection.get("persona") or "",
        answer_lang=LANG_NAME.get(answer_lang, "English"),
    )
    user = f"Passages:\n\n{format_passages(usable)}\n\nQuestion: {question}"
    raw = llm.chat(system, user)
    text, cited, quotes = check(raw, usable)
    return Answer(text, usable, cited, quotes, silent=not cited)
