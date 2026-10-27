"""Model access. Everything runs through a local Ollama server; nothing leaves the machine.

`FakeLLM` is a deterministic stand-in so the pipeline, tests and UI work without Ollama.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
from typing import Protocol

import httpx

from .config import Settings

# EmbeddingGemma was trained with these task prefixes; they noticeably help retrieval.
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "


class LLM(Protocol):
    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]: ...
    def chat(self, system: str, user: str, *, images: list[bytes] | None = None) -> str: ...


class OllamaLLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.http = httpx.Client(base_url=settings.ollama_url, timeout=httpx.Timeout(600.0))

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        prefix = QUERY_PREFIX if is_query else DOC_PREFIX
        out: list[list[float]] = []
        for i in range(0, len(texts), 16):
            batch = [prefix + t for t in texts[i : i + 16]]
            r = self.http.post("/api/embed", json={"model": self.s.embed_model, "input": batch})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return out

    def chat(self, system: str, user: str, *, images: list[bytes] | None = None) -> str:
        msg: dict = {"role": "user", "content": user}
        model = self.s.chat_model
        if images:
            msg["images"] = [base64.b64encode(b).decode() for b in images]
            model = self.s.vision_model
        r = self.http.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "system", "content": system}, msg],
                "stream": False,
                "options": {"num_ctx": self.s.num_ctx, "temperature": 0.3},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class FakeLLM:
    """Hashed character-trigram embeddings + an echo 'model'. Script-agnostic, fully offline."""

    dims = 512

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        vecs = []
        for t in texts:
            v = [0.0] * self.dims
            s = re.sub(r"\s+", " ", t.lower())
            for j in range(len(s) - 2):
                h = int(hashlib.md5(s[j : j + 3].encode()).hexdigest(), 16)
                v[h % self.dims] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / n for x in v])
        return vecs

    def chat(self, system: str, user: str, *, images: list[bytes] | None = None) -> str:
        if images:
            return "[fake transcription]"
        m = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\[2\]|\Z)", user, re.S)
        quote = (m.group(1).strip()[:200] if m else "").replace("\n", " ")
        return f'The letters say: "{quote}" [1]' if quote else "The letters are silent on this."


def get_llm(settings: Settings) -> LLM:
    return FakeLLM() if settings.llm_backend == "fake" else OllamaLLM(settings)
