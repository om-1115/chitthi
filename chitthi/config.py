"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("CHITTHI_DATA", "data")))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "gemma4:e4b"))
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", "gemma4:e4b"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "embeddinggemma"))
    num_ctx: int = field(default_factory=lambda: int(_env("NUM_CTX", "8192")))
    # "fake" swaps in a deterministic offline stub (used by tests / UI work without Ollama).
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "ollama"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chitthi.db"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"


def get_settings() -> Settings:
    return Settings()
