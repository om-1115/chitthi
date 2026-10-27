"""Hybrid retrieval: FTS5 keyword match + multilingual embeddings, fused with RRF.

Keywords catch exact names and dates; embeddings let an English question find a Hindi or Urdu
passage (EmbeddingGemma is multilingual), which matters when one collection mixes languages.
"""

from __future__ import annotations

import re

import numpy as np

from .llm import LLM
from .store import Chunk, Store

RRF_K = 60
_TOKEN = re.compile(r"[\wऀ-ॿ؀-ۿ]+", re.UNICODE)


def fts_query(question: str) -> str | None:
    terms = [t for t in _TOKEN.findall(question) if len(t) > 2]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms[:16])


def search(store: Store, llm: LLM, collection: str, question: str, k: int = 6) -> list[Chunk]:
    pool = max(k * 4, 20)
    ranked: dict[int, float] = {}
    cosine: dict[int, float] = {}

    q = fts_query(question)
    if q:
        for rank, (cid, _) in enumerate(store.fts(collection, q, pool)):
            ranked[cid] = ranked.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    ids, mat = store.vectors(collection)
    if len(ids):
        qv = np.asarray(llm.embed([question], is_query=True)[0], dtype=np.float32)
        qv /= np.linalg.norm(qv) or 1.0
        if qv.shape[0] == mat.shape[1]:
            sims = mat @ qv
            for rank, i in enumerate(np.argsort(-sims)[:pool]):
                cid = int(ids[i])
                cosine[cid] = float(sims[i])
                ranked[cid] = ranked.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    top = sorted(ranked, key=ranked.get, reverse=True)[:k]
    chunks = store.chunks_by_id(top)
    out = []
    for cid in top:
        c = chunks[cid]
        c.score = cosine.get(cid, 0.0)
        out.append(c)
    return out
