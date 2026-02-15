from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass

from . import sqlite_store
from .embedder import Embedder
from .vector_store import VectorIndex, topk_cosine


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    score: float
    source_ref: str
    text: str


class HybridRetriever:
    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        db_path: str,
        embedder: Embedder,
        vector_index: VectorIndex,
        alpha: float = 0.7,
    ):
        self.conn = conn
        self.db_path = db_path
        self.embedder = embedder
        self.vector_index = vector_index
        self.alpha = float(alpha)

    def retrieve(self, query: str, k: int = 8, lexical_k: int | None = None, vector_k: int | None = None) -> list[RetrievedChunk]:
        k = int(k)
        lexical_k = int(lexical_k or max(20, k * 3))
        vector_k = int(vector_k or max(20, k * 3))

        qvec = self.embedder.embed_query(query)
        vec_hits = topk_cosine(self.vector_index, qvec, k=vector_k)
        lex_hits = sqlite_store.fts_search(self.conn, _fts_query(query), limit=lexical_k)

        # Combine scores on union of ids.
        scores: dict[int, float] = {}
        vec_map = {cid: score for cid, score in vec_hits}
        lex_map = {cid: score for cid, score in lex_hits}

        all_ids = set(vec_map) | set(lex_map)
        if not all_ids:
            return []

        for cid in all_ids:
            v = vec_map.get(cid, 0.0)
            l = lex_map.get(cid, 0.0)
            scores[cid] = self.alpha * v + (1.0 - self.alpha) * l

        top_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        ordered_ids = [cid for cid, _ in top_ids]

        rows = sqlite_store.get_chunks_by_ids(self.conn, ordered_ids)
        out: list[RetrievedChunk] = []
        for cid, sc in top_ids:
            row = next((r for r in rows if int(r["chunk_id"]) == cid), None)
            if row is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=float(sc),
                    source_ref=str(row["source_ref"]),
                    text=str(row["text"]),
                )
            )
        return out


def _fts_query(q: str) -> str:
    """
    Build an FTS5 MATCH query from arbitrary user input.

    FTS5 has its own query language; passing raw user text can cause syntax
    errors (e.g. punctuation like '?' or unmatched quotes). For this project we
    want "always works" behavior, so we fall back to a simple OR query over
    tokens.
    """

    tokens = re.findall(r"[A-Za-z0-9_]+", q.lower())
    # Keep queries bounded and avoid extremely long MATCH strings.
    tokens = [t for t in tokens if t][:20]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)
