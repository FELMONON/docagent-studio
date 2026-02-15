from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from .embedder import Embedder
from . import sqlite_store
from . import vector_store


@dataclass(frozen=True)
class BuildResult:
    num_chunks: int
    embedding_dim: int


def build_indexes(
    *,
    conn: sqlite3.Connection,
    db_path: str,
    embed_model: str,
    batch_size: int = 64,
) -> BuildResult:
    sqlite_store.rebuild_fts(conn)

    rows = list(sqlite_store.iter_chunks(conn))
    chunk_ids: list[int] = []
    texts: list[str] = []
    for r in rows:
        t = str(r["text"]).strip()
        if not t:
            continue
        chunk_ids.append(int(r["chunk_id"]))
        texts.append(t)

    embedder = Embedder(embed_model)

    embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = embedder.embed_texts(batch)
        embs.append(res.vectors)

    if embs:
        embeddings = np.vstack(embs).astype(np.float32)
        dim = int(embeddings.shape[1])
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)
        dim = 0

    vector_store.save(
        db_path,
        chunk_ids=np.array(chunk_ids, dtype=np.int64),
        embeddings=embeddings,
    )

    return BuildResult(num_chunks=len(chunk_ids), embedding_dim=dim)
