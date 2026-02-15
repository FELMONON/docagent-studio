from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VectorIndex:
    chunk_ids: np.ndarray  # shape [n], int64
    embeddings: np.ndarray  # shape [n, d], float32, L2-normalized


def paths_for_db(db_path: str | os.PathLike[str]) -> tuple[Path, Path]:
    db = Path(db_path)
    return Path(str(db) + ".chunk_ids.npy"), Path(str(db) + ".embeddings.npy")


def save(db_path: str | os.PathLike[str], *, chunk_ids: np.ndarray, embeddings: np.ndarray) -> None:
    ids_path, emb_path = paths_for_db(db_path)
    ids_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(ids_path, chunk_ids.astype(np.int64), allow_pickle=False)
    np.save(emb_path, embeddings.astype(np.float32), allow_pickle=False)


def load(db_path: str | os.PathLike[str]) -> VectorIndex:
    ids_path, emb_path = paths_for_db(db_path)
    if not ids_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"Vector index not found. Expected {ids_path.name} and {emb_path.name} next to {Path(db_path).name}."
        )

    chunk_ids = np.load(ids_path, allow_pickle=False)
    embeddings = np.load(emb_path, allow_pickle=False)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return VectorIndex(chunk_ids=chunk_ids, embeddings=embeddings)


def topk_cosine(index: VectorIndex, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
    """Return [(chunk_id, cosine_sim)] sorted best-first."""
    if index.embeddings.size == 0:
        return []

    q = query_vec.astype(np.float32)
    sims = index.embeddings @ q  # [n]

    k = int(max(1, min(k, sims.shape[0])))
    # argpartition is O(n)
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_sorted = top_idx[np.argsort(-sims[top_idx])]

    out: list[tuple[int, float]] = []
    for i in top_sorted:
        out.append((int(index.chunk_ids[i]), float(sims[i])))
    return out
