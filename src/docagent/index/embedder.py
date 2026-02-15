from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray  # shape [n, d], float32, L2-normalized


class Embedder:
    def __init__(self, model_name: str):
        # Import here so CLI can still run ingest without embedding deps.
        from fastembed import TextEmbedding  # type: ignore

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=np.zeros((0, 0), dtype=np.float32))

        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        vectors = _l2_normalize(vectors)
        return EmbeddingResult(vectors=vectors)

    def embed_query(self, query: str) -> np.ndarray:
        vec = np.array(list(self._model.embed([query]))[0], dtype=np.float32)
        vec = _l2_normalize(vec.reshape(1, -1))[0]
        return vec


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)
