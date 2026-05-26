from typing import List, Tuple

import faiss
import numpy as np

from .base import VectorDB


class FaissFlat(VectorDB):
    """FAISS IndexFlatIP — exact search, 100% recall baseline."""

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._ids: List[str] = []

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors.astype(np.float32))
        self._ids = list(ids)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        q = query_vec.astype(np.float32).reshape(1, -1)
        scores, indices = self._index.search(q, top_k)
        return [(self._ids[i], float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    def disk_size_mb(self) -> float:
        return 0.0
