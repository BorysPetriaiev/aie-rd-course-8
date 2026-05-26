from typing import List, Tuple

import faiss
import numpy as np

from .base import VectorDB


class FaissHNSW(VectorDB):
    """FAISS IndexHNSWFlat — approximate nearest neighbour with HNSW graph."""

    def __init__(self, M: int = 32, ef_construction: int = 200, ef_search: int = 64) -> None:
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index: faiss.IndexHNSWFlat | None = None
        self._ids: List[str] = []

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        dim = vectors.shape[1]
        # HNSW uses L2 internally; vectors are already L2-normalised so
        # inner-product and L2 rankings are equivalent.
        self._index = faiss.IndexHNSWFlat(dim, self.M)
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.hnsw.efSearch = self.ef_search
        self._index.add(vectors.astype(np.float32))
        self._ids = list(ids)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        q = query_vec.astype(np.float32).reshape(1, -1)
        distances, indices = self._index.search(q, top_k)
        # HNSW returns L2 distances; convert to similarity score (lower = more similar)
        return [(self._ids[i], float(-d)) for i, d in zip(indices[0], distances[0]) if i >= 0]

    def disk_size_mb(self) -> float:
        return 0.0
