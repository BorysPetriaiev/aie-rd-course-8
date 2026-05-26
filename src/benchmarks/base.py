from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class VectorDB(ABC):
    """Shared interface for FAISS / Qdrant / Chroma / pgvector."""

    @abstractmethod
    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build index from vectors.
        vectors: shape (N, dim), float32, L2-normalised for cosine similarity
        ids: string IDs parallel to vectors
        """

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find top-K nearest vectors.
        query_vec: shape (dim,) — 1D
        Returns: [(doc_id, score), ...] of length top_k
        """

    @abstractmethod
    def disk_size_mb(self) -> float:
        """Index size on disk in MB (0 if in-memory only)."""

    def cleanup(self) -> None:
        """Close connections / remove temp files. No-op by default."""
        pass
