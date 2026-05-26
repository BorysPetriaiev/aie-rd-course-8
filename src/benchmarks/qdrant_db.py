from typing import List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .base import VectorDB

COLLECTION = "bench_quora"
BATCH_SIZE = 256


class QdrantDB(VectorDB):
    """Qdrant running in Docker at localhost:6333."""

    def __init__(self, url: str = "http://localhost:6333") -> None:
        self._client = QdrantClient(url=url, timeout=60)

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        dim = vectors.shape[1]

        if self._client.collection_exists(COLLECTION):
            self._client.delete_collection(COLLECTION)

        self._client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(id=idx, vector=vectors[idx].tolist(), payload={"doc_id": ids[idx]})
            for idx in range(len(ids))
        ]

        for i in range(0, len(points), BATCH_SIZE):
            self._client.upsert(collection_name=COLLECTION, points=points[i : i + BATCH_SIZE])

        self._ids = list(ids)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        results = self._client.search(
            collection_name=COLLECTION,
            query_vector=query_vec.astype(np.float32).tolist(),
            limit=top_k,
        )
        return [(hit.payload["doc_id"], hit.score) for hit in results]

    def disk_size_mb(self) -> float:
        info = self._client.get_collection(COLLECTION)
        try:
            return info.result.disk_data_size / (1024 * 1024)
        except Exception:
            return 0.0

    def cleanup(self) -> None:
        try:
            self._client.delete_collection(COLLECTION)
        except Exception:
            pass
