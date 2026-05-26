import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .base import VectorDB

COLLECTION = "bench_quora"
BATCH_SIZE = 5000


class ChromaDB(VectorDB):
    """ChromaDB embedded persistent store."""

    def __init__(self, persist_dir: str | None = None) -> None:
        self._tmp_dir: str | None = None
        if persist_dir is None:
            self._tmp_dir = tempfile.mkdtemp(prefix="chroma_bench_")
            persist_dir = self._tmp_dir
        self._persist_dir = persist_dir
        self._collection = None

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=self._persist_dir)

        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

        self._collection = client.create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i : i + BATCH_SIZE]
            batch_vecs = vectors[i : i + BATCH_SIZE].tolist()
            self._collection.add(ids=batch_ids, embeddings=batch_vecs)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        results = self._collection.query(
            query_embeddings=[query_vec.astype(np.float32).tolist()],
            n_results=top_k,
        )
        ids_out = results["ids"][0]
        distances = results["distances"][0]
        # Chroma cosine returns distance (0=identical); convert to similarity
        return [(doc_id, 1.0 - dist) for doc_id, dist in zip(ids_out, distances)]

    def disk_size_mb(self) -> float:
        total = sum(
            f.stat().st_size for f in Path(self._persist_dir).rglob("*") if f.is_file()
        )
        return total / (1024 * 1024)

    def cleanup(self) -> None:
        if self._tmp_dir and Path(self._tmp_dir).exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
