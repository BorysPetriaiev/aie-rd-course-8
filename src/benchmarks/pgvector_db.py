from typing import List, Tuple

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .base import VectorDB

BATCH_SIZE = 512
TABLE = "bench_vectors"


class PgVectorDB(VectorDB):
    """PostgreSQL + pgvector HNSW index via Docker."""

    def __init__(
        self,
        dsn: str = "postgresql://bench:bench@localhost:5432/bench",
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 40,
    ) -> None:
        self._dsn = dsn
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._conn: psycopg.Connection | None = None

    def _connect(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._dsn, autocommit=True)
            register_vector(self._conn)
        return self._conn

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        conn = self._connect()
        dim = vectors.shape[1]

        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        conn.execute(
            f"CREATE TABLE {TABLE} (id TEXT PRIMARY KEY, embedding vector({dim}))"
        )

        with conn.cursor() as cur:
            for i in range(0, len(ids), BATCH_SIZE):
                batch_ids = ids[i : i + BATCH_SIZE]
                batch_vecs = vectors[i : i + BATCH_SIZE]
                cur.executemany(
                    f"INSERT INTO {TABLE} (id, embedding) VALUES (%s, %s)",
                    [(bid, vec) for bid, vec in zip(batch_ids, batch_vecs)],
                )

        conn.execute(
            f"CREATE INDEX ON {TABLE} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {self._m}, ef_construction = {self._ef_construction})"
        )
        conn.execute(f"SET hnsw.ef_search = {self._ef_search}")

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        conn = self._connect()
        q = query_vec.astype(np.float32)
        rows = conn.execute(
            f"SELECT id, 1 - (embedding <=> %s) AS score FROM {TABLE} "
            f"ORDER BY embedding <=> %s LIMIT %s",
            (q, q, top_k),
        ).fetchall()
        return [(row[0], float(row[1])) for row in rows]

    def disk_size_mb(self) -> float:
        conn = self._connect()
        row = conn.execute(
            "SELECT pg_total_relation_size($1) / 1024.0 / 1024.0", (TABLE,)
        ).fetchone()
        return float(row[0]) if row else 0.0

    def cleanup(self) -> None:
        if self._conn and not self._conn.closed:
            try:
                self._conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
            except Exception:
                pass
            self._conn.close()
