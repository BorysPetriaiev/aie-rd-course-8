"""Recall@K, MRR@K, and latency percentile helpers."""
from typing import List

import numpy as np


def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / min(k, len(relevant))


def mrr_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def latency_percentiles(latencies_ms: np.ndarray) -> dict:
    return {
        "p50": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99": round(float(np.percentile(latencies_ms, 99)), 3),
    }
