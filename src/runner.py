"""Run all 5 vector DB benchmarks and write results.csv.

Usage:
    python src/runner.py --output results/results.csv [--queries 1000]
"""
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

WARMUP_QUERIES = 50
NUM_REPEATS = 3
TOP_K = 10

DATA_DIR = Path(__file__).parent.parent / "data"


def load_qrels(path: Path) -> Dict[str, set]:
    qrels: Dict[str, set] = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def benchmark_db(
    db,
    doc_vectors: np.ndarray,
    doc_ids: List[str],
    query_vectors: np.ndarray,
    query_ids: List[str],
    qrels: Dict[str, set],
    top_k: int = TOP_K,
) -> Dict:
    from metrics import recall_at_k, mrr_at_k

    # INDEX
    t0 = time.perf_counter()
    db.index(doc_vectors, ids=doc_ids)
    index_time = time.perf_counter() - t0

    # WARMUP
    for q_vec in query_vectors[:WARMUP_QUERIES]:
        db.search(q_vec, top_k=top_k)

    # MEASURED (3 repeats, take per-query median)
    all_latencies: List[List[float]] = []
    recalls: List[float] = []
    mrrs: List[float] = []

    for repeat in range(NUM_REPEATS):
        latencies = []
        for q_vec, q_id in tqdm(
            zip(query_vectors, query_ids),
            total=len(query_ids),
            desc=f"  repeat {repeat + 1}/{NUM_REPEATS}",
            leave=False,
        ):
            t0 = time.perf_counter()
            results = db.search(q_vec, top_k=top_k)
            latencies.append((time.perf_counter() - t0) * 1000)

            if repeat == 0:
                retrieved_ids = [doc_id for doc_id, _ in results]
                relevant = qrels.get(q_id, set())
                recalls.append(recall_at_k(retrieved_ids, relevant, top_k))
                mrrs.append(mrr_at_k(retrieved_ids, relevant, top_k))
        all_latencies.append(latencies)

    latencies_arr = np.median(np.array(all_latencies), axis=0)

    return {
        "index_time_sec": round(index_time, 2),
        "disk_mb": round(db.disk_size_mb(), 1),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "mrr_at_10": round(float(np.mean(mrrs)), 4),
        "num_queries": len(query_vectors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/results.csv")
    parser.add_argument("--queries", type=int, default=1000, help="Number of queries to run")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="DB names to skip: faiss_flat faiss_hnsw qdrant chroma pgvector",
    )
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    # Load embeddings
    print("Loading doc embeddings...")
    doc_vecs = np.load(DATA_DIR / "embeddings.npy").astype(np.float32)
    doc_ids: List[str] = json.loads((DATA_DIR / "embeddings.ids.json").read_text())

    print("Loading query embeddings...")
    q_vecs = np.load(DATA_DIR / "query_embeddings.npy").astype(np.float32)
    q_ids: List[str] = json.loads((DATA_DIR / "query_embeddings.ids.json").read_text())

    print("Loading qrels...")
    qrels = load_qrels(DATA_DIR / "qrels.tsv")

    # Limit queries to those with ground truth
    valid_pairs = [(vec, qid) for vec, qid in zip(q_vecs, q_ids) if qid in qrels]
    valid_pairs = valid_pairs[: args.queries]
    q_vecs_run = np.array([p[0] for p in valid_pairs])
    q_ids_run = [p[1] for p in valid_pairs]
    print(f"Running on {len(q_ids_run)} queries with ground truth")

    from benchmarks.faiss_flat import FaissFlat
    from benchmarks.faiss_hnsw import FaissHNSW
    from benchmarks.qdrant_db import QdrantDB
    from benchmarks.chroma_db import ChromaDB
    from benchmarks.pgvector_db import PgVectorDB

    dbs = {
        "faiss_flat": FaissFlat,
        "faiss_hnsw": FaissHNSW,
        "qdrant": QdrantDB,
        "chroma": ChromaDB,
        "pgvector": PgVectorDB,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    fieldnames = [
        "db", "index_time_sec", "disk_mb",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "recall_at_10", "mrr_at_10", "num_queries",
    ]

    rows = []
    for name, DbClass in dbs.items():
        if name in args.skip:
            print(f"\n--- Skipping {name} ---")
            continue
        print(f"\n--- Benchmarking {name} ---")
        db = DbClass()
        try:
            metrics = benchmark_db(db, doc_vecs, doc_ids, q_vecs_run, q_ids_run, qrels)
            row = {"db": name, **metrics}
            rows.append(row)
            print(f"  recall@10={metrics['recall_at_10']:.4f}  "
                  f"p50={metrics['latency_p50_ms']:.2f}ms  "
                  f"index={metrics['index_time_sec']:.1f}s  "
                  f"disk={metrics['disk_mb']:.1f}MB")
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            db.cleanup()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'DB':<14} {'Recall@10':>10} {'MRR@10':>8} {'p50ms':>7} {'p95ms':>7} {'p99ms':>7} {'IndexS':>7} {'DiskMB':>7}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['db']:<14} {row['recall_at_10']:>10.4f} {row['mrr_at_10']:>8.4f} "
            f"{row['latency_p50_ms']:>7.2f} {row['latency_p95_ms']:>7.2f} "
            f"{row['latency_p99_ms']:>7.2f} {row['index_time_sec']:>7.1f} "
            f"{row['disk_mb']:>7.1f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
