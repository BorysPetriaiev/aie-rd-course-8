"""Download BeIR/quora and save corpus.jsonl, queries.jsonl, qrels.tsv."""
import json
import os
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent / "data"


def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("Loading BeIR/quora corpus...")
    corpus_ds = load_dataset("BeIR/quora", "corpus", split="corpus")
    corpus = [{"_id": str(row["_id"]), "text": row["text"]} for row in corpus_ds]
    save_jsonl(corpus, DATA_DIR / "corpus.jsonl")
    print(f"  corpus.jsonl — {len(corpus):,} docs")

    print("Loading queries...")
    queries_ds = load_dataset("BeIR/quora", "queries", split="queries")
    queries = [{"_id": str(row["_id"]), "text": row["text"]} for row in queries_ds]
    save_jsonl(queries, DATA_DIR / "queries.jsonl")
    print(f"  queries.jsonl — {len(queries):,} queries")

    print("Loading qrels...")
    qrels_ds = load_dataset("BeIR/quora-qrels", split="validation")
    qrels_path = DATA_DIR / "qrels.tsv"
    with open(qrels_path, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for row in qrels_ds:
            f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")
    print(f"  qrels.tsv — {len(qrels_ds):,} pairs")

    print("Done.")


if __name__ == "__main__":
    main()
