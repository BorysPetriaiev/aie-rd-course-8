"""Generate and cache embeddings for corpus and queries.

Usage:
    python src/embed.py --model BAAI/bge-small-en-v1.5 \
                        --input data/corpus.jsonl \
                        --output data/embeddings.npy

    python src/embed.py --model BAAI/bge-small-en-v1.5 \
                        --input data/queries.jsonl \
                        --output data/query_embeddings.npy
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_texts(jsonl_path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["_id"])
            texts.append(row["text"])
    return ids, texts


def embed_batch(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=normalize, show_progress_bar=False)
        all_vecs.append(vecs.astype(np.float32))
    return np.vstack(all_vecs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    ids_path = output_path.with_suffix(".ids.json")

    if output_path.exists():
        print(f"{output_path} already exists, skipping.")
        return

    print(f"Loading {input_path}...")
    ids, texts = load_texts(input_path)
    print(f"  {len(texts):,} texts")

    print(f"Loading model {args.model}...")
    model = SentenceTransformer(args.model)

    vecs = embed_batch(model, texts, batch_size=args.batch_size)
    print(f"  vectors shape: {vecs.shape}")

    np.save(output_path, vecs)
    ids_path.write_text(json.dumps(ids, ensure_ascii=False))
    print(f"Saved {output_path} and {ids_path}")


if __name__ == "__main__":
    main()
