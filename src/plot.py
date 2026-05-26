"""Generate benchmark charts from results.csv.

Usage:
    python src/plot.py --input results/results.csv --output results/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

COLORS = {
    "faiss_flat":  "#2196F3",
    "faiss_hnsw":  "#4CAF50",
    "qdrant":      "#FF9800",
    "chroma":      "#9C27B0",
    "pgvector":    "#F44336",
}


def pareto_frontier(df: pd.DataFrame) -> list[int]:
    """Return indices of Pareto-optimal rows (max recall, min latency)."""
    dominated = []
    for i, ri in df.iterrows():
        for j, rj in df.iterrows():
            if i == j:
                continue
            if rj["recall_at_10"] >= ri["recall_at_10"] and rj["latency_p50_ms"] <= ri["latency_p50_ms"]:
                dominated.append(i)
                break
    return [i for i in df.index if i not in dominated]


def plot_pareto(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    pareto_idx = pareto_frontier(df)
    pareto_df = df.loc[pareto_idx].sort_values("latency_p50_ms")

    ax.plot(
        pareto_df["latency_p50_ms"],
        pareto_df["recall_at_10"],
        "--",
        color="gray",
        linewidth=1,
        alpha=0.6,
        label="Pareto frontier",
    )

    for _, row in df.iterrows():
        color = COLORS.get(row["db"], "#607D8B")
        ax.scatter(row["latency_p50_ms"], row["recall_at_10"], color=color, s=150, zorder=5)
        ax.annotate(
            row["db"],
            (row["latency_p50_ms"], row["recall_at_10"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    ax.set_xlabel("Query latency p50 (ms)", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("Pareto Frontier: Recall vs Latency", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "pareto_frontier.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_latency_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    dbs = df["db"].tolist()
    x = np.arange(len(dbs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_p50 = ax.bar(x - width, df["latency_p50_ms"], width, label="p50", color="#42A5F5")
    bars_p95 = ax.bar(x, df["latency_p95_ms"], width, label="p95", color="#FFA726")
    bars_p99 = ax.bar(x + width, df["latency_p99_ms"], width, label="p99", color="#EF5350")

    ax.set_xticks(x)
    ax.set_xticklabels(dbs, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Query Latency Distribution (p50 / p95 / p99)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in [*bars_p50, *bars_p95, *bars_p99]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    path = out_dir / "latency_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_disk_size(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS.get(db, "#607D8B") for db in df["db"]]
    bars = ax.bar(df["db"], df["disk_mb"], color=colors)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h:.0f} MB", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Disk size (MB)", fontsize=12)
    ax.set_title("Index Size on Disk", fontsize=14, fontweight="bold")
    ax.set_xticklabels(df["db"], rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "disk_size_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_results_table(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, len(df) * 0.6 + 1.5))
    ax.axis("off")

    cols = ["db", "recall_at_10", "mrr_at_10", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "index_time_sec", "disk_mb"]
    col_labels = ["DB", "Recall@10", "MRR@10", "p50 ms", "p95 ms", "p99 ms", "Index s", "Disk MB"]
    cell_text = []
    for _, row in df[cols].iterrows():
        cell_text.append([
            row["db"],
            f"{row['recall_at_10']:.4f}",
            f"{row['mrr_at_10']:.4f}",
            f"{row['latency_p50_ms']:.2f}",
            f"{row['latency_p95_ms']:.2f}",
            f"{row['latency_p99_ms']:.2f}",
            f"{row['index_time_sec']:.1f}",
            f"{row['disk_mb']:.1f}",
        ])

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ECEFF1")

    ax.set_title("Vector DB Benchmark Results", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    path = out_dir / "results_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/results.csv")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    plot_pareto(df, out_dir)
    plot_latency_distribution(df, out_dir)
    plot_disk_size(df, out_dir)
    plot_results_table(df, out_dir)
    print("All charts generated.")


if __name__ == "__main__":
    main()
