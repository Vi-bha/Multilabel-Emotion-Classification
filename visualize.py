"""
Visualization of Multi-Label Classification Results

Generates three figures matching the paper:
  1. Heatmap — Micro-F1 across methods × classifiers
  2. Heatmap — Hamming Loss across methods × classifiers
  3. Radar chart — all metrics for best model (RAkEL + Random Forest)

Output: results/figures/
Usage : python visualize.py
        python visualize.py --from-csv results/results.csv  (use your own run)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from math import pi

# ---------------------------------------------------------------------------
# Results from the paper (used when no CSV is provided)
# ---------------------------------------------------------------------------

PAPER_RESULTS = [
    # Method                Classifier              SubAcc  HammLoss  P_mi   R_mi   F1_mi   P_ma   R_ma   F1_ma  Jac_mi  Jac_sa
    ("Binary Relevance",   "Naive Bayes",           0.2269, 0.2451,  0.5098,0.4885,0.4989, 0.4912,0.4730,0.4819, 0.3325, 0.3282),
    ("Binary Relevance",   "Logistic Regression",   0.3109, 0.2073,  0.6406,0.6154,0.6278, 0.6224,0.5907,0.6062, 0.4573, 0.4476),
    ("Binary Relevance",   "Random Forest",         0.3277, 0.1765,  0.7692,0.7189,0.7433, 0.7401,0.6882,0.6985, 0.5924, 0.5791),
    ("Binary Relevance",   "SVM (Linear)",          0.3109, 0.2031,  0.6531,0.6154,0.6337, 0.6289,0.5861,0.6068, 0.4640, 0.4542),
    ("Classifier Chain",   "Naive Bayes",           0.2269, 0.2353,  0.5224,0.5160,0.5192, 0.5061,0.4964,0.5012, 0.3508, 0.3469),
    ("Classifier Chain",   "Logistic Regression",   0.3109, 0.2157,  0.6224,0.5929,0.6073, 0.5981,0.5611,0.5790, 0.4360, 0.4251),
    ("Classifier Chain",   "Random Forest",         0.3109, 0.1863,  0.7143,0.6918,0.7029, 0.6892,0.6601,0.6744, 0.5418, 0.5287),
    ("Classifier Chain",   "SVM (Linear)",          0.3193, 0.2143,  0.6327,0.6042,0.6181, 0.6084,0.5741,0.5908, 0.4472, 0.4356),
    ("Label Powerset",     "Naive Bayes",           0.3025, 0.2108,  0.6531,0.6314,0.6420, 0.6382,0.6231,0.6103, 0.4731, 0.4718),
    ("Label Powerset",     "Logistic Regression",   0.3193, 0.2010,  0.6531,0.5739,0.6109, 0.6452,0.5809,0.6126, 0.4398, 0.4328),
    ("Label Powerset",     "Random Forest",         0.4202, 0.1863,  0.7143,0.6384,0.6737, 0.7294,0.6769,0.7020, 0.5082, 0.5168),
    ("Label Powerset",     "SVM (Linear)",          0.3025, 0.2059,  0.6327,0.5739,0.6018, 0.6195,0.5958,0.6195, 0.4302, 0.4272),
    ("RAkEL",              "Naive Bayes",           0.2941, 0.2206,  0.6122,0.5739,0.5895, 0.5963,0.5731,0.6054, 0.4179, 0.4148),
    ("RAkEL",              "Logistic Regression",   0.3529, 0.1863,  0.6735,0.6132,0.6420, 0.6564,0.6308,0.6437, 0.4731, 0.4631),
    ("RAkEL",              "Random Forest",         0.3950, 0.1176,  0.7347,0.6441,0.6867, 0.7386,0.6843,0.7111, 0.5229, 0.5163),
    ("RAkEL",              "SVM (Linear)",          0.3529, 0.1569,  0.7143,0.6029,0.6540, 0.6984,0.6490,0.6667, 0.4865, 0.4779),
]

COLUMNS = [
    "Method", "Classifier",
    "Subset Accuracy", "Hamming Loss",
    "Precision (micro)", "Recall (micro)", "F1 Score (micro)",
    "Precision (macro)", "Recall (macro)", "F1 Score (macro)",
    "Jaccard (micro)", "Jaccard (samples)",
]

METHODS      = ["Binary Relevance", "Classifier Chain", "Label Powerset", "RAkEL"]
CLASSIFIERS  = ["Naive Bayes", "Logistic Regression", "Random Forest", "SVM (Linear)"]
FIGURES_DIR  = Path("results/figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(csv_path=None) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        print(f"Loading results from {csv_path}")
        return pd.read_csv(csv_path)
    print("Using paper results")
    return pd.DataFrame(PAPER_RESULTS, columns=COLUMNS)


def pivot(df, metric) -> pd.DataFrame:
    """Return a classifiers × methods pivot table for one metric."""
    return df.pivot(index="Classifier", columns="Method", values=metric).reindex(
        index=CLASSIFIERS, columns=METHODS
    )


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 1 & 2 — Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmap(df, metric, cmap, title, filename, fmt=".3f", annot_size=11):
    data = pivot(df, metric)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor="#e0e0e0",
        annot_kws={"size": annot_size, "weight": "bold"},
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Method", fontsize=11, labelpad=8)
    ax.set_ylabel("Classifier", fontsize=11, labelpad=8)
    ax.tick_params(axis="x", labelsize=10, rotation=15)
    ax.tick_params(axis="y", labelsize=10, rotation=0)

    fig.tight_layout()
    save(fig, filename)


# ---------------------------------------------------------------------------
# Figure 3 — Radar chart
# ---------------------------------------------------------------------------

def plot_radar(df):
    """Radar chart for the best model: RAkEL + Random Forest."""
    row = df[(df["Method"] == "RAkEL") & (df["Classifier"] == "Random Forest")].iloc[0]

    # Metrics to display (all higher-is-better; invert Hamming Loss)
    labels  = ["1 - Hamming\nLoss", "Subset\nAccuracy", "F1\n(micro)", "F1\n(macro)", "Jaccard\n(samples)"]
    values  = [
        1 - row["Hamming Loss"],
        row["Subset Accuracy"],
        row["F1 Score (micro)"],
        row["F1 Score (macro)"],
        row["Jaccard (samples)"],
    ]

    N      = len(labels)
    angles = [n / N * 2 * pi for n in range(N)]
    angles += angles[:1]          # close the polygon
    values += values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    # Grid styling
    ax.set_facecolor("#f8f9fa")
    ax.spines["polar"].set_color("#cccccc")
    ax.grid(color="#cccccc", linewidth=0.6, linestyle="--")

    # Ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9.5, color="#333333")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8, color="#888888")
    ax.set_ylim(0, 1)

    # Plot
    ax.plot(angles, values, color="#4C72B0", linewidth=2, linestyle="solid")
    ax.fill(angles, values, color="#4C72B0", alpha=0.25)

    # Annotate values
    for angle, value in zip(angles[:-1], values[:-1]):
        ax.text(
            angle, value + 0.07,
            f"{value:.3f}",
            ha="center", va="center",
            fontsize=8.5, color="#4C72B0", fontweight="bold"
        )

    ax.set_title(
        "RAkEL + Random Forest\n(Best Model)",
        fontsize=12, fontweight="bold", pad=20, color="#222222"
    )

    fig.tight_layout()
    save(fig, "rakel_rf_radar.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate result figures")
    parser.add_argument("--from-csv", metavar="PATH", help="Path to results CSV from experiments.py")
    args = parser.parse_args()

    df = load_data(args.from_csv)

    print("\nGenerating figures...")

    plot_heatmap(
        df,
        metric   = "F1 Score (micro)",
        cmap     = "YlGn",
        title    = "Micro-F1 Score across Methods & Classifiers",
        filename = "f1_heatmap.png",
    )

    plot_heatmap(
        df,
        metric   = "Hamming Loss",
        cmap     = "YlOrRd_r",       # reversed: lower = greener = better
        title    = "Hamming Loss across Methods & Classifiers",
        filename = "hamming_heatmap.png",
    )

    plot_radar(df)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Add them to your README with:")
    print("  ![F1 Heatmap](results/figures/f1_heatmap.png)")
    print("  ![Hamming Loss Heatmap](results/figures/hamming_heatmap.png)")
    print("  ![Radar Chart](results/figures/rakel_rf_radar.png)")
