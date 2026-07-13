"""
Fig. 7 – Pearson Correlation Heatmap of Network-Flow Features.

Generates a publication-ready correlation heatmap suitable for an IEEE
conference paper. Saves output to paper/results/fig7_correlation_heatmap.png
at 300 DPI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "processed" / "flows_features_balanced.csv"
OUT_DIR = ROOT / "paper" / "results"
OUT_PNG = OUT_DIR / "fig7_correlation_heatmap.png"

# ── 17 numerical feature columns (in desired display order) ──────────
FEATURE_COLUMNS = [
    "total_packets",
    "total_bytes",
    "flow_duration",
    "bytes_per_second",
    "packets_per_second",
    "mean_packet_size",
    "std_packet_size",
    "min_packet_size",
    "max_packet_size",
    "mean_interarrival_time",
    "std_interarrival_time",
    "direction_ratio",
    "burst_count",
    "burst_avg_size",
    "burst_mean_size_variance",
    "small_to_large_ratio",
    "inter_burst_gap",
]

# Pretty labels for the axes (shorter, paper-friendly names)
PRETTY_LABELS = [
    "Total Pkts",
    "Total Bytes",
    "Flow Duration",
    "Bytes/s",
    "Pkts/s",
    "Mean Pkt Size",
    "Std Pkt Size",
    "Min Pkt Size",
    "Max Pkt Size",
    "Mean IAT",
    "Std IAT",
    "Direction Ratio",
    "Burst Count",
    "Burst Avg Size",
    "Burst Variance",
    "Small/Large Ratio",
    "Inter-Burst Gap",
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH.name}")

    # ── 2. Select numerical features only ────────────────────────────
    X = df[FEATURE_COLUMNS].copy()
    print(f"Selected {X.shape[1]} numerical features")

    # ── 3. Pearson correlation matrix ────────────────────────────────
    corr = X.corr(method="pearson")

    # ── 4–6. Publication-quality heatmap ─────────────────────────────
    # IEEE column width ≈ 3.5 in; use full-width (7 in) for readability
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a mask for the upper triangle to reduce visual clutter
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    cmap = sns.diverging_palette(240, 10, s=80, l=55, as_cmap=True)  # blue ↔ red

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "shrink": 0.75,
            "label": "Pearson Correlation Coefficient",
        },
        annot_kws={"size": 7},
        xticklabels=PRETTY_LABELS,
        yticklabels=PRETTY_LABELS,
        ax=ax,
    )

    ax.set_title(
        "Fig. 7.  Pearson Correlation Matrix of Network-Flow Features",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.tick_params(axis="x", labelsize=9, rotation=45)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()

    # ── 7. Save at 300 DPI ───────────────────────────────────────────
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved heatmap to {OUT_PNG}")

    # ── 8. Top-10 most strongly correlated pairs ─────────────────────
    # Flatten the upper triangle (exclude diagonal)
    pairs = (
        corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
        .stack()
        .reset_index()
    )
    pairs.columns = ["Feature_1", "Feature_2", "Correlation"]
    pairs["abs_corr"] = pairs["Correlation"].abs()
    top10 = pairs.nlargest(10, "abs_corr")

    print("\n" + "=" * 72)
    print("  Top 10 Most Strongly Correlated Feature Pairs")
    print("=" * 72)
    print(f"{'Rank':<6}{'Feature 1':<28}{'Feature 2':<28}{'r':>8}")
    print("-" * 72)
    for rank, (_, row) in enumerate(top10.iterrows(), start=1):
        print(
            f"{rank:<6}{row['Feature_1']:<28}{row['Feature_2']:<28}"
            f"{row['Correlation']:>8.4f}"
        )

    # ── 9. Brief interpretation ──────────────────────────────────────
    strongest_pos = top10.iloc[0] if top10.iloc[0]["Correlation"] > 0 else top10[top10["Correlation"] > 0].iloc[0]
    neg_pairs = top10[top10["Correlation"] < 0]

    print("\n" + "=" * 72)
    print("  Interpretation")
    print("=" * 72)
    print(
        f"\n  Strongest POSITIVE correlation:  "
        f"{strongest_pos['Feature_1']} <-> {strongest_pos['Feature_2']}  "
        f"(r = {strongest_pos['Correlation']:.4f})"
    )
    print(
        "    -> These features are highly linearly related; including both "
        "may introduce\n      multicollinearity. Consider removing one or "
        "applying PCA for dimensionality reduction."
    )

    if not neg_pairs.empty:
        strongest_neg = neg_pairs.iloc[0]
        print(
            f"\n  Strongest NEGATIVE correlation: "
            f"{strongest_neg['Feature_1']} <-> {strongest_neg['Feature_2']}  "
            f"(r = {strongest_neg['Correlation']:.4f})"
        )
        print(
            "    -> An inverse relationship exists; as one feature increases "
            "the other tends\n      to decrease. This can provide complementary "
            "discriminative power for classification."
        )
    else:
        print("\n  No negative correlations found among the top-10 pairs.")

    print(
        "\n  Overall: Most features show weak-to-moderate correlations, "
        "indicating that\n  the selected feature set captures diverse aspects "
        "of encrypted network traffic\n  behaviour with limited redundancy."
    )
    print()


if __name__ == "__main__":
    main()
