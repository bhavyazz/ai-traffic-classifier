"""
5-Fold Stratified Cross Validation for all three classifiers.

Loads raw features from data/processed/flows_features.csv, applies the same
class balancing used during training (undersample non_ai to 400), then runs
StratifiedKFold(n_splits=5) on Random Forest, XGBoost, and 1D CNN.

Outputs:
  - paper/results/kfold_results.csv   (per-fold metrics)
  - paper/results/kfold_summary.md    (formatted summary table)
  - paper/results/kfold_boxplot.png   (accuracy box plot)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.extractor import FEATURE_COLUMNS  # noqa: E402
from models.train_cnn import FlowConvNet          # noqa: E402

# ── paths ────────────────────────────────────────────────────────────────
DATA_CSV = ROOT / "data" / "processed" / "flows_features.csv"
RESULTS_DIR = ROOT / "paper" / "results"

CLASS_NAMES = ["chatgpt", "claude", "copilot", "non_ai"]
N_SPLITS = 5
RANDOM_STATE = 42

warnings.filterwarnings("ignore", category=UserWarning)


# ═════════════════════════════════════════════════════════════════════════
# Data loading & balancing
# ═════════════════════════════════════════════════════════════════════════

def load_and_balance(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Load features CSV and apply the same balancing as training."""
    df = pd.read_csv(csv_path)
    df = df[df["label"].isin(CLASS_NAMES)].copy()

    # Undersample non_ai to 400 flows (same random_state as training)
    df_non_ai = df[df["label"] == "non_ai"]
    df_rest = df[df["label"] != "non_ai"]
    if len(df_non_ai) > 400:
        df_non_ai = df_non_ai.sample(n=400, random_state=RANDOM_STATE)
    df = pd.concat([df_rest, df_non_ai], ignore_index=True)

    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y = le.transform(df["label"].astype(str))
    X = df[list(FEATURE_COLUMNS)]

    print(f"Loaded {len(df)} samples after balancing:")
    for cls in CLASS_NAMES:
        print(f"  {cls:>10s}: {(df['label'] == cls).sum()}")
    print()
    return X, y, le


# ═════════════════════════════════════════════════════════════════════════
# Model builders  (fresh instance per fold)
# ═════════════════════════════════════════════════════════════════════════

def build_rf() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])


def build_xgb(num_classes: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", xgb.XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=num_classes,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss",
        )),
    ])


def train_predict_cnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
    num_classes: int,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> np.ndarray:
    """Train a fresh 1D CNN and return predictions on X_test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    Xi_tr = imputer.fit_transform(X_train)
    Xs_tr = scaler.fit_transform(Xi_tr).astype(np.float32)[:, np.newaxis, :]
    Xi_te = imputer.transform(X_test)
    Xs_te = scaler.transform(Xi_te).astype(np.float32)[:, np.newaxis, :]

    F = X_train.shape[1]
    model = FlowConvNet(in_len=F, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xs_tr).to(device)
    ytr_t = torch.from_numpy(y_train.astype(np.int64)).to(device)
    n = Xtr_t.size(0)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    Xte_t = torch.from_numpy(Xs_te).to(device)
    with torch.no_grad():
        preds = model(Xte_t).argmax(dim=1).cpu().numpy()
    return preds


# ═════════════════════════════════════════════════════════════════════════
# Run k-fold CV for one model type
# ═════════════════════════════════════════════════════════════════════════

def run_kfold(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
) -> List[Dict]:
    """Return a list of per-fold metric dicts."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    num_classes = len(le.classes_)
    fold_records: List[Dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_name == "Random Forest":
            pipe = build_rf()
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
        elif model_name == "XGBoost":
            pipe = build_xgb(num_classes)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
        elif model_name == "1D CNN":
            y_pred = train_predict_cnn(X_train, y_train, X_test, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        per_class_f1 = f1_score(
            y_test, y_pred,
            average=None,
            labels=list(range(num_classes)),
            zero_division=0,
        )

        record = {
            "model": model_name,
            "fold": fold_idx,
            "accuracy": acc,
            "macro_f1": macro_f1,
        }
        for i, cls in enumerate(le.classes_):
            record[f"f1_{cls}"] = per_class_f1[i]

        fold_records.append(record)
        print(f"  Fold {fold_idx}: Accuracy={acc:.4f}  Macro F1={macro_f1:.4f}")

    return fold_records


# ═════════════════════════════════════════════════════════════════════════
# Display helpers
# ═════════════════════════════════════════════════════════════════════════

def print_model_summary(model_name: str, records: List[Dict], le: LabelEncoder) -> str:
    """Pretty-print and return a markdown summary block."""
    accs = [r["accuracy"] for r in records]
    f1s  = [r["macro_f1"] for r in records]

    lines: List[str] = []
    lines.append(f"\nModel: {model_name}")
    for r in records:
        lines.append(f"  Fold {r['fold']}: Accuracy={r['accuracy']:.4f}  Macro F1={r['macro_f1']:.4f}")
    lines.append(f"  Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    lines.append(f"  Mean Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    lines.append("")
    lines.append("  Per Class F1:")
    for cls in le.classes_:
        vals = [r[f"f1_{cls}"] for r in records]
        lines.append(f"    {cls:>10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    block = "\n".join(lines)
    print(block)
    return block


# ═════════════════════════════════════════════════════════════════════════
# Save results
# ═════════════════════════════════════════════════════════════════════════

def save_results_csv(all_records: List[Dict], out_path: Path) -> None:
    df = pd.DataFrame(all_records)
    df.to_csv(out_path, index=False)
    print(f"\nSaved per-fold results to {out_path}")


def save_summary_md(all_records: List[Dict], le: LabelEncoder, out_path: Path) -> None:
    models = ["Random Forest", "XGBoost", "1D CNN"]
    lines: List[str] = [
        "# K-Fold Cross Validation Results",
        "",
        f"- **Folds**: {N_SPLITS}",
        f"- **Random state**: {RANDOM_STATE}",
        f"- **Balancing**: non_ai undersampled to 400 flows",
        "",
        "## Summary Table",
        "",
        "| Model | Mean Accuracy | Std Accuracy | Mean Macro F1 | Std Macro F1 |",
        "|-------|--------------|-------------|--------------|-------------|",
    ]

    for m in models:
        recs = [r for r in all_records if r["model"] == m]
        if not recs:
            continue
        accs = [r["accuracy"] for r in recs]
        f1s  = [r["macro_f1"]  for r in recs]
        lines.append(
            f"| {m} | {np.mean(accs):.4f} | {np.std(accs):.4f} "
            f"| {np.mean(f1s):.4f} | {np.std(f1s):.4f} |"
        )

    lines += ["", "## Per-Class F1 Scores", ""]
    for m in models:
        recs = [r for r in all_records if r["model"] == m]
        if not recs:
            continue
        lines.append(f"### {m}")
        lines.append("")
        lines.append("| Class | Mean F1 | Std F1 |")
        lines.append("|-------|---------|--------|")
        for cls in le.classes_:
            vals = [r[f"f1_{cls}"] for r in recs]
            lines.append(f"| {cls} | {np.mean(vals):.4f} | {np.std(vals):.4f} |")
        lines.append("")

    lines += ["", "## Per-Fold Detail", ""]
    lines.append("| Model | Fold | Accuracy | Macro F1 | " +
                 " | ".join(f"F1 {c}" for c in le.classes_) + " |")
    lines.append("|-------|------|----------|----------|" +
                 "|".join(["-------"] * len(le.classes_)) + "|")
    for r in all_records:
        row = (f"| {r['model']} | {r['fold']} | {r['accuracy']:.4f} "
               f"| {r['macro_f1']:.4f} |")
        for cls in le.classes_:
            row += f" {r[f'f1_{cls}']:.4f} |"
        lines.append(row)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary to {out_path}")


def save_boxplot(all_records: List[Dict], out_path: Path) -> None:
    """Accuracy distribution box plot across folds for all 3 models."""
    models = ["Random Forest", "XGBoost", "1D CNN"]
    data = []
    labels = []
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for m in models:
        accs = [r["accuracy"] for r in all_records if r["model"] == m]
        data.append(accs)
        labels.append(m)

    fig, ax = plt.subplots(figsize=(9, 6))

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=7),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(d))
        ax.scatter(
            [i + 1 + j for j in jitter],
            d,
            color=color,
            edgecolor="black",
            s=50,
            zorder=5,
            alpha=0.9,
        )

    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("5-Fold Cross Validation — Accuracy Distribution", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved box plot to {out_path}")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_df, y, le = load_and_balance(DATA_CSV)
    X = X_df.values  # numpy for indexing in k-fold

    all_records: List[Dict] = []

    for model_name in ["Random Forest", "XGBoost", "1D CNN"]:
        print(f"\n{'=' * 60}")
        print(f"  {model_name}  —  {N_SPLITS}-Fold Stratified CV")
        print(f"{'=' * 60}")
        records = run_kfold(model_name, X, y, le)
        all_records.extend(records)
        print_model_summary(model_name, records, le)

    # Persist
    save_results_csv(all_records, RESULTS_DIR / "kfold_results.csv")
    save_summary_md(all_records, le, RESULTS_DIR / "kfold_summary.md")
    save_boxplot(all_records, RESULTS_DIR / "kfold_boxplot.png")

    print("\n[DONE] K-Fold Cross Validation complete.")
