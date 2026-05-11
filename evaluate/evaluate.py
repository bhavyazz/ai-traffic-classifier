"""
Evaluation harness: metrics, confusion matrices, calibration curves, model comparison.

Loads saved bundles from models/artifacts. Assumes the same CSV row order and
train_test_split(test_size=, random_state=42) as training for a fair test subset.
TODO: If you change the dataset, retrain models and keep the CSV export unchanged
      or fix the split to match your protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.extractor import FEATURE_COLUMNS  # noqa: E402
from models.train_cnn import predict_with_confidence as cnn_predict  # noqa: E402
from models.train_rf import predict_with_confidence as rf_predict  # noqa: E402
from models.train_xgboost import predict_with_confidence as xgb_predict  # noqa: E402

# TODO: default features CSV (output of extractor.py)
DEFAULT_DATA_CSV = "data/processed/flows_features_balanced.csv"
DEFAULT_RESULTS_DIR = "paper/results"

CLASS_NAMES = ["chatgpt", "claude", "copilot", "non_ai"]


def load_xy(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column.")
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    df = df[df["label"].isin(CLASS_NAMES)].copy()
    y = le.transform(df["label"].astype(str))
    X = df[list(FEATURE_COLUMNS)]
    return X, y, le


def _proba_matrix_rf_xgb(pipe, X: pd.DataFrame) -> np.ndarray:
    """Extract n x K probability matrix from sklearn Pipeline."""
    return pipe.predict_proba(X)


def _proba_matrix_cnn(bundle: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from models.train_cnn import FlowConvNet

    imputer: SimpleImputer = bundle["imputer"]
    scaler: StandardScaler = bundle["scaler"]
    model = FlowConvNet(
        in_len=int(bundle["num_features"]),
        num_classes=int(bundle["num_classes"]),
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    Xi = imputer.transform(X)
    Xs = scaler.transform(Xi).astype(np.float32)[:, np.newaxis, :]
    tens = torch.from_numpy(Xs)
    with torch.no_grad():
        logits = model(tens)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs


def confusion_matrix_png(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_calibration_multiclass(
    y_true_idx: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    out_path: str,
    n_bins: int = 10,
) -> None:
    """One panel per class: fraction of positives vs mean predicted probability."""
    Y = label_binarize(y_true_idx, classes=list(range(len(class_names))))
    cols = min(2, len(class_names))
    rows = int(np.ceil(len(class_names) / cols))
    plt.figure(figsize=(5 * cols, 4 * rows))
    for k, name in enumerate(class_names):
        y_k = Y[:, k]
        if y_k.sum() == 0:
            continue
        prob_pos = probs[:, k]
        try:
            fraction_pos, mean_pred = calibration_curve(y_k, prob_pos, n_bins=n_bins, strategy="uniform")
        except ValueError:
            continue
        ax = plt.subplot(rows, cols, k + 1)
        ax.plot(mean_pred, fraction_pos, "s-", label=name)
        ax.plot([0, 1], [0, 1], "k:", label="perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positives")
        ax.set_title(f"Calibration — {name}")
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved classifiers.")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_CSV)
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--test-size", type=float, default=0.2, help="Must match training scripts.")
    p.add_argument(
        "--rf-path",
        type=str,
        default="models/artifacts/rf_model.joblib",
    )
    p.add_argument(
        "--xgb-path",
        type=str,
        default="models/artifacts/xgb_model.joblib",
    )
    p.add_argument(
        "--cnn-path",
        type=str,
        default="models/artifacts/cnn_bundle.joblib",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    X, y, le = load_xy(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    models_spec: List[Tuple[str, str, Any, Callable]] = []
    if os.path.isfile(args.rf_path):
        bundle_rf = joblib.load(args.rf_path)
        models_spec.append(
            ("RandomForest", "rf", bundle_rf, lambda b, Xt: rf_predict(b, Xt)),
        )
    if os.path.isfile(args.xgb_path):
        bundle_xgb = joblib.load(args.xgb_path)
        models_spec.append(
            ("XGBoost", "xgb", bundle_xgb, lambda b, Xt: xgb_predict(b, Xt)),
        )
    if os.path.isfile(args.cnn_path):
        bundle_cnn = joblib.load(args.cnn_path)
        models_spec.append(
            ("CNN1D", "cnn", bundle_cnn, lambda b, Xt: cnn_predict(b, Xt)),
        )

    if not models_spec:
        raise SystemExit(
            "No model bundles found. Train models first (train_rf.py, train_xgboost.py, train_cnn.py)."
        )

    rows_summary = []
    for display_name, short, bundle, predict_fn in models_spec:
        y_pred_str, conf = predict_fn(bundle, X_test)
        y_pred = le.transform(y_pred_str)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0
        )
        report = classification_report(
            y_test,
            y_pred,
            target_names=list(le.classes_),
            output_dict=True,
            zero_division=0,
        )
        rows_summary.append(
            {
                "model": display_name,
                "accuracy": acc,
                "macro_f1": report["macro avg"]["f1-score"],
                "mean_confidence_pct": float(np.mean(conf)),
            }
        )

        cm_path = os.path.join(args.results_dir, f"confusion_matrix__{short}.png")
        confusion_matrix_png(y_test, y_pred, list(le.classes_), cm_path)

        # Calibration from probability matrix
        if short in ("rf", "xgb"):
            probs = _proba_matrix_rf_xgb(bundle["pipeline"], X_test)
        else:
            probs = _proba_matrix_cnn(bundle, X_test)
        cal_path = os.path.join(args.results_dir, f"calibration__{short}.png")
        save_calibration_multiclass(y_test, probs, list(le.classes_), cal_path)

        metrics_path = os.path.join(args.results_dir, f"metrics__{short}.json")
        out_metrics = {
            "accuracy": float(acc),
            "per_class": {
                CLASS_NAMES[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
                for i in range(len(CLASS_NAMES))
            },
            "classification_report": report,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(out_metrics, f, indent=2)

        print(f"\n=== {display_name} ===\naccuracy={acc:.4f}\n", flush=True)
        print(classification_report(y_test, y_pred, target_names=list(le.classes_)), flush=True)

    cmp_df = pd.DataFrame(rows_summary)
    cmp_path = os.path.join(args.results_dir, "model_comparison_table.csv")
    cmp_df.to_csv(cmp_path, index=False)
    md_path = os.path.join(args.results_dir, "model_comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        cols = list(cmp_df.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in cmp_df.iterrows():
            f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
    print(f"\nSaved comparison table to {cmp_path} and {md_path}", flush=True)
