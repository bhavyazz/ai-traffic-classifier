"""
Random Forest baseline for encrypted-traffic flow classification.

Reads processed CSV from features/extractor.py, trains sklearn RandomForestClassifier,
saves model + metadata for evaluate.py and the dashboard.

Confidence score: max class probability * 100 (0–100%).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.extractor import FEATURE_COLUMNS  # noqa: E402

# TODO: path to your merged features CSV (output of extractor.py)
DEFAULT_DATA_CSV = "data/processed/flows_features.csv"
DEFAULT_MODEL_PATH = "models/artifacts/rf_model.joblib"

CLASS_NAMES = ["chatgpt", "claude", "copilot", "non_ai"]


def load_xy(
    csv_path: str,
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column.")
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    # Drop rows with unknown labels
    df = df[df["label"].isin(CLASS_NAMES)].copy()
    y = le.transform(df["label"].astype(str))
    X = df[list(FEATURE_COLUMNS)]
    return X, y, le


def build_pipeline() -> Pipeline:
    # Impute TLS NaNs (non-TLS flows) with median learned on train
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def predict_with_confidence(
    bundle: Dict[str, Any],
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (predicted_label_str, confidence_0_100).
    """
    pipe: Pipeline = bundle["pipeline"]
    le: LabelEncoder = bundle["label_encoder"]
    probs = pipe.predict_proba(X)
    idx = np.argmax(probs, axis=1)
    conf = probs[np.arange(len(idx)), idx] * 100.0
    return le.inverse_transform(idx), conf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Random Forest flow classifier.")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_CSV, help="Features CSV.")
    p.add_argument("--model-out", type=str, default=DEFAULT_MODEL_PATH, help="joblib path.")
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    X, y, le = load_xy(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Hold-out accuracy: {acc:.4f}", flush=True)
    print(classification_report(y_test, pred, target_names=list(le.classes_)), flush=True)

    bundle = {
        "pipeline": pipe,
        "label_encoder": le,
        "feature_columns": list(FEATURE_COLUMNS),
        "classes": list(le.classes_),
    }
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(bundle, args.model_out)
    print(f"Saved model bundle to {args.model_out}", flush=True)
