"""
XGBoost classifier for encrypted-traffic flow classification.

Same feature CSV and confidence semantics as train_rf.py (max softmax * 100).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.extractor import FEATURE_COLUMNS  # noqa: E402

# TODO: path to your merged features CSV
DEFAULT_DATA_CSV = "data/processed/flows_features.csv"
DEFAULT_MODEL_PATH = "models/artifacts/xgb_model.joblib"

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


def build_pipeline(num_classes: int) -> Pipeline:
    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=num_classes,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
    )


def predict_with_confidence(
    bundle: Dict[str, Any],
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    pipe: Pipeline = bundle["pipeline"]
    le: LabelEncoder = bundle["label_encoder"]
    probs = pipe.predict_proba(X)
    idx = np.argmax(probs, axis=1)
    conf = probs[np.arange(len(idx)), idx] * 100.0
    return le.inverse_transform(idx), conf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost flow classifier.")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_CSV)
    p.add_argument("--model-out", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    X, y, le = load_xy(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    pipe = build_pipeline(num_classes=len(le.classes_))
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
