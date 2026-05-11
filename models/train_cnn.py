"""
1D CNN over the flow feature vector (treated as a single-channel sequence).

Input shape: (batch, 1, F) where F = len(FEATURE_COLUMNS). This is a lightweight
CNN baseline for papers comparing classical vs deep models on the same features.

Confidence: softmax max * 100.
Uses CPU by default; set CUDA_VISIBLE_DEVICES as needed.
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
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.extractor import FEATURE_COLUMNS  # noqa: E402

# TODO: path to your merged features CSV
DEFAULT_DATA_CSV = "data/processed/flows_features_balanced.csv"
DEFAULT_BUNDLE_PATH = "models/artifacts/cnn_bundle.joblib"
DEFAULT_CKPT_PATH = "models/artifacts/cnn_model.pt"

CLASS_NAMES = ["chatgpt", "claude", "copilot", "non_ai"]


class FlowConvNet(nn.Module):
    """Small 1D CNN: two Conv1d blocks + GAP + linear classifier."""

    def __init__(self, in_len: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)  # (B, 64, 1)
        h = h.flatten(1)
        return self.fc(h)


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


def prepare_tensors(
    X: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    *,
    fit: bool,
) -> np.ndarray:
    Xi = imputer.fit_transform(X) if fit else imputer.transform(X)
    Xs = scaler.fit_transform(Xi) if fit else scaler.transform(Xi)
    # (N, 1, F)
    return Xs.astype(np.float32)[:, np.newaxis, :]


@torch.no_grad()
def predict_with_confidence(
    bundle: Dict[str, Any],
    X: pd.DataFrame,
    *,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cpu")
    imputer: SimpleImputer = bundle["imputer"]
    scaler: StandardScaler = bundle["scaler"]
    le: LabelEncoder = bundle["label_encoder"]
    model = FlowConvNet(
        in_len=int(bundle["num_features"]),
        num_classes=int(bundle["num_classes"]),
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()

    Xi = imputer.transform(X)
    Xs = scaler.transform(Xi).astype(np.float32)[:, np.newaxis, :]
    tens = torch.from_numpy(Xs).to(device)
    logits = model(tens)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    idx = np.argmax(probs, axis=1)
    conf = probs[np.arange(len(idx)), idx] * 100.0
    return le.inverse_transform(idx), conf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 1D CNN flow classifier.")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_CSV)
    p.add_argument("--bundle-out", type=str, default=DEFAULT_BUNDLE_PATH)
    p.add_argument("--ckpt-out", type=str, default=DEFAULT_CKPT_PATH)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, le = load_xy(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = prepare_tensors(X_train, imputer, scaler, fit=True)
    Xte = prepare_tensors(X_test, imputer, scaler, fit=False)

    num_classes = len(le.classes_)
    F = len(FEATURE_COLUMNS)
    model = FlowConvNet(in_len=F, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    ytr_t = torch.from_numpy(y_train.astype(np.int64)).to(device)
    yte_t = torch.from_numpy(y_test.astype(np.int64)).to(device)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)

    n = Xtr_t.size(0)
    bs = int(args.batch_size)
    model.train()
    for epoch in range(int(args.epochs)):
        perm = torch.randperm(n, device=Xtr_t.device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = Xtr_t[idx]
            yb = ytr_t[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                pred = model(Xte_t).argmax(dim=1)
                acc = (pred.view(-1) == yte_t.view(-1)).float().mean().item()
            print(f"epoch {epoch + 1}/{args.epochs}  val_acc={acc:.4f}", flush=True)
            model.train()

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(dim=1).cpu().numpy()
    print(f"Hold-out accuracy: {accuracy_score(y_test, pred):.4f}", flush=True)
    print(classification_report(y_test, pred, target_names=list(le.classes_)), flush=True)

    os.makedirs(os.path.dirname(args.ckpt_out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.ckpt_out)

    # joblib + weights only (no live nn.Module — portable across machines)
    model_cpu = FlowConvNet(in_len=F, num_classes=num_classes).cpu()
    model_cpu.load_state_dict(model.cpu().state_dict())
    bundle = {
        "kind": "cnn",
        "imputer": imputer,
        "scaler": scaler,
        "label_encoder": le,
        "feature_columns": list(FEATURE_COLUMNS),
        "classes": list(le.classes_),
        "state_dict": model_cpu.state_dict(),
        "num_features": F,
        "num_classes": num_classes,
    }
    joblib.dump(bundle, args.bundle_out)
    print(f"Saved PyTorch weights to {args.ckpt_out}", flush=True)
    print(f"Saved inference bundle to {args.bundle_out}", flush=True)
