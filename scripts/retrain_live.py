"""
Interactive live capture → feature extraction → retraining pipeline.

This script walks you through:
1. Capturing 3 minutes each of ChatGPT, Claude, Copilot, and Non-AI traffic
2. Extracting features from the captures
3. Combining with existing training data
4. Retraining all 3 models (CNN, RF, XGBoost)
5. Displaying accuracy metrics

Usage:
    python scripts/retrain_live.py

Requires admin/root privileges to capture network traffic on most systems.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scapy.all import IP, TCP, UDP, sniff, wrpcap

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import project modules
from features.extractor import FEATURE_COLUMNS, extract_from_pcap
from models.train_cnn import FlowConvNet, load_xy as cnn_load_xy, predict_with_confidence as cnn_predict
from models.train_rf import build_pipeline as build_rf_pipeline, load_xy as rf_load_xy
from models.train_xgboost import build_pipeline as build_xgb_pipeline, load_xy as xgb_load_xy

# Configuration
CAPTURE_DURATION = 180  # 3 minutes in seconds
CAPTURE_LABELS = ("chatgpt", "claude", "copilot", "non_ai")
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"
ORIGINAL_CSV = PROCESSED_DIR / "flows_features.csv"
NEW_CAPTURES_CSV = PROCESSED_DIR / "new_captures_features.csv"
COMBINED_CSV = PROCESSED_DIR / "combined_features.csv"

CNN_BUNDLE_PATH = ARTIFACTS_DIR / "cnn_bundle.joblib"
CNN_CKPT_PATH = ARTIFACTS_DIR / "cnn_model.pt"
RF_MODEL_PATH = ARTIFACTS_DIR / "rf_model.joblib"
XGB_MODEL_PATH = ARTIFACTS_DIR / "xgb_model.joblib"

CLASS_NAMES = ["chatgpt", "claude", "copilot", "non_ai"]


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step: int, text: str) -> None:
    """Print a step number and description."""
    print(f"\n[Step {step}] {text}")


def next_pcap_index(label: str, outdir: Path) -> int:
    """Find the next NNN for label_NNN.pcap."""
    pattern = str(outdir / f"{label}_*.pcap")
    exists = glob.glob(pattern)
    best = 0
    for p in exists:
        m = re.match(rf".*{re.escape(label)}_(\d+)\.pcap$", p.replace("\\", "/"))
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def get_user_confirmation(prompt: str) -> bool:
    """Get yes/no confirmation from user."""
    while True:
        response = input(f"\n{prompt} [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        else:
            print("Please enter 'y' or 'n'.")


def capture_labeled_traffic(
    label: str,
    duration: float = CAPTURE_DURATION,
    outdir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Capture traffic for the specified label.
    
    Returns path to the saved PCAP file, or None if capture failed.
    """
    if outdir is None:
        outdir = RAW_DIR
    
    outdir.mkdir(parents=True, exist_ok=True)
    idx = next_pcap_index(label, outdir)
    out_path = outdir / f"{label}_{idx:03d}.pcap"
    
    print(f"\n  Capturing {label} traffic for {duration} seconds...")
    print(f"  Output will be saved to: {out_path}")
    print(f"  NOTE: If running on Windows, ensure you're running as Administrator!")
    
    if not get_user_confirmation("Ready to start capture?"):
        print("  Skipping this capture.")
        return None
    
    try:
        print(f"  ⏱️  CAPTURING FOR {duration} SECONDS...")
        start = time.time()
        pkts = sniff(timeout=float(duration))
        elapsed = time.time() - start
        
        wrpcap(str(out_path), pkts)
        print(f"  ✓ Captured {len(pkts)} packets in {elapsed:.1f}s")
        print(f"  ✓ Saved to {out_path}")
        return out_path
    except Exception as e:
        print(f"  ✗ Capture failed: {e}")
        print(f"  Make sure you're running with administrator privileges on Windows.")
        return None


def extract_features_from_new_captures(
    pcap_files: List[Path],
    output_csv: Path,
) -> pd.DataFrame:
    """
    Extract features from newly captured PCAP files.
    
    Returns the combined features DataFrame.
    """
    print(f"\n  Processing {len(pcap_files)} PCAP files...")
    
    dfs = []
    for pcap_file in pcap_files:
        print(f"    Extracting features from {pcap_file.name}...", end=" ", flush=True)
        try:
            # extract_from_pcap infers label from filename automatically
            df_flows = extract_from_pcap(str(pcap_file))
            dfs.append(df_flows)
            print(f"({len(df_flows)} flows)")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    if not dfs:
        raise ValueError("No features extracted from captures!")
    
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"\n  ✓ Saved {len(df)} flow features to {output_csv}")
    return df


def combine_datasets(
    original_csv: Path,
    new_csv: Path,
    combined_csv: Path,
) -> pd.DataFrame:
    """
    Combine original training data with new captures.
    
    Returns the combined DataFrame.
    """
    print(f"\n  Loading original training data...")
    original_df = pd.read_csv(original_csv)
    print(f"    ✓ Loaded {len(original_df)} flows")
    
    print(f"  Loading new capture features...")
    new_df = pd.read_csv(new_csv)
    print(f"    ✓ Loaded {len(new_df)} flows")
    
    combined_df = pd.concat([original_df, new_df], ignore_index=True)
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n  ✓ Combined dataset: {len(combined_df)} total flows")
    print(f"  ✓ Saved to {combined_csv}")
    
    return combined_df


def train_rf_model(
    data_csv: Path,
    model_out: Path,
) -> Tuple[float, str]:
    """Train Random Forest model and return (accuracy, report_string)."""
    print(f"\n  Loading data...")
    X, y, le = rf_load_xy(str(data_csv))
    
    print(f"  Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training Random Forest classifier...")
    pipe = build_rf_pipeline()
    pipe.fit(X_train, y_train)
    
    print(f"  Evaluating on test set...")
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, target_names=list(le.classes_))
    
    bundle = {
        "pipeline": pipe,
        "label_encoder": le,
        "feature_columns": list(FEATURE_COLUMNS),
        "classes": list(le.classes_),
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    print(f"  ✓ Saved RF model to {model_out}")
    
    return acc, report


def train_xgb_model(
    data_csv: Path,
    model_out: Path,
) -> Tuple[float, str]:
    """Train XGBoost model and return (accuracy, report_string)."""
    print(f"\n  Loading data...")
    X, y, le = xgb_load_xy(str(data_csv))
    
    print(f"  Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training XGBoost classifier...")
    pipe = build_xgb_pipeline(num_classes=len(le.classes_))
    pipe.fit(X_train, y_train)
    
    print(f"  Evaluating on test set...")
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, target_names=list(le.classes_))
    
    bundle = {
        "pipeline": pipe,
        "label_encoder": le,
        "feature_columns": list(FEATURE_COLUMNS),
        "classes": list(le.classes_),
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    print(f"  ✓ Saved XGBoost model to {model_out}")
    
    return acc, report


def train_cnn_model(
    data_csv: Path,
    bundle_out: Path,
    ckpt_out: Path,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Tuple[float, str]:
    """Train CNN model and return (accuracy, report_string)."""
    print(f"\n  Loading data...")
    X, y, le = cnn_load_xy(str(data_csv))
    
    print(f"  Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    print(f"  Preparing training data...")
    Xi_train = imputer.fit_transform(X_train)
    Xs_train = scaler.fit_transform(Xi_train).astype(np.float32)[:, np.newaxis, :]
    
    print(f"  Preparing test data...")
    Xi_test = imputer.transform(X_test)
    Xs_test = scaler.transform(Xi_test).astype(np.float32)[:, np.newaxis, :]
    
    Xtr = torch.from_numpy(Xs_train)
    Xte = torch.from_numpy(Xs_test)
    ytr = torch.from_numpy(y_train).long()
    yte = torch.from_numpy(y_test).long()
    
    num_classes = len(le.classes_)
    F = len(FEATURE_COLUMNS)
    model = FlowConvNet(in_len=F, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    
    print(f"  Training CNN for {epochs} epochs...")
    best_acc = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        indices = np.arange(len(Xtr))
        np.random.shuffle(indices)
        
        train_loss = 0.0
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_x = Xtr[batch_idx].to(device)
            batch_y = ytr[batch_idx].to(device)
            
            opt.zero_grad()
            logits = model(batch_x)
            loss = crit(logits, batch_y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            logits_te = model(Xte.to(device))
            pred_te = logits_te.argmax(dim=1).cpu().numpy()
            acc_te = accuracy_score(y_test, pred_te)
            if acc_te > best_acc:
                best_acc = acc_te
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{epochs}: train_loss={train_loss / (len(Xtr) / batch_size):.4f}, "
                  f"val_acc={acc_te:.4f}")
    
    print(f"  Evaluating final model...")
    model.eval()
    with torch.no_grad():
        logits_te = model(Xte.to(device))
        pred_te = logits_te.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, pred_te)
    
    report = classification_report(y_test, pred_te, target_names=list(le.classes_))
    
    # Save bundle and checkpoint
    bundle_out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    
    bundle = {
        "imputer": imputer,
        "scaler": scaler,
        "label_encoder": le,
        "state_dict": model.state_dict(),
        "num_features": F,
        "num_classes": num_classes,
        "feature_columns": list(FEATURE_COLUMNS),
        "classes": list(le.classes_),
    }
    joblib.dump(bundle, bundle_out)
    torch.save(model.state_dict(), ckpt_out)
    print(f"  ✓ Saved CNN bundle to {bundle_out}")
    print(f"  ✓ Saved CNN checkpoint to {ckpt_out}")
    
    return acc, report


def get_old_metrics() -> Dict[str, Any]:
    """
    Load existing model metrics before retraining.
    
    Returns a dict of model → accuracy.
    """
    metrics = {}
    for model_name in ["rf", "xgb", "cnn"]:
        try:
            metric_file = ARTIFACTS_DIR / f"metrics__{model_name}.json"
            if metric_file.exists():
                with open(metric_file) as f:
                    data = json.load(f)
                    metrics[model_name] = data.get("accuracy", None)
        except Exception:
            pass
    return metrics


def save_metrics(
    model_name: str,
    accuracy: float,
    report_dict: Dict[str, Any],
) -> None:
    """Save model metrics to JSON."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_file = ARTIFACTS_DIR / f"metrics__{model_name}.json"
    
    data = {
        "model": model_name,
        "accuracy": accuracy,
        "classification_report": report_dict,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    with open(metric_file, "w") as f:
        json.dump(data, f, indent=2)


def parse_report_to_dict(report_str: str) -> Dict[str, Any]:
    """Parse sklearn classification_report string to dict."""
    lines = report_str.split("\n")
    result = {"raw": report_str}
    return result


def main():
    print_header("AI Traffic Classifier - Live Capture & Retrain Pipeline")
    
    print("""
This interactive script will guide you through:

  1. Capturing live network traffic (3 min each):
     - ChatGPT traffic
     - Claude traffic
     - Copilot traffic
     - Non-AI (regular internet) traffic

  2. Extracting flow features from captures

  3. Combining with existing training data

  4. Retraining all 3 models:
     - Random Forest
     - XGBoost
     - CNN

  5. Comparing accuracy before/after

IMPORTANT NOTES:
  • Windows: Run as Administrator (right-click → Run as Administrator)
  • Linux/Mac: May need sudo or appropriate network privileges
  • Each capture takes 3 minutes
  • Total time: ~15 min + model training (5-10 min)

Press Enter to begin...
    """)
    input()
    
    # Step 1: Capture traffic
    print_step(1, "CAPTURING LIVE TRAFFIC")
    
    captured_files = []
    for label in CAPTURE_LABELS:
        print(f"\n{'─' * 70}")
        print(f"Capturing {label.upper()} traffic")
        print(f"{'─' * 70}")
        
        # Provide guidance for each capture type
        if label == "chatgpt":
            print("""
  Instructions:
  1. Open a terminal or PowerShell
  2. Run: curl https://api.openai.com/health -v
     (or visit ChatGPT in a browser)
  3. This script will capture during the 3-minute window
            """)
        elif label == "claude":
            print("""
  Instructions:
  1. Open a terminal or PowerShell
  2. Run: curl https://claude.ai -v
     (or visit Claude.ai in a browser)
  3. This script will capture during the 3-minute window
            """)
        elif label == "copilot":
            print("""
  Instructions:
  1. Open a terminal or PowerShell
  2. Run: curl https://www.bing.com/chat -v
     (or use Copilot in Edge/VS Code)
  3. This script will capture during the 3-minute window
            """)
        else:  # non_ai
            print("""
  Instructions:
  1. Open a terminal or PowerShell
  2. Run some regular internet activity:
     - Browse a website (e.g., curl https://example.com)
     - Download a file
     - Check email
  3. This script will capture during the 3-minute window
            """)
        
        pcap_file = capture_labeled_traffic(label, duration=CAPTURE_DURATION)
        if pcap_file:
            captured_files.append((label, pcap_file))
    
    if not captured_files:
        print("\n✗ No captures completed. Exiting.")
        return
    
    print(f"\n✓ Captured {len(captured_files)} PCAP files:")
    for label, path in captured_files:
        print(f"  • {label}: {path}")
    
    # Step 2: Extract features
    print_step(2, "EXTRACTING FEATURES")
    
    new_df = extract_features_from_new_captures(
        [path for _, path in captured_files],
        NEW_CAPTURES_CSV,
    )
    
    # Step 3: Combine with existing data
    print_step(3, "COMBINING WITH EXISTING TRAINING DATA")
    
    combined_df = combine_datasets(
        ORIGINAL_CSV,
        NEW_CAPTURES_CSV,
        COMBINED_CSV,
    )
    
    # Get old metrics
    print_step(4, "LOADING OLD MODEL METRICS")
    old_metrics = get_old_metrics()
    print(f"\n  Previous model accuracies:")
    for model_name in ["rf", "xgb", "cnn"]:
        if model_name in old_metrics:
            print(f"    • {model_name.upper()}: {old_metrics[model_name]:.4f}")
        else:
            print(f"    • {model_name.upper()}: No previous metrics")
    
    # Step 5: Retrain models
    print_step(5, "RETRAINING ALL MODELS")
    
    new_metrics = {}
    
    # Random Forest
    print_header("Retraining Random Forest Model")
    try:
        acc_rf, report_rf = train_rf_model(COMBINED_CSV, RF_MODEL_PATH)
        new_metrics["rf"] = acc_rf
        save_metrics("rf", acc_rf, parse_report_to_dict(report_rf))
        print(f"\n  Hold-out accuracy: {acc_rf:.4f}")
        print(f"\n{report_rf}")
    except Exception as e:
        print(f"\n  ✗ RF training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # XGBoost
    print_header("Retraining XGBoost Model")
    try:
        acc_xgb, report_xgb = train_xgb_model(COMBINED_CSV, XGB_MODEL_PATH)
        new_metrics["xgb"] = acc_xgb
        save_metrics("xgb", acc_xgb, parse_report_to_dict(report_xgb))
        print(f"\n  Hold-out accuracy: {acc_xgb:.4f}")
        print(f"\n{report_xgb}")
    except Exception as e:
        print(f"\n  ✗ XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # CNN
    print_header("Retraining CNN Model")
    try:
        acc_cnn, report_cnn = train_cnn_model(COMBINED_CSV, CNN_BUNDLE_PATH, CNN_CKPT_PATH)
        new_metrics["cnn"] = acc_cnn
        save_metrics("cnn", acc_cnn, parse_report_to_dict(report_cnn))
        print(f"\n  Hold-out accuracy: {acc_cnn:.4f}")
        print(f"\n{report_cnn}")
    except Exception as e:
        print(f"\n  ✗ CNN training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Show comparison
    print_step(6, "RETRAINING SUMMARY")
    print_header("Model Accuracy Comparison")
    
    print(f"{'Model':<15} {'Before':<15} {'After':<15} {'Change':<15}")
    print("─" * 60)
    
    for model_name in ["rf", "xgb", "cnn"]:
        before = old_metrics.get(model_name, None)
        after = new_metrics.get(model_name, None)
        
        if before is None:
            before_str = "N/A"
        else:
            before_str = f"{before:.4f}"
        
        if after is None:
            after_str = "N/A"
        else:
            after_str = f"{after:.4f}"
        
        if before is not None and after is not None:
            change = after - before
            change_str = f"{change:+.4f} ({change/before*100:+.1f}%)"
        else:
            change_str = "N/A"
        
        print(f"{model_name.upper():<15} {before_str:<15} {after_str:<15} {change_str:<15}")
    
    print("\n" + "=" * 70)
    print("✓ Retraining complete!")
    print("=" * 70)
    print(f"""
Saved files:
  • New PCAP captures: {[f[1] for f in captured_files]}
  • New features CSV: {NEW_CAPTURES_CSV}
  • Combined dataset: {COMBINED_CSV}
  • Updated models:
    - {RF_MODEL_PATH}
    - {XGB_MODEL_PATH}
    - {CNN_BUNDLE_PATH}
    - {CNN_CKPT_PATH}

Next steps:
  • Run: python evaluate/evaluate.py
    To get full evaluation metrics and confusion matrices
  
  • Check: paper/results/ for saved metrics
  
  • The models in models/artifacts/ are now retrained with
    your new data and will be used by the dashboard and
    classification scripts.
    """)


if __name__ == "__main__":
    main()
