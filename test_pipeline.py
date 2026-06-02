#!/usr/bin/env python3
"""
Quick test of the classification pipeline.
Tests: model loading, feature extraction, predictions.
"""

import sys
from pathlib import Path

print("=" * 60)
print("Testing AI Traffic Classifier Pipeline")
print("=" * 60)

# 1. Test imports
print("\n[1/4] Testing imports...")
try:
    import joblib
    import numpy as np
    import pandas as pd
    from scapy.all import rdpcap
    print("✓ Core libraries imported")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# 2. Test feature extractor
print("\n[2/4] Testing feature extraction...")
try:
    PROJECT_ROOT = Path(__file__).parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from features.extractor import extract_from_pcap, FEATURE_COLUMNS, LABELS
    print(f"✓ Feature extractor loaded. Features: {len(FEATURE_COLUMNS)}")
    print(f"  Labels: {LABELS}")
except Exception as e:
    print(f"✗ Feature extractor failed: {e}")
    sys.exit(1)

# 3. Test model loading
print("\n[3/4] Testing model loading...")
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
models = {}

for model_name in ["rf", "xgb", "cnn"]:
    if model_name == "cnn":
        path = MODELS_DIR / "cnn_bundle.joblib"
    else:
        path = MODELS_DIR / f"{model_name}_model.joblib"
    
    if path.exists():
        try:
            models[model_name] = joblib.load(path)
            print(f"✓ Loaded {model_name.upper()} model from {path.name}")
        except Exception as e:
            print(f"✗ Failed to load {model_name.upper()}: {e}")
    else:
        print(f"⚠ {model_name.upper()} model not found: {path}")

if not models:
    print("✗ No models loaded!")
    sys.exit(1)

print(f"\n✓ {len(models)} model(s) loaded successfully")

# 4. Test with a sample PCAP if available
print("\n[4/4] Testing with sample PCAP...")
sample_pcaps = sorted(Path(PROJECT_ROOT / "data" / "raw").glob("*.pcap"))[:1]

if sample_pcaps:
    pcap_path = sample_pcaps[0]
    print(f"  Testing with: {pcap_path.name}")
    
    try:
        features_df = extract_from_pcap(str(pcap_path))
        if features_df is not None and len(features_df) > 0:
            print(f"✓ Extracted {len(features_df)} flows from PCAP")
            
            # Try prediction with RF
            if "rf" in models:
                try:
                    rf_pipe = models["rf"]["pipeline"]
                    rf_le = models["rf"]["label_encoder"]
                    
                    X = features_df[list(FEATURE_COLUMNS)]
                    preds = rf_pipe.predict(X)
                    preds_str = rf_le.inverse_transform(preds)
                    
                    print(f"✓ RF predictions: {preds_str[:3]}")
                    print(f"  Sample row: {X.iloc[0].to_dict()}")
                except Exception as e:
                    print(f"✗ Prediction failed: {e}")
        else:
            print("⚠ No flows extracted from PCAP")
    except Exception as e:
        print(f"✗ PCAP processing failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ No PCAP files found in data/raw/")

print("\n" + "=" * 60)
print("Pipeline test complete!")
print("=" * 60)
