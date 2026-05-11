import os
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from features.extractor import extract_from_pcap, FEATURE_COLUMNS

app = FastAPI(title="AI Traffic Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"
FRONTEND_DIR = Path(__file__).parent / "frontend"
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Load models in memory
models_cache = {}

def load_models():
    if not models_cache:
        try:
            rf_path = MODELS_DIR / "rf_model.joblib"
            xgb_path = MODELS_DIR / "xgb_model.joblib"
            cnn_path = MODELS_DIR / "cnn_bundle.joblib"
            
            if rf_path.exists(): models_cache["rf"] = joblib.load(rf_path)
            if xgb_path.exists(): models_cache["xgb"] = joblib.load(xgb_path)
            if cnn_path.exists(): models_cache["cnn"] = joblib.load(cnn_path)
        except Exception as e:
            print(f"Error loading models: {e}")

def get_predictions(features_df: pd.DataFrame) -> Dict:
    results = {
        "rf": {"predictions": [], "confidence": []},
        "xgb": {"predictions": [], "confidence": []},
        "cnn": {"predictions": [], "confidence": []},
    }
    
    if features_df is None or len(features_df) == 0:
        return results
        
    X = features_df[list(FEATURE_COLUMNS)]
    
    # RF
    if "rf" in models_cache:
        rf_pipe = models_cache["rf"]["pipeline"]
        rf_le = models_cache["rf"]["label_encoder"]
        rf_probs = rf_pipe.predict_proba(X)
        rf_idx = np.argmax(rf_probs, axis=1)
        rf_conf = rf_probs[np.arange(len(rf_idx)), rf_idx] * 100.0
        results["rf"]["predictions"] = rf_le.inverse_transform(rf_idx).tolist()
        results["rf"]["confidence"] = rf_conf.tolist()
        
    # XGBoost
    if "xgb" in models_cache:
        xgb_pipe = models_cache["xgb"]["pipeline"]
        xgb_le = models_cache["xgb"]["label_encoder"]
        xgb_probs = xgb_pipe.predict_proba(X)
        xgb_idx = np.argmax(xgb_probs, axis=1)
        xgb_conf = xgb_probs[np.arange(len(xgb_idx)), xgb_idx] * 100.0
        results["xgb"]["predictions"] = xgb_le.inverse_transform(xgb_idx).tolist()
        results["xgb"]["confidence"] = xgb_conf.tolist()
        
    # CNN
    if "cnn" in models_cache:
        import torch
        from models.train_cnn import predict_with_confidence
        try:
            preds, confs = predict_with_confidence(models_cache["cnn"], X, device=torch.device("cpu"))
            results["cnn"]["predictions"] = preds.tolist()
            results["cnn"]["confidence"] = confs.tolist()
        except Exception as e:
            print(f"CNN error: {e}")
            
    return results

def get_majority_vote(predictions: Dict, num_flows: int) -> list:
    votes = []
    for i in range(num_flows):
        flow_votes = {}
        for m in ["rf", "xgb", "cnn"]:
            if predictions[m]["predictions"] and len(predictions[m]["predictions"]) > i:
                pred = predictions[m]["predictions"][i]
                flow_votes[pred] = flow_votes.get(pred, 0) + 1
        
        if flow_votes:
            majority = max(flow_votes, key=flow_votes.get)
            votes.append(majority)
        else:
            votes.append("unknown")
    return votes

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.post("/api/upload")
async def upload_pcap(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        features_df = extract_from_pcap(tmp_path)
        if features_df is None or len(features_df) == 0:
            raise HTTPException(status_code=400, detail="No valid flows found in PCAP.")
            
        predictions = get_predictions(features_df)
        majority_votes = get_majority_vote(predictions, len(features_df))
        
        # Prepare response
        flows_data = []
        for i in range(len(features_df)):
            row = features_df.iloc[i]
            flows_data.append({
                "id": i + 1,
                "packets": int(row["total_packets"]),
                "bytes": int(row["total_bytes"]),
                "duration": float(row["flow_duration"]),
                "rf_pred": predictions["rf"]["predictions"][i] if predictions["rf"]["predictions"] else None,
                "rf_conf": float(predictions["rf"]["confidence"][i]) if predictions["rf"]["confidence"] else None,
                "xgb_pred": predictions["xgb"]["predictions"][i] if predictions["xgb"]["predictions"] else None,
                "xgb_conf": float(predictions["xgb"]["confidence"][i]) if predictions["xgb"]["confidence"] else None,
                "cnn_pred": predictions["cnn"]["predictions"][i] if predictions["cnn"]["predictions"] else None,
                "cnn_conf": float(predictions["cnn"]["confidence"][i]) if predictions["cnn"]["confidence"] else None,
                "majority": majority_votes[i]
            })
            
        overall_majority = max(set(majority_votes), key=majority_votes.count) if majority_votes else "unknown"
        
        return {
            "overall_prediction": overall_majority,
            "total_flows": len(features_df),
            "flows": flows_data
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

class CaptureRequest(BaseModel):
    duration: int = 60

@app.post("/api/capture")
async def capture_traffic(req: CaptureRequest):
    from scapy.all import sniff, wrpcap
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp_path = tmp.name
        
    try:
        packets = sniff(timeout=req.duration)
        wrpcap(tmp_path, packets)
        
        features_df = extract_from_pcap(tmp_path)
        if features_df is None or len(features_df) == 0:
            return {"error": "No significant flows captured.", "total_flows": 0}
            
        # Filter tiny flows like in original app
        features_df = features_df[
            (features_df['total_packets'] >= 15) & 
            (features_df['flow_duration'] >= 0.5)
        ].reset_index(drop=True)
        
        if len(features_df) == 0:
            return {"error": "No significant flows captured after filtering.", "total_flows": 0}
            
        predictions = get_predictions(features_df)
        majority_votes = get_majority_vote(predictions, len(features_df))
        
        flows_data = []
        for i in range(len(features_df)):
            row = features_df.iloc[i]
            flows_data.append({
                "id": i + 1,
                "packets": int(row["total_packets"]),
                "bytes": int(row["total_bytes"]),
                "duration": float(row["flow_duration"]),
                "rf_pred": predictions["rf"]["predictions"][i] if predictions["rf"]["predictions"] else None,
                "rf_conf": float(predictions["rf"]["confidence"][i]) if predictions["rf"]["confidence"] else None,
                "xgb_pred": predictions["xgb"]["predictions"][i] if predictions["xgb"]["predictions"] else None,
                "xgb_conf": float(predictions["xgb"]["confidence"][i]) if predictions["xgb"]["confidence"] else None,
                "cnn_pred": predictions["cnn"]["predictions"][i] if predictions["cnn"]["predictions"] else None,
                "cnn_conf": float(predictions["cnn"]["confidence"][i]) if predictions["cnn"]["confidence"] else None,
                "majority": majority_votes[i]
            })
            
        overall_majority = max(set(majority_votes), key=majority_votes.count) if majority_votes else "unknown"
        
        return {
            "overall_prediction": overall_majority,
            "total_flows": len(features_df),
            "flows": flows_data
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/api/analytics")
def get_analytics():
    data = {"models": {}}
    for m in ["rf", "xgb", "cnn"]:
        path = RESULTS_DIR / f"metrics__{m}.json"
        if path.exists():
            with open(path) as f:
                data["models"][m] = json.load(f)
    return data
