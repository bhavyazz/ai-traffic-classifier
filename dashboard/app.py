"""
AI Traffic Classifier Dashboard - Streamlit Application

Classify encrypted network traffic as ChatGPT, Claude, Copilot, or Non-AI
using machine learning models trained on network flow statistics.

Run from project root:
    streamlit run dashboard/app.py
"""

import json
import os
import subprocess
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Import feature extractor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.extractor import (
    extract_from_pcap,
    FEATURE_COLUMNS,
    label_from_pcap_path,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Traffic Classifier",
        layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme via custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fdfdfd 0%, #e5e5e5 100%);
        color: #1a1a1a;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    .main-title {
        font-size: 3.5em;
        font-weight: 900;
        background: linear-gradient(90deg, #111 0%, #777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1em;
        letter-spacing: -1px;
    }
    
    .sub-title {
        font-size: 1.2em;
        font-weight: 500;
        color: #555555;
        margin-bottom: 2em;
        letter-spacing: 0.5px;
        border-bottom: 2px solid rgba(0,0,0,0.1);
        padding-bottom: 1em;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.8);
        border-left: 5px solid #111;
        padding: 1.5em;
        border-radius: 14px;
        margin: 0.5em 0;
        color: #1a1a1a;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }
    
    .alert-banner {
        background: linear-gradient(135deg, #000000 0%, #333333 100%);
        color: #ffffff;
        padding: 1.5em;
        border-radius: 12px;
        margin: 1.5em 0;
        font-weight: 700;
        border: 1px solid #444444;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0,0,0,0.1); }
        70% { box-shadow: 0 0 0 15px rgba(0,0,0,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,0,0,0); }
    }
    
    .success-banner {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        color: #1a1a1a;
        padding: 1.5em;
        border-radius: 12px;
        margin: 1em 0;
        font-weight: 700;
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    }
    
    /* Enhance Streamlit buttons */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #111 0%, #444 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 0.6em 1.2em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
        color: white;
        border: none;
    }
    
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 2.2rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Color mapping for classes (Black & White Theme)
COLOR_MAP = {
    "chatgpt": "#000000",      # black
    "claude": "#333333",       # dark grey
    "copilot": "#666666",      # grey
    "non_ai": "#999999"        # light grey
}

CLASS_NAMES = ["chatgpt", "claude", "copilot", "non_ai"]
MODEL_NAMES = ["Random Forest", "XGBoost", "CNN"]

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "capture_results" not in st.session_state:
    st.session_state.capture_results = None
if "upload_results" not in st.session_state:
    st.session_state.upload_results = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_models() -> Dict:
    """Load all three trained models."""
    models = {}
    try:
        # Random Forest
        rf_path = MODELS_DIR / "rf_model.joblib"
        if rf_path.exists():
            models["rf"] = joblib.load(rf_path)
        else:
            st.warning(f"Random Forest model not found at {rf_path}")
        
        # XGBoost
        xgb_path = MODELS_DIR / "xgb_model.joblib"
        if xgb_path.exists():
            models["xgb"] = joblib.load(xgb_path)
        else:
            st.warning(f"XGBoost model not found at {xgb_path}")
        
        # CNN
        cnn_path = MODELS_DIR / "cnn_bundle.joblib"
        if cnn_path.exists():
            models["cnn"] = joblib.load(cnn_path)
        else:
            st.warning(f"CNN model not found at {cnn_path}")
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models


def predict_with_models(features_df: pd.DataFrame, models: Dict) -> Optional[Dict]:
    """Run predictions on extracted features with all 3 models."""
    results = {
        "rf": {"predictions": None, "confidence": None},
        "xgb": {"predictions": None, "confidence": None},
        "cnn": {"predictions": None, "confidence": None},
    }
    
    try:
        # Get feature matrix
        X = features_df[list(FEATURE_COLUMNS)]
        
        # Random Forest
        if "rf" in models and models["rf"]:
            rf_bundle = models["rf"]
            rf_pipe = rf_bundle["pipeline"]
            rf_le = rf_bundle["label_encoder"]
            rf_probs = rf_pipe.predict_proba(X)
            rf_idx = np.argmax(rf_probs, axis=1)
            rf_conf = rf_probs[np.arange(len(rf_idx)), rf_idx] * 100.0
            results["rf"]["predictions"] = rf_le.inverse_transform(rf_idx)
            results["rf"]["confidence"] = rf_conf
        
        # XGBoost
        if "xgb" in models and models["xgb"]:
            xgb_bundle = models["xgb"]
            xgb_pipe = xgb_bundle["pipeline"]
            xgb_le = xgb_bundle["label_encoder"]
            xgb_probs = xgb_pipe.predict_proba(X)
            xgb_idx = np.argmax(xgb_probs, axis=1)
            xgb_conf = xgb_probs[np.arange(len(xgb_idx)), xgb_idx] * 100.0
            results["xgb"]["predictions"] = xgb_le.inverse_transform(xgb_idx)
            results["xgb"]["confidence"] = xgb_conf
        
        # CNN
        if "cnn" in models and models["cnn"]:
            import torch
            from models.train_cnn import predict_with_confidence
            
            cnn_bundle = models["cnn"]
            try:
                preds, confs = predict_with_confidence(cnn_bundle, X, device=torch.device("cpu"))
                results["cnn"]["predictions"] = preds
                results["cnn"]["confidence"] = confs
            except Exception as e:
                import streamlit as st
                st.error(f"CNN prediction error: {e}")
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None
    
    return results


def get_majority_vote(predictions: Dict) -> Tuple[Optional[str], int]:
    """Get majority vote across all models for each flow."""
    votes = {}
    for model_key in ["rf", "xgb", "cnn"]:
        if predictions[model_key]["predictions"] is not None:
            for pred in predictions[model_key]["predictions"]:
                votes[pred] = votes.get(pred, 0) + 1
    
    if votes:
        majority = max(votes, key=votes.get)
        count = votes[majority]
        return majority, count
    return None, 0


def extract_features(pcap_path: str) -> Optional[pd.DataFrame]:
    """Extract features from a PCAP file."""
    try:
        with st.spinner("Extracting features..."):
            df = extract_from_pcap(pcap_path)
            if df is not None and len(df) > 0:
                return df
            else:
                st.error("No flows extracted from PCAP file")
                return None
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


def display_prediction_result(class_name: str, confidence: float, model_name: str):
    """Display a single model prediction with confidence bar."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        color = COLOR_MAP.get(class_name, "#666666")
        st.markdown(
            f"<div style='background-color: {color}; color: #ffffff; "
            f"padding: 0.5em; border-radius: 0.3em; text-align: center; font-weight: bold;'>"
            f"{class_name.upper()}</div>",
            unsafe_allow_html=True
        )
    
    with col2:
        # Cast to float to avoid numpy float32 error in Streamlit
        st.progress(float(confidence / 100.0))
    
    with col3:
        st.markdown(f"**{confidence:.1f}%**")


def display_majority_vote(predictions: Dict):
    """Display majority vote prediction prominently."""
    majority_class, vote_count = get_majority_vote(predictions)
    
    if majority_class:
        color = COLOR_MAP.get(majority_class, "#666666")
        st.markdown(
            f"""
            <div style='background-color: {color}; color: #ffffff; 
            padding: 2.5em; border-radius: 1em; text-align: center; 
            margin: 1.5em 0; font-size: 2.5em; font-weight: bold;'>
            {majority_class.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if majority_class != "non_ai":
            st.markdown(
                """
                <div class='alert-banner'>
                AI TOOL DETECTED - This traffic is likely from an AI service!
                </div>
                """,
                unsafe_allow_html=True
            )


def create_traffic_pie_chart(predictions: Dict) -> None:
    """Create pie chart of traffic classification."""
    import matplotlib.pyplot as plt
    
    vote_counts = {}
    for model_key in ["rf", "xgb", "cnn"]:
        if predictions[model_key]["predictions"] is not None:
            for pred in predictions[model_key]["predictions"]:
                vote_counts[pred] = vote_counts.get(pred, 0) + 1
    
    if vote_counts:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [COLOR_MAP.get(class_name, "#666666") for class_name in vote_counts.keys()]
        ax.pie(vote_counts.values(), labels=[k.upper() for k in vote_counts.keys()],
               autopct="%1.1f%%", colors=colors, startangle=90)
        ax.set_title("Traffic Classification Distribution", fontsize=14, fontweight='bold', color='#000000')
        ax.set_facecolor('#ffffff')
        fig.patch.set_facecolor('#ffffff')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def display_per_flow_table(features_df: pd.DataFrame, predictions: Dict) -> None:
    """Display per-flow predictions in a table."""
    rows = []
    
    for i in range(len(features_df)):
        row = {
            "Flow #": i + 1,
            "Packets": int(features_df.iloc[i]["total_packets"]),
            "Bytes": int(features_df.iloc[i]["total_bytes"]),
            "Duration (s)": f"{features_df.iloc[i]['flow_duration']:.2f}",
            "RF": predictions["rf"]["predictions"][i] if predictions["rf"]["predictions"] is not None else "N/A",
            "RF Conf": f"{predictions['rf']['confidence'][i]:.1f}%" if predictions["rf"]["confidence"] is not None else "N/A",
            "XGB": predictions["xgb"]["predictions"][i] if predictions["xgb"]["predictions"] is not None else "N/A",
            "XGB Conf": f"{predictions['xgb']['confidence'][i]:.1f}%" if predictions["xgb"]["confidence"] is not None else "N/A",
            "CNN": predictions["cnn"]["predictions"][i] if predictions["cnn"]["predictions"] is not None else "N/A",
            "CNN Conf": f"{predictions['cnn']['confidence'][i]:.1f}%" if predictions["cnn"]["confidence"] is not None else "N/A",
        }
        rows.append(row)
    
    df_display = pd.DataFrame(rows)
    st.dataframe(df_display, use_container_width=True)


# ============================================================================
# HEADER
# ============================================================================

st.markdown("<div class='main-title'>AI Traffic Classifier</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>ChatGPT vs Claude vs Copilot — Encrypted Traffic Classification Without Decryption</div>",
    unsafe_allow_html=True
)
st.divider()

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["Live Capture", "Upload & Classify", "Results & Analytics"])

# ============================================================================
# TAB 1: LIVE CAPTURE
# ============================================================================

with tab1:
    st.subheader("Live Traffic Capture & Classification")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        capture_duration = st.number_input("Capture Duration (seconds)", min_value=10, max_value=300, value=60, step=10)
    
    with col2:
        st.markdown("###")
        if st.button("Start Capture", key="capture_btn", use_container_width=True):
            st.session_state.capture_running = True
    
    if st.session_state.get("capture_running", False):
        # Show countdown timer
        placeholder_timer = st.empty()
        placeholder_status = st.empty()
        
        with placeholder_status.container():
            st.markdown("### Capturing... Please wait")
        
        # Live network capture using scapy
        import threading
        from scapy.all import sniff, wrpcap
        import tempfile
        import os
        
        tmp_pcap = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
        tmp_pcap.close()
        tmp_path = tmp_pcap.name
        
        capture_error = []
        
        def run_capture():
            try:
                packets = sniff(timeout=capture_duration)
                wrpcap(tmp_path, packets)
            except Exception as e:
                capture_error.append(str(e))
                
        capture_thread = threading.Thread(target=run_capture)
        capture_thread.start()

        # Capture countdown
        for remaining in range(capture_duration, 0, -1):
            with placeholder_timer.container():
                st.markdown(f"### {remaining} seconds remaining")
            time.sleep(1)
            
        capture_thread.join()
        
        with placeholder_timer.container():
            st.markdown("### Capture Complete!")
        with placeholder_status.container():
            st.success("Capture finished! Processing...")
        
        if capture_error:
            st.error(f"Error during capture: {capture_error[0]}")
        else:
            try:
                # Extract features from live capture
                features_df = extract_features(tmp_path)
                
                if features_df is not None:
                    # Filter out short background noise flows
                    features_df = features_df[
                        (features_df['total_packets'] >= 15) & 
                        (features_df['flow_duration'] >= 0.5)
                    ].reset_index(drop=True)
                    
                    if len(features_df) == 0:
                        st.warning("No significant network flows detected. Try generating more AI traffic during the capture window.")
                    else:
                        # Get predictions
                        models = load_models()
                        if any(v for v in models.values()):
                        predictions = predict_with_models(features_df, models)
                        
                        if predictions:
                            st.session_state.capture_results = {
                                "features": features_df,
                                "predictions": predictions
                            }
                            
                            # Display results
                            st.divider()
                            st.subheader("Classification Results")
                            
                            # Show flow statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Flows", len(features_df))
                            with col2:
                                st.metric("Avg Packets", f"{features_df['total_packets'].mean():.1f}")
                            with col3:
                                st.metric("Avg Bytes", f"{features_df['total_bytes'].mean():.0f}")
                            with col4:
                                st.metric("Avg Duration", f"{features_df['flow_duration'].mean():.2f}s")
                            
                            st.divider()
                            
                            # Show individual model predictions
                            st.markdown("### Model Predictions (First Flow)")
                            for i, model_name in enumerate(MODEL_NAMES):
                                model_key = ["rf", "xgb", "cnn"][i]
                                if predictions[model_key]["predictions"] is not None:
                                    pred_class = predictions[model_key]["predictions"][0]
                                    confidence = predictions[model_key]["confidence"][0]
                                    st.markdown(f"**{model_name}:**")
                                    display_prediction_result(pred_class, confidence, model_name)
                            
                            st.divider()
                            
                            # Show majority vote
                            st.markdown("### Majority Vote")
                            display_majority_vote(predictions)
                            
                            # Show traffic breakdown
                            st.markdown("### Traffic Distribution")
                            create_traffic_pie_chart(predictions)
                            
                            # Show per-flow table
                            st.markdown("### Per-Flow Predictions")
                            display_per_flow_table(features_df, predictions)
            
            except Exception as e:
                st.error(f"Error processing capture: {e}")
            finally:
                # Clean up temporary PCAP file
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except:
                    pass
        
        st.session_state.capture_running = False
    
    else:
        st.info("💡 Click the button above to start a live capture. "
                "The system will capture network traffic for the specified duration and automatically classify it.")


# ============================================================================
# TAB 2: UPLOAD & CLASSIFY
# ============================================================================

with tab2:
    st.subheader("Upload PCAP File & Classify Traffic")
    
    uploaded_file = st.file_uploader("Choose a PCAP file", type=["pcap"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Extract features
        features_df = extract_features(tmp_path)
        
        if features_df is not None:
            st.success(f"Extracted {len(features_df)} flows from PCAP")
            
            # Load models
            models = load_models()
            
            if any(v for v in models.values()):
                # Run predictions
                st.info("Running all 3 models...")
                predictions = predict_with_models(features_df, models)
                
                if predictions:
                    st.session_state.upload_results = {
                        "filename": uploaded_file.name,
                        "features": features_df,
                        "predictions": predictions
                    }
                    
                    st.divider()
                    st.subheader("Classification Results")
                    
                    # Flow statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Flows", len(features_df))
                    with col2:
                        st.metric("Avg Packets", f"{features_df['total_packets'].mean():.1f}")
                    with col3:
                        st.metric("Avg Bytes", f"{features_df['total_bytes'].mean():.0f}")
                    with col4:
                        st.metric("Avg Duration", f"{features_df['flow_duration'].mean():.2f}s")
                    
                    st.divider()
                    
                    # Majority vote
                    st.markdown("### Majority Vote Classification")
                    display_majority_vote(predictions)
                    
                    # Model predictions
                    st.markdown("### Individual Model Predictions (First Flow)")
                    for i, model_name in enumerate(MODEL_NAMES):
                        model_key = ["rf", "xgb", "cnn"][i]
                        if predictions[model_key]["predictions"] is not None:
                            pred_class = predictions[model_key]["predictions"][0]
                            confidence = predictions[model_key]["confidence"][0]
                            st.markdown(f"**{model_name}:**")
                            display_prediction_result(pred_class, confidence, model_name)
                    
                    st.divider()
                    
                    # Traffic breakdown pie chart
                    st.markdown("### Traffic Classification Distribution")
                    create_traffic_pie_chart(predictions)
                    
                    st.divider()
                    
                    # Per-flow table
                    st.markdown("### Per-Flow Predictions")
                    display_per_flow_table(features_df, predictions)
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    else:
        st.info("Upload a PCAP file to get started")


# ============================================================================
# TAB 3: RESULTS & ANALYTICS
# ============================================================================

with tab3:
    st.subheader("Model Results & Analytics")
    
    # Check what results are available
    if not RESULTS_DIR.exists():
        st.error(f"Results directory not found: {RESULTS_DIR}")
    else:
        # Load model comparison table
        comparison_csv = RESULTS_DIR / "model_comparison_table.csv"
        if comparison_csv.exists():
            st.markdown("### Model Comparison")
            comparison_df = pd.read_csv(comparison_csv)
            st.dataframe(comparison_df, use_container_width=True)
        
        st.divider()
        
        # Load metrics JSON
        metrics_json = RESULTS_DIR / "final_metrics.json"
        if metrics_json.exists():
            with open(metrics_json) as f:
                metrics = json.load(f)
            
            st.markdown("### Key Findings")
            
            # Find best model by accuracy
            models_acc = {}
            for model_name in ["RandomForest", "XGBoost", "CNN"]:
                if model_name in metrics.get("models", {}):
                    models_acc[model_name] = metrics["models"][model_name].get("accuracy", 0)
            
            if models_acc:
                best_model = max(models_acc, key=models_acc.get)
                best_accuracy = models_acc[best_model]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                        <div style='font-size: 0.9em; color: #555555;'>Best Model Accuracy</div>
                        <div style='font-size: 1.8em; font-weight: bold; color: #000000;'>{best_accuracy:.2%}</div>
                        <div style='font-size: 0.85em; color: #555555;'>{best_model}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Find best performing class
                if best_model in metrics["models"]:
                    per_class_f1 = metrics["models"][best_model].get("per_class_f1", {})
                    if per_class_f1:
                        best_class = max(per_class_f1, key=per_class_f1.get)
                        best_f1 = per_class_f1[best_class]
                        
                        with col2:
                            st.markdown(
                                f"""
                                <div class='metric-card'>
                                <div style='font-size: 0.9em; color: #555555;'>Best Class (F1)</div>
                                <div style='font-size: 1.8em; font-weight: bold; color: {COLOR_MAP.get(best_class, "#000000")};'>{best_f1:.2%}</div>
                                <div style='font-size: 0.85em; color: #555555;'>{best_class.upper()}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # Find worst performing class
                        worst_class = min(per_class_f1, key=per_class_f1.get)
                        worst_f1 = per_class_f1[worst_class]
                        
                        with col3:
                            st.markdown(
                                f"""
                                <div class='metric-card'>
                                <div style='font-size: 0.9em; color: #555555;'>Worst Class (F1)</div>
                                <div style='font-size: 1.8em; font-weight: bold; color: {COLOR_MAP.get(worst_class, "#000000")};'>{worst_f1:.2%}</div>
                                <div style='font-size: 0.85em; color: #555555;'>{worst_class.upper()}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
                with col4:
                    total_test = metrics.get("test_flows", 0)
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                        <div style='font-size: 0.9em; color: #555555;'>Test Dataset Size</div>
                        <div style='font-size: 1.8em; font-weight: bold; color: #000000;'>{total_test}</div>
                        <div style='font-size: 0.85em; color: #555555;'>flows</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            st.divider()
            
            # Detailed metrics per model
            st.markdown("### Detailed Model Metrics")
            for model_name in ["RandomForest", "XGBoost", "CNN"]:
                if model_name in metrics.get("models", {}):
                    model_metrics = metrics["models"][model_name]
                    
                    with st.expander(f"{model_name}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.4f}")
                        with col2:
                            st.metric("Macro F1", f"{model_metrics.get('macro_f1', 0):.4f}")
                        with col3:
                            st.metric("Mean Confidence", f"{model_metrics.get('mean_confidence_pct', 0):.1f}%")
                        
                        # Per-class metrics
                        if "per_class_f1" in model_metrics:
                            st.markdown("**Per-Class F1 Scores:**")
                            per_class_data = []
                            for class_name, f1_score in model_metrics["per_class_f1"].items():
                                per_class_data.append({
                                    "Class": class_name.upper(),
                                    "F1 Score": f"{f1_score:.4f}"
                                })
                            st.dataframe(pd.DataFrame(per_class_data), use_container_width=True)
        
        st.divider()
        
        # Display confusion matrices
        st.markdown("### Confusion Matrices")
        
        confusion_files = {
            "Random Forest": RESULTS_DIR / "confusion_matrix__rf.png",
            "XGBoost": RESULTS_DIR / "confusion_matrix__xgb.png",
            "CNN": RESULTS_DIR / "confusion_matrix__cnn.png",
        }
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for (model_name, filepath), col in zip(confusion_files.items(), cols):
            if filepath.exists():
                with col:
                    st.markdown(f"**{model_name}**")
                    img = Image.open(filepath)
                    st.image(img, use_column_width=True)
        
        st.divider()
        
        # Display calibration curves
        st.markdown("### Calibration Curves")
        
        calibration_files = {
            "Random Forest": RESULTS_DIR / "calibration__rf.png",
            "XGBoost": RESULTS_DIR / "calibration__xgb.png",
            "CNN": RESULTS_DIR / "calibration__cnn.png",
        }
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for (model_name, filepath), col in zip(calibration_files.items(), cols):
            if filepath.exists():
                with col:
                    st.markdown(f"**{model_name}**")
                    img = Image.open(filepath)
                    st.image(img, use_column_width=True)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
**AI Traffic Classifier** — Classifying encrypted network traffic using machine learning
- ChatGPT, Claude, Copilot, and Non-AI traffic detection
- Based on network flow statistics (packet sizes, timings, directions)
- No traffic decryption required
- Built with Random Forest, XGBoost, and CNN models

📚 For more information, see the project README.
""")
