"""
Streamlit dashboard: upload a pcap or run live capture, then classify flows with
saved models (Random Forest, XGBoost, 1D CNN). Shows confidence bars and an alert
when any AI-tool class is predicted.

Run from project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.extractor import extract_from_pcap, FEATURE_COLUMNS, flows_from_packets  # noqa: E402
from models.train_cnn import predict_with_confidence as cnn_predict  # noqa: E402
from models.train_rf import predict_with_confidence as rf_predict  # noqa: E402
from models.train_xgboost import predict_with_confidence as xgb_predict  # noqa: E402

try:
    from scapy.all import sniff
except ImportError:
    sniff = None  # type: ignore

MODEL_PATHS = {
    "Random Forest": ROOT / "models" / "artifacts" / "rf_model.joblib",
    "XGBoost": ROOT / "models" / "artifacts" / "xgb_model.joblib",
    "1D CNN": ROOT / "models" / "artifacts" / "cnn_bundle.joblib",
}

COLOR = {
    "chatgpt": "#2e7d32",
    "claude": "#ef6c00",
    "copilot": "#1565c0",
    "non_ai": "#757575",
}

VALID_LABELS = ["chatgpt", "claude", "copilot", "non_ai"]


@st.cache_resource
def load_bundle(path: Path):
    if not path.is_file():
        return None
    return joblib.load(path)


def aggregate_prediction(pred_df: pd.DataFrame) -> tuple[str, float]:
    """Majority vote weighted by total_bytes."""
    if pred_df.empty:
        return "non_ai", 0.0
    w = pred_df["total_bytes"].astype(float).values
    labels = pred_df["pred_label"].astype(str).values
    conf = pred_df["confidence_pct"].astype(float).values
    scores: dict[str, float] = {c: 0.0 for c in VALID_LABELS}
    for lab, wt, cf in zip(labels, w, conf):
        scores[lab] = scores.get(lab, 0.0) + wt * (cf / 100.0)
    top = max(scores, key=scores.get)
    sub = pred_df[pred_df["pred_label"] == top]
    mean_conf = float(sub["confidence_pct"].mean()) if len(sub) else 0.0
    return top, mean_conf


def run_classifiers(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, path in MODEL_PATHS.items():
        bundle = load_bundle(path)
        if bundle is None:
            continue
        if name == "1D CNN":
            pred, conf = cnn_predict(bundle, X)
        elif name == "XGBoost":
            pred, conf = xgb_predict(bundle, X)
        else:
            pred, conf = rf_predict(bundle, X)
        d = X.copy()
        d["pred_label"] = pred
        d["confidence_pct"] = conf
        out[name] = d
    return out


def main() -> None:
    st.set_page_config(page_title="AI Traffic Classifier", layout="wide")
    st.title("Fingerprinting AI Tools from Encrypted Traffic")
    st.caption("Metadata-only features — no decryption.")

    if "feature_df" not in st.session_state:
        st.session_state.feature_df = None

    tab_up, tab_live, tab_sub = st.tabs(["Upload PCAP", "Live capture", "Run capture script"])

    with tab_up:
        up = st.file_uploader("PCAP file", type=["pcap", "cap"])
        if up is not None and st.button("Classify upload", type="primary"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
            tmp.write(up.getvalue())
            tmp.close()
            try:
                st.session_state.feature_df = extract_from_pcap(tmp.name)
            finally:
                os.unlink(tmp.name)

    with tab_live:
        st.info(
            "Live capture requires OS permissions (often Administrator on Windows). "
            "If sniff fails, use the Capture script tab or: "
            "`python capture/capture.py --label chatgpt --duration 60`."
        )
        cap_label = st.selectbox("Session label (metadata only)", VALID_LABELS, index=3, key="caplab")
        duration = st.number_input("Duration (seconds)", min_value=5.0, value=30.0, step=5.0)
        iface = st.text_input("Interface (blank = default)", value="", key="iface")
        if st.button("Sniff & classify", type="primary", key="sniffbtn"):
            if sniff is None:
                st.error("Scapy not available.")
            else:
                with st.spinner(f"Sniffing for {duration}s..."):
                    kwargs: dict = {"timeout": float(duration)}
                    if iface.strip():
                        kwargs["iface"] = iface.strip()
                    pkts = sniff(**kwargs)
                st.write(f"Captured {len(pkts)} packets")
                if not pkts:
                    st.warning("No packets — try Administrator mode or set Interface.")
                    st.session_state.feature_df = None
                else:
                    df = flows_from_packets(pkts)
                    df["source_pcap"] = "live_sniff"
                    df["label"] = cap_label
                    st.session_state.feature_df = df

    with tab_sub:
        st.markdown(
            "Runs `capture/capture.py` as a subprocess (good when in-app sniff lacks permissions)."
        )
        sub_label = st.selectbox("Label", VALID_LABELS, index=0, key="sublab")
        sub_dur = st.number_input("Duration (s)", min_value=5.0, value=60.0, key="subdur")
        sub_iface = st.text_input("iface (optional)", value="", key="subiface")
        if st.button("Run capture script", key="subrun"):
            cmd = [
                sys.executable,
                str(ROOT / "capture" / "capture.py"),
                "--label",
                sub_label,
                "--duration",
                str(sub_dur),
            ]
            if sub_iface.strip():
                cmd.extend(["--iface", sub_iface.strip()])
            try:
                r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
                st.code(r.stdout + r.stderr)
            except OSError as e:
                st.error(str(e))

    frames: pd.DataFrame | None = st.session_state.feature_df
    if frames is None or frames.empty:
        st.stop()

    X = frames[list(FEATURE_COLUMNS)]
    st.subheader("Flow features (sample)")
    st.dataframe(frames.head(50), use_container_width=True)

    results = run_classifiers(X)
    if not results:
        st.error(
            "No trained models found under models/artifacts/. "
            "Run models/train_rf.py, train_xgboost.py, and train_cnn.py first."
        )
        st.stop()

    for model_name, pred_df in results.items():
        st.markdown(f"### {model_name}")
        top_label, top_conf = aggregate_prediction(pred_df)
        if top_label != "non_ai":
            st.markdown(
                f"<div style='padding:12px;background:#fff3e0;border-left:6px solid {COLOR.get(top_label, '#333')};'>"
                f"<b>AI tool traffic detected:</b> {top_label.upper()} "
                f"(aggregate confidence ~ {top_conf:.1f}%)"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.success("No AI tool class won aggregate vote (closest: non_ai).")

        cols = st.columns(4)
        for i, cls in enumerate(VALID_LABELS):
            sub = pred_df[pred_df["pred_label"] == cls]
            mean_c = float(sub["confidence_pct"].mean()) if len(sub) else 0.0
            frac_flows = len(sub) / max(len(pred_df), 1)
            with cols[i % 4]:
                st.caption(cls.replace("_", " ").title())
                st.progress(min(mean_c / 100.0, 1.0))
                st.caption(f"{mean_c:.0f}% mean conf · {frac_flows * 100:.0f}% flows")

        vc = pred_df["pred_label"].astype(str).value_counts().reset_index()
        vc.columns = ["class", "flows"]
        fig = px.bar(
            vc,
            x="class",
            y="flows",
            color="class",
            color_discrete_map=COLOR,
            title=f"{model_name} — flows per predicted class",
        )
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Per-flow predictions"):
            show = pred_df[
                list(FEATURE_COLUMNS)
                + [c for c in ("pred_label", "confidence_pct", "flow_key", "source_pcap") if c in pred_df.columns]
            ]
            st.dataframe(show, use_container_width=True)


if __name__ == "__main__":
    main()
