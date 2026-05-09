# Scripts Directory

This directory contains automation scripts for the AI Traffic Classifier project.

## retrain_live.py

**Interactive live capture → feature extraction → model retraining pipeline.**

### Overview

This script automates the complete workflow for:
1. **Live traffic capture** (3 minutes each) for ChatGPT, Claude, Copilot, and Non-AI traffic
2. **Feature extraction** from captured PCAP files
3. **Dataset combination** with existing training data
4. **Model retraining** for all 3 classifiers (Random Forest, XGBoost, CNN)
5. **Accuracy comparison** before vs. after retraining

### Usage

```bash
python scripts/retrain_live.py
```

### Prerequisites

#### Environment Setup

1. **Python 3.10+** with all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Administrator/Root Privileges**:
   - **Windows**: Right-click PowerShell/Command Prompt → "Run as Administrator"
   - **Linux/Mac**: Use `sudo python scripts/retrain_live.py` or ensure network capture permissions

3. **Network Interface Access**:
   - On Windows: Requires admin privileges to access raw sockets
   - On Linux: May require `cap_net_admin` capability or sudo
   - On Mac: May require `ChmodBPF` or sudo

### How It Works

#### Step 1: Interactive Guided Capture

The script walks you through capturing 3 minutes of traffic for each type:

```
Capturing CHATGPT traffic
────────────────────────────────────────

Instructions:
1. Open a terminal or PowerShell
2. Run: curl https://api.openai.com/health -v
   (or visit ChatGPT in a browser)
3. This script will capture during the 3-minute window

Ready to start capture? [y/n]:
```

**For each traffic type:**
- **ChatGPT**: Make API calls or browse ChatGPT interface
- **Claude**: Access Claude.ai or use API
- **Copilot**: Use Bing Chat or Copilot in Edge/VS Code
- **Non-AI**: Regular browsing, downloads, file transfers, etc.

The script automatically:
- Captures all network packets during the 3-minute window
- Saves to `data/raw/[label]_NNN.pcap` with automatic numbering
- Displays packet count and confirmation

#### Step 2: Feature Extraction

The script automatically:
- Reads each newly captured PCAP file
- Extracts 14 flow-level features (packet count, bytes, timing, direction, bursts, etc.)
- Saves features to `data/processed/new_captures_features.csv`

Example features extracted:
- `total_packets`, `total_bytes`, `flow_duration`
- `mean_packet_size`, `std_packet_size`
- `mean_interarrival_time`, `std_interarrival_time`
- `direction_ratio`, `burst_count`, `burst_avg_size`
- `bytes_per_second`, `packets_per_second`

#### Step 3: Dataset Combination

The script:
- Loads original training data from `data/processed/flows_features.csv`
- Combines with newly extracted features
- Creates `data/processed/combined_features.csv`
- Reports total flow count and split

#### Step 4: Model Retraining

All 3 models are retrained on the combined dataset with 80/20 train/test split:

**Random Forest** (models/artifacts/rf_model.joblib)
- 300 trees, max_depth=None, balanced class weights
- Training time: ~1-2 minutes

**XGBoost** (models/artifacts/xgb_model.joblib)
- 400 boosting rounds, max_depth=8, learning_rate=0.05
- Training time: ~2-3 minutes

**CNN** (models/artifacts/cnn_bundle.joblib + cnn_model.pt)
- 1D Conv layers, 40 epochs, batch_size=64
- Training time: ~3-5 minutes (depends on GPU availability)

For each model, the script displays:
- Per-class precision, recall, F1-score
- Overall accuracy on held-out test set

#### Step 5: Accuracy Comparison

Final summary table showing:

```
Model          Before           After            Change
───────────────────────────────────────────────────────
RF             0.8523           0.8761           +0.0238 (+2.8%)
XGB            0.8614           0.8892           +0.0278 (+3.2%)
CNN            0.8401           0.8625           +0.0224 (+2.7%)
```

### Output Files

After running, check:

| Path | Description |
|------|-------------|
| `data/raw/chatgpt_*.pcap` | New ChatGPT capture files |
| `data/raw/claude_*.pcap` | New Claude capture files |
| `data/raw/copilot_*.pcap` | New Copilot capture files |
| `data/raw/nonai_*.pcap` | New Non-AI capture files |
| `data/processed/new_captures_features.csv` | Features extracted from captures |
| `data/processed/combined_features.csv` | Original + new combined dataset |
| `models/artifacts/rf_model.joblib` | Retrained Random Forest (overwritten) |
| `models/artifacts/xgb_model.joblib` | Retrained XGBoost (overwritten) |
| `models/artifacts/cnn_bundle.joblib` | Retrained CNN bundle (overwritten) |
| `models/artifacts/cnn_model.pt` | Retrained CNN checkpoint (overwritten) |
| `models/artifacts/metrics__rf.json` | RF accuracy metrics |
| `models/artifacts/metrics__xgb.json` | XGBoost accuracy metrics |
| `models/artifacts/metrics__cnn.json` | CNN accuracy metrics |

### Next Steps

After retraining:

1. **Full Evaluation** - Get detailed metrics:
   ```bash
   python evaluate/evaluate.py
   ```
   Generates confusion matrices, calibration curves, and model comparison charts.

2. **Update Dashboard** - Retrained models are automatically used:
   ```bash
   python dashboard/app.py
   ```

3. **Classify New Traffic** - Use retrained models:
   ```bash
   python check_cnn.py --pcap some_capture.pcap
   ```

### Troubleshooting

#### "Cannot sniff on... No such device exists"
- **Windows**: Run as Administrator
- **Linux**: Use `sudo` or check available interfaces with `ip link`
- **Mac**: Use `sudo` or install `ChmodBPF`

#### "Zero packets captured"
- **Windows**: Make sure you're running as Administrator
- **Linux**: Check permissions: `getcap`
- Ensure network interface is active and traffic is flowing

#### "No features extracted"
- PCAP file may be empty or corrupted
- Try capturing again with active network traffic
- Verify the capture saved correctly: `tcpdump -n -r data/raw/chatgpt_001.pcap | head`

#### Training fails on CNN
- If using CPU (slow): Reduce `--epochs` or `--batch-size`
- If using GPU: Ensure CUDA drivers and PyTorch CUDA are installed
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

#### "ImportError: No module named 'scapy'"
```bash
pip install -r requirements.txt
```

### Tips for Better Results

1. **Capture diverse traffic**:
   - For ChatGPT: Varied prompts, coding questions, creative writing
   - For Claude: Mix of conversations, code analysis, summarization
   - For Copilot: Different query types in Bing Chat or IDE
   - For Non-AI: Browse different sites, download files, stream video

2. **Capture long enough**:
   - 3 minutes should capture 20-100 flows per service
   - More captures improve model robustness

3. **Low-traffic environments**:
   - If getting few flows, keep terminal open and do more activities
   - Consider capturing during peak internet usage

4. **Monitor training progress**:
   - Watch console output for per-epoch CNN accuracy
   - Check if models are improving over previous baselines

### Architecture Notes

- **Features**: Extracted from packet metadata only (no decryption)
- **Flow Definition**: 5-tuple bidirectional grouping (TCP/UDP)
- **Scaling**: StandardScaler applied per model
- **Imputation**: Median imputation for NaN values
- **Class Balance**: RF and XGB use balanced class weights

### See Also

- [Main README](../README.md)
- [Feature Extraction Details](../features/extractor.py)
- [Model Training Code](../models/)
- [Evaluation Script](../evaluate/evaluate.py)
