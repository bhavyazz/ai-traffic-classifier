# AI Traffic Classifier

A machine learning project to classify network traffic from AI services (ChatGPT, Claude, Copilot) versus non-AI applications using deep learning and ensemble methods.

## Project Overview

This project analyzes network traffic patterns captured from different AI services and trains multiple classification models to distinguish AI-generated traffic from non-AI traffic. The dataset includes PCAP files from various AI services and non-AI applications.

## Project Structure

```
├── capture/                    # Network packet capture utilities
│   └── capture.py
├── dashboard/                  # Visualization dashboard
│   └── app.py
├── data/
│   ├── raw/                   # Raw PCAP files from different services
│   │   ├── chatgpt_*.pcap
│   │   ├── claude_*.pcap
│   │   ├── copilot_*.pcap
│   │   └── nonai_*.pcap
│   └── processed/             # Extracted features
│       └── flows_features.csv
├── features/                  # Feature extraction module
│   ├── __init__.py
│   └── extractor.py
├── models/                    # Model training scripts
│   ├── __init__.py
│   ├── train_cnn.py          # CNN model training
│   ├── train_rf.py           # Random Forest model training
│   ├── train_xgboost.py      # XGBoost model training
│   └── artifacts/            # Trained model files
│       ├── cnn_model.pt
│       ├── cnn_bundle.joblib
│       ├── rf_model.joblib
│       └── xgb_model.joblib
├── evaluate/                 # Model evaluation
│   └── evaluate.py
├── paper/                    # Research results
│   └── results/
│       ├── metrics__cnn.json
│       ├── metrics__rf.json
│       ├── metrics__xgb.json
│       ├── final_metrics.json
│       ├── model_comparison_table.csv
│       └── model_comparison_table.md
└── requirements.txt          # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-traffic-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Feature Extraction

Extract features from raw PCAP files:
```bash
python features/extractor.py --input data/raw/ --output data/processed/flows_features.csv
```

### Model Training

Train individual models:

**CNN Model:**
```bash
python models/train_cnn.py
```

**Random Forest Model:**
```bash
python models/train_rf.py
```

**XGBoost Model:**
```bash
python models/train_xgboost.py
```

### Model Evaluation

Evaluate trained models:
```bash
python evaluate/evaluate.py
```

### Dashboard Visualization

Launch the interactive dashboard:
```bash
python dashboard/app.py
```

## Dataset

The project includes network traffic samples from:
- **ChatGPT**: 8 PCAP files
- **Claude**: 6 PCAP files
- **Copilot**: 6 PCAP files
- **Non-AI Applications**: 8 PCAP files

Total: 28 traffic capture files

### Features

Extracted network flow features include:
- Packet statistics (size, count, timing)
- Protocol information
- Flow duration and rate
- Statistical aggregations

Features are stored in `data/processed/flows_features.csv`

## Models

### 1. CNN (Convolutional Neural Network)
- Deep learning approach for traffic pattern recognition
- Artifacts: `models/artifacts/cnn_model.pt`, `models/artifacts/cnn_bundle.joblib`

### 2. Random Forest
- Ensemble method using decision trees
- Artifacts: `models/artifacts/rf_model.joblib`

### 3. XGBoost
- Gradient boosting approach
- Artifacts: `models/artifacts/xgb_model.joblib`

## Results

Model performance metrics are available in `paper/results/`:
- Individual model metrics (JSON format)
- Comparative analysis (CSV and Markdown)
- Final aggregated metrics

View the model comparison:
```bash
cat paper/results/model_comparison_table.md
```

## Requirements

See `requirements.txt` for all dependencies. Key packages include:
- PyTorch (for CNN)
- scikit-learn (for Random Forest)
- XGBoost (for gradient boosting)
- Scapy or dpkt (for packet processing)
- Flask (for dashboard)

## Network Traffic Capture

Use the capture module to capture new traffic:
```bash
python capture/capture.py --output data/raw/new_capture.pcap
```

## Contributing

To contribute to this project, please ensure:
1. All models are retrained and evaluated
2. Metrics are updated in `paper/results/`
3. Dashboard visualizations reflect latest model performance

## License

[Add your license information here]

## Contact

[Add contact information here]
