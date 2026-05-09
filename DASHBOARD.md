# AI Traffic Classifier Dashboard

A professional, polished Streamlit dashboard for classifying encrypted network traffic as ChatGPT, Claude, Copilot, or Non-AI using machine learning models.

## Features

### 🎨 Design & Interface
- **Dark Theme**: Professional dark background with accent colors
- **Color-Coded Classification**: 
  - ChatGPT = Green (#28a745)
  - Claude = Orange (#fd7e14)
  - Copilot = Blue (#0dcaf0)
  - Non-AI = Grey (#6c757d)
- **Mobile-Friendly**: Responsive layout with proper spacing
- **Professional Styling**: Clear typography and visual hierarchy
- **Large Prediction Text**: Prominent majority vote display
- **Progress Bars**: Visual confidence representation

### 📡 Tab 1: Live Capture
- **60-Second Capture Button**: Start live network traffic capture
- **Countdown Timer**: Real-time display of remaining capture time
- **Animated Status**: "Capturing..." status with visual feedback
- **Automatic Feature Extraction**: Auto-runs after capture completes
- **All 3 Model Predictions**: Random Forest, XGBoost, CNN results
- **Confidence Score Progress Bars**: Visual representation of model confidence
- **Majority Vote Highlighting**: Large text showing consensus prediction
- **Color-Coded Results**: Each model prediction color-coded by class
- **AI Detection Alert**: Red banner warns if AI tool detected
- **Traffic Distribution Pie Chart**: Breakdown of predicted classes
- **Per-Flow Predictions Table**: Detailed results for each flow

### 📤 Tab 2: Upload & Classify
- **PCAP File Uploader**: Accept .pcap files
- **Flow Count Display**: Shows number of flows extracted
- **Automatic Feature Extraction**: Auto-runs on upload
- **All 3 Model Execution**: RF, XGBoost, CNN predictions
- **Flow Statistics Cards**: 
  - Total flows
  - Average packets
  - Average bytes
  - Average duration
- **Majority Vote Classification**: Bold display of consensus
- **Individual Model Results**: Confidence bars for each model
- **Pie Chart**: Traffic classification distribution
- **Per-Flow Table**: Complete predictions with all details:
  - Flow number
  - Packet and byte counts
  - Duration
  - Predictions from each model
  - Confidence scores

### 📊 Tab 3: Results & Analytics
- **Model Comparison Table**: CSV loaded from paper/results/
- **Key Findings Summary Cards**:
  - Best model accuracy
  - Best performing class (F1)
  - Worst performing class (F1)
  - Test dataset size
- **Detailed Model Metrics**: Expandable sections for each model
  - Accuracy
  - Macro F1
  - Mean confidence
  - Per-class F1 scores
- **Confusion Matrices**: Images for all 3 models side-by-side
- **Calibration Curves**: Visual calibration analysis for all models

## Technical Implementation

### Model Loading
- **Random Forest** (`rf_model.joblib`): Pipeline with imputer + classifier
- **XGBoost** (`xgb_model.joblib`): Pipeline with imputer + classifier
- **CNN** (`cnn_bundle.joblib`): PyTorch model + label encoder

### Predictions
- All models generate probabilities
- Confidence = max class probability × 100
- Majority vote across all 3 models
- Per-flow predictions with individual model results

### Feature Extraction
- Integrates `features/extractor.py`
- Extracts 14 network flow features:
  - total_packets, total_bytes, flow_duration
  - mean/std/min/max packet sizes
  - mean/std interarrival times
  - direction ratio, burst count, burst average size
  - bytes/packets per second

### Error Handling
- Graceful handling when models not found
- File upload validation
- Feature extraction error messages
- Model prediction error handling
- Clear error/warning messages for user

## Running the Dashboard

### Prerequisites
```bash
pip install -r requirements.txt
```

### Start the Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will open at: `http://localhost:8501`

## Usage Examples

### Upload a PCAP
1. Go to "📤 Upload & Classify" tab
2. Click file uploader and select a .pcap file
3. Features auto-extract
4. All 3 models auto-run
5. View results with majority vote, confidence bars, and per-flow table

### Live Capture (if supported)
1. Go to "📡 Live Capture" tab
2. Set capture duration (default 60 seconds)
3. Click "🎬 Start Capture"
4. Watch countdown timer
5. Results appear after capture completes

### View Analytics
1. Go to "📊 Results & Analytics" tab
2. See model comparison metrics
3. View confusion matrices (3 side-by-side)
4. View calibration curves (3 side-by-side)
5. Explore detailed per-model metrics

## File Structure

```
dashboard/
├── app.py              # Main Streamlit dashboard
models/
├── artifacts/
│   ├── rf_model.joblib      # Random Forest trained model
│   ├── xgb_model.joblib     # XGBoost trained model
│   └── cnn_bundle.joblib    # CNN trained model
paper/
└── results/
    ├── model_comparison_table.csv
    ├── model_comparison_table.md
    ├── final_metrics.json
    ├── metrics__*.json
    ├── confusion_matrix__*.png
    └── calibration__*.png
```

## Features Summary

| Feature | Tab 1 | Tab 2 | Tab 3 |
|---------|-------|-------|-------|
| Live Capture | ✓ | - | - |
| File Upload | - | ✓ | - |
| Auto Feature Extract | ✓ | ✓ | - |
| All 3 Models | ✓ | ✓ | - |
| Confidence Bars | ✓ | ✓ | - |
| Majority Vote | ✓ | ✓ | - |
| Color Coding | ✓ | ✓ | - |
| AI Detection Alert | ✓ | ✓ | - |
| Pie Chart | ✓ | ✓ | - |
| Per-Flow Table | ✓ | ✓ | - |
| Model Metrics | - | - | ✓ |
| Confusion Matrices | - | - | ✓ |
| Calibration Curves | - | - | ✓ |

## Design Highlights

✨ **Professional Dark Theme**: Easy on the eyes during extended use

🎨 **Color-Coded Throughout**: Instant visual recognition of traffic class

📊 **Data Visualization**: Pie charts, progress bars, and tables

📈 **Comprehensive Analytics**: Side-by-side model comparison

⚠️ **Alert Banners**: Clear warning when AI tool detected

🔄 **Automatic Processing**: No manual steps after upload/capture

📱 **Mobile Responsive**: Works on various screen sizes

💪 **Robust Error Handling**: Graceful failures with clear messages
