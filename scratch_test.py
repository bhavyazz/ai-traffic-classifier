import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, r"d:\ai-traffic-classifier")
from dashboard.backend import load_models, get_predictions, get_majority_vote
from features.extractor import extract_from_pcap

load_models()

for pcap_name in ["chatgpt_001.pcap", "claude_001.pcap", "copilot_001.pcap", "nonai_001.pcap"]:
    pcap_file = rf"d:\ai-traffic-classifier\data\raw\{pcap_name}"
    print(f"\n--- Testing {pcap_name} ---")
    df = extract_from_pcap(pcap_file)
    predictions = get_predictions(df)
    majority_votes = get_majority_vote(predictions, len(df))
    from collections import Counter
    print(Counter(majority_votes))
