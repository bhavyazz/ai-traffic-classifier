import os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys

# 1. Confusion Matrix for XGB
# Wait, evaluate.py does train_test_split with random_state=42.
# Wait, evaluate.py does train_test_split with random_state=42.
# Let's just run it the same way
from sklearn.model_selection import train_test_split
from features.extractor import FEATURE_COLUMNS

df = pd.read_csv('data/processed/flows_features.csv')
X = df[list(FEATURE_COLUMNS)]
y_str = df['label']
# le from bundle
bundle = joblib.load('models/artifacts/xgb_model.joblib')
le = bundle['label_encoder']
pipe = bundle['pipeline']

y = le.transform(y_str)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

y_pred = pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("XGBoost Confusion Matrix:")
print("Classes:", le.classes_)
print(cm)

# 2. Raw File Stats
raw_dir = 'data/raw'
stats = {}
total_size = 0
for f in os.listdir(raw_dir):
    if f.endswith('.pcap'):
        cls = f.split('_')[0]
        if cls == 'nonai': cls = 'non_ai'
        if cls not in stats: stats[cls] = {'count': 0, 'size': 0}
        size = os.path.getsize(os.path.join(raw_dir, f))
        stats[cls]['count'] += 1
        stats[cls]['size'] += size
        total_size += size

print("\nRaw File Stats:")
for cls, data in stats.items():
    print(f"{cls}: {data['count']} files, {data['size'] / (1024*1024):.2f} MB")
print(f"Total: {sum(d['count'] for d in stats.values())} files, {total_size / (1024*1024):.2f} MB")

# 3. Number of features
print(f"\nNumber of features: {len(FEATURE_COLUMNS)}")
