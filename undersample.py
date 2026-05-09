import pandas as pd
import numpy as np

# Load original data
df = pd.read_csv('data/processed/flows_features.csv')

print("Original dataset:")
print(df['label'].value_counts().sort_index())
print(f"Total flows: {len(df)}\n")

# Separate by class
non_ai_flows = df[df['label'] == 'non_ai']
ai_flows = df[df['label'] != 'non_ai']

print(f"Non-AI flows: {len(non_ai_flows)}")
print(f"AI flows: {len(ai_flows)}\n")

# Undersample non_ai to 400 with random_state=42
non_ai_balanced = non_ai_flows.sample(n=400, random_state=42)

# Combine
df_balanced = pd.concat([ai_flows, non_ai_balanced], ignore_index=True)

print("Balanced dataset:")
print(df_balanced['label'].value_counts().sort_index())
print(f"Total flows: {len(df_balanced)}\n")

# Save balanced dataset
df_balanced.to_csv('data/processed/flows_features_balanced.csv', index=False)
print("Saved balanced dataset to: data/processed/flows_features_balanced.csv")
