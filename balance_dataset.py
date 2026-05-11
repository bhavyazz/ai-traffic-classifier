import pandas as pd

df = pd.read_csv('data/processed/flows_features.csv')
print("--- Flow counts per class (before balancing) ---")
print(df['label'].value_counts())

# Balancing logic
# 1. Undersample non_ai to 400 flows
df_non_ai = df[df['label'] == 'non_ai']
if len(df_non_ai) > 400:
    df_non_ai = df_non_ai.sample(n=400, random_state=42)

# 2. Keep all claude and copilot
df_claude = df[df['label'] == 'claude']
df_copilot = df[df['label'] == 'copilot']

# 3. Undersample chatgpt to match claude
df_chatgpt = df[df['label'] == 'chatgpt']
if len(df_chatgpt) > len(df_claude):
    df_chatgpt = df_chatgpt.sample(n=len(df_claude), random_state=42)
else:
    df_chatgpt = df_chatgpt.sample(n=len(df_chatgpt), random_state=42) # Should just keep them if less

# Concatenate back together
df_balanced = pd.concat([df_non_ai, df_claude, df_copilot, df_chatgpt])

print("\n--- Flow counts per class (after balancing) ---")
print(df_balanced['label'].value_counts())

df_balanced.to_csv('data/processed/flows_features.csv', index=False)
