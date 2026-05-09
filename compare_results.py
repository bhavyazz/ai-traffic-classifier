import pandas as pd

# Results from unbalanced (original) dataset - 3771 flows
unbalanced_results = {
    'RF': {'accuracy': 0.8609, 'macro_f1': 0.69, 'per_class': {'chatgpt': 0.47, 'claude': 0.65, 'copilot': 0.68, 'non_ai': 0.95}},
    'XGBoost': {'accuracy': 0.8768, 'macro_f1': 0.72, 'per_class': {'chatgpt': 0.54, 'claude': 0.66, 'copilot': 0.71, 'non_ai': 0.96}},
    'CNN': {'accuracy': 0.7788, 'macro_f1': 0.45, 'per_class': {'chatgpt': 0.07, 'claude': 0.33, 'copilot': 0.49, 'non_ai': 0.90}}
}

# Results from balanced dataset - 1380 flows (non_ai undersampled to 400)
balanced_results = {
    'RF': {'accuracy': 0.6739, 'macro_f1': 0.67, 'per_class': {'chatgpt': 0.53, 'claude': 0.62, 'copilot': 0.68, 'non_ai': 0.84}},
    'XGBoost': {'accuracy': 0.7464, 'macro_f1': 0.74, 'per_class': {'chatgpt': 0.62, 'claude': 0.71, 'copilot': 0.78, 'non_ai': 0.86}},
    'CNN': {'accuracy': 0.5181, 'macro_f1': 0.47, 'per_class': {'chatgpt': 0.24, 'claude': 0.39, 'copilot': 0.51, 'non_ai': 0.76}}
}

print("=" * 80)
print("ACCURACY COMPARISON: UNBALANCED vs BALANCED")
print("=" * 80)
print(f"{'Model':<12} {'Unbalanced':>12} {'Balanced':>12} {'Delta':>12}")
print("-" * 80)
for model in ['RF', 'XGBoost', 'CNN']:
    ub_acc = unbalanced_results[model]['accuracy']
    b_acc = balanced_results[model]['accuracy']
    delta = b_acc - ub_acc
    print(f"{model:<12} {ub_acc:>12.2%} {b_acc:>12.2%} {delta:>12.2%}")

print("\n" + "=" * 80)
print("MACRO F1 COMPARISON: UNBALANCED vs BALANCED")
print("=" * 80)
print(f"{'Model':<12} {'Unbalanced':>12} {'Balanced':>12} {'Delta':>12}")
print("-" * 80)
for model in ['RF', 'XGBoost', 'CNN']:
    ub_f1 = unbalanced_results[model]['macro_f1']
    b_f1 = balanced_results[model]['macro_f1']
    delta = b_f1 - ub_f1
    print(f"{model:<12} {ub_f1:>12.2%} {b_f1:>12.2%} {delta:>12.2%}")

print("\n" + "=" * 80)
print("PER-CLASS F1 SCORES: UNBALANCED vs BALANCED")
print("=" * 80)

for model in ['RF', 'XGBoost', 'CNN']:
    print(f"\n{model}:")
    print(f"{'Class':<12} {'Unbalanced':>12} {'Balanced':>12} {'Delta':>12}")
    print("-" * 48)
    for class_name in ['chatgpt', 'claude', 'copilot', 'non_ai']:
        ub_f1 = unbalanced_results[model]['per_class'][class_name]
        b_f1 = balanced_results[model]['per_class'][class_name]
        delta = b_f1 - ub_f1
        print(f"{class_name:<12} {ub_f1:>12.2%} {b_f1:>12.2%} {delta:>12.2%}")
