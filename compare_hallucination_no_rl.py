import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

RESULTS_FILE = "evaluation_metrics_results_no_rl_deterministic.csv"
GOLD_FILE = "evaluate_batch_with_metrics_no_rl_gold.csv"
OUTPUT_FILE = "evaluation_metrics_with_no_rl.csv"

results = pd.read_csv(RESULTS_FILE)
gold = pd.read_csv(GOLD_FILE)

# Keep query + gold label
gold_labels = gold[
    ["query", "is_hallucination_gold"]
].copy()

# Remove any accidental duplicate gold column
if "is_hallucination_gold" in results.columns:
    results = results.drop(columns=["is_hallucination_gold"])

# Merge gold labels using query
df = results.merge(
    gold_labels,
    on="query",
    how="left"
)

# Check that every query received a gold label
if df["is_hallucination_gold"].isna().any():
    print("ERROR: Some queries have no gold label.")
    print(
        df[df["is_hallucination_gold"].isna()][["query"]]
    )
    raise SystemExit(1)

# Convert Yes/No to binary
y_true = df["is_hallucination_gold"].map({
    "Yes": 1,
    "No": 0
})

y_pred = df["is_hallucination"].map({
    "Yes": 1,
    "No": 0
})

if y_true.isna().any() or y_pred.isna().any():
    print("ERROR: Unexpected Yes/No values.")
    print("Gold:", df["is_hallucination_gold"].unique())
    print("Prediction:", df["is_hallucination"].unique())
    raise SystemExit(1)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, zero_division=0
)
recall = recall_score(
    y_true, y_pred, zero_division=0
)
f1 = f1_score(
    y_true, y_pred, zero_division=0
)

cm = confusion_matrix(y_true, y_pred)

print("\n" + "=" * 60)
print("HALLUCINATION DETECTION EVALUATION")
print("=" * 60)

print(f"\nTotal Queries : {len(df)}")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "No Hallucination",
            "Hallucination"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix Details:")
print(f"True Positives  (TP): {cm[1,1]}")
print(f"True Negatives  (TN): {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

# ---------------------------------------------------------
# Incorrect predictions
# ---------------------------------------------------------

wrong = df[
    df["is_hallucination"] !=
    df["is_hallucination_gold"]
]

print("\n" + "=" * 60)
print("INCORRECT PREDICTIONS")
print("=" * 60)

print(f"\nIncorrect: {len(wrong)} / {len(df)}")

for _, row in wrong.iterrows():

    print("\nQuery      :", row["query"])
    print("Prediction :", row["is_hallucination"])
    print("Gold       :", row["is_hallucination_gold"])
    print("Response   :", row["final_response"])
    print("-" * 60)

