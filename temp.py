import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("evaluate_batch_with_metrics_deterministic.csv")

# Convert Yes/No to 1/0
y_true = df["is_hallucination_gold"].map({"Yes": 1, "No": 0})
y_pred = df["is_hallucination"].map({"Yes": 1, "No": 0})

print("\n========== HALLUCINATION DETECTION ==========\n")

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
print("F1 Score :", f1_score(y_true, y_pred, zero_division=0))

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["No Hallucination", "Hallucination"],
        zero_division=0
    )
)

print("\nTP:", cm[1,1])
print("TN:", cm[0,0])
print("FP:", cm[0,1])
print("FN:", cm[1,0])