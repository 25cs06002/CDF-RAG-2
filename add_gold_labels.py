import pandas as pd

df = pd.read_csv("evaluate_batch_with_metrics.csv")

gold_labels = [
    "Yes",  # 1
    "No",   # 2
    "No",   # 3
    "Yes",  # 4
    "Yes",  # 5
    "No",   # 6
    "No",   # 7
    "No",   # 8
    "No",   # 9
    "No",   # 10
    "No",   # 11
    "No",   # 12
    "No",   # 13
    "Yes",  # 14
    "No",   # 15
    "No",   # 16
    "No",   # 17
    "No",   # 18
    "No",   # 19
    "No",   # 20
    "No",   # 21
    "Yes"   # 22
]

if len(df) != len(gold_labels):
    raise ValueError(
        f"Expected {len(gold_labels)} queries, "
        f"but CSV contains {len(df)} rows."
    )

df["is_hallucination_gold"] = gold_labels

df.to_csv(
    "evaluate_batch_with_metrics_gold.csv",
    index=False
)

print("✅ Gold labels added.")
print()
print(df[
    [
        "query",
        "is_hallucination",
        "is_hallucination_gold"
    ]
].to_string(index=False))