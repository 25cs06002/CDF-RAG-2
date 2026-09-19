import pandas as pd

rl = pd.read_csv("evaluation_metrics_results_rl_deterministic.csv")
no = pd.read_csv("evaluation_metrics_results_no_rl_deterministic.csv")

cols = [
    "query",
    "matched_concept",
    "num_direct_causes",
    "num_multi_hop_paths",
    "is_hallucination"
]

print("\n========== RL vs NO-RL ==========\n")

for i in range(len(rl)):
    print("Query:", rl.loc[i, "query"])
    print("RL:")
    print(rl.loc[i, cols[1:]].to_dict())
    print("No-RL:")
    print(no.loc[i, cols[1:]].to_dict())
    print("-" * 70)
