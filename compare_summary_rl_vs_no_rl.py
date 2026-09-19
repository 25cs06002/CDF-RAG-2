import pandas as pd

rl = pd.read_csv("evaluation_summary_rl_deterministic.csv")
no_rl = pd.read_csv("evaluation_summary_no_rl_deterministic.csv")

rl = rl.set_index("Metric")["Value"]
no_rl = no_rl.set_index("Metric")["Value"]

metrics = [
    "Causal Retrieval Coverage",
    "Causal Chain Depth",
    "Semantic Refinement Score",
    "Context Relevance",
    "Groundedness",
    "Average Answer Length",
    "Hallucination Rate",
    "Retrieval Success Rate",
    "Average Direct Causes",
    "Total Direct Causes",
    "Total Execution Time (s)",
    "Average Latency (s)",
    "Average Latency (ms)",
    "Median Latency (ms)",
    "Average Throughput (QPS)",
    "Refinement Success Rate"
]

comparison = pd.DataFrame({
    "Metric": metrics,
    "No-RL": [no_rl.get(m) for m in metrics],
    "RL": [rl.get(m) for m in metrics]
})

comparison["RL - No-RL"] = (
    comparison["RL"] - comparison["No-RL"]
)

comparison.to_csv(
    "rl_vs_no_rl_comparison.csv",
    index=False
)

print("\n")
print(comparison.to_string(index=False))

print("\nSaved: rl_vs_no_rl_comparison.csv")