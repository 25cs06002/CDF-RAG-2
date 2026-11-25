# Frozen B9-v2 policy ablation

Use the same 132 questions, G1/corpus fingerprints, Ollama model digest, generation settings and top_k=5 as the archived adaptive run. The code is unchanged. Do not change environment settings between runs. Run sequentially; do not execute both at once. CPU load/thermal state and warm model state are not controlled by this script.

    python run_b9_v2_132.py --allow-live --policy fixed1
    python run_b9_v2_132.py --allow-live --policy fixed3

fixed1 runs only the first scheduled retrieval action (not necessarily a document search); fixed3 executes three scheduled actions then selects once; adaptive can stop or retry selection after a valid decline. Safety/unsupported-comparison guards remain unchanged and use zero rounds. This compares entire policies, not a pure independent round-count effect.

Use compare_b9_ablation_132.py with --fixed1-dir and --fixed3-dir pointing at completed runs; archived adaptive is the default. Optional --package-dir and --review-dir attach the validated original adaptive AI quality labels. Do not copy these labels to fixed policies. Operational report is not a medical quality comparison; fixed policies need fresh review. All outputs are under presentation_20261118/thesis132/b9_ablation_v2/ablation_132_<timestamp>/.

Report both all-query and common-answered latency. QPS uses measured wall time; mean rounds includes guarded zero-round cases and the active-only mean is also available. Partial pathway answers are distinguished from strict typed paths. No medical gold, memory or energy measurement claimed. For stronger performance conclusions repeat all policies under matched conditions and report variability.
