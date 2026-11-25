# B9 end-to-end adaptive candidate-evidence RAG

Historical B9 retrieval-only and B1-B8 are unchanged. This implementation uses a
rule-based query-only target/intent gate, bounded typed retrieval and BM25
candidate-document fallback, and one local Ollama selection call when candidate
evidence is available. The model selects evidence IDs; the final answer is
rendered deterministically from their original qualified source-summary wording.
No free-form generator answer is accepted. This is an extractive/constrained
generation design, not PPO and not a clinical proof of entailment.

For multi-hop questions, only two-link CAUSES chains with both component IDs
and source URLs are allowed; no transitive causal conclusion is asserted.
Comparative, unresolved targets and patient-specific inference requests abstain.
Document candidates are positive-score, explicit-target matches, not verified
clinical evidence. G1 narrowing-warning edges are excluded. No evaluator labels
or AI-reference audit inputs are read during inference.

Run from root with the existing venv:
    python -m unittest discover -s tests -p 'test_thesis132_b9_local.py' -v
    export OLLAMA_MODEL=llama3.2:latest
    python -m research_v2.ollama_generation check
    python run_b9_132.py --allow-live --smoke
    python run_b9_132.py --allow-live --policy adaptive

Smoke uses only first five questions and cannot serve as the 132-query result.
Full output: presentation_20261118/thesis132/b9_end_to_end/b9_adaptive_<stamp>/
with records.jsonl, round_traces.jsonl, B9_queries_132.csv, summary.json,
manifest.json. Run is checkpointed by appending per-query records; interrupted
runs stay RUNNING, not complete. No automatic resume yet. Use a new run if
interrupted. Nonzero execution errors fail the ready gate.

Ablations, after the adaptive full run:
    python run_b9_132.py --allow-live --policy fixed1
    python run_b9_132.py --allow-live --policy fixed3

Same evidence, selection/rendering and model; only round budget/stopping varies.
Adaptive may stop after one round, fixed3 executes all available scheduled rounds.
A positive candidate gate is not true evidence sufficiency. The local selector
can decline and invalid output abstains; transport failures are errors.
No answer cache, only exact retrieval-query cache; no memory/energy metrics yet.
The five-question smoke and CI use different purposes: CI uses deterministic
selector stubs, not measured live-model quality.

The B9 evidence snapshot differs from historical B1-B8. Compare descriptively,
not as a causal attribution to the controller. A fair controlled ablation uses
the same snapshot and generator across B9 policies. B9 needs a fresh answer-level
AI review; do not copy B1-B8 labels or call citation membership a hallucination
detector. Full quality/claim-level evaluation remains a subsequent step.
