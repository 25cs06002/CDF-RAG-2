# November 18 M.Tech presentation sprint

All experiment inputs, logs, run outputs, metrics and figures for this sprint belong under **`CDF-RAG-2/presentation_20261118/`**. The original G0 snapshot, Phase 2 artifacts, 22-query baseline CSVs and the user's uncommitted local scripts must not be modified.

## Status and scope

- Available now: legacy `no_rl` and `rl` pipelines, the 100-question AI-drafted pilot, G0 structural snapshot, evaluation runner and existing 22-query legacy comparisons.
- Implemented as exploratory **G0-derived** retrieval configurations: naive lexical retrieval, BM25, local dense retrieval, BM25+dense RRF, graph-neighborhood retrieval, and directed causal KG retrieval. They still require local test execution and a separately sourced medical document corpus for a full medical RAG comparison.
- Target: run eight **functionally distinct, independently executable** configurations on 100 common queries, record complete results, and assess structural retrieval on a separately justified G0-covered subset.
- **Do not claim** that the pilot contains independently reviewed medical gold. The 100-question draft queries are valuable for operational and category-wise behavioral testing, not externally verified medical answer accuracy.
- Source-review status is separate from any automatic code validation.

## First local deployment

From the Ubuntu machine, without deleting local uncommitted work:

```bash
cd ~/Folder/CDF-RAG-2
git status --short
# Remote origin may still use HTTPS with a network certificate issue.
# Explicit SSH over 443 uses the GitHub key already configured.
git -c core.sshCommand="ssh -p 443 -i $HOME/.ssh/id_ed25519_github -o IdentitiesOnly=yes" \
  fetch git@ssh.github.com:25cs06002/CDF-RAG-2.git \
  +refs/heads/research-v2:refs/remotes/origin/research-v2
# Only fast-forward if there is no local divergence.
git merge --ff-only origin/research-v2

python3 -m unittest tests.test_presentation_sprint -v
python3 -m research_v2.presentation_sprint prepare
python3 -m research_v2.presentation_sprint status
```

The `prepare` command validates the exact 100-question distribution, exports inference-only queries, freezes copies of the legacy result CSVs, hashes inputs, and records the G0 version and original git commit. It is safe against accidentally overwriting its own frozen query export; use `status` on subsequent runs. It does not run the model or Neo4j.

Check files:

```bash
find presentation_20261118 -maxdepth 3 -type f | sort
python3 -m research_v2.experiment_runner \
  --pipeline no_rl \
  --input presentation_20261118/benchmark/questions_only.jsonl \
  --validate-only
python3 -m research_v2.preflight \
  --config research_v2/configs/g0_no_rl_smoke_v1.json
```

**Caution:** `git merge --ff-only` refuses to merge if local edits would be overwritten or if branches diverge. Stop and inspect rather than stashing, resetting or deleting anything automatically.

## Exploratory six-baseline smoke tests and 100-query runs

The following six configurations use G0 relationship verbalizations as their shared **graph-derived** evidence corpus. They do **not** establish medical document retrieval performance or clinical accuracy. By default, they are **retrieval-only** experiments; add `--generate` to evaluate grounded Ollama generation too. Local SentenceTransformer weights must already be cached for dense and hybrid retrieval (no model download is intentionally triggered).

```bash
cd ~/Folder/CDF-RAG-2
python3 -m unittest tests.test_presentation_baselines -v
# Initial offline retrieval smoke test: six configurations x three queries
python3 -m research_v2.presentation_baselines \
  --pipeline all \
  --input presentation_20261118/benchmark/questions_only.jsonl \
  --limit 3
# Then the full retrieval-only pilot: six configurations x 100 queries
python3 -m research_v2.presentation_baselines \
  --pipeline all \
  --input presentation_20261118/benchmark/questions_only.jsonl \
  --limit 100
# Optional shared LLM generation: run after verifying Ollama availability
python3 -m research_v2.presentation_baselines \
  --pipeline bm25_rag \
  --input presentation_20261118/benchmark/questions_only.jsonl \
  --limit 3 --generate
```

All run outputs are stored as separate directories under `presentation_20261118/runs/`, including failed/abstained records. The existing RL/No-RL adapters still need their own 100-query configs and live preflight; the six new presentation baselines are not yet integrated into the version-locked `configured_experiment` command.

## Baseline smoke tests (only existing pipelines)

The `prepare` command now creates SHA256-locked configs for **100 pilot queries**, under `presentation_20261118/configs/`. The two commands below run the existing legacy adapters only after the original G0/Ollama/PPO preflight passes. Each run takes substantial local time. **Run them sequentially** to avoid contending for Ollama resources:

```bash
cd ~/Folder/CDF-RAG-2
# Validate (no model execution):
python3 -m research_v2.configured_experiment \
  --config presentation_20261118/configs/g0_no_rl_pilot100_v1.json --validate-only
python3 -m research_v2.configured_experiment \
  --config presentation_20261118/configs/g0_rl_pilot100_v1.json --validate-only

# Execute actual 100 queries; the runner includes live runtime preflight:
python3 -m research_v2.configured_experiment \
  --config presentation_20261118/configs/g0_no_rl_pilot100_v1.json \
  --output-root presentation_20261118/runs

python3 -m research_v2.configured_experiment \
  --config presentation_20261118/configs/g0_rl_pilot100_v1.json \
  --output-root presentation_20261118/runs
```

Both use the same frozen input. They are exploratory pilot runs, not independent medical correctness evaluation. Never silently bypass failed preflight or call the old `limit: 1` configs a 100-query experiment.

## Recording and pushing output

Before publishing files from this **public repository**, examine all generated content for private medical data, credentials, copyrighted passages and other sensitive material. Commit source code, safe manifests and aggregate results deliberately, not everything in `presentation_20261118/` indiscriminately.

```bash
git status --short
git add research_v2/presentation_sprint.py tests/test_presentation_sprint.py
# Stage only reviewed artifacts, no secrets.
git add presentation_20261118/benchmark/manifest.json
git commit -m "Archive presentation benchmark provenance" || true
git -c core.sshCommand="ssh -p 443 -i $HOME/.ssh/id_ed25519_github -o IdentitiesOnly=yes" \
  push git@ssh.github.com:25cs06002/CDF-RAG-2.git HEAD:refs/heads/research-v2
```

Keep the three previously modified/untracked local scripts untouched.

## Next implementation gates

1. Inspect available medical document text and licensing, not just candidate URLs.
2. Freeze a common document corpus. If only graph-derived verbalizations exist, explicitly label that as graph-derived retrieval smoke data rather than a medical document benchmark.
3. Implement the six missing baselines with unique retrieval behavior, common answer generator, uniform logging and unit tests.
4. Smoke-test on 3, then 10 representative questions before running the full 100.
5. Run 100 questions on each pipeline using the same frozen inference input; store failures and abstentions, not just successful responses.
6. Measure stage timings, retrieval quality with valid references, query-category behavior and wall-time throughput. The legacy runtime lock currently applies only to legacy RL/No-RL.
7. Produce results, graphs, slides, and honest limitations.
