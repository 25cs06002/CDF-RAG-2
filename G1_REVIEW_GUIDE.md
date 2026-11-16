# Review-gated G1 candidate evidence expansion

**Purpose:** preserve the archived 35-node/37-edge G0 baseline and
triage actual B5/B6 G0 retrieval failures. Nothing in this workflow
constitutes medical advice, independent clinical validation or gold labels.

1. Run `bash scripts/run_g1_review.sh` on the Ubuntu checkout **after**
   completing the 100-query pilot and diagnostics. It creates a new,
   timestamped directory in `presentation_20261118/g1_review/`.
2. Open `review_queue.csv`. It contains only queries identified as
   `G0_SNAPSHOT_COVERAGE_GAP` or `DIRECTED_RETRIEVAL_GAP`, with
   *draft* target concepts as search hints—not approved knowledge.
3. Research sources manually (such as authoritative clinical guidelines
   and official health-agency publications). The generated
   `evidence_candidates.csv` is intentionally **empty**.
4. Enter one row per independently reviewed source supporting a specific,
   directional, causal relationship. Use the exact CSV columns:
   `evidence_id,query_id,cause,effect,relation,source_url,source_title,source_publisher,evidence_note,review_status,reviewer,reviewed_date`
   The permitted `relation` is currently only `CAUSES`, with status
   `proposed`, `rejected`, or `approved`. Unsupported medical assertions,
   statistical associations, and risk factors must **not** be converted
   automatically to causal graph edges. Include a detailed paraphrased
   evidence note, the source HTTPS URL, and a reviewer's identity and ISO date.
   URLs are checked for syntax, **not** visited or validated automatically.
5. Validate your ledger (replace path with the actual queue directory):

   ```bash
   python -m research_v2.g1_review check --queue-dir presentation_20261118/g1_review/queue_...
   ```

6. Only after reviewing/approving evidence, create a *new candidate* graph:

   ```bash
   python -m research_v2.g1_review build --queue-dir presentation_20261118/g1_review/queue_...
   ```

7. Optionally measure B5/B6 retrieval yield against G0 without running Ollama:

   ```bash
   python -m research_v2.g1_review probe --graph-dir presentation_20261118/g1_review/queue_.../graphs/G1_CANDIDATE_...
   ```

The resulting snapshot contains `nodes.csv`, `edges.csv`,
`approved_edge_attributions.csv`, and `manifest.json`. New edges have
stable deterministic IDs. Existing G0 edge IDs remain unchanged, and the
source and candidate ledger checksums are recorded.

**Critical limitations:** all 37 inherited G0 edges remain labeled
unreviewed. Approval is a human assertion, not external source verification.
A candidate G1 snapshot is **not** called clinically verified, even when
some new edges have review attestations. New evaluation labels must be
independently adjudicated, not constructed from the graph being scored.
The frozen 100-question pilot must not be changed to improve coverage.
Do not commit unreviewed personal health information or copyrighted
long-form source text.
