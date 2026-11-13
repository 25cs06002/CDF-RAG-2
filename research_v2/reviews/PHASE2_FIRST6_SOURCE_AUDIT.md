# Phase 2G — First six questions: source-based annotation audit

**Status:** first-pass source inspection only; **not** independent clinical review,
final gold, or an answer-quality experiment.

**Source inspection date:** 2026-11-13 UTC.
**Source URLs:** NIH/NHLBI and NIH/NIDDK official health pages.
**Queries:** PILOT_001 / _016, PILOT_004 / _019, PILOT_007 / _022.
**Candidate relationship tasks:** EVP001:R01; EVP002:R01–R03; EVP003:R01–R02.

This document is a **researcher's review aid**, not a set of completed
`claim_review.csv` or `question_review.csv` decisions.
Original pilot questions, G0, PPO, candidate evidence proposals and prior
review workbooks remain unchanged.

## 1. Coronary artery disease — EVP001

**Read and cross-check:**

- [NHLBI: What Is Coronary Heart Disease?](https://www.nhlbi.nih.gov/health/coronary-heart-disease),
  particularly the definition and obstructive/nonobstructive descriptions.
- [NHLBI: Coronary Heart Disease Causes and Prevention](https://www.nhlbi.nih.gov/health/coronary-heart-disease/causes),
  especially *What causes coronary heart disease?* and *Plaque buildup*.

**Source-derived finding:** The definition page states that coronary heart
disease is also *sometimes* called coronary artery disease or ischemic heart
disease. Plaque buildup in the large surface coronary arteries is described
as a leading cause. Other vessel dysfunction mechanisms are also discussed.

**Implication for EVP001:R01:** Plaque buildup can remain a
**candidate** `CAUSES` relation, subject to independent review. The
term equivalence issue is less serious than originally feared, but annotate
the particular arterial scope and avoid saying plaque is the only disease
mechanism. If a more specific target is required, consider the distinction
between obstructive, nonobstructive and microvascular coronary conditions.

**Question review:** PILOT_001 and PILOT_016 ask broadly about factors
contributing to disease development. This page supports an example cause
but not necessarily a **complete** answer listing every factor. Do not
approve comprehensive-reference-answer coverage from this link alone.

## 2. Type 2 diabetes — EVP002

**Read:** [NIDDK: Risk Factors for Type 2 Diabetes](https://www.niddk.nih.gov/health-information/diabetes/overview/risk-factors-type-2-diabetes),
especially *What are the risk factors for type 2 diabetes?* and
*Does your weight put you at risk for type 2 diabetes?*

**Source-derived finding:** The NIDDK page lists overweight or obesity,
a family history of diabetes and physical inactivity among recognized
**risk factors** for developing type 2 diabetes.

**Implications for candidate relationships:**

| Task | Proposed cause/source factor | Preliminary assessment |
| --- | --- | --- |
| EVP002:R01 | Overweight or obesity | Keep `RISK_FACTOR_FOR` as a candidate |
| EVP002:R02 | Physical inactivity | Keep `RISK_FACTOR_FOR` as a candidate |
| EVP002:R03 | Family history of diabetes | Keep `RISK_FACTOR_FOR` as a candidate |

This page **does not** establish that any single risk factor deterministically
causes the disease in an individual, or verify a complete molecular pathway
through insulin resistance. Do not convert these to `CAUSES` without
separate mechanism-specific evidence.

**Question review:** PILOT_004 and PILOT_019 use general language about
conditions, exposures and onset. A possible reference answer should
carefully say *risk factors* and not assert a complete causal-mechanistic
explanation based solely on this page.

## 3. COPD — EVP003

**Read:** [NHLBI: COPD Causes and Risk Factors](https://www.nhlbi.nih.gov/health/copd/causes),
especially the opening overview and *What raises your risk of COPD?*

**Source-derived finding:** NHLBI identifies cigarette smoking as the
most common cause of COPD **in the United States**. The source also lists
long-term exposure to polluted air, dust, fumes and other lung irritants
as potential risk factors. Susceptibility, age and other conditions matter.

**Implications for candidate relationships:**

| Task | Source factor | Preliminary assessment |
| --- | --- | --- |
| EVP003:R01 | Cigarette smoking | Keep proposed `CAUSES`, with geographic/population qualification |
| EVP003:R02 | Long-term exposure to air pollution and dust | Keep `RISK_FACTOR_FOR` provisionally; split exposures if gold graph needs individually testable edges |

The current combined `air pollution and dust` source factor is not
ideal for an exact-match causal graph: pollution, dust, fumes and indoor
fuel smoke are distinct exposures. Do not represent the union as a
single proven exposure mechanism without explicit ontology decisions.

**Question review:** PILOT_007 and PILOT_022 can use this page to
describe some established causes/risks, but the page is not necessarily
an exhaustive explanation and does not determine an individual's diagnosis.

## Annotation actions

1. Open the **existing** generated `claim_review.csv` and
   `question_review.csv` for the 30-question worklist.
2. Start with the six named claim tasks and six named question IDs.
   Read source passages independently; confirm terminology, population
   and distinction among causes, risks and associations.
3. Fill `review_decision` only when you personally checked the evidence:
   `accept`, `revise`, `reject`, or `needs_more_evidence`.
   Use `source_passage_locator`, `review_note`, **actual** reviewer ID
   and date. If revising a relation, use the separate
   `corrected_cause/effect/relation` fields without editing original text.
4. The `question_review.csv` decision should assess **answerability
   from the defined evidence corpus** and whether the reference answer is
   complete enough for the question's scope.
5. It is acceptable to record `needs_more_evidence` or partial
   coverage. Do not complete rows just to satisfy the strict workflow audit.
6. Validate the workbook with
   `python3 -m research_v2.reviewer_workspace --audit-dir "$OUT"`.
   A passing format audit does not attest clinical correctness.

**Review status:** 0 independently verified gold labels are created
by these source-audit notes. The three topic-level recommendations are
stored separately in
`data/benchmark/first6_source_audit_notes_v1.jsonl`, which explicitly
sets `human_review_completed=false` and `gold_annotation_created=false`.

**Research progression:** After actual source review of these six questions,
resolve remaining evidence gaps and seek domain-expert adjudication where
appropriate. Only then create a separate, versioned gold-label dataset,
followed by train/validation/test partitions. Do not copy the AI candidate
edges directly into the Neo4j G1 graph.
