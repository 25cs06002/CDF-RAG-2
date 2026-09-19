import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ============================================================
# CONFIGURATION
# ============================================================

# INPUT_FILE = "evaluate_batch_with_metrics_rl_gold.csv"
# OUTPUT_FILE = "evaluation_metrics_results_rl_gold.csv"
# SUMMARY_FILE = "evaluation_summary_rl_gold.csv"

INPUT_FILE = "evaluate_batch_with_metrics_rl_deterministic.csv"
OUTPUT_FILE = "evaluation_metrics_results_rl_deterministic.csv"
SUMMARY_FILE = "evaluation_summary_rl_deterministic.csv"



# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\n" + "=" * 70)
print("              CDF-RAG METRICS EVALUATION")
print("=" * 70)

print(f"\n📁 Input file: {INPUT_FILE}")
print(f"📊 Total rows: {len(df)}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n🧠 Loading SentenceTransformer model...")

encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Embedding model loaded.")


# ============================================================
# ROBUST EMBEDDING FUNCTION
# ============================================================

def embed(text):

    if not isinstance(text, str) or not text.strip():
        return np.zeros(384)

    return encoder.encode(
        text.strip(),
        normalize_embeddings=True
    )


# ============================================================
# STORAGE FOR PER-QUERY METRICS
# ============================================================

semantic_scores = []
context_scores = []
grounded_scores = []

final_lengths = []

ccd_values = []

has_causal = []

hallucinations = []

latency_seconds = []
latency_ms = []

throughput_qps = []

retrieval_counts = []

refinement_success = []


# ============================================================
# PER-QUERY EVALUATION
# ============================================================

for index, row in df.iterrows():

    print(
        f"\n🔎 Evaluating query "
        f"{index + 1}/{len(df)}: "
        f"{row.get('query', '')}"
    )


    # --------------------------------------------------------
    # Extract fields
    # --------------------------------------------------------

    query = str(
        row.get("query", "")
        or ""
    )

    refined_query = str(
        row.get("refined_query_text", "")
        or ""
    )

    final_response = str(
        row.get("final_response", "")
        or ""
    )

    knowledge = str(
        row.get("rewritten_knowledge", "")
        or ""
    )

    direct_causes = str(
        row.get("direct_causes", "")
        or ""
    )

    causal_paths = str(
        row.get("multi_hop_paths", "")
        or ""
    )


    retrieved = (
        direct_causes +
        " " +
        causal_paths
    )


    # ========================================================
    # EMBEDDINGS
    # ========================================================

    query_emb = embed(query)

    refined_emb = embed(refined_query)

    final_emb = embed(final_response)

    knowledge_emb = embed(knowledge)

    retrieved_emb = embed(retrieved)


    # ========================================================
    # 1. SEMANTIC REFINEMENT SCORE
    # ========================================================

    srs = cosine_similarity(
        [query_emb],
        [refined_emb]
    )[0][0]

    semantic_scores.append(srs)


    # ========================================================
    # 2. CONTEXT RELEVANCE
    # ========================================================

    context_score = cosine_similarity(
        [query_emb],
        [knowledge_emb]
    )[0][0]

    context_scores.append(
        context_score
    )


    # ========================================================
    # 3. GROUNDEDNESS
    # ========================================================

    grounded_score = cosine_similarity(
        [knowledge_emb],
        [retrieved_emb]
    )[0][0]

    grounded_scores.append(
        grounded_score
    )


    # ========================================================
    # 4. FINAL ANSWER LENGTH
    # ========================================================

    answer_length = len(
        final_response.split()
    )

    final_lengths.append(
        answer_length
    )


    # ========================================================
    # 5. CAUSAL RETRIEVAL COVERAGE
    # ========================================================

    retrieval_success_value = str(
        row.get("retrieval_success", "")
    ).strip().lower()

    if retrieval_success_value == "yes":

        has_causal.append(1)

    elif retrieval_success_value == "no":

        has_causal.append(0)

    else:

        has_causal.append(
            1 if retrieved.strip() else 0
        )


    # ========================================================
    # 6. CAUSAL CHAIN DEPTH
    # ========================================================

    if causal_paths.strip():

        paths = [
            p.strip()
            for p in causal_paths.split(";")
            if p.strip()
        ]

        path_lengths = []

        for path in paths:

            nodes = path.split("→")

            nodes = [
                node.strip()
                for node in nodes
                if node.strip()
            ]

            path_lengths.append(
                len(nodes)
            )

        if path_lengths:

            ccd_values.append(
                np.mean(path_lengths)
            )

        else:

            ccd_values.append(0)

    else:

        ccd_values.append(0)


    # ========================================================
    # 7. HALLUCINATION
    # ========================================================

    hallucination_value = str(
        row.get(
            "is_hallucination",
            "No"
        )
    ).strip().lower()

    if hallucination_value == "yes":

        hallucinations.append(1)

    else:

        hallucinations.append(0)


    # ========================================================
    # 8. LATENCY
    # ========================================================

    try:

        latency_sec = float(
            row.get(
                "latency_seconds",
                0
            )
        )

    except:

        latency_sec = 0


    try:

        latency_millis = float(
            row.get(
                "latency_ms",
                latency_sec * 1000
            )
        )

    except:

        latency_millis = latency_sec * 1000


    latency_seconds.append(
        latency_sec
    )

    latency_ms.append(
        latency_millis
    )


    # ========================================================
    # 9. THROUGHPUT
    # ========================================================

    try:

        qps = float(
            row.get(
                "query_throughput_qps",
                0
            )
        )

    except:

        qps = 0


    throughput_qps.append(
        qps
    )


    # ========================================================
    # 10. NUMBER OF RETRIEVED CAUSES
    # ========================================================

    try:

        retrieval_count = int(
            row.get(
                "num_direct_causes",
                0
            )
        )

    except:

        retrieval_count = 0


    retrieval_counts.append(
        retrieval_count
    )


    # ========================================================
    # 11. REFINEMENT SUCCESS
    # ========================================================

    if refined_query.strip():

        refinement_success.append(1)

    else:

        refinement_success.append(0)


# ============================================================
# ADD METRICS TO DATAFRAME
# ============================================================

df["SRS"] = semantic_scores

df["Context_Relevance"] = context_scores

df["Groundedness"] = grounded_scores

df["Final_Answer_Length"] = final_lengths

df["CCD"] = ccd_values

df["Causal_Retrieval"] = has_causal

df["Hallucinated"] = hallucinations

df["Latency_Seconds_Calculated"] = latency_seconds

df["Latency_ms_Calculated"] = latency_ms

df["Throughput_QPS_Calculated"] = throughput_qps

df["Retrieved_Cause_Count"] = retrieval_counts

df["Refinement_Success"] = refinement_success


# ============================================================
# SAVE PER-QUERY RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\n💾 Detailed metrics saved to: "
    f"{OUTPUT_FILE}"
)


# ============================================================
# AGGREGATE METRICS
# ============================================================

total_queries = len(df)


# ------------------------------------------------------------
# Quality Metrics
# ------------------------------------------------------------

CRC = np.mean(has_causal)

CCD = np.mean(ccd_values)

SRS = np.mean(semantic_scores)

ContextRel = np.mean(context_scores)

Grounded = np.mean(grounded_scores)

AnswerLen = np.mean(final_lengths)

HR = np.mean(hallucinations)


# ------------------------------------------------------------
# Performance Metrics
# ------------------------------------------------------------

AvgLatencySec = np.mean(
    latency_seconds
)

AvgLatencyMs = np.mean(
    latency_ms
)

MinLatencyMs = np.min(
    latency_ms
)

MaxLatencyMs = np.max(
    latency_ms
)

MedianLatencyMs = np.median(
    latency_ms
)

AvgThroughput = np.mean(
    throughput_qps
)

MinThroughput = np.min(
    throughput_qps
)

MaxThroughput = np.max(
    throughput_qps
)

TotalExecutionTime = np.sum(
    latency_seconds
)


# ============================================================
# RETRIEVAL STATISTICS
# ============================================================

AvgDirectCauses = np.mean(
    retrieval_counts
)

TotalDirectCauses = np.sum(
    retrieval_counts
)

RetrievalSuccessRate = np.mean(
    has_causal
)


# ============================================================
# REFINEMENT STATISTICS
# ============================================================

RefinementSuccessRate = np.mean(
    refinement_success
)


# ============================================================
# PRINT QUALITY METRICS
# ============================================================

print("\n")
print("=" * 70)
print("                 QUALITY METRICS")
print("=" * 70)

print(
    f"Causal Retrieval Coverage (CRC): "
    f"{CRC:.2%}"
)

print(
    f"Causal Chain Depth (CCD):         "
    f"{CCD:.2f}"
)

print(
    f"Semantic Refinement Score (SRS):  "
    f"{SRS:.3f}"
)

print(
    f"Context Relevance Score:           "
    f"{ContextRel:.3f}"
)

print(
    f"Groundedness Score:                "
    f"{Grounded:.3f}"
)

print(
    f"Average Final Answer Length:       "
    f"{AnswerLen:.1f} words"
)

print(
    f"Hallucination Rate (HR):           "
    f"{HR:.2%}"
)


# ============================================================
# PRINT RETRIEVAL METRICS
# ============================================================

print("\n")
print("=" * 70)
print("                 RETRIEVAL METRICS")
print("=" * 70)

print(
    f"Retrieval Success Rate:            "
    f"{RetrievalSuccessRate:.2%}"
)

print(
    f"Average Direct Causes Retrieved:   "
    f"{AvgDirectCauses:.2f}"
)

print(
    f"Total Direct Causes Retrieved:     "
    f"{TotalDirectCauses}"
)


# ============================================================
# PRINT PERFORMANCE METRICS
# ============================================================

print("\n")
print("=" * 70)
print("                PERFORMANCE METRICS")
print("=" * 70)

print(
    f"Total Execution Time:              "
    f"{TotalExecutionTime:.2f} seconds"
)

print(
    f"Average Latency:                   "
    f"{AvgLatencySec:.2f} seconds"
)

print(
    f"Average Latency:                   "
    f"{AvgLatencyMs:.2f} ms"
)

print(
    f"Minimum Latency:                   "
    f"{MinLatencyMs:.2f} ms"
)

print(
    f"Maximum Latency:                   "
    f"{MaxLatencyMs:.2f} ms"
)

print(
    f"Median Latency:                    "
    f"{MedianLatencyMs:.2f} ms"
)

print(
    f"Average Throughput:                "
    f"{AvgThroughput:.4f} queries/sec"
)

print(
    f"Minimum Throughput:                "
    f"{MinThroughput:.4f} queries/sec"
)

print(
    f"Maximum Throughput:                "
    f"{MaxThroughput:.4f} queries/sec"
)


# ============================================================
# REFINEMENT METRICS
# ============================================================

print("\n")
print("=" * 70)
print("              QUERY REFINEMENT METRICS")
print("=" * 70)

print(
    f"Refinement Success Rate:            "
    f"{RefinementSuccessRate:.2%}"
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

print("\n")
print("=" * 70)
print("             HALLUCINATION CLASSIFICATION")
print("=" * 70)


if "is_hallucination_gold" in df.columns:

    y_true = (
        df["is_hallucination_gold"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    y_pred = (
        df["is_hallucination"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )


    valid = y_true.notna()

    y_true = y_true[valid]

    y_pred = y_pred[valid]


    if len(y_true) > 0:

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_true,
            y_pred
        )


        print(
            f"Accuracy:                         "
            f"{accuracy:.3f}"
        )

        print(
            f"Precision:                        "
            f"{precision:.3f}"
        )

        print(
            f"Recall:                           "
            f"{recall:.3f}"
        )

        print(
            f"F1 Score:                         "
            f"{f1:.3f}"
        )

    else:

        print(
            "⚠️ No valid gold labels found."
        )

else:

    print(
        "⚠️ is_hallucination_gold column "
        "not found."
    )

    print(
        "   Precision, Recall, F1 and Accuracy "
        "cannot be calculated yet."
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Queries Evaluated",
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
        "Minimum Latency (ms)",
        "Maximum Latency (ms)",
        "Median Latency (ms)",
        "Average Throughput (QPS)",
        "Minimum Throughput (QPS)",
        "Maximum Throughput (QPS)",
        "Refinement Success Rate"
    ],

    "Value": [

        total_queries,
        CRC,
        CCD,
        SRS,
        ContextRel,
        Grounded,
        AnswerLen,
        HR,
        RetrievalSuccessRate,
        AvgDirectCauses,
        TotalDirectCauses,
        TotalExecutionTime,
        AvgLatencySec,
        AvgLatencyMs,
        MinLatencyMs,
        MaxLatencyMs,
        MedianLatencyMs,
        AvgThroughput,
        MinThroughput,
        MaxThroughput,
        RefinementSuccessRate
    ]
})


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("                 EVALUATION COMPLETE")
print("=" * 70)

print(
    "\n📁 Per-query metrics:"
    f"\n   {OUTPUT_FILE}"
)

print(
    "\n📁 Aggregate metrics:"
    f"\n   {SUMMARY_FILE}"
)

print(
    "\n✅ All 22 queries have been evaluated."
)