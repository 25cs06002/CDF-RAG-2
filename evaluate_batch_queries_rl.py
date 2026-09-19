import csv
import time
import statistics

from agents.document_causal_retriever_rl import (
    graph,
    CDFState,
    CausalGraphRetriever
)
from config import CONFIG


# ============================================================
# Evaluation Queries
# ============================================================

test_queries = [

    # --------------------------------------------------------
    # Heart Disease
    # --------------------------------------------------------
    "What causes heart disease?",
    "What leads to heart disease?",
    "What are the causes of heart disease?",
    "What factors contribute to heart disease?",
    "What causes cardiovascular disease?",

    # --------------------------------------------------------
    # Lung Cancer
    # --------------------------------------------------------
    "What causes lung cancer?",
    "What leads to lung cancer?",
    "What are the causes of lung cancer?",
    "What factors contribute to lung cancer?",

    # --------------------------------------------------------
    # Migraine
    # --------------------------------------------------------
    "What causes migraine?",
    "What causes migraines?",
    "What triggers migraine?",
    "What factors contribute to migraine?",

    # --------------------------------------------------------
    # Asthma
    # --------------------------------------------------------
    "What causes asthma?",
    "What leads to asthma?",
    "What factors contribute to asthma?",

    # --------------------------------------------------------
    # Depression
    # --------------------------------------------------------
    "What causes depression?",
    "What leads to depression?",
    "What factors contribute to depression?",

    # --------------------------------------------------------
    # Stroke
    # --------------------------------------------------------
    "What causes stroke?",
    "What leads to stroke?",
    "What factors contribute to stroke?"
]


# ============================================================
# Output CSV
# ============================================================

# output_file = "evaluate_batch_with_metrics.csv"
# output_file = "evaluate_batch_with_metrics_deterministic.csv"
output_file = "evaluate_batch_with_metrics_rl_deterministic.csv"

# ============================================================
# Initialize Neo4j
# ============================================================

graph_obj = CausalGraphRetriever(
    CONFIG["NEO4J_URI"],
    CONFIG["NEO4J_USER"],
    CONFIG["NEO4J_PASSWORD"]
)


# ============================================================
# CSV Columns
# ============================================================

fieldnames = [

    # Query information
    "query",

    # Query refinement
    "refinement_type",
    "refined_query_text",
    "refined_query",
    "matched_concept",

    # Retrieval
    "direct_causes",
    "num_direct_causes",
    "multi_hop_paths",
    "num_multi_hop_paths",
    "retrieval_success",

    # Generation
    "rewritten_knowledge",
    "final_response",

    # Hallucination
    "is_hallucination",

    # Performance
    "latency_seconds",
    "latency_ms",

    # Throughput for this query
    "query_throughput_qps",

    # Pipeline status
    "status"
]


# ============================================================
# Batch-level timing
# ============================================================

batch_start_time = time.perf_counter()

successful_queries = 0
failed_queries = 0

latencies = []

total_direct_causes = 0
total_multi_hop_paths = 0

hallucination_count = 0
retrieval_success_count = 0


# ============================================================
# Open CSV
# ============================================================

with open(
    output_file,
    mode="w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()


    # ========================================================
    # Process Queries
    # ========================================================

    for i, query in enumerate(test_queries, start=1):

        print("\n" + "=" * 70)
        print(f"QUERY {i}/{len(test_queries)}")
        print("=" * 70)

        print(f"⚙️ Processing: {query}")

        query_start_time = time.perf_counter()

        try:

            # ------------------------------------------------
            # Initial State
            # ------------------------------------------------

            initial_state = CDFState(
                query=query,
                graph=graph_obj
            )


            # ------------------------------------------------
            # Run complete CDF-RAG pipeline
            # ------------------------------------------------

            result = graph.invoke(initial_state)


            # ------------------------------------------------
            # Query latency
            # ------------------------------------------------

            query_end_time = time.perf_counter()

            latency_seconds = (
                query_end_time - query_start_time
            )

            latency_ms = latency_seconds * 1000


            # ------------------------------------------------
            # Extract results
            # ------------------------------------------------

            refinement_type = result.get(
                "refinement_type",
                ""
            )

            refined_query_text = result.get(
                "refined_query_text",
                ""
            )

            refined_query = result.get(
                "refined_query",
                ""
            )

            retrieved_docs = result.get(
                "retrieved_docs",
                []
            )

            causal_docs = result.get(
                "causal_docs",
                []
            )

            rewritten_knowledge = result.get(
                "rewritten_knowledge",
                ""
            )

            final_response = result.get(
                "final_response",
                ""
            )

            is_hallucination = result.get(
                "is_hallucination",
                False
            )


            # ------------------------------------------------
            # Retrieval metrics
            # ------------------------------------------------

            num_direct_causes = len(
                retrieved_docs
            )

            num_multi_hop_paths = len(
                causal_docs
            )

            retrieval_success = (
                num_direct_causes > 0
                or
                num_multi_hop_paths > 0
            )


            # ------------------------------------------------
            # Per-query throughput
            #
            # This represents how many queries/second the
            # system could process if this query's latency
            # were representative.
            # ------------------------------------------------

            query_throughput_qps = (
                1.0 / latency_seconds
                if latency_seconds > 0
                else 0
            )


            # ------------------------------------------------
            # Update counters
            # ------------------------------------------------

            successful_queries += 1

            latencies.append(
                latency_seconds
            )

            total_direct_causes += (
                num_direct_causes
            )

            total_multi_hop_paths += (
                num_multi_hop_paths
            )

            if retrieval_success:
                retrieval_success_count += 1

            if is_hallucination:
                hallucination_count += 1


            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            print("\n📌 Original Query:")
            print(query)

            print(
                "\n🧠 Refinement Type:"
            )
            print(refinement_type)

            print(
                "\n🔁 Refined Query:"
            )
            print(refined_query_text)

            print(
                "\n🎯 Matched Concept:"
            )
            print(refined_query)

            print(
                "\n📚 Direct Causes:"
            )

            for doc in retrieved_docs:
                print(f"   - {doc}")


            print(
                "\n🔗 Multi-hop Paths:"
            )

            for path in causal_docs:
                print(f"   - {path}")


            print(
                "\n📝 Structured Knowledge:"
            )
            print(rewritten_knowledge)

            print(
                "\n💬 Final Response:"
            )
            print(final_response)

            print(
                f"\n📊 Direct causes:"
                f" {num_direct_causes}"
            )

            print(
                f"📊 Multi-hop paths:"
                f" {num_multi_hop_paths}"
            )

            print(
                f"📊 Retrieval success:"
                f" {'Yes' if retrieval_success else 'No'}"
            )

            print(
                f"⏱️ Latency:"
                f" {latency_seconds:.3f} seconds"
            )

            print(
                f"⚡ Query throughput:"
                f" {query_throughput_qps:.3f} QPS"
            )

            print(
                f"🛡️ Hallucination:"
                f" {'Yes' if is_hallucination else 'No'}"
            )


            # ------------------------------------------------
            # Save query results
            # ------------------------------------------------

            writer.writerow({

                "query":
                    query,

                "refinement_type":
                    refinement_type,

                "refined_query_text":
                    refined_query_text,

                "refined_query":
                    refined_query,

                "matched_concept":
                    refined_query,

                "direct_causes":
                    "; ".join(
                        retrieved_docs
                    ),

                "num_direct_causes":
                    num_direct_causes,

                "multi_hop_paths":
                    "; ".join(
                        causal_docs
                    ),

                "num_multi_hop_paths":
                    num_multi_hop_paths,

                "retrieval_success":
                    "Yes"
                    if retrieval_success
                    else "No",

                "rewritten_knowledge":
                    rewritten_knowledge,

                "final_response":
                    final_response,

                "is_hallucination":
                    "Yes"
                    if is_hallucination
                    else "No",

                "latency_seconds":
                    f"{latency_seconds:.4f}",

                "latency_ms":
                    f"{latency_ms:.2f}",

                "query_throughput_qps":
                    f"{query_throughput_qps:.4f}",

                "status":
                    "SUCCESS"
            })


        except Exception as e:

            # ------------------------------------------------
            # Measure latency even when query fails
            # ------------------------------------------------

            query_end_time = time.perf_counter()

            latency_seconds = (
                query_end_time - query_start_time
            )

            latency_ms = latency_seconds * 1000

            failed_queries += 1

            print(
                f"\n❌ Error processing query:"
                f"\n{e}"
            )


            # ------------------------------------------------
            # Save failed query
            # ------------------------------------------------

            writer.writerow({

                "query":
                    query,

                "refinement_type":
                    "ERROR",

                "refined_query_text":
                    "",

                "refined_query":
                    "",

                "matched_concept":
                    "",

                "direct_causes":
                    "",

                "num_direct_causes":
                    0,

                "multi_hop_paths":
                    "",

                "num_multi_hop_paths":
                    0,

                "retrieval_success":
                    "No",

                "rewritten_knowledge":
                    "",

                "final_response":
                    "",

                "is_hallucination":
                    "ERROR",

                "latency_seconds":
                    f"{latency_seconds:.4f}",

                "latency_ms":
                    f"{latency_ms:.2f}",

                "query_throughput_qps":
                    (
                        f"{1.0 / latency_seconds:.4f}"
                        if latency_seconds > 0
                        else "0"
                    ),

                "status":
                    "ERROR"
            })


# ============================================================
# Batch-level metrics
# ============================================================

batch_end_time = time.perf_counter()

total_batch_time = (
    batch_end_time - batch_start_time
)


# ============================================================
# Calculate Metrics
# ============================================================

total_queries = len(test_queries)

average_latency = (
    statistics.mean(latencies)
    if latencies
    else 0
)

median_latency = (
    statistics.median(latencies)
    if latencies
    else 0
)

minimum_latency = (
    min(latencies)
    if latencies
    else 0
)

maximum_latency = (
    max(latencies)
    if latencies
    else 0
)


# ------------------------------------------------------------
# Batch throughput
# ------------------------------------------------------------

batch_throughput_qps = (
    successful_queries / total_batch_time
    if total_batch_time > 0
    else 0
)


# ------------------------------------------------------------
# Retrieval coverage
# ------------------------------------------------------------

retrieval_coverage = (
    retrieval_success_count / successful_queries * 100
    if successful_queries > 0
    else 0
)


# ------------------------------------------------------------
# Hallucination rate
# ------------------------------------------------------------

hallucination_rate = (
    hallucination_count / successful_queries * 100
    if successful_queries > 0
    else 0
)


# ------------------------------------------------------------
# Supported response rate
# ------------------------------------------------------------

supported_response_rate = (
    (
        successful_queries
        - hallucination_count
    )
    / successful_queries
    * 100
    if successful_queries > 0
    else 0
)


# ------------------------------------------------------------
# Average retrieval information
# ------------------------------------------------------------

average_direct_causes = (
    total_direct_causes / successful_queries
    if successful_queries > 0
    else 0
)

average_multi_hop_paths = (
    total_multi_hop_paths / successful_queries
    if successful_queries > 0
    else 0
)


# ============================================================
# Print Final Evaluation
# ============================================================

print("\n")
print("=" * 70)
print("                 CDF-RAG EVALUATION")
print("=" * 70)

print(
    f"\n📊 Total queries:"
    f" {total_queries}"
)

print(
    f"✅ Successful queries:"
    f" {successful_queries}"
)

print(
    f"❌ Failed queries:"
    f" {failed_queries}"
)

print(
    f"\n⏱️ Total batch time:"
    f" {total_batch_time:.3f} seconds"
)

print(
    f"⏱️ Average latency:"
    f" {average_latency:.3f} seconds"
)

print(
    f"⏱️ Median latency:"
    f" {median_latency:.3f} seconds"
)

print(
    f"⚡ Minimum latency:"
    f" {minimum_latency:.3f} seconds"
)

print(
    f"⚡ Maximum latency:"
    f" {maximum_latency:.3f} seconds"
)

print(
    f"\n🚀 Batch throughput:"
    f" {batch_throughput_qps:.4f} queries/sec"
)

print(
    f"\n📚 Retrieval coverage:"
    f" {retrieval_coverage:.2f}%"
)

print(
    f"📚 Average direct causes/query:"
    f" {average_direct_causes:.2f}"
)

print(
    f"🔗 Average multi-hop paths/query:"
    f" {average_multi_hop_paths:.2f}"
)

print(
    f"\n🛡️ Hallucinations:"
    f" {hallucination_count}/{successful_queries}"
)

print(
    f"🛡️ Hallucination rate:"
    f" {hallucination_rate:.2f}%"
)

print(
    f"✅ Supported response rate:"
    f" {supported_response_rate:.2f}%"
)

print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETE")
print("=" * 70)

print(
    f"📁 Results saved to:"
    f" {output_file}"
)


# ============================================================
# Safe Neo4j Shutdown
# ============================================================

try:

    graph_obj.close()

    print(
        "🔌 Neo4j connection closed."
    )

except Exception as e:

    print(
        "⚠️ Warning: Failed to close "
        "Neo4j driver gracefully."
    )