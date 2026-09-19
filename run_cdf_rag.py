import sys

from agents.document_causal_retriever_rl import (
    graph,
    CDFState,
    CausalGraphRetriever
)
from config import CONFIG


def main(query):

    print("\n========================================")
    print("       CDF-RAG MEDICAL QA PIPELINE")
    print("========================================")

    graph_obj = CausalGraphRetriever(
        CONFIG["NEO4J_URI"],
        CONFIG["NEO4J_USER"],
        CONFIG["NEO4J_PASSWORD"]
    )

    try:

        initial_state = CDFState(
            query=query,
            graph=graph_obj
        )

        final_state = graph.invoke(initial_state)

        print("\n========================================")
        print("              RESULTS")
        print("========================================")

        print(f"\nOriginal Query:")
        print(query)

        print(f"\nRefinement Type:")
        print(final_state.get("refinement_type"))

        print(f"\nRefined Query:")
        print(final_state.get("refined_query_text"))

        print(f"\nMatched Concept:")
        print(final_state.get("refined_query"))

        print(f"\nDirect Causes:")
        for cause in final_state.get("retrieved_docs", []):
            print(f"  - {cause}")

        print(f"\nMulti-hop Causal Paths:")
        for path in final_state.get("causal_docs", []):
            print(f"  - {path}")

        print("\nStructured Knowledge:")
        print(final_state.get("rewritten_knowledge"))

        print("\nFinal Response:")
        print(final_state.get("final_response"))

        print(
            "\nHallucination Detected:",
            "Yes"
            if final_state.get("is_hallucination")
            else "No"
        )

        print("\n========================================")

    finally:
        graph_obj.close()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Usage: python run_cdf_rag.py "Your question here"'
        )
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    main(query)
