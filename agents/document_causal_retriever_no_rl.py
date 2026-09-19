# ============================================================
# CDF-RAG Document Causal Retriever
# ============================================================
#
# Architecture:
#
# Query
#   ↓
#   ↓
# Ollama Query Refinement
#   ↓
# Neo4j Concept Matching
#   ↓
# Neo4j Causal Retrieval
#   ↓
# Ollama Knowledge Rewriting
#   ↓
# Ollama Final Response Generation
#   ↓
# Embedding Hallucination Detector
#   ↓
# Deterministic Neo4j Causal-Evidence Validator
#   ↓
# Final Result
#
# IMPORTANT:
# No OpenAI API is used in this file.
# ============================================================


# ============================================================
# Project path configuration
# ============================================================

import os
import sys

# Get CDF-RAG-2 project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Add project root to Python import path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Local imports
# ============================================================

from agents.query_refiner_rl import QueryRefinementEnv
from agents.knowledge_rewriter_rl import (
    RewritingState,
    KnowledgeRewritingAgent
)

from agents.knowledge_rewriter_rl import (
    RewritingState,
    KnowledgeRewritingAgent
)

from neo4j import GraphDatabase


# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langgraph.graph import StateGraph
from pydantic import BaseModel

import numpy as np
import requests
import re

from dotenv import load_dotenv

from agents.llm_inference_response_generator import (
    LLMResponseGenerator
)

from agents.hallucination_detector_rl import (
    HallucinationDetector
)


# ============================================================
# Environment
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()


# ============================================================
# Ollama Configuration
# ============================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
)

OLLAMA_MODEL = os.getenv(
    "QUERY_REFINEMENT_MODEL",
    "llama3.2:latest"
)


# ============================================================
# 1. Neo4j Graph Wrapper
# ============================================================

class CausalGraphRetriever:

    def __init__(
        self,
        uri,
        user,
        password
    ):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )

    # --------------------------------------------------------
    # Close connection
    # --------------------------------------------------------

    def close(self):

        self.driver.close()

    # --------------------------------------------------------
    # Get direct causes
    # --------------------------------------------------------

    def retrieve_direct_causes(
        self,
        effect
    ):

        query = """
        MATCH (c:Concept)-[:CAUSES]->(e:Concept)
        WHERE toLower(e.name) = toLower($effect)
        RETURN c.name AS cause
        ORDER BY c.name
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                effect=effect
            )

            return [
                record["cause"]
                for record in result
            ]

    # --------------------------------------------------------
    # Get multi-hop causal paths
    # --------------------------------------------------------

    def retrieve_multi_hop_causes(
        self,
        effect
    ):

        query = """
        MATCH path =
            (c:Concept)-[:CAUSES*]->(e:Concept)

        WHERE toLower(e.name) = toLower($effect)

        RETURN
            [node IN nodes(path) | node.name]
            AS causal_path
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                effect=effect
            )

            return [
                record["causal_path"]
                for record in result
            ]

    # --------------------------------------------------------
    # Get all concepts
    # --------------------------------------------------------

    def get_all_concepts(self):

        query = """
        MATCH (c:Concept)
        RETURN DISTINCT c.name AS concept
        ORDER BY c.name
        """

        with self.driver.session() as session:

            result = session.run(query)

            return [
                record["concept"]
                for record in result
            ]

    def match_concept(self, query_text):

        concepts = self.get_all_concepts()

        if not concepts:
            return None, 0.0

        query_lower = query_text.lower().strip()

        # --------------------------------------------------------
        # 1. Exact concept appearing inside the query
        # --------------------------------------------------------

        exact_matches = []

        for concept in concepts:

            concept_lower = concept.lower().strip()

            if concept_lower in query_lower:

                exact_matches.append(concept)

        if exact_matches:

            # Prefer the longest matching concept
            best = max(
                exact_matches,
                key=len
            )

            return best, 1.0

        # --------------------------------------------------------
        # 2. Semantic matching fallback
        # --------------------------------------------------------

        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        query_embedding = encoder.encode(
            query_text,
            normalize_embeddings=True
        )

        concept_embeddings = encoder.encode(
            concepts,
            normalize_embeddings=True
        )

        scores = np.dot(
            concept_embeddings,
            query_embedding
        )

        best_index = int(
            np.argmax(scores)
        )

        best_concept = concepts[best_index]

        best_score = float(
            scores[best_index]
        )

        return best_concept, best_score

    # --------------------------------------------------------
    # Check exact direct causal relationship
    # --------------------------------------------------------

    def causal_edge_exists(
        self,
        cause,
        effect
    ):

        query = """
        MATCH
            (c:Concept)-[:CAUSES]->(e:Concept)

        WHERE
            toLower(c.name) = toLower($cause)
            AND
            toLower(e.name) = toLower($effect)

        RETURN count(*) AS count
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                cause=cause,
                effect=effect
            )

            record = result.single()

            return record["count"] > 0

    # --------------------------------------------------------
    # Check exact multi-hop path
    # --------------------------------------------------------

    def causal_path_exists(
        self,
        nodes
    ):

        if len(nodes) < 2:
            return False

        query = """
        MATCH path =
            (start:Concept)-[:CAUSES*]->(end:Concept)

        WHERE
            toLower(start.name) = toLower($start)
            AND
            toLower(end.name) = toLower($end)

        WITH
            [n IN nodes(path) | toLower(n.name)]
            AS path_nodes

        WHERE path_nodes = $nodes

        RETURN count(*) AS count
        """

        normalized_nodes = [
            str(node).strip().lower()
            for node in nodes
        ]

        with self.driver.session() as session:

            result = session.run(
                query,
                start=nodes[0],
                end=nodes[-1],
                nodes=normalized_nodes
            )

            record = result.single()

            return record["count"] > 0


# ============================================================
# 2. Deterministic Neo4j Causal-Evidence Validator
# ============================================================

class DeterministicCausalValidator:

    """
    Deterministic validator.

    It does NOT use an LLM.

    It checks causal relationships directly against
    the Neo4j CAUSES edges.
    """

    def __init__(
        self,
        graph
    ):

        self.graph = graph

    # --------------------------------------------------------
    # Find concepts mentioned in answer
    # --------------------------------------------------------

    def find_concepts(
        self,
        text,
        concepts
    ):

        text_lower = text.lower()

        found = []

        for concept in concepts:

            concept_lower = concept.lower()

            pattern = (
                r"\b"
                + re.escape(concept_lower)
                + r"\b"
            )

            if re.search(
                pattern,
                text_lower
            ):

                found.append(concept)

        # Longest concepts first
        found.sort(
            key=len,
            reverse=True
        )

        return found

    # --------------------------------------------------------
    # Extract simple causal claims
    # --------------------------------------------------------

    def extract_causal_claims(
        self,
        sentence,
        concepts
    ):

        claims = []

        found_concepts = self.find_concepts(
            sentence,
            concepts
        )

        if len(found_concepts) < 2:
            return claims

        # ----------------------------------------------------
        # Pattern:
        #
        # A causes B
        # A leads to B
        # A contributes to B
        # A results in B
        # ----------------------------------------------------

        forward_patterns = [
            r"(.+?)\s+causes\s+(.+)",
            r"(.+?)\s+leads\s+to\s+(.+)",
            r"(.+?)\s+contributes\s+to\s+(.+)",
            r"(.+?)\s+results\s+in\s+(.+)",
            r"(.+?)\s+is\s+a\s+cause\s+of\s+(.+)",
            r"(.+?)\s+can\s+cause\s+(.+)",
            r"(.+?)\s+can\s+lead\s+to\s+(.+)"
        ]

        # ----------------------------------------------------
        # Pattern:
        #
        # B is caused by A
        # B is influenced by A
        # ----------------------------------------------------

        reverse_patterns = [
            r"(.+?)\s+is\s+caused\s+by\s+(.+)",
            r"(.+?)\s+is\s+influenced\s+by\s+(.+)"
        ]

        sentence_clean = sentence.strip()

        # ----------------------------------------------------
        # Forward causal statements
        # ----------------------------------------------------

        for pattern in forward_patterns:

            match = re.search(
                pattern,
                sentence_clean,
                flags=re.IGNORECASE
            )

            if match:

                left = match.group(1)
                right = match.group(2)

                left_concepts = self.find_concepts(
                    left,
                    concepts
                )

                right_concepts = self.find_concepts(
                    right,
                    concepts
                )

                for cause in left_concepts:

                    for effect in right_concepts:

                        claims.append({
                            "cause": cause,
                            "effect": effect,
                            "sentence": sentence_clean
                        })

                return claims

        # ----------------------------------------------------
        # Reverse causal statements
        # ----------------------------------------------------

        for pattern in reverse_patterns:

            match = re.search(
                pattern,
                sentence_clean,
                flags=re.IGNORECASE
            )

            if match:

                effect_text = match.group(1)
                cause_text = match.group(2)

                effect_concepts = self.find_concepts(
                    effect_text,
                    concepts
                )

                cause_concepts = self.find_concepts(
                    cause_text,
                    concepts
                )

                for cause in cause_concepts:

                    for effect in effect_concepts:

                        claims.append({
                            "cause": cause,
                            "effect": effect,
                            "sentence": sentence_clean
                        })

                return claims

        return claims

    # --------------------------------------------------------
    # Validate complete answer
    # --------------------------------------------------------

    def validate(
        self,
        answer
    ):

        concepts = self.graph.get_all_concepts()

        if not answer or not answer.strip():

            return {
                "passed": False,
                "supported_claims": [],
                "unsupported_claims": [],
                "num_claims": 0,
                "num_supported_claims": 0,
                "num_unsupported_claims": 0
            }

        # ----------------------------------------------------
        # Split answer into sentences
        # ----------------------------------------------------

        sentences = re.split(
            r"(?<=[.!?])\s+",
            answer.strip()
        )

        supported_claims = []
        unsupported_claims = []

        # ----------------------------------------------------
        # Process every sentence
        # ----------------------------------------------------

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            claims = self.extract_causal_claims(
                sentence,
                concepts
            )

            # ------------------------------------------------
            # Validate every individual causal claim
            # ------------------------------------------------

            for claim in claims:

                claim_record = {
                    "cause": claim["cause"],
                    "effect": claim["effect"],
                    "sentence": claim["sentence"]
                }

                exists = self.graph.causal_edge_exists(
                    claim["cause"],
                    claim["effect"]
                )

                if exists:

                    supported_claims.append(
                        claim_record
                    )

                else:

                    unsupported_claims.append(
                        claim_record
                    )

        # ----------------------------------------------------
        # Remove duplicate causal claims
        # ----------------------------------------------------

        def unique_claims(claims):

            unique = []
            seen = set()

            for claim in claims:

                key = (
                    claim["cause"].lower().strip(),
                    claim["effect"].lower().strip()
                )

                if key not in seen:

                    seen.add(key)
                    unique.append(claim)

            return unique

        supported_claims = unique_claims(
            supported_claims
        )

        unsupported_claims = unique_claims(
            unsupported_claims
        )

        # ----------------------------------------------------
        # Calculate counts
        # ----------------------------------------------------

        num_supported_claims = len(
            supported_claims
        )

        num_unsupported_claims = len(
            unsupported_claims
        )

        num_claims = (
            num_supported_claims
            +
            num_unsupported_claims
        )

        # ----------------------------------------------------
        # Validation result
        # ----------------------------------------------------

        if num_claims == 0:

            # No causal claims detected.
            # Embedding detector handles general
            # semantic consistency.

            passed = True

        else:

            passed = (
                num_unsupported_claims == 0
            )

        # ----------------------------------------------------
        # Return results
        # ----------------------------------------------------

        return {

            "passed": passed,

            "supported_claims":
                supported_claims,

            "unsupported_claims":
                unsupported_claims,

            "num_claims":
                num_claims,

            "num_supported_claims":
                num_supported_claims,

            "num_unsupported_claims":
                num_unsupported_claims
        }


# ============================================================
# 3. Ollama Query No Refinement Agent
# ============================================================

class QueryRefinementAgent:

    def __init__(self):
        print("🚫 RL disabled: deterministic concept matching only.")

    def refine(self, state):

        original_query = state.query
        graph = state.graph

        print(
            f"\n🔁 Original Query (No-RL): "
            f"{original_query}"
        )

        # --------------------------------------------------------
        # Get all concepts from Neo4j
        # --------------------------------------------------------

        concepts = graph.get_all_concepts()

        if not concepts:

            print("⚠️ No concepts found in Neo4j.")

            state.refinement_type = "No Refinement"
            state.refined_query_text = original_query
            state.refined_query = original_query
            state.matched_concept = ""

            return state

        # --------------------------------------------------------
        # Normalize query
        # --------------------------------------------------------

        query_lower = original_query.lower().strip()

        # Remove common question words
        stop_words = {
            "what",
            "are",
            "is",
            "the",
            "a",
            "an",
            "causes",
            "cause",
            "caused",
            "by",
            "of",
            "for",
            "to",
            "does",
            "do",
            "factors",
            "factor",
            "contribute",
            "contributes",
            "leads",
            "lead",
            "triggers",
            "trigger",
            "underlying",
            "primary",
            "risk",
            "development"
        }

        query_tokens = set(
            query_lower
            .replace("?", "")
            .replace(",", "")
            .split()
        )

        query_content_tokens = (
            query_tokens - stop_words
        )

        # --------------------------------------------------------
        # 1. Exact concept phrase match
        # --------------------------------------------------------

        matched_concept = None

        exact_matches = []

        for concept in concepts:

            concept_lower = concept.lower().strip()

            if concept_lower in query_lower:

                exact_matches.append(
                    concept
                )

        # Prefer the longest exact concept.
        # This prevents a shorter concept from being selected
        # when a longer concept is present.

        if exact_matches:

            matched_concept = max(
                exact_matches,
                key=lambda x: len(x)
            )

        # --------------------------------------------------------
        # 2. Deterministic token-overlap matching
        # --------------------------------------------------------

        if matched_concept is None:

            best_score = 0
            best_concept = None

            for concept in concepts:

                concept_tokens = set(
                    concept.lower()
                    .replace("?", "")
                    .replace(",", "")
                    .split()
                )

                if not concept_tokens:
                    continue

                overlap = (
                    query_content_tokens
                    &
                    concept_tokens
                )

                if not overlap:
                    continue

                # Fraction of concept tokens matched
                concept_coverage = (
                    len(overlap)
                    /
                    len(concept_tokens)
                )

                # Fraction of query content represented
                query_coverage = (
                    len(overlap)
                    /
                    max(
                        len(query_content_tokens),
                        1
                    )
                )

                # Weighted deterministic score
                score = (
                    0.7 * concept_coverage
                    +
                    0.3 * query_coverage
                )

                if score > best_score:

                    best_score = score
                    best_concept = concept

            if best_concept is not None:

                matched_concept = best_concept

                print(
                    f"🔎 Token-overlap match: "
                    f"{matched_concept} "
                    f"(score={best_score:.3f})"
                )

        # --------------------------------------------------------
        # 3. No concept matched
        # --------------------------------------------------------

        if matched_concept is None:

            print(
                "⚠️ No graph concept matched."
            )

            state.refinement_type = "No Refinement"

            state.refined_query_text = original_query

            # Keep original query if no graph concept
            # can be identified.

            state.refined_query = original_query

            state.matched_concept = ""

            return state

        # --------------------------------------------------------
        # 4. Successful deterministic matching
        # --------------------------------------------------------

        print(
            f"🔎 Deterministic concept match: "
            f"{matched_concept}"
        )

        state.refinement_type = "No Refinement"

        # Preserve original query for evaluation
        state.refined_query_text = original_query

        # Retrieval uses graph concept
        state.refined_query = matched_concept

        state.matched_concept = matched_concept

        print(
            f"🔁 Retrieval Query (No-RL): "
            f"{state.refined_query}"
        )

        return state


# ============================================================
# 4. Retrieval Agent
# ============================================================

class RetrievalAgent:

    def retrieve(self, state):

        query = state.refined_query
        graph = state.graph

        print(
            f"\n🔎 Retrieving knowledge for: "
            f"{query}"
        )

        direct = graph.retrieve_direct_causes(query)

        multi = graph.retrieve_multi_hop_causes(query)

        print(
            f"✅ Direct causes: {len(direct)}"
        )

        print(
            f"✅ Multi-hop paths: {len(multi)}"
        )

        if not direct and not multi:

            print(
                f"⚠️ No causal evidence found "
                f"in Neo4j for: {query}"
            )

        state.retrieved_docs = direct

        state.causal_docs = [
            " → ".join(path)
            for path in multi
        ]

        return state


# ============================================================
# 5. Knowledge Rewriting Agent
# ============================================================

class RewritingAgent:

    def __init__(self):

        self.rewriter = (
            KnowledgeRewritingAgent()
            .optimize_rewriting
        )

    def rewrite(
        self,
        state
    ):

        rewriting_state = RewritingState(
            query=state.query,
            retrieved_docs=state.retrieved_docs,
            causal_docs=state.causal_docs
        )

        result = self.rewriter(
            rewriting_state
        )

        state.rewritten_knowledge = (
            result.rewritten_knowledge
        )

        return state


# ============================================================
# 6. Response Generation Agent
# ============================================================

class ResponseGenerationAgent:

    def __init__(self):

        self.generator = (
            LLMResponseGenerator()
        )

    def generate_response(
        self,
        state
    ):

        response = self.generator.generate(
            knowledge=state.rewritten_knowledge,
            original_query=state.query
        )

        state.final_response = response

        return state


# ============================================================
# 7. Embedding Hallucination Detector
# ============================================================

class HallucinationDetectionAgent:

    def __init__(self):

        self.detector = (
            HallucinationDetector()
        )

    def detect(
        self,
        state
    ):

        is_bad = self.detector.detect(
            state.final_response,
            state.rewritten_knowledge
        )

        state.embedding_hallucination = (
            bool(is_bad)
        )

        print(
            f"🧠 Embedding Hallucination: "
            f"{is_bad}"
        )

        return state


# ============================================================
# 8. Deterministic Validator Agent
# ============================================================

class DeterministicValidationAgent:

    def validate(
        self,
        state
    ):

        print(
            "\n🔐 Deterministic Neo4j "
            "Causal Validation"
        )

        validator = (
            DeterministicCausalValidator(
                state.graph
            )
        )

        result = validator.validate(
            state.final_response
        )

        state.validator_passed = (
            result["passed"]
        )

        state.supported_claims = (
            result["supported_claims"]
        )

        state.unsupported_claims = (
            result["unsupported_claims"]
        )

        state.num_claims = (
            result["num_claims"]
        )

        state.num_supported_claims = (
            result["num_supported_claims"]
        )

        state.num_unsupported_claims = (
            result["num_unsupported_claims"]
        )

        # ----------------------------------------------------
        # Final hallucination flag
        #
        # Either detector can flag the answer.
        # ----------------------------------------------------

        embedding_bad = getattr(
            state,
            "embedding_hallucination",
            False
        )

        deterministic_bad = (
            not state.validator_passed
        )

        state.is_hallucination = (
            embedding_bad
            or deterministic_bad
        )

        print(
            f"   Claims detected: "
            f"{state.num_claims}"
        )

        print(
            f"   Supported claims: "
            f"{state.num_supported_claims}"
        )

        print(
            f"   Unsupported claims: "
            f"{state.num_unsupported_claims}"
        )

        print(
            f"   Neo4j validator: "
            f"{'PASS' if state.validator_passed else 'FAIL'}"
        )

        print(
            f"   Final hallucination: "
            f"{'YES' if state.is_hallucination else 'NO'}"
        )

        if state.unsupported_claims:

            for claim in state.unsupported_claims:

                print(
                    f"   - {claim['cause']} → {claim['effect']}"
                )

                print(
                    f"     Sentence: {claim['sentence']}"
                )

        return state


# ============================================================
# 9. Regeneration Agent
# ============================================================

class HallucinationFallbackAgent:

    def __init__(self):

        self.generator = (
            LLMResponseGenerator()
        )

    def regenerate(
        self,
        state
    ):

        if not state.is_hallucination:

            return state

        print(
            "\n🔁 Regenerating response "
            "because validation failed..."
        )

        response = self.generator.generate(
            knowledge=state.rewritten_knowledge,
            original_query=state.query
        )

        state.final_response = response

        # ----------------------------------------------------
        # Revalidate regenerated response
        # ----------------------------------------------------

        validator = (
            DeterministicCausalValidator(
                state.graph
            )
        )

        validation = validator.validate(
            state.final_response
        )

        state.validator_passed = (
            validation["passed"]
        )

        state.supported_claims = (
            validation[
                "supported_claims"
            ]
        )

        state.unsupported_claims = (
            validation[
                "unsupported_claims"
            ]
        )

        state.num_claims = (
            validation[
                "num_claims"
            ]
        )

        state.num_supported_claims = (
            validation[
                "num_supported_claims"
            ]
        )

        state.num_unsupported_claims = (
            validation[
                "num_unsupported_claims"
            ]
        )

        # ----------------------------------------------------
        # Recalculate final hallucination
        # ----------------------------------------------------

        embedding_bad = getattr(
            state,
            "embedding_hallucination",
            False
        )

        deterministic_bad = (
            not state.validator_passed
        )

        state.is_hallucination = (
            embedding_bad
            or deterministic_bad
        )

        print(
            "\n🔐 Re-validation:"
        )

        print(
            f"   Neo4j validator: "
            f"{'PASS' if state.validator_passed else 'FAIL'}"
        )

        print(
            f"   Final hallucination: "
            f"{'YES' if state.is_hallucination else 'NO'}"
        )

        return state


# ============================================================
# 10. CDF State
# ============================================================

class CDFState(BaseModel):

    query: str

    refinement_type: str = ""

    refined_query_text: str = ""

    refined_query: str = ""

    matched_concept: str = ""

    retrieved_docs: list[str] = []

    causal_docs: list[str] = []

    rewritten_knowledge: str = ""

    final_response: str = ""

    # --------------------------------------------------------
    # Existing hallucination result
    # --------------------------------------------------------

    is_hallucination: bool = False

    # --------------------------------------------------------
    # Embedding detector result
    # --------------------------------------------------------

    embedding_hallucination: bool = False

    # --------------------------------------------------------
    # Deterministic validator results
    # --------------------------------------------------------

    validator_passed: bool = False

    supported_claims: list[str] = []

    unsupported_claims: list[str] = []

    num_claims: int = 0

    num_supported_claims: int = 0

    num_unsupported_claims: int = 0

    # Neo4j object
    graph: object

    class Config:

        arbitrary_types_allowed = True


# ============================================================
# 11. LangGraph Pipeline
# ============================================================

print(
    "⚙️ Initializing LangGraph pipeline..."
)

pipeline = StateGraph(
    CDFState
)


pipeline.add_node(
    "refine_query",
    QueryRefinementAgent().refine
)


pipeline.add_node(
    "retrieve_knowledge",
    RetrievalAgent().retrieve
)


pipeline.add_node(
    "rewrite_knowledge",
    RewritingAgent().rewrite
)


pipeline.add_node(
    "generate_response",
    ResponseGenerationAgent().generate_response
)


pipeline.add_node(
    "detect_hallucination",
    HallucinationDetectionAgent().detect
)


pipeline.add_node(
    "validate_causal_evidence",
    DeterministicValidationAgent().validate
)


pipeline.add_node(
    "regenerate_response_if_needed",
    HallucinationFallbackAgent().regenerate
)



# ============================================================
# Pipeline edges
# ============================================================

graph = (

    pipeline

    .set_entry_point(
        "refine_query"
    )

    .add_edge(
        "refine_query",
        "retrieve_knowledge"
    )

    .add_edge(
        "retrieve_knowledge",
        "rewrite_knowledge"
    )

    .add_edge(
        "rewrite_knowledge",
        "generate_response"
    )

    .add_edge(
        "generate_response",
        "detect_hallucination"
    )

    .add_edge(
        "detect_hallucination",
        "validate_causal_evidence"
    )

    .add_edge(
        "validate_causal_evidence",
        "regenerate_response_if_needed"
    )

    .set_finish_point(
        "regenerate_response_if_needed"
    )

    .compile()
)


print(
    "✅ LangGraph pipeline compiled."
)


# ============================================================
# 12. Standalone Test
# ============================================================

if __name__ == "__main__":

    from config import CONFIG

    graph_obj = CausalGraphRetriever(
        CONFIG["NEO4J_URI"],
        CONFIG["NEO4J_USER"],
        CONFIG["NEO4J_PASSWORD"]
    )

    initial_state = {
        "query":
            "What causes heart disease?",

        "graph":
            graph_obj
    }

    try:

        final_state = graph.invoke(
            initial_state
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "FINAL STRUCTURED KNOWLEDGE"
        )

        print(
            "=" * 60
        )

        print(
            final_state.get(
                "rewritten_knowledge"
            )
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "FINAL RESPONSE"
        )

        print(
            "=" * 60
        )

        print(
            final_state.get(
                "final_response"
            )
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "DETERMINISTIC VALIDATION"
        )

        print(
            "=" * 60
        )

        print(
            "Claims:",
            final_state.get(
                "num_claims",
                0
            )
        )

        print(
            "Supported:",
            final_state.get(
                "num_supported_claims",
                0
            )
        )

        print(
            "Unsupported:",
            final_state.get(
                "num_unsupported_claims",
                0
            )
        )

        print(
            "Validator:",
            "PASS"
            if final_state.get(
                "validator_passed"
            )
            else "FAIL"
        )

        print(
            "\nUnsupported claims:"
        )

        for claim in final_state.get(
            "unsupported_claims",
            []
        ):

            print(
                f"  ❌ {claim}"
            )

        print(
            "\nEmbedding detector:",
            "YES"
            if final_state.get(
                "embedding_hallucination"
            )
            else "NO"
        )

        print(
            "Final hallucination:",
            "YES"
            if final_state.get(
                "is_hallucination"
            )
            else "NO"
        )

    finally:

        graph_obj.close()

        print(
            "\n🔌 Neo4j connection closed."
        )