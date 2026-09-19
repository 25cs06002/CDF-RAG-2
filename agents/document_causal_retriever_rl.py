# ============================================================
# CDF-RAG Document Causal Retriever
# ============================================================
#
# Architecture:
#
# Query
#   ↓
# PPO Query Refinement Decision
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


from stable_baselines3 import PPO
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
# 3. Ollama Query Refinement Agent
# ============================================================

class QueryRefinementAgent:

    def __init__(self):

        # ----------------------------------------------------
        # PPO model
        # ----------------------------------------------------

        self.model = PPO.load(
            "ppo_query_refiner_2"
        )

        # ========================================================
        # Ollama LLM used for query refinement
        # ========================================================

        self.ollama_model = os.getenv(
            "QUERY_REFINEMENT_MODEL",
            "llama3.2:latest"
        )

        self.ollama_url = (
            f"{OLLAMA_BASE_URL}/api/generate"
        )

        # ----------------------------------------------------
        # Ollama embedding model
        # MUST be the same model used during PPO training
        # ----------------------------------------------------

        self.embedding_model = os.getenv(
            "ENCODER_MODEL",
            "nomic-embed-text:latest"
        )

        self.embedding_url = (
            f"{OLLAMA_BASE_URL}/api/embed"
        )

        print(
            f"🧠 PPO embedding model: "
            f"{self.embedding_model}"
        )

    # --------------------------------------------------------
    # Ollama embedding
    # --------------------------------------------------------

    def get_embedding(self, text):

        response = requests.post(
            self.embedding_url,
            json={
                "model": self.embedding_model,
                "input": text
            },
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        embedding = data.get("embeddings")

        if not embedding:

            raise RuntimeError(
                "Ollama returned no embedding."
            )

        embedding = np.asarray(
            embedding[0],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # PPO expects 768 dimensions
        # ----------------------------------------------------

        if embedding.shape != (768,):

            raise ValueError(
                f"Ollama embedding has incorrect "
                f"shape {embedding.shape}. "
                f"Expected (768,)."
            )

        # ----------------------------------------------------
        # Normalize embedding
        # ----------------------------------------------------

        norm = np.linalg.norm(embedding)

        if norm > 0:

            embedding = (
                embedding / norm
            )

        return embedding

    # --------------------------------------------------------
    # Ollama refinement
    # --------------------------------------------------------

    def refine_with_ollama(
        self,
        query,
        refinement_type
    ):

        prompt = f"""
You are a query refinement assistant.

Original query:
{query}

Refinement operation:
{refinement_type}

Rewrite the query into one clear sentence
that preserves the original meaning.

Rules:

1. Keep the same medical/domain concept.
2. Do not introduce new concepts.
3. Do not introduce new causes.
4. Do not introduce new effects.
5. Do not answer the query.
6. Only rewrite the query.
7. Return exactly one sentence.
"""

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 1024
            }
        }

        response = requests.post(
            self.ollama_url,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        refined = result.get(
            "response",
            ""
        ).strip()

        if not refined:

            refined = query

        return refined

    # --------------------------------------------------------
    # Main refinement
    # --------------------------------------------------------

    def refine(
        self,
        state
    ):

        query = state.query

        # ========================================================
        # 1. PPO decides refinement strategy
        # ========================================================

        env = QueryRefinementEnv(
            causal_graph={}
        )

        obs, _ = env.reset()

        # --------------------------------------------------------
        # IMPORTANT:
        # PPO was trained using nomic-embed-text = 768 dimensions
        # --------------------------------------------------------

        observation = self.get_embedding(query)

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        if observation.shape != (768,):

            raise ValueError(
                f"PPO observation has incorrect shape: "
                f"{observation.shape}. Expected (768,)."
            )

        action, _ = self.model.predict(
            observation,
            deterministic=True
        )

        action_int = int(action)

        action_map = {
            0: "Expand Query",
            1: "Simplify Query",
            2: "Decompose Query"
        }

        refinement_type = action_map.get(
            action_int,
            "Expand Query"
        )

        state.refinement_type = refinement_type

        print(
            f"🧠 Suggested refinement: "
            f"{refinement_type}"
        )

        # ========================================================
        # 2. Ollama performs the actual query refinement
        # ========================================================

        refined_query_text = (
            self.refine_with_ollama(
                query,
                refinement_type
            )
        )

        print(
            f"🔁 Refined Query (Ollama): "
            f"{refined_query_text}"
        )

        # ========================================================
        # 3. Embed refined query using SAME model as PPO
        # ========================================================

        query_embedding = self.get_embedding(
            refined_query_text
        )

        # ========================================================
        # 4. Retrieve Neo4j concepts
        # ========================================================

        concepts = state.graph.get_all_concepts()

        if not concepts:

            state.refined_query = ""

            state.refined_query_text = (
                refined_query_text
            )

            return state

        # ========================================================
        # 5. Embed Neo4j concepts
        # ========================================================

        concept_embeddings = np.asarray(
            [
                self.get_embedding(concept)
                for concept in concepts
            ],
            dtype=np.float32
        )

        # ========================================================
        # 6. Semantic concept matching
        # ========================================================

        similarities = cosine_similarity(
            [query_embedding],
            concept_embeddings
        )[0]

        best_idx = int(
            np.argmax(similarities)
        )

        matched_concept = concepts[
            best_idx
        ]

        score = similarities[
            best_idx
        ]

        print(
            f"🔎 Best concept match: "
            f"{matched_concept} "
            f"(score={score:.3f})"
        )

        # ========================================================
        # 7. Save state
        # ========================================================

        state.matched_concept = matched_concept
        state.refined_query = matched_concept

        state.refined_query_text = (
            refined_query_text
        )

        return state


# ============================================================
# 4. Retrieval Agent
# ============================================================

class RetrievalAgent:

    def retrieve(
        self,
        state
    ):

        query = state.matched_concept

        graph = state.graph

        print(
            f"\n🔎 Retrieving knowledge for: "
            f"{query}"
        )

        direct = (
            graph.retrieve_direct_causes(
                query
            )
        )

        multi = (
            graph.retrieve_multi_hop_causes(
                query
            )
        )

        print(
            f"✅ Direct causes: "
            f"{len(direct)}"
        )

        print(
            f"✅ Multi-hop paths: "
            f"{len(multi)}"
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # No LLM causal enrichment is performed here.
        #
        # Neo4j remains the authoritative causal source.
        # ----------------------------------------------------

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