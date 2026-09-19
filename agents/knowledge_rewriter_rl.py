import requests
from langgraph.graph import StateGraph
from pydantic import BaseModel
from config import CONFIG


# ============================================================
# LangGraph State
# ============================================================

class RewritingState(BaseModel):
    query: str
    retrieved_docs: list[str]
    causal_docs: list[str]
    rewritten_knowledge: str = ""


# ============================================================
# Ollama Knowledge Rewriter
# ============================================================

class KnowledgeRewriter:
    """
    Evidence-grounded knowledge rewriting using local Ollama.

    IMPORTANT:
    - No OpenAI API
    - No external LLM
    - Only retrieved Neo4j evidence is used
    """

    def __init__(self):

        self.model = CONFIG.get(
            "KNOWLEDGE_REWRITING_MODEL",
            "llama3.2:latest"
        )

        self.base_url = CONFIG.get(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434"
        )

    # ========================================================
    # Rewrite
    # ========================================================

    def rewrite(
        self,
        direct_docs: list[str],
        causal_docs: list[str],
        query: str
    ) -> str:

        # ----------------------------------------------------
        # Direct evidence
        # ----------------------------------------------------

        direct_context = "\n".join(
            f"{i + 1}. {doc}"
            for i, doc in enumerate(direct_docs)
        )

        # ----------------------------------------------------
        # Multi-hop evidence
        # ----------------------------------------------------

        causal_context = "\n".join(
            f"{i + 1}. {doc}"
            for i, doc in enumerate(causal_docs)
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = f"""
You are an evidence-grounded medical knowledge rewriting system.

Your job is ONLY to organize the evidence retrieved from a causal
knowledge graph.

You must NOT use your own medical knowledge.

QUERY:
{query}


============================================================
DIRECT CAUSAL EVIDENCE
============================================================

{direct_context}


============================================================
MULTI-HOP CAUSAL EVIDENCE
============================================================

{causal_context}


============================================================
STRICT RULES
============================================================

RULE 1:
Use ONLY information explicitly contained in the evidence.

RULE 2:
A direct causal statement has the form:

A → B

If the evidence contains:

A → B

you may write:

"A causes B."

RULE 3:
A multi-hop path has the form:

A → B → C

NEVER rewrite this as:

"A causes C."

unless "A → C" is separately present in the DIRECT CAUSAL
EVIDENCE.

RULE 4:
Never convert a multi-hop relationship into a direct relationship.

RULE 5:
Never infer a medical relationship.

RULE 6:
Never add causes, risk factors, symptoms, treatments, mechanisms,
or explanations that are not explicitly present.

RULE 7:
Do not combine multiple causal paths into a new causal statement.

RULE 8:
The final answer should primarily use DIRECT CAUSAL EVIDENCE.

RULE 9:
Multi-hop paths may only be mentioned if the complete chain is
preserved exactly.

For example:

Evidence:
A → B → C

Allowed:
"A is connected to C through B."

Not allowed:
"A causes C."

RULE 10:
If a direct cause exists, report the direct cause exactly.

RULE 11:
Do not mention information merely because it is medically plausible.

RULE 12:
Do not introduce words such as "therefore", "thus", "because",
or "which leads to" if doing so creates a new causal relationship.

RULE 13:
Write concise factual sentences.

RULE 14:
Return ONLY the evidence-grounded explanation.

============================================================
OUTPUT REQUIREMENT
============================================================

For a query asking "What causes X?", prefer the following format:

"X is caused by A, B, and C."

ONLY include A, B, and C if the corresponding relationships
A → X, B → X, and C → X are explicitly present in the
DIRECT CAUSAL EVIDENCE.

Do not include upstream nodes from multi-hop paths as direct causes.
"""

        # ----------------------------------------------------
        # Ollama request
        # ----------------------------------------------------

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 4096
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get("response", "").strip()

        return answer


# ============================================================
# Deterministic Evidence Fallback
# ============================================================

def deterministic_direct_answer(
    direct_docs: list[str]
) -> str:
    """
    Creates an answer directly from retrieved causal evidence.

    This is intentionally deterministic and does not use an LLM.
    """

    if not direct_docs:
        return (
            "No direct causal evidence was retrieved "
            "from the knowledge graph."
        )

    statements = []

    for doc in direct_docs:

        doc = str(doc).strip()

        if not doc:
            continue

        # ----------------------------------------------------
        # Convert only the explicit relationship.
        #
        # Example:
        # Smoking → heart disease
        #
        # becomes:
        # Smoking causes heart disease.
        # ----------------------------------------------------

        if "→" in doc:

            parts = [
                p.strip()
                for p in doc.split("→")
            ]

            # Only accept a DIRECT relationship.
            if len(parts) == 2:

                cause = parts[0]
                effect = parts[1]

                statements.append(
                    f"{cause} causes {effect}."
                )

        else:

            # If document is already a natural-language
            # causal statement, preserve it.
            statements.append(doc)

    if not statements:

        return (
            "No direct causal evidence was available "
            "for the query."
        )

    return " ".join(statements)


# ============================================================
# Knowledge Rewriting Agent
# ============================================================

class KnowledgeRewritingAgent:

    def __init__(self):

        self.rewriter = KnowledgeRewriter()

    # ========================================================
    # Optimize Rewriting
    # ========================================================

    def optimize_rewriting(
        self,
        state: RewritingState
    ) -> RewritingState:

        direct_docs = state.retrieved_docs
        causal_docs = state.causal_docs

        # ----------------------------------------------------
        # No evidence
        # ----------------------------------------------------

        if not direct_docs and not causal_docs:

            state.rewritten_knowledge = (
                "No supporting evidence was retrieved "
                "from the knowledge graph."
            )

            return state

        # ----------------------------------------------------
        # Try Ollama rewriting
        # ----------------------------------------------------

        try:

            rewritten = self.rewriter.rewrite(
                direct_docs=direct_docs,
                causal_docs=causal_docs,
                query=state.query
            )

            # ------------------------------------------------
            # Empty response
            # ------------------------------------------------

            if not rewritten.strip():

                print(
                    "⚠️ Ollama returned an empty response."
                )

                rewritten = deterministic_direct_answer(
                    direct_docs
                )

            state.rewritten_knowledge = rewritten.strip()

        except Exception as e:

            print(
                f"⚠️ Knowledge rewriting failed: {e}"
            )

            print(
                "🔄 Using deterministic evidence-based fallback."
            )

            state.rewritten_knowledge = (
                deterministic_direct_answer(
                    direct_docs
                )
            )

        return state


# ============================================================
# LangGraph Workflow
# ============================================================

rewriting_graph = StateGraph(
    RewritingState
)

rewriting_graph.add_node(
    "optimize_rewriting",
    KnowledgeRewritingAgent().optimize_rewriting
)

graph = (
    rewriting_graph
    .set_entry_point("optimize_rewriting")
    .set_finish_point("optimize_rewriting")
    .compile()
)


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    test_state = RewritingState(

        query="What causes heart disease?",

        retrieved_docs=[
            "Genetic predisposition → heart disease",
            "Poor diet → heart disease",
            "Diabetes → heart disease",
            "High blood pressure → heart disease",
            "Smoking → heart disease"
        ],

        causal_docs=[
            "Genetic predisposition → Diabetes → heart disease",
            "Poor diet → Diabetes → heart disease",
            "Lack of physical activity → Diabetes → heart disease",
            "Family history of disease → Genetic predisposition → heart disease"
        ]
    )

    final_state = graph.invoke(
        test_state
    )

    print(
        "\n🔹 Final Structured Explanation:\n"
    )

    print(
        final_state.get(
            "rewritten_knowledge"
        )
    )


'''
import requests
from langgraph.graph import StateGraph
from pydantic import BaseModel
from config import CONFIG


# ============================================================
# LangGraph state
# ============================================================

class RewritingState(BaseModel):
    query: str
    retrieved_docs: list[str]
    causal_docs: list[str]
    rewritten_knowledge: str = ""


# ============================================================
# Ollama-based Knowledge Rewriter
# ============================================================

class KnowledgeRewriter:
    """
    Rewrites and structures retrieved semantic and causal evidence
    using a local Ollama LLM.
    """

    def __init__(self):
        self.model = CONFIG.get(
            "KNOWLEDGE_REWRITING_MODEL",
            "llama3.2:latest"
        )

        self.base_url = CONFIG.get(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434"
        )

    def rewrite(self, documents: list[str], query: str) -> str:

        # Combine retrieved semantic and causal evidence
        context = "\n".join(
            f"- {doc}" for doc in documents
        )

        prompt = f"""
    You are a medical knowledge synthesis assistant.

    Your task is to organize the retrieved evidence into a concise,
    factually grounded explanation that answers the query.

    Query:
    {query}

    Retrieved Evidence:
    {context}

    Strict instructions:

    1. Use ONLY information explicitly present in the retrieved evidence.

    2. Do NOT introduce new medical facts, causes, effects, symptoms,
    treatments, risk factors, or explanations.

    3. Do NOT use general medical knowledge that is not present in
    the retrieved evidence.

    4. Preserve the direction of every causal relationship.
    If the evidence says A causes B, do not reverse it to B causes A.

    5. Preserve multi-hop causal relationships when they are explicitly
    present in the evidence.

    6. Do not infer additional causal relationships.

    7. Do not combine separate facts in a way that creates a new
    unsupported conclusion.

    8. If the evidence is insufficient to answer part of the query,
    do not fill the gap with your own knowledge.

    9. Keep the explanation concise.

    10. Write each factual statement as a separate sentence.

    Return ONLY the evidence-grounded explanation.
    """

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 2048
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        return result["response"].strip()


# ============================================================
# LangGraph Agent
# ============================================================

class KnowledgeRewritingAgent:

    def __init__(self):
        self.rewriter = KnowledgeRewriter()

    def optimize_rewriting(
        self,
        state: RewritingState
    ) -> RewritingState:

        # Combine semantic and causal documents
        combined_docs = (
            state.retrieved_docs +
            state.causal_docs
        )

        # Rewrite the combined evidence
        structured_summary = self.rewriter.rewrite(
            combined_docs,
            state.query
        )

        state.rewritten_knowledge = structured_summary

        return state


# ============================================================
# LangGraph workflow
# ============================================================

rewriting_graph = StateGraph(RewritingState)

rewriting_graph.add_node(
    "optimize_rewriting",
    KnowledgeRewritingAgent().optimize_rewriting
)

graph = (
    rewriting_graph
    .set_entry_point("optimize_rewriting")
    .set_finish_point("optimize_rewriting")
    .compile()
)


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    test_state = RewritingState(
        query="What causes heart disease?",

        retrieved_docs=[
            "Hypercholesterolemia causes heart disease.",
            "Obstructive sleep apnea contributes to cardiovascular conditions."
        ],

        causal_docs=[
            "Hypertension is a common intermediary between stress and heart disease.",
            "Smoking leads to atherosclerosis, which leads to heart disease."
        ]
    )

    final_state = graph.invoke(test_state)

    print("\n🔹 Final Structured Explanation:\n")
    print(final_state.get("rewritten_knowledge"))
'''

