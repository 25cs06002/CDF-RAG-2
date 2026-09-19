import os
import requests
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from sentence_transformers.util import cos_sim
from stable_baselines3.common.env_checker import check_env

# ============================================================
# Ollama Configuration
# ============================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest"
)

ENCODER_MODEL = os.getenv(
    "ENCODER_MODEL",
    "nomic-embed-text:latest"
)


# ============================================================
# Ollama LLM
# ============================================================

def call_ollama(prompt: str) -> str:
    """
    Send a prompt to the local Ollama LLM with retry handling.

    Ollama may occasionally close a connection while a model is
    loading or generating. Retry a few times before failing.
    """

    max_retries = 3

    for attempt in range(1, max_retries + 1):

        try:

            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0
                    }
                },
                timeout=300
            )

            response.raise_for_status()

            data = response.json()

            result = data.get("response", "").strip()

            if not result:
                raise RuntimeError(
                    "Ollama returned an empty response."
                )

            return result

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException
        ) as e:

            print(
                f"\n⚠️ Ollama request failed "
                f"(attempt {attempt}/{max_retries})"
            )

            print(f"Error: {e}")

            if attempt < max_retries:

                import time

                wait_time = 5 * attempt

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                raise RuntimeError(
                    "Ollama request failed after "
                    f"{max_retries} attempts."
                ) from e

    raise RuntimeError("Unexpected Ollama execution path.")



# ============================================================
# Ollama Embedding
# ============================================================

def get_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding using Ollama nomic-embed-text.
    """

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": ENCODER_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    embedding = response.json()["embeddings"][0]

    return np.asarray(embedding, dtype=np.float32)


# ============================================================
# Simulated Retrieval
# ============================================================

def retrieve_causal_docs(query):
    """
    Temporary retrieval function.

    This will later be replaced by the actual
    Pinecone + Neo4j retrieval pipeline.
    """

    return (
        [f"Retrieved document relevant to: {query}"],
        ["cause → effect"]
    )


# ============================================================
# Knowledge Rewriting
# ============================================================

def call_rewriting_llm(docs, paths):

    prompt = f"""
You are a causal reasoning assistant.

Given the retrieved documents and causal paths below,
write a concise structured causal explanation.

Retrieved Documents:
{docs}

Causal Paths:
{paths}

Requirements:
- Use only the supplied information.
- Preserve the causal relationships.
- Do not invent facts.
- Clearly describe cause → effect relationships.
"""

    return call_ollama(prompt)


# ============================================================
# Final Response Generation
# ============================================================

def call_final_response_llm(query, explanation):

    prompt = f"""
Answer the following question using only the supplied
structured explanation.

Question:
{query}

Structured Explanation:
{explanation}

Requirements:
- Give a clear natural-language answer.
- Do not introduce unsupported facts.
- Stay consistent with the explanation.
"""

    return call_ollama(prompt)


# ============================================================
# Hallucination Detection
# ============================================================

def detect_hallucination(rewritten, final_answer):

    prompt = f"""
You are a strict semantic entailment evaluator.

Your job is to determine whether every factual claim in the
FINAL ANSWER is supported by the STRUCTURED EXPLANATION.

IMPORTANT:
Judge MEANING, not exact wording.

Equivalent paraphrases MUST be considered supported.

Examples:

STRUCTURED EXPLANATION:
"Transportation problems result in missed appointments."

FINAL ANSWER:
"Transportation issues can cause patients to miss appointments."

Result: SUPPORTED

The following differences do NOT constitute hallucination:
- cause vs result in
- can cause vs contributes to
- problems vs issues
- patients vs people
- different sentence structure
- grammatical reformulation

However, classify as HALLUCINATED if the final answer:
- introduces a completely new fact
- introduces a new cause
- introduces a new effect
- introduces a new entity
- introduces unsupported numbers/statistics
- contradicts the explanation
- makes a substantially stronger unsupported claim

------------------------------------------------------------
STRUCTURED EXPLANATION
------------------------------------------------------------

{rewritten}

------------------------------------------------------------
FINAL ANSWER
------------------------------------------------------------

{final_answer}

------------------------------------------------------------

Return exactly ONE word:

SUPPORTED

or

HALLUCINATED

Do not provide an explanation.
"""

    try:

        result = call_ollama(prompt)

        result = result.strip().upper()

        print(
            "[Hallucination Detector] Raw LLM output:",
            result
        )

        # ----------------------------------------------------
        # Extract classification
        # ----------------------------------------------------

        if result.startswith("SUPPORTED"):
            return False

        if result.startswith("HALLUCINATED"):
            return True

        # Sometimes Ollama may put extra text before the label.
        if "HALLUCINATED" in result:
            return True

        if "SUPPORTED" in result:
            return False

        print(
            "[Hallucination Detector] WARNING: "
            "Invalid detector output."
        )

        return True

    except Exception as e:

        print(
            "[Hallucination Detector] ERROR:",
            str(e)
        )

        # Conservative fallback
        return True



# ============================================================
# Average Causal Depth
# ============================================================

def average_depth(paths):

    if not paths:
        return 0.0

    return np.mean([
        path.count("→") + 1
        for path in paths
    ])


# ============================================================
# Query / Explanation Similarity
# ============================================================

def cosine_similarity(query, rewritten):

    query_embedding = get_embedding(query)
    rewritten_embedding = get_embedding(rewritten)

    return float(
        cos_sim(
            query_embedding,
            rewritten_embedding
        ).item()
    )


# ============================================================
# CDF-RAG Simulation
# ============================================================

def simulate_cdf_pipeline(refined_query):

    # --------------------------------------------------------
    # 1. Retrieval
    # --------------------------------------------------------

    retrieved_docs, causal_paths = retrieve_causal_docs(
        refined_query
    )

    # --------------------------------------------------------
    # 2. Knowledge rewriting
    # --------------------------------------------------------

    rewritten = call_rewriting_llm(
        retrieved_docs,
        causal_paths
    )

    # --------------------------------------------------------
    # 3. Final response
    # --------------------------------------------------------

    final_answer = call_final_response_llm(
        refined_query,
        rewritten
    )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print("\n========================================")
    print("DEBUG - STRUCTURED EXPLANATION")
    print("========================================")
    print(rewritten)

    print("\n========================================")
    print("DEBUG - FINAL ANSWER")
    print("========================================")
    print(final_answer)

    print("\n========================================")

    # --------------------------------------------------------
    # 4. Hallucination detection
    # --------------------------------------------------------

    hallucinated = detect_hallucination(
        rewritten,
        final_answer
    )

    # --------------------------------------------------------
    # 5. Reward components
    # --------------------------------------------------------

    # Retrieval coverage
    retrieval_coverage = (
        1.0 if retrieved_docs else 0.0
    )

    # --------------------------------------------------------
    # Normalize causal depth
    #
    # Depth 1 -> 1.0
    # Depth 2 -> 0.75
    # Depth 3+ -> approaches 1.0
    #
    # We do not want depth alone to dominate the reward.
    # --------------------------------------------------------

    raw_depth = average_depth(causal_paths)

    if raw_depth <= 0:
        depth_score = 0.0
    else:
        depth_score = min(raw_depth / 3.0, 1.0)

    # --------------------------------------------------------
    # Query / explanation relevance
    # --------------------------------------------------------

    context_relevance = cosine_similarity(
        refined_query,
        rewritten
    )

    # Cosine similarity should normally be [-1, 1].
    # Convert it to [0, 1] for reward calculation.
    context_relevance = (
        context_relevance + 1.0
    ) / 2.0

    context_relevance = float(
        np.clip(context_relevance, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # Hallucination score
    #
    # No hallucination -> 1
    # Hallucination    -> 0
    # --------------------------------------------------------

    hallucination_score = (
        0.0 if hallucinated else 1.0
    )

    # --------------------------------------------------------
    # Final weighted reward
    # --------------------------------------------------------

    reward = (
        0.30 * retrieval_coverage +
        0.20 * depth_score +
        0.20 * context_relevance +
        0.30 * hallucination_score
    )

    reward = float(
        np.clip(reward, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # Debug information
    # --------------------------------------------------------

    print(
        f"Retrieval Score : {retrieval_coverage:.3f}"
    )

    print(
        f"Depth Score     : {depth_score:.3f}"
    )

    print(
        f"Relevance Score : {context_relevance:.3f}"
    )

    print(
        f"Hallucination   : {hallucination_score:.3f}"
    )

    print(
        f"Final Reward    : {reward:.3f}"
    )

    return reward, hallucinated


# ============================================================
# Precompute RL Rewards
# ============================================================

def build_reward_cache(queries):
    """
    Precompute the expensive CDF-RAG evaluation once.

    Each query is evaluated with all three refinement strategies.
    The resulting rewards are stored in memory so PPO training
    does not repeatedly call Ollama.
    """

    strategies = [
        "Expand",
        "Simplify",
        "Decompose"
    ]

    reward_cache = {}

    for query in queries:

        print("\n" + "=" * 60)
        print("Precomputing rewards for:")
        print(query)
        print("=" * 60)

        reward_cache[query] = {}

        for strategy in strategies:

            # ------------------------------------------------
            # Generate refined query
            # ------------------------------------------------

            if strategy == "Expand":

                refined_query = (
                    f"What are the detailed causes "
                    f"and contributing factors of {query}?"
                )

            elif strategy == "Simplify":

                refined_query = (
                    f"What are the main causes "
                    f"related to {query}?"
                )

            else:

                refined_query = (
                    f"What are the subcauses and "
                    f"cause-effect relationships involved in {query}?"
                )

            print("\n----------------------------------------")
            print("Original Query :", query)
            print("Strategy       :", strategy)
            print("Refined Query  :", refined_query)
            print("----------------------------------------")

            # ------------------------------------------------
            # Expensive CDF-RAG evaluation
            # ------------------------------------------------

            reward, hallucinated = simulate_cdf_pipeline(
                refined_query
            )

            reward_cache[query][strategy] = {
                "reward": float(reward),
                "hallucinated": bool(hallucinated),
                "refined_query": refined_query
            }

            print(
                f"Cached Reward  : {reward:.4f}"
            )

            print(
                f"Hallucinated   : {hallucinated}"
            )

    return reward_cache



# ============================================================
# RL Environment
# ============================================================

class QueryRefinementEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, queries, reward_cache):

        super().__init__()

        self.queries = queries
        self.reward_cache = reward_cache
        self.query_index = 0

        # nomic-embed-text produces 768-dimensional embeddings
        self.embedding_dim = 768

        # Actions:
        # 0 = Expand
        # 1 = Simplify
        # 2 = Decompose
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.embedding_dim,),
            dtype=np.float32
        )

        self.current_query = None

        # --------------------------------------------------------
        # Precompute query embeddings
        # --------------------------------------------------------

        self.embedding_cache = {
            query: get_embedding(query).astype(np.float32)
            for query in self.queries
        }


    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.query_index = (
            self.query_index + 1
        ) % len(self.queries)

        self.current_query = self.queries[
            self.query_index
        ]

        embedding = self.embedding_cache[
            self.current_query
        ]


        return embedding, {}


    # --------------------------------------------------------
    # Step
    # --------------------------------------------------------

    def step(self, action):

        query = self.current_query

        strategies = [
            "Expand",
            "Simplify",
            "Decompose"
        ]

        strategy = strategies[int(action)]

        # ----------------------------------------------------
        # Get precomputed result
        # ----------------------------------------------------

        result = self.reward_cache[query][strategy]

        reward = result["reward"]
        hallucinated = result["hallucinated"]
        refined_query = result["refined_query"]

        # ----------------------------------------------------
        # Next observation
        # ----------------------------------------------------

        observation = get_embedding(query).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Episode terminates after one refinement decision
        # ----------------------------------------------------

        terminated = True
        truncated = False

        # ----------------------------------------------------
        # Debug information
        # ----------------------------------------------------

        print("\n" + "-" * 40)
        print("Original Query :", query)
        print("Strategy       :", strategy)
        print("Refined Query  :", refined_query)
        print("Reward         :", reward)
        print("Hallucinated   :", hallucinated)
        print("Step completed")
        print("-" * 40)

        info = {
            "strategy": strategy,
            "refined_query": refined_query,
            "hallucinated": hallucinated
        }

        return observation, reward, terminated, truncated, info



# ============================================================
# Training
# ============================================================


if __name__ == "__main__":

    queries = [
        "Why do patients miss appointments?",
        "Why do people develop diabetes?",
        "Why do people quit jobs?",
        "Why is climate change a concern?",
        "What leads to student underperformance?",
        "Why is homelessness increasing?"
    ]

    print("\n" + "=" * 60)
    print("BUILDING REWARD CACHE")
    print("=" * 60)

    reward_cache = build_reward_cache(queries)

    print("\n" + "=" * 60)
    print("REWARD CACHE READY")
    print("=" * 60)

    env = QueryRefinementEnv(
        queries,
        reward_cache
    )

    print("\nCreating PPO model...")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=8,
        batch_size=8
    )

    print("\nStarting PPO training...")

    model.learn(
        total_timesteps=600
    )

    model.save("ppo_query_refiner_2")

    print("\n" + "=" * 60)
    print("PPO TRAINING COMPLETE")
    print("=" * 60)

    print("Model saved as:")
    print("ppo_query_refiner_2.zip")
