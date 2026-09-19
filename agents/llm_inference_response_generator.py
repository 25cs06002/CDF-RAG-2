import requests
from config import CONFIG


class LLMResponseGenerator:
    """
    Generates the final response using a local Ollama LLM.
    """

    def __init__(self):
        self.model = CONFIG.get("LLM_MODEL", "llama3.2:latest")
        self.base_url = CONFIG.get(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434"
        )

    def generate(
        self,
        knowledge: str,
        original_query: str,
        force_retrieval: bool = False
    ) -> str:

        # --------------------------------------------------------
        # Prevent generation when no evidence was retrieved
        # --------------------------------------------------------

        if (
            not knowledge.strip()
            or
            knowledge.strip()
            == "No supporting evidence was retrieved from the knowledge graph."
        ):
            print(
                "⚠️ No supporting evidence available. "
                "Skipping LLM answer generation."
            )

            return (
                "No supporting evidence was retrieved from the "
                "knowledge graph to answer this query."
            )

        
        prompt = f"""
        You are a domain-specific question answering assistant.

        Answer the user's query using ONLY the information contained
        in the Structured Knowledge.

        Query:
        {original_query}

        Structured Knowledge:
        {knowledge}

        Rules:
        - Use only facts explicitly supported by the Structured Knowledge.
        - Do not introduce new causes, effects, explanations, examples,
        statistics, or details.
        - Do not use your own background knowledge to add information.
        - You may paraphrase the provided knowledge.
        - Preserve the meaning and causal relationships of the knowledge.
        - If the knowledge is insufficient to answer the query, say that
        the available evidence is insufficient.
        - Keep the answer concise and factual.
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