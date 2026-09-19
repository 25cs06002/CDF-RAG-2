import requests
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from config import CONFIG


class HallucinationDetector:

    def __init__(self):

        # ----------------------------------------------------
        # Local embedding model
        # ----------------------------------------------------

        self.encoder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # Semantic similarity threshold
        self.threshold = 0.65

        self.model = CONFIG.get(
            "HALLUCINATION_DETECTION_MODEL",
            "llama3.2:latest"
        )

        self.base_url = CONFIG.get(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434"
        )

        print(
            "[Hallucination Detector] "
            "Embedding-based detector initialized."
        )

    # --------------------------------------------------------
    # Normalize text for exact evidence matching
    # --------------------------------------------------------

    def _normalize(self, text):

        text = text.lower()

        # Remove punctuation
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # --------------------------------------------------------
    # Split explanation into individual factual statements
    # --------------------------------------------------------

    def _split_sentences(self, text):

        text = text.replace("\n", " ")

        sentences = []

        for sentence in text.split("."):

            sentence = sentence.strip()

            if sentence:
                sentences.append(sentence)

        return sentences

    # --------------------------------------------------------
    # Split final answer into claims
    # --------------------------------------------------------

    def _extract_claims(self, answer):

        prompt = f"""
Extract the distinct factual claims from the following answer.

ANSWER:
{answer}

Rules:
- Preserve the meaning of every factual claim.
- Separate different causes, effects, reasons, consequences,
  conditions, and additional factual details.
- Do not invent information.
- Do not judge whether the claims are true.
- Return one claim per line.
- Do not number the claims.
- Do not provide explanations.

Return only the claims.
"""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 100
                }
            },
            timeout=120
        )

        response.raise_for_status()

        output = response.json().get(
            "response",
            ""
        ).strip()

        claims = []

        for line in output.splitlines():

            line = line.strip()

            # Remove numbering/bullets
            line = line.lstrip("0123456789.-) ")

            if line:
                claims.append(line)

        return claims

    # --------------------------------------------------------
    # Check whether claim is explicitly contained in evidence
    # --------------------------------------------------------

    def _exact_evidence_match(
        self,
        claim,
        explanation
    ):

        claim_norm = self._normalize(claim)
        explanation_norm = self._normalize(explanation)

        if not claim_norm:
            return False

        return claim_norm in explanation_norm

    # --------------------------------------------------------
    # Main detection function
    # --------------------------------------------------------

    def detect(
        self,
        final_answer,
        structured_explanation
    ):

        print("\n[Hallucination Detector]")
        print("Checking semantic consistency...")

        explanation_claims = self._split_sentences(
            structured_explanation
        )

        answer_claims = self._extract_claims(
            final_answer
        )

        if not explanation_claims:

            print(
                "⚠️ Empty structured explanation."
            )

            return True

        if not answer_claims:

            print(
                "⚠️ Empty final answer."
            )

            return True

        # ----------------------------------------------------
        # First perform exact evidence matching
        # ----------------------------------------------------

        for claim in answer_claims:

            exact_match = False

            for explanation in explanation_claims:

                if self._exact_evidence_match(
                    claim,
                    explanation
                ):

                    exact_match = True

                    print(
                        f"\nAnswer claim:\n"
                        f"  {claim}"
                    )

                    print(
                        f"Evidence match:\n"
                        f"  {explanation}"
                    )

                    print(
                        "✅ Exact evidence match."
                    )

                    break

            if exact_match:
                continue

            # ------------------------------------------------
            # Semantic fallback
            # ------------------------------------------------

            explanation_embeddings = self.encoder.encode(
                explanation_claims,
                normalize_embeddings=True
            )

            claim_embedding = self.encoder.encode(
                claim,
                normalize_embeddings=True
            )

            similarities = np.dot(
                explanation_embeddings,
                claim_embedding
            )

            best_index = int(
                np.argmax(similarities)
            )

            best_score = float(
                similarities[best_index]
            )

            best_match = explanation_claims[
                best_index
            ]

            print(
                f"\nAnswer claim:\n"
                f"  {claim}"
            )

            print(
                f"Best explanation match:\n"
                f"  {best_match}"
            )

            print(
                f"Semantic similarity: "
                f"{best_score:.3f}"
            )

            if best_score < self.threshold:

                print(
                    "❌ Unsupported claim detected."
                )

                return True

            print(
                "✅ Claim semantically supported."
            )

        # ----------------------------------------------------
        # All claims supported
        # ----------------------------------------------------

        print(
            "\n[Hallucination Detector] "
            "Result: SUPPORTED"
        )

        return False
