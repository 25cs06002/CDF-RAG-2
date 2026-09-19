import requests
from config import CONFIG


class CausalEnricher:
    def __init__(self):
        self.model = CONFIG.get(
            "LLM_MODEL",
            "llama3.2:latest"
        )

        self.base_url = CONFIG.get(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434"
        )

    def generate_pairs(
        self,
        concept: str
    ) -> list[tuple[str, str]]:

        concept = concept.strip()

        prompt = f"""
You are an expert in causal reasoning.

Target EFFECT:
{concept}

Generate exactly 3 to 5 causal relationships for this target.

STRICT OUTPUT FORMAT:

Cause, {concept}

Rules:
1. Output one relationship per line.
2. Each line must contain exactly one comma.
3. The first part is the cause.
4. The second part MUST be exactly:
{concept}
5. Do not change the target effect.
6. Do not use commas inside the cause.
7. Do not use commas inside the effect.
8. Do not number the lines.
9. Do not use bullets.
10. Do not provide explanations.
11. Output only the causal pairs.

Example:

Smoking, heart disease
Obesity, heart disease
Hypertension, heart disease
Physical inactivity, heart disease
Genetic predisposition, heart disease
"""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 150
                }
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        output = result.get(
            "response",
            ""
        ).strip()

        print(
            f"[Causal Enricher] Raw LLM output:\n{output}"
        )

        pairs = []

        for line in output.splitlines():

            line = line.strip()

            if not line:
                continue

            # Remove markdown/code formatting
            line = line.strip("`")

            # Remove numbering/bullets
            line = line.lstrip(
                "0123456789.-) "
            )

            # Split only on the first comma
            parts = [
                p.strip()
                for p in line.split(",", 1)
            ]

            if len(parts) != 2:
                print(
                    f"⚠️ Skipping malformed line: {line}"
                )
                continue

            cause, effect = parts

            if not cause or not effect:
                print(
                    f"⚠️ Skipping empty pair: {line}"
                )
                continue

            # ------------------------------------------------
            # IMPORTANT: Validate target effect
            # ------------------------------------------------

            if effect.lower() != concept.lower():

                print(
                    f"⚠️ Skipping incorrect effect: "
                    f"{cause} → {effect}"
                )

                continue

            # Prevent self-causal relationship
            if cause.lower() == effect.lower():

                print(
                    f"⚠️ Skipping self-causal pair: "
                    f"{cause} → {effect}"
                )

                continue

            pairs.append(
                (cause, concept)
            )

        # Remove duplicate causes
        unique_pairs = []
        seen_causes = set()

        for cause, effect in pairs:

            key = cause.lower()

            if key not in seen_causes:

                seen_causes.add(key)

                unique_pairs.append(
                    (cause, effect)
                )

        print(
            f"🧠 Valid causal pairs: "
            f"{len(unique_pairs)}"
        )

        for cause, effect in unique_pairs:

            print(
                f"   ✓ {cause} → {effect}"
            )

        return unique_pairs