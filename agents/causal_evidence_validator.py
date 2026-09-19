from neo4j import GraphDatabase
from pydantic import BaseModel
import re


# ============================================================
# Validator State
# ============================================================

class ValidatorState(BaseModel):
    answer: str
    target_concept: str
    validation_passed: bool = False
    unsupported_claims: list[str] = []
    supported_claims: list[str] = []


# ============================================================
# Deterministic Neo4j Causal Validator
# ============================================================

class CausalEvidenceValidator:

    def __init__(self, uri, username, password):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )

    # --------------------------------------------------------
    # Close Neo4j
    # --------------------------------------------------------

    def close(self):

        self.driver.close()

    # --------------------------------------------------------
    # Get direct causes of target
    # --------------------------------------------------------

    def get_direct_causes(self, target):

        query = """
        MATCH (c:Concept)-[:CAUSES]->(e:Concept)
        WHERE toLower(e.name) = toLower($target)
        RETURN c.name AS cause
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                target=target
            )

            return [
                record["cause"]
                for record in result
            ]

    # --------------------------------------------------------
    # Check direct causal relationship
    # --------------------------------------------------------

    def check_direct_cause(
        self,
        cause,
        effect
    ):

        query = """
        MATCH (c:Concept)-[:CAUSES]->(e:Concept)
        WHERE toLower(c.name) = toLower($cause)
          AND toLower(e.name) = toLower($effect)
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
    # Check exact causal path
    # --------------------------------------------------------

    def check_causal_path(self, nodes):

        if len(nodes) < 2:
            return False

        query = """
        MATCH p = (start:Concept)-[:CAUSES*]->(end:Concept)
        WHERE toLower(start.name) = toLower($start)
          AND toLower(end.name) = toLower($end)

        WITH p,
             [n IN nodes(p) | toLower(n.name)] AS names

        WHERE names = $nodes

        RETURN count(*) AS count
        """

        normalized_nodes = [
            node.strip().lower()
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

    # --------------------------------------------------------
    # Extract causal statements
    # --------------------------------------------------------

    def extract_causal_claims(self, answer):

        claims = []

        sentences = re.split(
            r'(?<=[.!?])\s+',
            answer.strip()
        )

        causal_patterns = [

            r'(.+?)\s+causes\s+(.+?)[.]?$',

            r'(.+?)\s+leads to\s+(.+?)[.]?$',

            r'(.+?)\s+contributes to\s+(.+?)[.]?$',

            r'(.+?)\s+results in\s+(.+?)[.]?$',

            r'(.+?)\s+is a cause of\s+(.+?)[.]?$'
        ]

        for sentence in sentences:

            sentence = sentence.strip()

            for pattern in causal_patterns:

                match = re.match(
                    pattern,
                    sentence,
                    flags=re.IGNORECASE
                )

                if match:

                    cause = match.group(1).strip()
                    effect = match.group(2).strip()

                    claims.append({
                        "text": sentence,
                        "cause": cause,
                        "effect": effect
                    })

                    break

        return claims

    # --------------------------------------------------------
    # Validate answer
    # --------------------------------------------------------

    def validate(
        self,
        answer,
        target_concept
    ):

        claims = self.extract_causal_claims(answer)

        supported = []
        unsupported = []

        for claim in claims:

            cause = claim["cause"]
            effect = claim["effect"]

            # Remove common punctuation
            effect = effect.rstrip(".,!?")

            supported_edge = self.check_direct_cause(
                cause,
                effect
            )

            if supported_edge:

                supported.append(
                    claim["text"]
                )

            else:

                unsupported.append(
                    claim["text"]
                )

        passed = len(unsupported) == 0

        return ValidatorState(
            answer=answer,
            target_concept=target_concept,
            validation_passed=passed,
            unsupported_claims=unsupported,
            supported_claims=supported
        )