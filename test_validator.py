from agents.causal_evidence_validator import (
    CausalEvidenceValidator
)

from config import CONFIG


validator = CausalEvidenceValidator(
    CONFIG["NEO4J_URI"],
    CONFIG["NEO4J_USER"],
    CONFIG["NEO4J_PASSWORD"]
)


# ============================================================
# Test 1
# ============================================================

answer = """
Smoking causes heart disease.
Poor diet causes heart disease.
Diabetes causes heart disease.
"""

result = validator.validate(
    answer,
    "heart disease"
)

print("\nTEST 1")
print("Passed:", result.validation_passed)
print("Supported:", result.supported_claims)
print("Unsupported:", result.unsupported_claims)


# ============================================================
# Test 2
# ============================================================

answer = """
Smoking causes diabetes.
"""

result = validator.validate(
    answer,
    "heart disease"
)

print("\nTEST 2")
print("Passed:", result.validation_passed)
print("Supported:", result.supported_claims)
print("Unsupported:", result.unsupported_claims)


# ============================================================
# Test 3
# ============================================================

answer = """
Family history of disease causes heart disease.
"""

result = validator.validate(
    answer,
    "heart disease"
)

print("\nTEST 3")
print("Passed:", result.validation_passed)
print("Supported:", result.supported_claims)
print("Unsupported:", result.unsupported_claims)


validator.close()