from agents.hallucination_detector_rl import HallucinationDetector


# ============================================================
# Hallucination Detector Test Suite
# ============================================================

detector = HallucinationDetector()


# ============================================================
# Test Cases
# ============================================================

test_cases = [

    # --------------------------------------------------------
    # TEST 1: Exact supported claim
    # --------------------------------------------------------
    {
        "name": "Exact supported claim",
        "explanation": """
        Transportation problems result in missed appointments.
        """,
        "answer": """
        Transportation problems can cause patients to miss appointments.
        """,
        "expected": False
    },

    # --------------------------------------------------------
    # TEST 2: Paraphrased supported claim
    # --------------------------------------------------------
    {
        "name": "Paraphrased supported claim",
        "explanation": """
        Poor communication between patients and healthcare providers
        contributes to missed appointments.
        """,
        "answer": """
        Patients may miss appointments because communication with
        healthcare providers is inadequate.
        """,
        "expected": False
    },

    # --------------------------------------------------------
    # TEST 3: Synonym replacement
    # --------------------------------------------------------
    {
        "name": "Synonym replacement",
        "explanation": """
        Transportation issues can make it difficult for patients
        to reach appointments on time.
        """,
        "answer": """
        Transportation problems may prevent patients from arriving
        at their appointments on time.
        """,
        "expected": False
    },

    # --------------------------------------------------------
    # TEST 4: Completely unsupported claim
    # --------------------------------------------------------
    {
        "name": "Completely unsupported claim",
        "explanation": """
        Transportation problems result in missed appointments.
        """,
        "answer": """
        Patients miss appointments because hospitals charge very
        high fees.
        """,
        "expected": True
    },

    # --------------------------------------------------------
    # TEST 5: One supported + one unsupported claim
    # --------------------------------------------------------
    {
        "name": "Supported + unsupported claim",
        "explanation": """
        Transportation problems contribute to missed appointments.
        """,
        "answer": """
        Transportation problems contribute to missed appointments,
        and hospitals charge high fees for every appointment.
        """,
        "expected": True
    },

    # --------------------------------------------------------
    # TEST 6: New cause added
    # --------------------------------------------------------
    {
        "name": "New unsupported cause",
        "explanation": """
        Poor communication between patients and healthcare providers
        contributes to missed appointments.
        """,
        "answer": """
        Poor communication and expensive hospital fees cause patients
        to miss appointments.
        """,
        "expected": True
    },

    # --------------------------------------------------------
    # TEST 7: New effect added
    # --------------------------------------------------------
    {
        "name": "New unsupported effect",
        "explanation": """
        Transportation problems can cause patients to miss appointments.
        """,
        "answer": """
        Transportation problems cause patients to miss appointments
        and permanently worsen their health.
        """,
        "expected": True
    },

    # --------------------------------------------------------
    # TEST 8: Extra unsupported detail
    # --------------------------------------------------------
    {
        "name": "Extra unsupported detail",
        "explanation": """
        Personal emergencies can cause patients to miss appointments.
        """,
        "answer": """
        Family emergencies can cause patients to miss appointments,
        especially when they occur shortly before the scheduled visit.
        """,
        "expected": True
    },

    # --------------------------------------------------------
    # TEST 9: Same causal relationship, different wording
    # --------------------------------------------------------
    {
        "name": "Equivalent causal wording",
        "explanation": """
        Inadequate transportation contributes to missed appointments.
        """,
        "answer": """
        Limited transportation access is one factor that may lead
        patients to miss scheduled visits.
        """,
        "expected": False
    },

    # --------------------------------------------------------
    # TEST 10: Completely unrelated answer
    # --------------------------------------------------------
    {
        "name": "Unrelated answer",
        "explanation": """
        Patients may miss appointments because of transportation
        difficulties and communication problems.
        """,
        "answer": """
        Heart disease is primarily caused by high cholesterol and
        hypertension.
        """,
        "expected": True
    },

]


# ============================================================
# Run Tests
# ============================================================

print("\n")
print("=" * 70)
print("HALLUCINATION DETECTOR TEST SUITE")
print("=" * 70)


passed = 0
failed = 0
results = []

for i, test in enumerate(test_cases, start=1):

    print("\n" + "-" * 70)
    print(f"TEST {i}: {test['name']}")
    print("-" * 70)

    print("\nSTRUCTURED EXPLANATION:")
    print(test["explanation"].strip())

    print("\nFINAL ANSWER:")
    print(test["answer"].strip())

    print("\nExpected:")
    print(
        "HALLUCINATED"
        if test["expected"]
        else "SUPPORTED"
    )

    try:

        result = detector.detect(
            test["answer"].strip(),
            test["explanation"].strip()
        )

        results.append({
            "test": i,
            "name": test["name"],
            "expected": test["expected"],
            "actual": result,
            "passed": result == test["expected"]
        })

        print("\nActual:")
        print(
            "HALLUCINATED"
            if result
            else "SUPPORTED"
        )

        if result == test["expected"]:

            print("STATUS: ✅ PASS")
            passed += 1

        else:

            print("STATUS: ❌ FAIL")
            failed += 1

    except Exception as e:

        print("\nSTATUS: ❌ ERROR")
        print("Error:", e)

        failed += 1


# ============================================================
# Final Summary
# ============================================================

total = len(test_cases)

accuracy = (
    passed / total * 100
    if total > 0
    else 0
)


print("\n")
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)

print(f"Total Tests : {total}")
print(f"Passed      : {passed}")
print(f"Failed      : {failed}")
print(f"Accuracy    : {accuracy:.2f}%")

print("\nFAILED TEST DETAILS")
print("=" * 70)

for item in results:
    if not item["passed"]:
        print(f"\nTEST {item['test']}: {item['name']}")
        print(
            f"Expected: "
            f"{'HALLUCINATED' if item['expected'] else 'SUPPORTED'}"
        )
        print(
            f"Actual:   "
            f"{'HALLUCINATED' if item['actual'] else 'SUPPORTED'}"
        )


print("=" * 70)