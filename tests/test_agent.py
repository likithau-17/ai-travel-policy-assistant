from src.agent import agent


def run_test(question):
    result = agent.invoke(
        {
            "question": question,
            "decision": "",
            "result": {},
            "employee_result": {},
            "messages": [],
        },
        config={
            "configurable": {
                "thread_id": "test-thread"
            }
        },
    )

    print(f"\nQuestion: {question}")
    print(f"Decision: {result['decision']}")
    print(f"Result: {result['result']}")

    return result


def test_employee_eligible():
    result = run_test("Is EMP001 eligible?")
    assert result["result"]["eligibility_status"] == "Eligible"


def test_employee_not_eligible():
    result = run_test("Is EMP004 eligible?")
    assert result["result"]["eligibility_status"] == "Not Eligible"


def test_employee_approval_required():
    result = run_test("Is EMP002 eligible?")
    assert result["result"]["eligibility_status"] == "Approval Required"


def test_reimbursement_within_limit():
    result = run_test("How much can I reimburse for a 1500 India trip?")
    assert result["result"]["reimbursable_amount"] == 1500


def test_reimbursement_above_limit():
    result = run_test("How much can I reimburse for a 2500 India trip?")
    assert result["result"]["reimbursable_amount"] == 2000
    assert result["result"]["excess_amount"] == 500


def test_valid_trip():
    result = run_test("Can EMP001 take a 1500 India business trip?")
    assert result["result"]["status"] == "Within Policy"


def test_trip_needs_approval():
    result = run_test("Can EMP001 take a 2500 India business trip?")
    assert result["result"]["status"] == "Needs Approval"


def test_not_eligible_trip():
    result = run_test("Can EMP004 take a 50 US business trip?")
    assert result["result"]["status"] == "Not Eligible"


def test_policy_rag():
    result = run_test("What is the standard ride limit in India?")
    assert "2,000" in result["result"]["answer"]
    assert len(result["result"]["sources"]) > 0


def test_unknown_employee():
    result = run_test("Can EMP999 take a 1000 India business trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["reason"] == "Employee ID not found."


def test_cancellation_fee_not_invented():
    result = run_test("What is the cancellation fee for a cancelled ride?")
    answer = result["result"]["answer"].lower()

    assert "insufficient" in answer
    assert "cancellation" in answer
    assert "fee" in answer
    assert len(result["result"]["sources"]) > 0


def test_unsupported_reimbursement_country():
    result = run_test("How much can I reimburse for a 1000 Canada trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["message"] == "No supported country was found in the question."


def test_trip_without_employee_id():
    result = run_test("Can I take a 1000 India business trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["reason"] == "Employee ID not found."


def test_trip_country_mismatch():
    result = run_test("Can EMP001 take a 1000 Canada business trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["reason"] == "Employee country does not match the trip country."


def test_trip_without_amount():
    result = run_test("Can EMP001 take an India business trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["reason"] == "No trip amount was found in the question."