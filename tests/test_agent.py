from src.agent import agent
from unittest.mock import patch
import uuid


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
                "thread_id": str(uuid.uuid4())
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


@patch(
    "src.gemini.generate_answer",
    return_value="Based on the policy context, the standard individual ride limit in India is INR 2,000.",
)
def test_policy_rag(mock_generate_answer):
    result = run_test("What is the standard ride limit in India?")

    assert "2,000" in result["result"]["answer"]
    assert len(result["result"]["sources"]) > 0

    mock_generate_answer.assert_called_once()


def test_unknown_employee():
    result = run_test("Can EMP999 take a 1000 India business trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["reason"] == "Employee ID not found."


@patch(
    "src.gemini.generate_answer",
    return_value="The policy information is insufficient to provide the specific cancellation fee.",
)
def test_cancellation_fee_not_invented(mock_generate_answer):
    result = run_test("What is the cancellation fee for a cancelled ride?")

    answer = result["result"]["answer"].lower()

    assert "insufficient" in answer
    assert "cancellation" in answer
    assert "fee" in answer
    assert len(result["result"]["sources"]) > 0

    mock_generate_answer.assert_called_once()


def test_unsupported_reimbursement_country():
    result = run_test("How much can I reimburse for a 1000 Canada trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["message"] == "No supported country was found in the question."


def test_trip_without_employee_id():
    result = run_test("Can I take a 1000 India business trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["reason"] == "No employee ID was found in the question."


def test_trip_country_mismatch():
    result = run_test("Can EMP001 take a 1000 Canada business trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["reason"] == "Employee country does not match the trip country."


def test_trip_without_amount():
    result = run_test("Can EMP001 take an India business trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["reason"] == "No trip amount was found in the question."


def test_us_reimbursement_within_limit():
    result = run_test("How much can I reimburse for a 50 US trip?")
    assert result["result"]["status"] == "Within Policy"
    assert result["result"]["reimbursable_amount"] == 50


def test_us_reimbursement_above_limit():
    result = run_test("How much can I reimburse for a 100 US trip?")
    assert result["result"]["status"] == "Needs Approval"
    assert result["result"]["reimbursable_amount"] == 75
    assert result["result"]["excess_amount"] == 25


def test_employee_approval_required_trip():
    result = run_test("Can EMP002 take a 1500 India business trip?")
    assert result["result"]["status"] == "Needs Approval"
    assert "requires approval" in result["result"]["reason"].lower()


def test_not_eligible_us_employee():
    result = run_test("Can EMP004 take a 50 US business trip?")
    assert result["result"]["status"] == "Not Eligible"


def test_unknown_reimbursement_country():
    result = run_test("How much can I reimburse for a 1000 Canada trip?")
    assert result["result"]["status"] == "Invalid"
    assert result["result"]["message"] == "No supported country was found in the question."


@patch(
    "src.gemini.generate_answer",
    return_value="The policy information is insufficient to provide the specific cancellation fee.",
)
def test_hallucination_unknown_cancellation_fee(mock_generate_answer):
    result = run_test("What exact cancellation fee will I be charged?")
    answer = result["result"]["answer"].lower()

    assert "insufficient" in answer
    assert "fee" in answer

    mock_generate_answer.assert_called_once()


@patch(
    "src.gemini.generate_answer",
    return_value="The policy information is insufficient to provide a standard ride limit for Canada.",
)
def test_hallucination_unknown_country_limit(mock_generate_answer):
    result = run_test("What is the standard ride limit in Canada?")
    answer = result["result"]["answer"].lower()

    assert "insufficient" in answer

    mock_generate_answer.assert_called_once()


@patch(
    "src.gemini.generate_answer",
    return_value="The policy information is insufficient to answer this question.",
)
def test_hallucination_unlisted_policy_rule(mock_generate_answer):
    result = run_test("What is the weekend travel allowance for employees?")
    answer = result["result"]["answer"].lower()

    assert "insufficient" in answer

    mock_generate_answer.assert_called_once()