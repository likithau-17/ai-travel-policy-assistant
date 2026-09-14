from unittest.mock import patch

from app.app import app

from unittest.mock import patch

from app.app import app


def test_ask_empty_question():
    client = app.test_client()

    response = client.post(
        "/ask",
        json={"question": ""}
    )

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json()["error"] == "Question is required."


def test_ask_agent_failure_returns_json_error():
    client = app.test_client()

    with patch(
        "app.app.agent.invoke",
        side_effect=Exception("Simulated agent failure")
    ):
        response = client.post(
            "/ask",
            json={"question": "What is the travel limit in India?"}
        )

    assert response.status_code == 500
    assert response.is_json

    data = response.get_json()

    assert data["error"] == (
        "The assistant could not process your request."
    )
    assert data["details"] == "Simulated agent failure"


def test_health_endpoint():
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.is_json
    assert response.get_json()["status"] == "ok"


def test_ask_employee_tool_failure_returns_json_error():
    client = app.test_client()

    with patch(
        "app.app.agent.invoke",
        side_effect=Exception("Simulated employee tool failure")
    ):
        response = client.post(
            "/ask",
            json={"question": "Is EMP001 eligible?"}
        )

    assert response.status_code == 500
    assert response.is_json

    data = response.get_json()

    assert data["error"] == (
        "The assistant could not process your request."
    )
    assert data["details"] == "Simulated employee tool failure"


def test_ask_trip_validation_failure_returns_json_error():
    client = app.test_client()

    with patch(
        "app.app.agent.invoke",
        side_effect=Exception("Simulated trip validation failure")
    ):
        response = client.post(
            "/ask",
            json={"question": "Can EMP001 take a 1500 India business trip?"}
        )

    assert response.status_code == 500
    assert response.is_json

    data = response.get_json()

    assert data["error"] == (
        "The assistant could not process your request."
    )
    assert data["details"] == "Simulated trip validation failure"


def test_ask_reimbursement_failure_returns_json_error():
    client = app.test_client()

    with patch(
        "app.app.agent.invoke",
        side_effect=Exception("Simulated reimbursement failure")
    ):
        response = client.post(
            "/ask",
            json={"question": "How much can I reimburse for a 1500 India trip?"}
        )

    assert response.status_code == 500
    assert response.is_json

    data = response.get_json()

    assert data["error"] == (
        "The assistant could not process your request."
    )
    assert data["details"] == "Simulated reimbursement failure"


def test_ask_rag_failure_returns_json_error():
    client = app.test_client()

    with patch(
        "app.app.agent.invoke",
        side_effect=Exception("Simulated RAG failure")
    ):
        response = client.post(
            "/ask",
            json={"question": "What is the standard travel limit in India?"}
        )

    assert response.status_code == 500
    assert response.is_json

    data = response.get_json()

    assert data["error"] == (
        "The assistant could not process your request."
    )
    assert data["details"] == "Simulated RAG failure"
