from typing import TypedDict
import re

from langgraph.graph import END, START, StateGraph

from src.tools import (
    calculate_reimbursement,
    check_employee_eligibility,
    validate_trip,
)
from src.rag import answer_question, load_vector_store
from src.embeddings import load_embedding_model

class AgentState(TypedDict):
    question: str
    decision: str
    result: dict


def decide_action(state: AgentState):
    question = state["question"].lower()

    if "eligible" in question or "eligibility" in question:
        decision = "employee_tool"
    elif "reimburse" in question or "how much" in question:
        decision = "reimbursement_tool"
    elif "validate" in question:
        decision = "trip_validation"
    elif "can i" in question or "allowed" in question:
        decision = "policy_rag"
    else:
        decision = "policy_rag"

    return {
        "decision": decision
    }


def run_employee_tool(state: AgentState):
    question = state["question"]

    words = question.split()

    employee_id = None

    for word in words:
        if word.upper().startswith("EMP"):
            employee_id = word.strip("?.!,").upper()
            break

    if employee_id is None:
        return {
            "result": {
                "found": False,
                "message": "No employee ID was found in the question.",
            }
        }

    result = check_employee_eligibility(employee_id)

    return {
        "result": result
    }

def run_reimbursement_tool(state: AgentState):
    question = state["question"].lower()

    country = None

    if "india" in question:
        country = "India"
    elif "us" in question or "usa" in question:
        country = "US"

    if country is None:
        return {
            "result": {
                "valid": False,
                "status": "Invalid",
                "message": "No supported country was found in the question.",
            }
        }

    amount = None

    for word in question.replace(",", "").split():
        cleaned = word.strip("₹$?.!")
        try:
            amount = float(cleaned)
            break
        except ValueError:
            continue

    if amount is None:
        return {
            "result": {
                "valid": False,
                "status": "Invalid",
                "message": "No trip amount was found in the question.",
            }
        }

    result = calculate_reimbursement(country, amount)

    return {
        "result": result
    }

def run_trip_validation(state):
    question = state["question"]

    words = question.split()

    employee_id = None

    match = re.search(r"\bEMP\d+\b", question.upper())

    if match:
        employee_id = match.group(0)

    if employee_id is None:
        return {
            "result": {
                "valid": False,
                "status": "Invalid",
                "reason": "No employee ID was found in the question.",
            }
        }

    question_lower = question.lower()

    if "india" in question_lower:
        country = "India"
    elif "us" in question_lower or "usa" in question_lower:
        country = "US"
    else:
        return {
            "result": {
                "valid": False,
                "status": "Invalid",
                "reason": "No supported country was found in the question.",
            }
        }

    amount = None
    for word in question_lower.replace(",", "").split():
        cleaned = word.strip("₹$?.!")
        try:
            amount = float(cleaned)
            break
        except ValueError:
            continue

    if amount is None:
        return {
            "result": {
                "valid": False,
                "status": "Invalid",
                "reason": "No trip amount was found in the question.",
            }
        }

    business_purpose = (
        "business" in question_lower
        or "work" in question_lower
        or "client" in question_lower
        or "meeting" in question_lower
    )

    result = validate_trip(
        employee_id=employee_id,
        country=country,
        trip_amount=amount,
        business_purpose=business_purpose,
    )

    return {"result": result}

def run_policy_rag(state: AgentState):
    model = load_embedding_model()
    index, chunks = load_vector_store()

    result = answer_question(
        question=state["question"],
        model=model,
        index=index,
        chunks=chunks,
    )

    return {
        "result": result
    }

def route_decision(state):
    if state["decision"] == "employee_tool":
        return "employee_tool"
    if state["decision"] == "reimbursement_tool":
        return "reimbursement_tool"
    if state["decision"] == "trip_validation":
        return "trip_validation"
    if state["decision"] == "policy_rag":
        return "policy_rag"
    return END


graph_builder = StateGraph(AgentState)


# -------------------------
# Nodes
# -------------------------

graph_builder.add_node("decide", decide_action)

graph_builder.add_node("employee_tool", run_employee_tool)
graph_builder.add_node("reimbursement_tool", run_reimbursement_tool)
graph_builder.add_node("trip_validation", run_trip_validation)
graph_builder.add_node("policy_rag", run_policy_rag)

# -------------------------
# Graph flow
# -------------------------

graph_builder.add_edge(START, "decide")

graph_builder.add_conditional_edges(
    "decide",
    route_decision,
)


# -------------------------
# End points
# -------------------------

graph_builder.add_edge("employee_tool", END)
graph_builder.add_edge("reimbursement_tool", END)
graph_builder.add_edge("trip_validation", END)
graph_builder.add_edge("policy_rag", END)

# -------------------------
# Compile graph
# -------------------------

agent = graph_builder.compile()


if __name__ == "__main__":
    questions = [
        "Is EMP001 eligible?",
        "How much can I reimburse for a 1500 India trip?",
        "Validate EMP001's 1500 India business trip.",
        "What is the standard ride limit in India?",
    ]

    for question in questions:
        result = agent.invoke({
            "question": question,
            "decision": "",
            "result": {},
        })

        print(f"\nQuestion: {question}")
        print(f"Decision: {result['decision']}")
        print(f"Result: {result['result']}")