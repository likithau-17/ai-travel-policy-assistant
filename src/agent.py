from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from tools import check_employee_eligibility


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


def route_decision(state: AgentState):
    if state["decision"] == "employee_tool":
        return "employee_tool"

    return END


graph_builder = StateGraph(AgentState)

graph_builder.add_node("decide", decide_action)
graph_builder.add_node("employee_tool", run_employee_tool)

graph_builder.add_edge(START, "decide")

graph_builder.add_conditional_edges(
    "decide",
    route_decision,
)

graph_builder.add_edge("employee_tool", END)

agent = graph_builder.compile()


if __name__ == "__main__":
    questions = [
        "Is EMP001 eligible?",
        "Is EMP004 eligible?",
        "Is EMP999 eligible?",
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