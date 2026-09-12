from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    question: str
    decision: str


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


graph_builder = StateGraph(AgentState)

graph_builder.add_node("decide", decide_action)

graph_builder.add_edge(START, "decide")
graph_builder.add_edge("decide", END)

agent = graph_builder.compile()


if __name__ == "__main__":
    questions = [
        "Is EMP001 eligible?",
        "How much can I reimburse?",
        "Are airport trips allowed?",
        "What is the travel policy?",
    ]

    for question in questions:
        result = agent.invoke({
            "question": question,
            "decision": "",
        })

        print(f"\nQuestion: {question}")
        print(f"Decision: {result['decision']}")