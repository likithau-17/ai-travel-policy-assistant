from typing import TypedDict, Annotated
import re

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.memory import create_memory

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
    employee_result: dict
    employee_id: str
    messages: Annotated[list, add_messages]


# ============================================================
# Helper Functions
# ============================================================

def get_message_content(message):
    """
    Safely extract message content from either:

    1. LangChain message objects
    2. Dictionary-style messages

    This is important because LangGraph memory may contain
    different message representations depending on where the
    message originated.
    """
    if isinstance(message, dict):
        return str(
            message.get("content", "")
        )

    return str(
        getattr(
            message,
            "content",
            ""
        )
    )


def get_last_employee_id(messages):
    """
    Find the most recent employee ID from conversation memory.

    Example:
        EMP001
    """
    for message in reversed(messages):
        content = get_message_content(
            message
        )

        match = re.search(
            r"\bEMP\d+\b",
            content.upper()
        )

        if match:
            return match.group(0)

    return None


def get_last_country(messages):
    """
    Find the most recent supported country from conversation memory.

    Supported countries:
        India
        US
    """
    for message in reversed(messages):
        text = get_message_content(
            message
        ).lower()

        if re.search(
            r"\bindia\b",
            text
        ):
            return "India"

        if (
            re.search(
                r"\bunited states\b",
                text
            )
            or re.search(
                r"\busa\b",
                text
            )
            or re.search(
                r"\bus\b",
                text
            )
        ):
            return "US"

    return None


def extract_employee_id(question):
    """
    Extract an employee ID from a question.

    Example:
        "Can EMP001 take a trip?"
        -> EMP001
    """
    match = re.search(
        r"\bEMP\d+\b",
        question.upper()
    )

    if match:
        return match.group(0)

    return None


def extract_country(question):
    """
    Extract a supported country from the question.

    Returns:
        India
        US
        None
    """
    question_lower = question.lower()

    if re.search(
        r"\bindia\b",
        question_lower
    ):
        return "India"

    if (
        re.search(
            r"\bunited states\b",
            question_lower
        )
        or re.search(
            r"\busa\b",
            question_lower
        )
        or re.search(
            r"\bus\b",
            question_lower
        )
    ):
        return "US"

    return None


def extract_explicit_country(question):
    """
    Extract an explicitly mentioned country.

    Unlike extract_country(), this function also preserves
    unsupported countries such as Canada.

    This is necessary for trip validation because an employee
    country mismatch should be detected by validate_trip().

    Example:
        EMP001 + Canada
        -> Canada
    """
    question_lower = question.lower()

    countries = {
        "india": "India",
        "united states": "US",
        "usa": "US",
        "us": "US",
        "canada": "Canada",
    }

    for name, country in countries.items():
        if re.search(
            rf"\b{re.escape(name)}\b",
            question_lower
        ):
            return country

    return None


def extract_amount(question):
    """
    Extract the first numeric amount from a question.

    Supports:
        1500
        2,500
        $75
        ₹2000
    """
    text = question.replace(
        ",",
        ""
    )

    # Currency-prefixed amount.
    currency_match = re.search(
        r"(?:₹|\$)\s*(\d+(?:\.\d+)?)",
        text
    )

    if currency_match:
        return float(
            currency_match.group(1)
        )

    # Standalone number.
    number_match = re.search(
        r"\b\d+(?:\.\d+)?\b",
        text
    )

    if number_match:
        return float(
            number_match.group(0)
        )

    return None


def split_questions(question):
    """
    Split multiple questions at question marks.

    This helper is retained for the multi-question enhancement.

    Example:
        "What is the limit? Is airport travel allowed?"

    Returns:
        [
            "What is the limit?",
            "Is airport travel allowed?"
        ]
    """
    question = question.strip()

    if not question:
        return []

    parts = re.split(
        r"\?+",
        question
    )

    questions = []

    for part in parts:
        cleaned = part.strip()

        if cleaned:
            questions.append(
                cleaned + "?"
            )

    return questions


# ============================================================
# Intent Detection
# ============================================================

def is_policy_question(question):
    """
    Detect informational corporate travel-policy questions.

    These questions should be answered using RAG.

    Examples:
        What is the standard travel limit?
        Are airport trips allowed?
        Are late-night trips allowed?
        What information is required for expenses?
        What happens if I exceed the limit?
        What is the cancellation fee?
    """
    question_lower = question.lower()

    policy_keywords = [
        "policy",
        "limit",
        "standard limit",
        "travel limit",
        "ride limit",
        "business ride",
        "airport",
        "airport trip",
        "airport trips",
        "expense",
        "expenses",
        "reimbursable",
        "reimbursement policy",
        "approval",
        "documentation",
        "required information",
        "required fields",
        "late night",
        "late-night",
        "allowed",
        "what happens if",
        "exceed",
        "maximum",
        "cancellation",
        "cancel",
        "fee",
        "fees",
    ]

    return any(
        keyword in question_lower
        for keyword in policy_keywords
    )


def is_trip_action_question(question):
    """
    Detect questions that actually require trip validation.

    A trip/travel keyword by itself is NOT enough.

    Examples that should validate:
        Can EMP001 take a 1500 India business trip?
        Can I take a 2500 India trip?
        Validate EMP001's trip.
        Can EMP001 travel to the US for 100 dollars?

    Examples that should NOT validate:
        Are airport trips allowed?
        What is the travel limit?
        Are late-night airport trips allowed?
    """
    question_lower = question.lower()

    # Explicit validation language.
    if "validate" in question_lower:
        return True

    # Questions involving a specific amount are strong
    # indicators that an actual trip is being evaluated.
    amount = extract_amount(
        question
    )

    if (
        amount is not None
        and (
            "trip" in question_lower
            or "travel" in question_lower
            or "ride" in question_lower
        )
    ):
        return True

    # Direct action questions.
    action_phrases = [
        "can i take",
        "can i travel",
        "can we take",
        "can we travel",
        "can emp",
        "am i allowed to take",
        "am i allowed to travel",
        "is my trip",
        "is this trip",
    ]

    if any(
        phrase in question_lower
        for phrase in action_phrases
    ):
        return True

    return False


def is_out_of_domain_question(question):
    """
    Detect obvious questions unrelated to corporate travel
    or employee operations.

    Important:
    Questions containing travel-policy terminology should NOT
    be classified as out-of-domain merely because the specific
    policy detail is unknown.

    Example:
        "What exact cancellation fee will I be charged?"
    remains a RAG question so the assistant can safely say
    that the policy does not specify the fee.
    """
    question_lower = question.lower()

    travel_keywords = [
        "travel",
        "trip",
        "employee",
        "business",
        "ride",
        "airport",
        "expense",
        "reimburse",
        "reimbursement",
        "policy",
        "approval",
        "eligible",
        "eligibility",
        "manager",
        "company",
        "work",
        "client",
        "meeting",
        "country",
        "limit",
        "fare",
        "cost",
        "cancellation",
        "cancel",
        "fee",
    ]

    return not any(
        keyword in question_lower
        for keyword in travel_keywords
    )


# ============================================================
# Decision / Routing
# ============================================================

def decide_action(state):
    """
    Decide which capability should handle the question.

    Priority:
        1. Employee eligibility
        2. Reimbursement
        3. Follow-up trip cost change
        4. Explicit validation
        5. Specific trip action
        6. Informational policy question
        7. Out-of-domain
        8. Safe RAG fallback
    """
    question = state["question"].strip()
    question_lower = question.lower()

    messages = state.get(
        "messages",
        []
    )

    current_employee_id = extract_employee_id(
        question
    )

    previous_employee_id = get_last_employee_id(
        messages
    )

    employee_id = (
        current_employee_id
        or previous_employee_id
        or ""
    )

    # Build reliable conversation text.
    memory_text = " ".join(
        get_message_content(
            message
        ).lower()
        for message in messages
    )

    has_previous_trip_context = (
        previous_employee_id is not None
        and (
            "trip" in memory_text
            or "travel" in memory_text
        )
    )

        # --------------------------------------------------------
    # 1. Employee eligibility
    #
    # Eligibility questions should stop at the employee tool.
    #
    # Examples:
    #   Is EMP001 eligible?
    #   Can EMP001 travel for business?
    #   Can EMP003 travel for business?
    #   Check whether EMP001 is eligible for business travel.
    #
    # Do NOT route specific trip requests here:
    #   Can EMP001 take a 1500 India business trip?
    # --------------------------------------------------------

    is_eligibility_question = (
        "eligible" in question_lower
        or "eligibility" in question_lower
        or (
            current_employee_id is not None
            and "travel for business" in question_lower
        )
        or (
            current_employee_id is not None
            and "business travel" in question_lower
            and "trip" not in question_lower
        )
    )

    if is_eligibility_question:
        return {
            "decision": "employee_tool",
            "employee_id": employee_id,
        }

    # --------------------------------------------------------
    # 2. Reimbursement
    # --------------------------------------------------------

    if (
        "reimburse" in question_lower
        or "reimbursement" in question_lower
    ):
        return {
            "decision": "reimbursement_tool",
            "employee_id": employee_id,
        }

    # --------------------------------------------------------
    # 3. Memory-based trip cost follow-up
    #
    # Example:
    #
    #   Can EMP001 take a 1500 India business trip?
    #
    #   What if it costs 2500?
    #
    # The second question should reuse the employee and
    # country context from memory.
    # --------------------------------------------------------

    if (
        has_previous_trip_context
        and (
            "what if" in question_lower
            or "costs" in question_lower
            or "cost" in question_lower
        )
    ):
        return {
            "decision": "trip_validation",
            "employee_id": employee_id,
        }

    # --------------------------------------------------------
    # 4. Explicit validation
    # --------------------------------------------------------

    if "validate" in question_lower:
        return {
            "decision": "trip_validation",
            "employee_id": employee_id,
        }

        # --------------------------------------------------------
    # 5. Specific trip action
    #
    # A direct request to take/validate a specific trip should
    # go to trip validation even when the amount is missing.
    #
    # This allows trip_validation() to return a clear error:
    #   "No trip amount was found in the question."
    #
    # Examples:
    #   Can EMP001 take a 50 US airport business trip?
    #   Can EMP001 take a 100 US airport business trip?
    #   Can EMP001 take an India business trip?
    #   Can I take a 2500 India trip?
    # --------------------------------------------------------

    amount = extract_amount(
        question
    )

    has_trip_or_travel_keyword = (
        "trip" in question_lower
        or "travel" in question_lower
        or "ride" in question_lower
    )

    has_direct_action = (
        current_employee_id is not None
        or "can i" in question_lower
        or "can we" in question_lower
        or "can emp" in question_lower
        or "am i allowed to take" in question_lower
        or "am i allowed to travel" in question_lower
        or "is my trip" in question_lower
        or "is this trip" in question_lower
    )

    if (
        has_trip_or_travel_keyword
        and has_direct_action
    ):
        return {
            "decision": (
                "employee_then_trip"
                if current_employee_id is not None
                else "trip_validation"
            ),
            "employee_id": employee_id,
        }

    # --------------------------------------------------------
    # 6. Informational policy question
    # --------------------------------------------------------

    if is_policy_question(
        question
    ):
        return {
            "decision": "policy_rag",
            "employee_id": employee_id,
        }

    # --------------------------------------------------------
    # 7. Out-of-domain
    # --------------------------------------------------------

    if is_out_of_domain_question(
        question
    ):
        return {
            "decision": "out_of_domain",
            "employee_id": employee_id,
        }

    # --------------------------------------------------------
    # 8. Safe default
    # --------------------------------------------------------

    return {
        "decision": "policy_rag",
        "employee_id": employee_id,
    }


# ============================================================
# Employee Tool
# ============================================================

def run_employee_tool(state: AgentState):
    question = state["question"]

    employee_id = extract_employee_id(
        question
    )

    if employee_id is None:
        employee_id = state.get(
            "employee_id",
            ""
        )

    if not employee_id:
        employee_id = get_last_employee_id(
            state.get(
                "messages",
                []
            )
        )

    if not employee_id:
        result = {
            "found": False,
            "message": (
                "No employee ID was found "
                "in the question."
            ),
        }

        return {
            "result": result,
            "employee_result": result,
        }

    result = check_employee_eligibility(
        employee_id
    )

    return {
        "employee_result": result,
        "result": result,
        "employee_id": employee_id,
        "messages": [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": str(result),
            },
        ],
    }


# ============================================================
# Reimbursement Tool
# ============================================================

def run_reimbursement_tool(state: AgentState):
    question = state["question"]

    # Reimbursement only supports countries explicitly
    # understood by the current policy.
    country = extract_country(
        question
    )

    if country is None:
        result = {
            "valid": False,
            "status": "Invalid",
            "message": (
                "No supported country was found "
                "in the question."
            ),
        }

        return {
            "result": result
        }

    amount = extract_amount(
        question
    )

    if amount is None:
        result = {
            "valid": False,
            "status": "Invalid",
            "message": (
                "No trip amount was found "
                "in the question."
            ),
        }

        return {
            "result": result
        }

    result = calculate_reimbursement(
        country,
        amount
    )

    return {
        "result": result
    }


# ============================================================
# Trip Validation
# ============================================================

def run_trip_validation(state: AgentState):
    question = state["question"]

    # --------------------------------------------------------
    # Employee
    # --------------------------------------------------------

    employee_id = extract_employee_id(
        question
    )

    if employee_id is None:
        employee_id = state.get(
            "employee_id",
            ""
        )

    if not employee_id:
        employee_id = get_last_employee_id(
            state.get(
                "messages",
                []
            )
        )

    if not employee_id:
        return {
            "result": {
                "valid": False,
                "status": "Invalid",
                "reason": (
                    "No employee ID was found "
                    "in the question."
                ),
            }
        }

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    # First preserve an explicitly mentioned country.
    #
    # This intentionally supports Canada here because
    # validate_trip() must be allowed to determine that
    # EMP001's India employee profile does not match Canada.
    country = extract_explicit_country(
        question
    )

    # If the current question does not mention a country,
    # reuse the most recent supported country from memory.
    if country is None:
        country = get_last_country(
            state.get(
                "messages",
                []
            )
        )

    if country is None:
        return {
            "result": {
                "valid": False,
                "status": "Invalid",
                "reason": (
                    "No supported country was found "
                    "in the question."
                ),
            }
        }

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    amount = extract_amount(
        question
    )

    if amount is None:
        return {
            "result": {
                "valid": False,
                "status": "Invalid",
                "reason": (
                    "No trip amount was found "
                    "in the question."
                ),
            }
        }

    # --------------------------------------------------------
    # Business purpose
    # --------------------------------------------------------

    messages = state.get(
        "messages",
        []
    )

    previous_context = " ".join(
        get_message_content(
            message
        )
        for message in messages
    )

    combined_context = (
        f"{previous_context} "
        f"{question.lower()}"
    )

    business_purpose = (
        "business" in combined_context
        or "work" in combined_context
        or "client" in combined_context
        or "meeting" in combined_context
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    result = validate_trip(
        employee_id=employee_id,
        country=country,
        trip_amount=amount,
        business_purpose=business_purpose,
    )

    return {
        "result": result,
        "employee_id": employee_id,
    }


# ============================================================
# Policy RAG
# ============================================================

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


# ============================================================
# Out-of-Domain
# ============================================================

def run_out_of_domain(state: AgentState):
    return {
        "result": {
            "valid": False,
            "status": "Out of Domain",
            "message": (
                "I can help with corporate travel, "
                "employee eligibility, travel policies, "
                "trip validation, and reimbursement "
                "questions."
            ),
        }
    }


# ============================================================
# Routing
# ============================================================

def route_decision(state):
    decision = state["decision"]

    if decision == "employee_tool":
        return "employee_tool"

    if decision == "employee_then_trip":
        return "employee_then_trip"

    if decision == "reimbursement_tool":
        return "reimbursement_tool"

    if decision == "trip_validation":
        return "trip_validation"

    if decision == "policy_rag":
        return "policy_rag"

    if decision == "out_of_domain":
        return "out_of_domain"

    return END


# ============================================================
# Graph
# ============================================================

graph_builder = StateGraph(
    AgentState
)


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

graph_builder.add_node(
    "decide",
    decide_action
)

graph_builder.add_node(
    "employee_tool",
    run_employee_tool
)

graph_builder.add_node(
    "employee_then_trip",
    run_employee_tool
)

graph_builder.add_node(
    "reimbursement_tool",
    run_reimbursement_tool
)

graph_builder.add_node(
    "trip_validation",
    run_trip_validation
)

graph_builder.add_node(
    "policy_rag",
    run_policy_rag
)

graph_builder.add_node(
    "out_of_domain",
    run_out_of_domain
)


# ============================================================
# Graph Flow
# ============================================================

graph_builder.add_edge(
    START,
    "decide"
)

graph_builder.add_conditional_edges(
    "decide",
    route_decision,
)

graph_builder.add_edge(
    "employee_then_trip",
    "trip_validation"
)


# ============================================================
# End Points
# ============================================================

graph_builder.add_edge(
    "employee_tool",
    END
)

graph_builder.add_edge(
    "reimbursement_tool",
    END
)

graph_builder.add_edge(
    "trip_validation",
    END
)

graph_builder.add_edge(
    "policy_rag",
    END
)

graph_builder.add_edge(
    "out_of_domain",
    END
)


# ============================================================
# Compile
# ============================================================

memory = create_memory()

agent = graph_builder.compile(
    checkpointer=memory
)


# ============================================================
# Local Test
# ============================================================

if __name__ == "__main__":
    import uuid

    thread_id = str(
        uuid.uuid4()
    )

    questions = [
        "Is EMP001 eligible?",
        "Can EMP001 take a 1500 India business trip?",
        "What is the standard travel limit in India?",
        "Are airport trips allowed for business travel?",
        "Are late-night airport trips allowed?",
        "Can EMP001 take a 1000 Canada business trip?",
        "What exact cancellation fee will I be charged?",
        "What is the color of sea?",
    ]

    for question in questions:

        result = agent.invoke(
            {
                "question": question,
                "decision": "",
                "result": {},
                "employee_result": {},
                "employee_id": "",
                "messages": [],
            },
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            },
        )

        print(
            "\n" + "=" * 70
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Decision: {result['decision']}"
        )

        print(
            f"Result: {result['result']}"
        )