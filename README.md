# AI-Powered Travel & Policy Assistant

An AI-powered corporate travel and policy assistant that answers fictional company travel-policy questions using **Retrieval-Augmented Generation (RAG)**, deterministic business tools, conversational memory, and an agentic workflow.

The project is designed as a training capstone for employee travel operations.

---

## Business Problem

Employees often need quick answers to questions such as:

* What is the standard travel limit for India?
* Are airport trips allowed?
* Is an employee eligible for company travel?
* Does a trip require additional approval?
* How much of a trip is reimbursable?
* What happens if a follow-up question changes the trip cost?

Manually searching policy documents for these answers can be slow and may lead to inconsistent interpretation.

This assistant combines **policy retrieval** with **deterministic business tools** so that policy questions are answered using retrieved policy context while calculations and eligibility decisions are handled programmatically.

---

## Solution

The assistant uses different components for different types of questions:

* **RAG** retrieves relevant policy information from company policy documents.
* **Business Tools** perform deterministic operations such as employee eligibility, trip validation, and reimbursement calculation.
* **LangGraph Agent** decides whether a question should use RAG or a business tool.
* **Conversational Memory** preserves relevant employee and trip context across follow-up questions.
* **MCP** exposes the core business tools through a Model Context Protocol server.
* **Flask** provides the web interface and API endpoints.
* **Gemini** generates the final employee-facing response.

---

## Architecture

```text
                              User
                                |
                                v
                       Flask Web Application
                                |
                                v
                       LangGraph AI Agent
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
            RAG Pipeline                 Business Tools
                 |                             |
                 v                    +--------+---------+
          FAISS Vector Store          |        |         |
                 |                    v        v         v
                 v                Employee  Trip    Reimbursement
          Policy Documents        Eligibility Validation Calculation
                 |
                 v
          Retrieved Context

                 +
                 |
                 v
       Conversational Memory
       (LangGraph Checkpointing)

                 |
                 v
              Gemini
                 |
                 v
             Response


              MCP Server
                  |
        +---------+---------+
        |         |         |
        v         v         v
    Eligibility  Trip   Reimbursement
                  Tools
```

The Flask application uses the LangGraph agent to route requests. The MCP server provides the same core business capabilities through the Model Context Protocol.

---

## Technology Stack

* Python
* Pandas
* Flask
* LangChain
* LangGraph
* Sentence Transformers
* FAISS
* Gemini API
* MCP
* pytest
* Git / GitHub

---

## Project Structure

```text
ai-travel-policy-assistant/
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── data/
│   ├── company_policy/
│   │   ├── airport_policy.txt
│   │   ├── approval_policy.txt
│   │   ├── cancellation_policy.txt
│   │   ├── employee_eligibility.txt
│   │   ├── expense_policy.txt
│   │   ├── travel_policy_india.txt
│   │   └── travel_policy_us.txt
│   └── employees.csv
│
├── mcp_server/
│   ├── client_demo.py
│   └── server.py
│
├── src/
│   ├── agent.py
│   ├── embeddings.py
│   ├── gemini.py
│   ├── ingestion.py
│   ├── memory.py
│   ├── rag.py
│   └── tools.py
│
├── tests/
│   ├── test_agent.py
│   └── test_app.py
│
├── vector_store/
│   ├── chunks.pkl
│   └── policy.index
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## RAG Workflow

The RAG pipeline follows these steps:

1. Load the company policy documents.
2. Clean and prepare the policy text.
3. Split the documents into chunks.
4. Generate sentence embeddings.
5. Store the embeddings in a FAISS vector index.
6. Retrieve the most relevant policy chunks for a user question.
7. Pass the retrieved context to Gemini.
8. Generate an employee-facing answer using the retrieved policy context.

### Multi-Question Handling

The RAG pipeline can also handle questions containing multiple policy requests.

For example:

```text
What is the US travel limit and is late-night business travel allowed?
```

The question is split into individual requests:

```text
Question 1: What is the US travel limit?

Question 2: Is late-night business travel allowed?
```

Each question is retrieved and answered independently before the results are combined into the final response.

This helps prevent information relevant to one part of a question from being incorrectly applied to another part.

---

## Agent Workflow

The LangGraph agent determines which action is appropriate for each question.

### Policy Questions

Example:

```text
What is the standard travel limit in India?
```

The agent routes the question to the **RAG pipeline**.

### Employee Eligibility

Example:

```text
Is EMP001 eligible?
```

The agent calls the **employee eligibility tool**.

### Trip Validation

Example:

```text
Can EMP001 take a 2500 India business trip?
```

The agent validates the relevant employee and trip information, including:

* Employee eligibility
* Country
* Trip amount
* Business purpose
* Applicable travel limit
* Approval requirement

### Reimbursement

Example:

```text
How much can I reimburse for a 2500 India trip?
```

The reimbursement tool calculates the applicable reimbursable amount.

### Improved Routing

The agent distinguishes between:

* Policy-information requests
* Employee eligibility requests
* Trip validation requests
* Reimbursement requests
* Out-of-domain questions

Specific trip-action questions are routed to deterministic tools instead of being treated as ordinary policy-retrieval questions.

---

## Available Business Tools

### 1. Employee Eligibility

Checks employee information using the employee dataset.

### 2. Trip Validation

Validates a business trip against the applicable travel policy.

### 3. Reimbursement Calculation

Calculates the reimbursable amount based on the applicable country limit.

These operations are deterministic and do not require the LLM to perform the underlying business calculation.

---

## Conversational Memory

The assistant maintains conversation state using **LangGraph checkpointing**.

For example:

```text
User:

Can EMP001 take a 1500 India business trip?

Assistant:

Within Policy

User:

What if it costs 2500?

Assistant:

Needs Approval
```

The second question does not repeat the employee ID because the assistant can reuse relevant context from the previous conversation.

The memory layer can preserve information such as:

* Employee ID
* Country
* Previous trip context
* Previous user messages

The Flask application also provides a `/clear` endpoint to start a fresh conversation.

---

## MCP

The project includes a basic **MCP server** exposing the following tools:

* `check_employee_eligibility`
* `validate_trip`
* `calculate_reimbursement`

The MCP client demo discovers the available tools and invokes them.

Run the MCP demo from the project root:

```bash
python -m mcp_server.client_demo
```

The MCP layer provides a standard interface through which compatible MCP clients can discover and invoke the business tools.

---

## Flask Application

The assistant is exposed through a Flask application.

### Start the Application

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Set the Gemini API key in a local `.env` file:

```text
GEMINI_API_KEY=your_api_key
```

Run the application:

```bash
python app/app.py
```

The application runs on:

```text
http://127.0.0.1:5000
```

### API Endpoints

| Method | Endpoint  | Purpose                         |
| ------ | --------- | ------------------------------- |
| GET    | `/`       | Web interface                   |
| POST   | `/ask`    | Submit a travel-policy question |
| POST   | `/clear`  | Clear the current conversation  |
| GET    | `/health` | Health check                    |

---

## Web Interface

The Flask application provides a chat-based interface for interacting with the assistant.

The interface supports:

* Sending travel-policy questions
* Displaying assistant responses
* Showing policy sources
* Maintaining the current conversation
* Starting a fresh conversation
* Displaying errors in a user-friendly format
* Rendering structured policy responses

Responses can contain sections such as:

```text
Answer

Limit or requirement

Approval needed
```

when those sections are applicable.

---

## Testing

The project contains automated tests covering:

* Agent behavior
* RAG grounding
* Business tools
* Conversational memory
* Flask endpoints
* Error handling

Run all tests:

```bash
python -m pytest -q
```

Current verified test result:

```text
38 passed
```

### Test Scenarios

The test suite includes scenarios for:

* Employee eligibility
* Manager approval requirements
* India travel limits
* US travel limits
* Airport travel
* Late-night travel
* Reimbursement calculations
* Trips requiring approval
* Invalid employee IDs
* Unsupported countries
* Missing employee IDs
* Missing trip amounts
* Country mismatches
* Conversational memory
* Follow-up cost changes
* Hallucination prevention
* Flask API failures
* Empty questions
* RAG failures
* Tool/agent failures

---

## Hallucination Testing

The assistant was tested against policy information that is **not present** in the provided documents.

Examples include questions about:

* Unsupported cancellation fees
* Unknown country limits
* Unlisted policy rules

The expected behavior is to avoid inventing unsupported policy information and instead state when the available policy context does not provide enough information.

The system's prompts instruct Gemini to use the retrieved policy context as the authoritative source for policy answers. This reduces unsupported responses but does not guarantee that an LLM can never generate incorrect information.

---

## Prompt Improvements

Two meaningful prompt improvements were implemented.

### Improvement 1 — Stronger Grounding

**4 Cs: Context + Constraints**

The prompt explicitly treats the retrieved policy context as the authoritative source and instructs the model to:

* Ignore irrelevant retrieved context.
* Avoid using outside knowledge.
* Avoid combining unrelated policy statements.
* Distinguish supported information from information that cannot be determined.

This improves the consistency of policy-grounded responses.

### Improvement 2 — Clearer Employee-Facing Responses

**4 Cs: Clarity + Customization**

The prompt provides a clearer response structure:

```text
Answer

Limit or requirement

Approval needed
```

when applicable.

The model is also instructed not to expose internal implementation details such as:

* Embeddings
* Vector stores
* Retrieval
* Agent internals

This keeps responses focused on the employee's actual question.

---

## Error Handling

The application handles several failure cases, including:

* Empty questions
* Malformed JSON requests
* Invalid employee IDs
* Missing trip amounts
* Unsupported countries
* Country mismatches
* Agent failures
* Tool failures
* RAG failures
* External LLM/API failures

Flask returns JSON error responses rather than exposing an HTML server error page to the client.

---

## Example Questions

### Policy Questions

```text
What is the standard travel limit in India?
```

```text
What is the US travel limit?
```

```text
Are airport trips allowed?
```

```text
Are late-night trips allowed?
```

```text
What information is required for an expense?
```

```text
What is the US travel limit and is late-night business travel allowed?
```

### Employee Eligibility

```text
Is EMP001 eligible?
```

```text
Is EMP002 eligible?
```

```text
Is EMP004 eligible?
```

### Trip Validation

```text
Can EMP001 take a 1500 India business trip?
```

```text
Can EMP001 take a 2500 India business trip?
```

```text
Can EMP003 take a 50 US airport trip?
```

```text
Can EMP003 take a 100 US airport trip?
```

### Follow-Up Questions

```text
Can EMP001 take a 1500 India business trip?
```

```text
What if it costs 2500?
```

The second question can use the context from the previous interaction.

---

## Limitations

* The policy documents are fictional training data.
* Employee records are sample data.
* The assistant should not be treated as a source of real company policy.
* Gemini API availability and quotas can affect live LLM responses.
* The current system supports the countries and rules represented in the provided policy documents.
* Complex policy questions outside the available policy context may require human review.
* LLM-generated responses may still require validation for high-impact business decisions.

---

## Project Status

The capstone implementation is completed with the following components:

* RAG pipeline
* FAISS semantic search
* Gemini LLM integration
* LangGraph agent workflow
* Improved intent and action routing
* Deterministic business tools
* Conversational memory
* Multi-question RAG handling
* MCP tool server
* Flask web application
* Structured employee-facing responses
* Policy source display
* Automated testing
* Hallucination testing
* Error handling
* Prompt evaluation and improvements
* Web interface improvements

The final automated test suite currently passes:

```text
38 passed
```

---

## Summary

The **AI-Powered Travel & Policy Assistant** demonstrates how RAG, deterministic business logic, agentic workflows, conversational memory, MCP, and a web API can be combined to build a practical enterprise AI assistant.

The architecture separates **policy knowledge retrieval** from **deterministic business decisions**, allowing the system to provide policy answers using retrieved documents while using programmatic tools for eligibility, trip validation, and reimbursement calculations.

The project also demonstrates practical considerations for enterprise AI systems, including:

* Tool-based decision making
* Conversation context
* Multi-question handling
* Source-aware responses
* Error handling
* Unsupported-information handling
* Automated testing
* LLM prompt improvement
* MCP-based tool exposure