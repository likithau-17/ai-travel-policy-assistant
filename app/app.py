import os
import uuid

from flask import Flask, jsonify, request, render_template, session

from src.agent import agent

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "development-secret-key"
)


@app.route("/")
def home():
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())

    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "error": "Question is required."
        }), 400

    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())

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
                "thread_id": session["thread_id"]
            }
        },
    )

    return jsonify({
        "question": question,
        "decision": result["decision"],
        "result": result["result"],
    })


if __name__ == "__main__":
    app.run(debug=True)