import json
import os
import sqlite3
import uuid
from datetime import datetime

from flask import Flask, jsonify, request, render_template, session

from src.agent import agent


app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "development-secret-key"
)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "conversation_history.db",
)


# -------------------------------------------------------------------
# Database helpers
# -------------------------------------------------------------------

def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def save_message(thread_id, role, content):
    connection = get_db()

    connection.execute(
        """
        INSERT INTO conversations (
            thread_id,
            role,
            content,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            thread_id,
            role,
            content,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    connection.commit()
    connection.close()


def get_history():
    connection = get_db()

    rows = connection.execute(
        """
        SELECT
            thread_id,
            MIN(created_at) AS created_at,
            MAX(id) AS last_id
        FROM conversations
        GROUP BY thread_id
        ORDER BY last_id DESC
        """
    ).fetchall()

    history = []

    for row in rows:
        first_message = connection.execute(
            """
            SELECT content
            FROM conversations
            WHERE thread_id = ?
              AND role = 'user'
            ORDER BY id ASC
            LIMIT 1
            """,
            (row["thread_id"],),
        ).fetchone()

        title = (
            first_message["content"]
            if first_message
            else "New conversation"
        )

        title = title.strip()

        if len(title) > 42:
            title = title[:42].rstrip() + "..."

        history.append({
            "thread_id": row["thread_id"],
            "title": title,
            "created_at": row["created_at"],
        })

    connection.close()

    return history


def get_conversation(thread_id):
    connection = get_db()

    rows = connection.execute(
        """
        SELECT role, content, created_at
        FROM conversations
        WHERE thread_id = ?
        ORDER BY id ASC
        """,
        (thread_id,),
    ).fetchall()

    connection.close()

    messages = []

    for row in rows:
        try:
            content = json.loads(row["content"])
        except (json.JSONDecodeError, TypeError):
            content = row["content"]

        messages.append({
            "role": row["role"],
            "content": content,
            "created_at": row["created_at"],
        })

    return messages


# Initialize the database when Flask starts.
init_db()


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/")
def home():
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())

    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "error": "Question is required."
        }), 400

    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())

    thread_id = session["thread_id"]

    try:
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

        assistant_result = result.get("result", {})
        decision = result.get("decision", "")

        # Save the user's message.
        save_message(
            thread_id,
            "user",
            json.dumps(question),
        )

        # Save the complete assistant result.
        save_message(
            thread_id,
            "assistant",
            json.dumps({
                "decision": decision,
                "result": assistant_result,
            }, default=str),
        )

        return jsonify({
            "question": question,
            "decision": decision,
            "result": assistant_result,
            "thread_id": thread_id,
        })

    except Exception as e:
        app.logger.exception("Agent request failed")

        return jsonify({
            "error": "The assistant could not process your request.",
            "details": str(e),
        }), 500


@app.route("/clear", methods=["POST"])
def clear():
    session["thread_id"] = str(uuid.uuid4())

    return jsonify({
        "message": "Conversation cleared.",
        "thread_id": session["thread_id"],
    })


@app.route("/history", methods=["GET"])
def history():
    return jsonify({
        "conversations": get_history()
    })


@app.route("/conversation/<thread_id>", methods=["GET"])
def conversation(thread_id):
    # Switch the active conversation to the selected thread.
    session["thread_id"] = thread_id

    messages = get_conversation(thread_id)

    return jsonify({
        "thread_id": thread_id,
        "messages": messages,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


if __name__ == "__main__":
    app.run(debug=True)