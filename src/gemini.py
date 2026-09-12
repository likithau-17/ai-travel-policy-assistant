import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


MODEL_NAME = "gemini-3.6-flash"


def create_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    return genai.Client(api_key=api_key)


def generate_answer(question, context):
    client = create_client()

    prompt = f"""
You are a corporate travel policy assistant.

Answer the user's question using ONLY the provided policy context.

If the policy context does not contain enough information to answer,
clearly say that the policy information is insufficient.

Do not invent policy rules, limits, approvals, or reimbursement amounts.

Policy context:
{context}

User question:
{question}

Answer:
"""

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt,
    )

    return interaction.output_text