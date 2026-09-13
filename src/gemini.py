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

Your job is to answer the user's question accurately using ONLY the provided policy context.

Follow these rules strictly:

1. Use only facts explicitly supported by the policy context.
2. If the context does not contain enough information to answer the question,
    say that the policy information is insufficient.
3. If the retrieved context is irrelevant to the question, do not use it
    to make assumptions or construct an answer.
4. Never invent policy rules, spending limits, approval status, eligibility,
    reimbursement amounts, cancellation fees, or other policy details.
5. Do not assume that an employee, trip, approval, or exception exists unless
    the provided context explicitly confirms it.
6. If the policy gives a standard limit but the user's situation requires
    additional approval, clearly state that approval is required.
7. Keep the answer concise and directly address the user's question.

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