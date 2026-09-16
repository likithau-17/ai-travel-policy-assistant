import faiss
import numpy as np

from .embeddings import load_embedding_model, generate_embeddings
from .ingestion import build_chunks

import re


VECTOR_STORE_DIR = "vector_store"
INDEX_PATH = f"{VECTOR_STORE_DIR}/policy.index"
CHUNKS_PATH = f"{VECTOR_STORE_DIR}/chunks.pkl"


def build_vector_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    return index


def save_vector_store(index, chunks):
    import pickle
    from pathlib import Path

    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)


def load_vector_store():
    import pickle

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


def search(query, model, index, chunks, top_k=3):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    ).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, index_position in zip(distances[0], indices[0]):
        results.append(
            {
                "chunk": chunks[index_position],
                "distance": float(distance),
            }
        )

    return results

def retrieve_policy(query, model, index, chunks, top_k=3):
    results = search(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=top_k,
    )

    retrieved = []

    for result in results:
        chunk = result["chunk"]

        retrieved.append(
            {
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "section": chunk["metadata"]["section_title"],
                "country": chunk["metadata"]["country"],
                "policy_type": chunk["metadata"]["policy_type"],
                "distance": result["distance"],
            }
        )

    return retrieved


def build_context(retrieved_results):
    context_parts = []

    for result in retrieved_results:
        context_parts.append(
            f"Source: {result['source']}\n"
            f"Section: {result['section']}\n"
            f"Policy:\n{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def split_questions_for_rag(question):
    """
    Split a user message into separate questions for RAG.

    Handles:
        What is the US limit?
        Is late-night travel allowed?

    Also handles a single question without a question mark.
    """

    question = question.strip()

    if not question:
        return []

    # Split on question marks first.
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

    # --------------------------------------------------------
    # Handle common "and" constructions.
    #
    # Example:
    #
    # What is the US travel limit and is late-night
    # business travel allowed?
    #
    # becomes:
    #
    # What is the US travel limit?
    # Is late-night business travel allowed?
    # --------------------------------------------------------

    if len(questions) == 1:

        text = questions[0]

        match = re.match(
            r"^(.*?\b(?:and)\b\s+"
            r"(?:is|are|can|does|do|what|how|"
            r"will|would|should)\b.*)$",
            text,
            flags=re.IGNORECASE,
        )

        if match:

            first_part = match.group(1)

            split_match = re.match(
                r"^(.*?)\s+\band\b\s+"
                r"((?:is|are|can|does|do|what|how|"
                r"will|would|should)\b.*)$",
                first_part,
                flags=re.IGNORECASE,
            )

            if split_match:

                first_question = (
                    split_match.group(1).strip()
                    + "?"
                )

                second_question = (
                    split_match.group(2).strip()
                )

                if not second_question.endswith("?"):
                    second_question += "?"

                return [
                    first_question,
                    second_question,
                ]

    return questions


def answer_question(question, model, index, chunks):
    """
    Answer one or more policy questions using RAG.

    If the user asks multiple questions in one message,
    each question is retrieved and answered independently.

    Example:
        What is the US travel limit and is late-night
        business travel allowed?

    becomes:

        1. What is the US travel limit?
        2. Is late-night business travel allowed?

    Each question gets its own FAISS retrieval and Gemini
    generation step before the answers are combined.
    """

    from .gemini import generate_answer

    # --------------------------------------------------------
    # Split multiple questions
    # --------------------------------------------------------

    questions = split_questions_for_rag(
        question
    )

    # Safety fallback.
    if not questions:
        questions = [question]

    all_answers = []
    all_sources = []

    # --------------------------------------------------------
    # Answer each question independently
    # --------------------------------------------------------

    for individual_question in questions:

        retrieved_results = retrieve_policy(
            query=individual_question,
            model=model,
            index=index,
            chunks=chunks,
        )

        context = build_context(
            retrieved_results
        )

        answer = generate_answer(
            question=individual_question,
            context=context,
        )

        all_answers.append(
            answer
        )

        # ----------------------------------------------------
        # Collect sources
        # ----------------------------------------------------

        for result in retrieved_results:

            source = {
                "source": result["source"],
                "section": result["section"],
            }

            if source not in all_sources:
                all_sources.append(
                    source
                )

    # --------------------------------------------------------
    # Combine answers
    # --------------------------------------------------------

    if len(all_answers) == 1:

        combined_answer = all_answers[0]

    else:

        combined_parts = []

        for number, (individual_question, answer) in enumerate(
            zip(questions, all_answers),
            start=1,
        ):
            combined_parts.append(
                f"Question {number}: {individual_question}\n"
                f"{answer}"
            )

        combined_answer = "\n\n".join(
            combined_parts
        )

    return {
        "answer": combined_answer,
        "sources": all_sources,
    }


if __name__ == "__main__":
    chunks = build_chunks()

    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    index = build_vector_index(embeddings)

    save_vector_store(index, chunks)

    print("\nFAISS vector store saved.")
    print("Indexed chunks:", index.ntotal)
    print("Index path:", INDEX_PATH)
    print("Chunks path:", CHUNKS_PATH)