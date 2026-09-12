import faiss
import numpy as np

from .embeddings import load_embedding_model, generate_embeddings
from .ingestion import build_chunks


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