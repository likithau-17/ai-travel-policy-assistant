import faiss
import numpy as np

from embeddings import load_embedding_model, generate_embeddings
from ingestion import build_chunks

from pathlib import Path
import pickle

VECTOR_STORE_DIR = Path("vector_store")
INDEX_PATH = VECTOR_STORE_DIR / "policy.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.pkl"

def build_vector_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    return index


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

def save_vector_store(index, chunks):
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)


def load_vector_store():
    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks

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