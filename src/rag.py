import faiss
import numpy as np

from embeddings import load_embedding_model, generate_embeddings
from ingestion import build_chunks


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


if __name__ == "__main__":
    chunks = build_chunks()

    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    index = build_vector_index(embeddings)

    print("\nFAISS index created.")
    print("Indexed chunks:", index.ntotal)

    query = "What is the standard ride limit in India?"

    results = search(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=3,
    )

    print(f"\nQuery: {query}")

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\n--- Result {rank} ---")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Source: {chunk['metadata']['source']}")
        print(f"Section: {chunk['metadata']['section_title']}")
        print(f"Text:\n{chunk['text']}")