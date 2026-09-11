from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return embeddings


if __name__ == "__main__":
    from ingestion import build_chunks

    chunks = build_chunks()

    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    print("\nEmbedding generation complete.")
    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)