from sentence_transformers import SentenceTransformer

def create_embeddings(texts):
    """Generate embeddings for all Txt."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts)
    return model, vectors