import numpy as np

def preprocess_vectors(vectors):
    """Normalize vectors once to speed up retrieval."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def retrieve(query, model, norm_vectors, texts, top_k=3):
    """Retrieve most similar texts using cosine similarity."""
    
    # Encode query and normalize
    q_vec = model.encode([query])[0]
    q_vec = q_vec / np.linalg.norm(q_vec)

    # Cosine similarity via efficient dot product
    sims = np.dot(norm_vectors, q_vec)

    # Get top_k highest similarity IDs
    top_ids = np.argsort(sims)[::-1][:top_k]

    return [texts[i] for i in top_ids]
