import numpy as np


def retrieve(query, model, vectors, texts, top_k=3):
    """Retrieve top_k most similar txts for a query."""
    q_vec = model.encode([query])[0]
    sims = np.dot(vectors, q_vec) / \
        (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec))
    top_ids = np.argsort(sims)[::-1][:top_k]
    return [texts[i] for i in top_ids]
