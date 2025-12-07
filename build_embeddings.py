import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from chatbot.data_loader import load_data

# ---------- 1. Load your JSON data ----------

texts, df = load_data()

# ---------- 3. Load MiniLM model ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- 4. Generate embeddings ----------
vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# ---------- 5. Save files ----------
np.save("embeddings/vectors.npy", vectors)

with open("embeddings/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

with open("embeddings/df.pkl", "wb") as f:
    pickle.dump(df, f)

print(" vectors.npy and texts.pkl generated successfully!")
