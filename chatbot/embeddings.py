import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import streamlit as st
import os


def create_embeddings(texts):
    """Generate embeddings for all Txt. This is local use only """

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = model.encode(texts)
        return model, vectors

    except Exception as e:
        print("Error while creating embeddings:", e)
        return None, None


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_vectors_texts_df():
    """Load prebuilt embeddings and data with safety checks."""

    # Paths
    vec_path = "embeddings/vectors.npy"
    text_path = "embeddings/texts.pkl"
    df_path = "embeddings/df.pkl"

    # --- File Checks ---
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"{vec_path} not found. Did you generate embeddings?")

    if not os.path.exists(text_path):
        raise FileNotFoundError(f"{text_path} not found. Did you generate texts.pkl?")

    if not os.path.exists(df_path):
        raise FileNotFoundError(f"{df_path} not found. Did you generate df.pkl?")

    # --- Load vectors ---
    vectors = np.load(vec_path)

    # --- Load texts ---
    with open(text_path, "rb") as f:
        texts = pickle.load(f)

    # --- Load df ---
    with open(df_path, "rb") as f:
        df = pickle.load(f)

    return vectors, texts, df


def load_prebuilt_embeddings():
    model = load_model()                     
    vectors, texts, df = load_vectors_texts_df()  
    return model, vectors, texts, df