import streamlit as st
from dotenv import load_dotenv
import os
from chatbot.data_loader import load_data
from chatbot.embeddings import create_embeddings
from chatbot.retrieval import retrieve
from chatbot.llm import ask_bot

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

# Load data and embeddings once
if "model" not in st.session_state:
    st.session_state.texts = load_data()
    st.session_state.model, st.session_state.vectors = create_embeddings(st.session_state.texts)
    st.session_state.history = []

st.title("RAG Chatbot")

# User input
user_input = st.text_input("Ask your question:")

if user_input.strip():
    if st.button("Send"):
        retrieved = retrieve(user_input, st.session_state.model, st.session_state.vectors, st.session_state.texts)
        answer = ask_bot(user_input, retrieved, api_key)
        st.session_state.history.append({"user": user_input, "bot": answer})

# Display chat history
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")
