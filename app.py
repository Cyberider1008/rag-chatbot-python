import streamlit as st
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import pandas as pd

from chatbot.data_loader import load_data
from chatbot.embeddings import create_embeddings
from chatbot.retrieval import retrieve
from chatbot.llm import ask_bot

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

#load data for chart & RAG
texts, df = load_data()

# Load data and embeddings once
if "model" not in st.session_state:
    st.session_state.texts = texts
    st.session_state.model, st.session_state.vectors = create_embeddings(st.session_state.texts)
    st.session_state.history = []

st.title("RAG Chatbot")

st.subheader("Monthly Spending Analysis")

with st.expander("Monthly Spending Analysis (Click to expand/collapse)"):

    monthly_spend = df.groupby("month")["amount"].sum()

    # Line Chart
    fig_line, ax_line = plt.subplots()
    ax_line.plot(monthly_spend.index, monthly_spend.values, marker="o")
    ax_line.set_title("Monthly Spend (Line Chart)")
    ax_line.set_xlabel("Month")
    ax_line.set_ylabel("Total Spend (₹)")
    plt.xticks(rotation=45)
    st.pyplot(fig_line)

    # Bar Chart
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(monthly_spend.index, monthly_spend.values)
    ax_bar.set_title("Monthly Spend (Bar Chart)")
    ax_bar.set_xlabel("Month")
    ax_bar.set_ylabel("Total Spend (₹)")
    plt.xticks(rotation=45)
    st.pyplot(fig_bar)

# User input
user_input = st.text_input("Ask your question:")

if user_input.strip():
    if st.button("Send"):
        retrieved = retrieve(user_input, st.session_state.model, st.session_state.vectors, st.session_state.texts)
        answer = ask_bot(user_input, retrieved, api_key)
        st.session_state.history.append({"user": user_input, "bot": answer})

# Memory button (Show last question)
if st.button("Show my last question"):
    if st.session_state.history:
        last_q = st.session_state.history[-1]["user"]
        st.info(f"Your last question was: **{last_q}**")
    else:
        st.warning("You haven't asked anything yet.")

# Display chat history
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")
