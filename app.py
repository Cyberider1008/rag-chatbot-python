import streamlit as st

import os
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import pandas as pd

from chatbot.data_loader import load_data
from chatbot.embeddings import create_embeddings
from chatbot.retrieval import retrieve
from chatbot.llm import ask_bot

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found in Render Environment!")

# Initialize state
if "loaded" not in st.session_state:
    st.session_state.loaded = False
    st.session_state.history = []

st.title("RAG Chatbot")

# LOAD BUTTON (prevents crashes)
if st.button("Load Data & Embeddings"):
    with st.spinner("Loading data..."):
        texts, df = load_data()
        st.session_state.texts = texts
        st.session_state.df = df

    with st.spinner("Creating embeddings (this may take some time)..."):
        st.session_state.model, st.session_state.vectors = create_embeddings(
            texts)

    st.session_state.loaded = True
    st.success("Data & Embeddings Loaded!")

# MONTHLY CHARTS
if st.session_state.loaded:
    df = st.session_state.df

    st.subheader("Monthly Spending Analysis")
    with st.expander("Click to view charts"):

        monthly_spend = df.groupby("month")["amount"].sum()

        # Line Chart
        fig_line, ax_line = plt.subplots()
        ax_line.plot(monthly_spend.index, monthly_spend.values, marker="o")
        ax_line.set_title("Monthly Spend (Line Chart)")
        ax_line.set_xlabel("Month")
        ax_line.set_ylabel("Total Spend (₹)")
        st.pyplot(fig_line)

        # Bar Chart
        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(monthly_spend.index, monthly_spend.values)
        ax_bar.set_title("Monthly Spend (Bar Chart)")
        st.pyplot(fig_bar)

else:
    st.info("Please click **Load Data & Embeddings** first.")

# CHATBOT
user_input = st.text_input("Ask your question:")

if st.session_state.loaded and st.button("Send"):
    retrieved = retrieve(user_input, st.session_state.model,
                         st.session_state.vectors, st.session_state.texts)
    answer = ask_bot(user_input, retrieved, api_key)
    st.session_state.history.append({"user": user_input, "bot": answer})

# History
for chat in st.session_state.history[::-1]:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")
