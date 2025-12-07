import streamlit as st

from chatbot.llm import ask_bot
from chatbot.retrieval import retrieve
from chatbot.embeddings import load_prebuilt_embeddings

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

api_key = os.environ.get("GOOGLE_API_KEY")

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
        st.session_state.model, st.session_state.vectors, texts, df = load_prebuilt_embeddings()
        st.session_state.texts = texts
        st.session_state.df = df

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
