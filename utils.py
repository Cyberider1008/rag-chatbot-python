import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def load_data(file_path="custdetail.json"):
    with open(file_path, "r") as f:
        data = json.load(f)

    texts = []
    for t in data:
        text = (
            f"On {t['date']}, {t['customer']} purchased a {t['product']} "
            f"for ₹{t['amount']}."
        )
        texts.append(text)

    return texts


def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts)
    return model, vectors

def retrieve_transactions(query, model, vectors, texts, top_k=3):
    q_vec = model.encode([query])[0]

    sims = np.dot(vectors, q_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec))
    top_ids = np.argsort(sims)[::-1][:top_k]

    retrieved = [texts[i] for i in top_ids]
    return retrieved

#LLM
def ask_bot(question, retrieved_texts, api_key):
    context = "\n".join(retrieved_texts)

    template = """
You are a retail transaction assistant. Answer strictly using ONLY this context:
{context}

Question: {question}

Answer:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)

    return response.content
