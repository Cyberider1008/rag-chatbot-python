from dotenv import load_dotenv
import os
from utils import load_data, create_embeddings, retrieve_transactions, ask_bot


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")


if __name__ == "__main__":
    print(" chatbot ...")

    texts = load_data()
    model, vectors = create_embeddings(texts)

    print("Ready! Type 'exit' or 'quit' to stop.\n")

    while True:
        user_q = input("You: ")
        if user_q.lower() in ["exit", "quit"]:
            break

        retrieved = retrieve_transactions(user_q, model, vectors, texts)
        answer = ask_bot(user_q, retrieved, api_key)

        print("\nBot:", answer, "\n")
