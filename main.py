import os
from dotenv import load_dotenv
from chatbot.data_loader import load_data
from chatbot.embeddings import create_embeddings
from chatbot.retrieval import preprocess_vectors, retrieve
from chatbot.llm import ask_bot

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

if __name__ == "__main__":

    print("Chatbot started...")

    # Load data and create embeddings
    texts, _ = load_data()
    model, vectors = create_embeddings(texts)

    print("Ready! Type 'exit' or 'quit' to stop.\n")

    norm_vectors = preprocess_vectors(vectors)
    try:
        while True:
            user_q = input("You: ").strip()
            if user_q.lower() in ["exit", "quit"]:
                print("Bot: Goodbye!")
                break
            if not user_q:
                print("Bot: Please ask me something.\n")
                continue

            retrieved = retrieve(user_q, model, norm_vectors, texts)
            answer = ask_bot(user_q, retrieved, api_key)
            print("\nBot:", answer, "\n")

    except KeyboardInterrupt:
        print("\nBot: Chat ended by user.\n")
