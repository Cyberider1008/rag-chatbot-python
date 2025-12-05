from dotenv import load_dotenv
import os
from utils import load_data, create_embeddings, retrieve_transactions, ask_bot


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")


if __name__ == "__main__":
    print(" chatbot start here ...")

    texts = load_data()
    model, vectors = create_embeddings(texts)

    print("Ready! Type 'exit' or 'quit' to stop.\n")

    try:

        while True:
            user_q = input("You: ").strip()

            if user_q.lower() in ["exit", "quit"]:
                print("Bot: Goodbye!")
                break

            if not user_q:
                print("\nBot: Plz Ask me.\n")
                continue

            # Retrieve similar chunks from vector store
            retrieved = retrieve_transactions(user_q, model, vectors, texts)

            # Ask LLM with retrieved context
            answer = ask_bot(user_q, retrieved, api_key)

            print("\nBot:", answer, "\n")
    
    except KeyboardInterrupt:
        print("\nBot: Chat ended by user.\n")
