from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Global LLM instance for reuse
_llm_instance = None


def get_llm(api_key):
    """Create or return the cached Gemini LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )
    return _llm_instance


def ask_bot(question, retrieved_texts, api_key):
    """Send question + retrieved context to Gemini LLM and get a short, clean answer."""

    # Combine retrieved text
    context = "\n".join(retrieved_texts)
    context = context[:400]

    # Clean, production-ready prompt
    template = """You are a friendly chatbot.
Keep answers short, simple, and natural.
Use the context only if helpful.

Context:
{context}

Question:
{question}
"""


    # Create template object
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Fill prompt
    final_prompt = prompt.format(context=context, question=question)

    # Get globally cached LLM instance (FAST)
    llm = get_llm(api_key)

    # Run model
    response = llm.invoke(final_prompt)

    return response.content
