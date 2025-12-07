from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def ask_bot(question, retrieved_texts, api_key):
    """Send question + context to Gemini LLM and get response."""
    context = "\n".join(retrieved_texts)

    template = """
You are a friendly chatbot.
Answer in short, simple, human-like sentences.
Use the context if relevant.

Instructions:
- Use the context to answer questions.
- If the user asks for total spending, compute it from context and answer in one sentence.
- If the user asks for purchase history, summarize all purchases naturally in one sentence.
- No bullet points unless user asks.

Context:
{context}

User question:
{question}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )

    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)
    return response.content
