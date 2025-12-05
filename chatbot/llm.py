from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def ask_bot(question, retrieved_texts, api_key):
    """Send question + context to LLM and get resp ."""
    context = "\n".join(retrieved_texts)

    template = """
You are a friendly chatbot.
Answer in short, simple, human-like sentences.
Use the context if relevant.

Instructions:
- Use the context to answer questions.
- If the user asks for total spending, compute it from context and report in one sentence.
- If the user asks for purchase history, summarize all purchases in one natural sentence.
- Do not use bullet points or extra breakdowns unless asked.

Context:
{context}

User question:
{question}
"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)
    return response.content
