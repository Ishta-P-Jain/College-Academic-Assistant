import os

from dotenv import load_dotenv

# override=True: values in .env win over stale variables already set in Windows.
load_dotenv(override=True)

NOT_AVAILABLE = "This information is not available in the current knowledge base."

# Configure in .env:
#   LLM_PROVIDER=gemini   (default)  or  ollama (local, no API key)
#   GEMINI_MODEL=gemini-3.8-flash
#   OLLAMA_MODEL=llama3.2
_llm = None


def get_llm():
    """Create the chat model on first use, so the app still loads without a key."""
    global _llm
    if _llm is None:
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            _llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI

            _llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                temperature=0,
            )
    return _llm


def generate_answer(question, context):
    prompt = f"""
You are the College Academic Assistant for the NMAMIT Computer Science
and Engineering department.

You must answer the user's question based ONLY on the context provided below.

CONTEXT:
{context}

USER QUESTION:
{question}

Instructions:
1. If the context directly contains information that answers the question,
   use that information to answer.
2. Do not say the information is unavailable if the context contains
   a relevant answer.
3. Do not add facts that are not present in the context.
4. If the context does not contain enough information to answer the question,
   respond exactly:
   "{NOT_AVAILABLE}"
5. Keep the answer clear and simple.

ANSWER:
"""

    response = get_llm().invoke(prompt)

    if isinstance(response.content, list):
        return "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
        )

    return response.content