import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


def generate_answer(question, context):
    prompt = f"""
You are the AI assistant for the NMAMIT Computer Science and Engineering department.

Answer the user's question using only the provided context.

Rules:
- Do not invent NMAMIT or CSE-specific information.
- If the answer is not available in the context, say that the information
  could not be verified from the available knowledge base.
- Give a clear and simple answer.
- Use the context relevant to the user's question.

Context:
{context}

User Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content