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
   "This information is not available in the current knowledge base."
5. Keep the answer clear and simple.

ANSWER:
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        return "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
        )

    return response.content