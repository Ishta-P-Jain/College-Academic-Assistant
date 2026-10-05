from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.llm import generate_answer, NOT_AVAILABLE


class WorkflowState(TypedDict):
    question: str
    question_type: str
    retrieved_info: str
    answer: str
    review: str
    final_answer: str


def analyze_question(state: WorkflowState):
    question = state["question"].lower()

    if any(word in question for word in [
        "faculty", "professor", "teacher", "hod"
    ]):
        question_type = "faculty"

    elif any(word in question for word in [
        "calendar", "exam date", "semester date",
        "semester exam", "semester exams",
        "registration", "academic event"
    ]):
        question_type = "calendar"

    elif any(word in question for word in [
        "syllabus", "subject", "course", "semester",
        "credits", "curriculum", "exam"
    ]):
        question_type = "academic"

    else:
        question_type = "general"

    return {"question_type": question_type}


def retrieve_information(state: WorkflowState):
    # The RAG context is fetched in app.py (src.rag.retrieve_context) and
    # passed in through the state; this node forwards it.
    return {"retrieved_info": state["retrieved_info"]}


def generate_response(state: WorkflowState):
    context = (state["retrieved_info"] or "").strip()

    # Nothing relevant retrieved: answer "not available" without calling the
    # LLM (no hallucination, no wasted API call).
    if not context:
        return {"answer": NOT_AVAILABLE}

    answer = generate_answer(state["question"], context)
    return {"answer": answer}


def review_response(state: WorkflowState):
    answer = state["answer"]

    if not answer or not answer.strip():
        review = "Answer is empty."
        final_answer = "I could not generate an answer."
    else:
        review = "Answer is valid."
        final_answer = answer

    return {"review": review, "final_answer": final_answer}


workflow = StateGraph(WorkflowState)

workflow.add_node("analyze", analyze_question)
workflow.add_node("retrieve", retrieve_information)
workflow.add_node("generate", generate_response)
workflow.add_node("review", review_response)

workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "review")
workflow.add_edge("review", END)

app = workflow.compile()