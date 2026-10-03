from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from src.llm import generate_answer


class WorkflowState(TypedDict):
    question: str
    question_type: str
    retrieved_info: str
    answer: str
    review: str


def analyze_question(state: WorkflowState):
    question = state["question"].lower()

    if any(word in question for word in [
        "faculty", "professor", "teacher", "hod"
    ]):
        question_type = "faculty"

    elif any(word in question for word in [
        "calendar", "exam date", "semester date",
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

    return {
        "question_type": question_type
    }


def retrieve_information(state: WorkflowState):
    # This will later connect to Member 1's RAG module.
    return {
        "retrieved_info": ""
    }


def generate_response(state: WorkflowState):
    answer = generate_answer(
        state["question"],
        state["retrieved_info"]
    )

    return {
        "answer": answer
    }


def review_response(state: WorkflowState):
    # This will later check the generated response.
    return {
        "review": "Response reviewed."
    }


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
