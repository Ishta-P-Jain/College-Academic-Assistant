from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class WorkflowState(TypedDict):
    question: str
    question_type: str
    retrieved_info: str
    answer: str
    review: str


def analyze_question(state: WorkflowState):
    question = state["question"]

    if "study" in question.lower() or "syllabus" in question.lower():
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
    # This will later connect to Member 2's LLM module.
    return {
        "answer": "Response generation will be connected here."
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
