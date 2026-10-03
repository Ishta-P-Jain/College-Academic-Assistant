from typing import TypedDict


class WorkflowState(TypedDict):
    question: str
    question_type: str
    retrieved_info: str
    answer: str
    review: str
