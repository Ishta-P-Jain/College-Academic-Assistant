# College AI Academic Assistant

## Project Overview

The **College AI Academic Assistant** is an AI-powered application designed to help students access college-related academic information and create personalized study plans.

It uses Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), LangChain, and LangGraph to answer questions using college documents and manage study-planning requests.

## Objectives

* Answer questions about college academics, syllabus, regulations.
* Retrieve relevant information from college documents using RAG.
* Generate context-aware answers using an LLM.
* Create personalized study plans based on subjects, available study hours, and exam dates.
* Allow students to modify their study plans through follow-up requests.
* Provide a simple and user-friendly interface.

## Key Features

* **Academic Question Answering:** Answers questions using available college documents.
* **Document Retrieval (RAG):** Retrieves relevant information before generating answers.
* **AI Conversation:** Supports follow-up questions.
* **Personalized Study Planner:** Creates study plans based on student requirements.
* **Plan Modification:** Updates plans when the student's schedule or requirements change.
* **Unknown-Answer Handling:** Indicates when sufficient information is unavailable.

## Technologies Used

* **Python** — Core programming language.
* **Streamlit** — User interface.
* **LangChain** — LLM integration and RAG components.
* **LangGraph** — Workflow orchestration.
* **Google Gemini** — Language model.
* **FAISS** — Vector similarity search.

## Project Structure

```text
academic-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
├── data/
│   └── college_docs/
├── src/
│   ├── rag.py
│   ├── llm.py
│   ├── planner.py
│   └── workflow.py
└── tests/
    └── test_app.py
```

## How It Works

1. The student enters an academic question or study-planning request.
2. The application identifies the type of request.
3. For academic questions, the RAG pipeline retrieves relevant college information.
4. The LLM generates an answer using the retrieved context.
5. For study-planning requests, the planner creates or modifies a personalized schedule.
6. The response is displayed through the Streamlit interface.

## Team Contributions

The project is divided into four main modules:

* **Member 1 — RAG Pipeline:** Document processing, embeddings, vector database, and retrieval.
* **Member 2 — LLM Integration:** Model configuration, prompt engineering, and conversation history.
* **Member 3 — LangGraph and Study Planner:** Workflow design, study-plan generation, and plan modifications.
* **Member 4 — UI and Integration:** Streamlit interface, GitHub coordination, module integration, and testing.


## Future Enhancements

* Calendar integration for study reminders.
* Student login and saved study plans.
* Support for additional document formats.
* Multilingual question answering.
* Progress tracking and study-plan recommendations.

## 📌 Important Note

The quality of academic answers depends on the accuracy and completeness of the provided college documents. The system should clearly communicate when relevant information cannot be found rather than inventing college-specific facts.

