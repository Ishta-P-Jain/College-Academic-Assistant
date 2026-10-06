# 🎓 College AI Academic Assistant

An AI-powered academic assistant that helps students get answers from college documents and create personalized study plans.

The project combines **RAG, Large Language Models, LangChain, and LangGraph** to provide context-aware academic assistance and manage study-planning requests.

## ✨ Features

* **Academic Q&A** — Answers questions about syllabus, academics, regulations, and college-related information.
* **RAG-based Retrieval** — Retrieves relevant information from college documents before generating an answer.
* **Conversational Follow-ups** — Allows students to ask follow-up questions.
* **Personalized Study Planner** — Creates study plans based on subjects, available study hours, and exam dates.
* **Plan Modification** — Updates an existing plan when the student's requirements change.
* **Unknown-Answer Handling** — Avoids generating unsupported college-specific information when relevant information is unavailable.
* **Simple UI** — Provides an easy-to-use interface through Streamlit.

## 🛠️ Tech Stack

| Technology    | Purpose                  |
| ------------- | ------------------------ |
| Python        | Core development         |
| Streamlit     | User interface           |
| LangChain     | RAG and LLM components   |
| LangGraph     | Workflow orchestration   |
| Google Gemini | Language model           |
| FAISS         | Vector similarity search |

## 📁 Project Structure

```text
academic-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── data/
│   └── college_docs/
│
├── src/
│   ├── rag.py
│   ├── llm.py
│   ├── planner.py
│   └── workflow.py
│
└── tests/
    └── test_app.py
```

## 🔄 How It Works

```text
Student Query
      ↓
Request Type Identification
      ↓
 ┌───────────────┬──────────────────┐
 │ Academic Query│ Study Plan Query  │
 ↓               ↓
RAG Retrieval    Study Planner
 ↓               ↓
Gemini LLM       LangGraph Workflow
 ↓               ↓
       Response
           ↓
     Streamlit UI
```

### Academic Questions

1. The student enters a question.
2. Relevant college documents are searched using FAISS.
3. Retrieved information is provided as context to Gemini.
4. Gemini generates the answer based on the available context.

### Study Planning

1. The student provides subjects, available study hours, and exam dates.
2. The planner generates a personalized schedule.
3. The student can request changes to the plan.
4. The workflow updates the plan based on the new requirements.

## 👥 Team Contributions

The project is divided into four main modules:

### Member 1 — RAG Pipeline

* Document loading and processing
* Text chunking
* Embeddings
* FAISS vector database
* Document retrieval

### Member 2 — LLM Integration

* Gemini integration
* Prompt design
* Conversation handling
* Response generation

### Member 3 — LangGraph & Study Planner

* LangGraph workflow
* Study-plan generation
* Study-plan modification
* Request routing

### Member 4 — UI & Integration

* Streamlit interface
* Module integration
* Testing
* GitHub coordination

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd academic-assistant
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your team's repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Add your actual Gemini API key locally.

**Important:** Never commit `.env` or your API key to GitHub.

### 5. Add college documents

Place relevant documents inside:

```text
data/college_docs/
```

Examples:

* Syllabus
* Academic regulations
* Examination information
* College FAQs
* Department information

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔐 Security

* Keep API keys inside `.env`.
* Add `.env` to `.gitignore`.
* Never commit API keys or other sensitive information to GitHub.
* Do not include private student or college data in the repository.

## 🚀 Future Enhancements

* Calendar integration and study reminders
* Student login and saved study plans
* Support for additional document formats
* Multilingual academic Q&A
* Study progress tracking
* Personalized study recommendations

## 📌 Limitations

The quality of academic answers depends on the accuracy and completeness of the provided college documents.

If relevant information cannot be found, the system should clearly indicate that the information is unavailable instead of generating unsupported college-specific facts.
