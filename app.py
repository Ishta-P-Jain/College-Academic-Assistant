import os
from datetime import date

import streamlit as st
from dotenv import load_dotenv

from src import ingest  # noqa: F401

# override=True: values in .env win over stale variables already set in Windows.
load_dotenv(override=True)

st.set_page_config(page_title="College AI Academic Assistant", layout="centered")

# ---------------------------------------------------------------------------
# Backend imports (fail gracefully so the UI always loads)
# ---------------------------------------------------------------------------
BACKEND_ERROR = None
try:
    from src.workflow import app as workflow_app
except Exception as e:  # missing package, bad path, etc.
    workflow_app = None
    BACKEND_ERROR = str(e)

# RAG module (local embeddings + FAISS). Returns "" when nothing is relevant.
RAG_ERROR = None
try:
    from src.rag import retrieve_context
    RAG_READY = True
except Exception as e:
    retrieve_context = None
    RAG_READY = False
    RAG_ERROR = str(e)


def get_context(question: str, history: list) -> str:
    if not RAG_READY or retrieve_context is None:
        return ""  # workflow answers "not available in the knowledge base"
    try:
        return retrieve_context(question, history)
    except Exception as e:
        st.warning(f"RAG retrieval failed: {e}")
        return ""


def with_history(question: str, history: list, max_turns: int = 3) -> str:
    """Prepend recent turns so follow-up questions make sense."""
    recent = history[-max_turns * 2:]
    if not recent:
        return question
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in recent]
    return "Previous conversation:\n" + "\n".join(lines) + f"\n\nCurrent question: {question}"


def ask_assistant(question: str, history: list) -> str:
    if workflow_app is None:
        return f"Backend not available: {BACKEND_ERROR}"
    result = workflow_app.invoke({
        "question": with_history(question, history),
        "question_type": "",
        "retrieved_info": get_context(question, history),
        "answer": "",
        "review": "",
        "final_answer": "",
    })
    return result["final_answer"]


# ---------------------------------------------------------------------------
# Study plan logic (simple, local, deterministic)
# ---------------------------------------------------------------------------
def build_plan(subjects, hours, exam_date, weak=None):
    weak = weak or []
    days_left = (exam_date - date.today()).days
    if days_left < 1:
        return None
    # Weak subjects appear twice in the rotation so they get more slots.
    rotation = []
    for s in subjects:
        rotation += [s] * (2 if s in weak else 1)

    slots_per_day = max(1, int(hours))
    plan, pointer = [], 0
    for d in range(days_left):
        if d == days_left - 1:
            plan.append({"Day": d + 1, "Focus": "Final revision of " + ", ".join(subjects)})
            continue
        todays = []
        for _ in range(slots_per_day):
            todays.append(rotation[pointer % len(rotation)])
            pointer += 1
        counts = {}
        for s in todays:
            counts[s] = counts.get(s, 0) + 1
        plan.append({
            "Day": d + 1,
            "Focus": ", ".join(f"{s} ({h}h)" for s, h in counts.items()),
        })
    return plan


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("chat", [])
st.session_state.setdefault("plan_inputs", None)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("College AI Academic Assistant")
st.write("Ask academic questions or create a personalized study plan.")

if os.getenv("LLM_PROVIDER", "gemini").lower() == "gemini" and not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Add it to your .env file and restart.")
if BACKEND_ERROR:
    st.error(f"Could not load src/workflow.py: {BACKEND_ERROR}")
if not RAG_READY:
    st.error(f"Could not load src/rag.py: {RAG_ERROR}")

academic_tab, planner_tab = st.tabs(["Academic Assistant", "Study Planner"])

# ----------------------------- Academic tab --------------------------------
with academic_tab:
    st.subheader("Ask an Academic Question")

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    question = st.chat_input("Example: Give me 4th sem syllabus for CSE department.")
    if question and question.strip():
        with st.chat_message("user"):
            st.write(question)
        with st.spinner("Thinking..."):
            try:
                answer = ask_assistant(question, st.session_state.chat)
            except Exception as e:
                answer = f"Error while generating the answer: {e}"
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.chat.append({"role": "user", "content": question})
        st.session_state.chat.append({"role": "assistant", "content": answer})

    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()

# ----------------------------- Planner tab ---------------------------------
with planner_tab:
    st.subheader("Create Your Study Plan")

    with st.form("study_plan_form"):
        subjects = st.text_input(
            "Subjects", placeholder="Example: DAA, DBMS, Computer Networks"
        )
        hours = st.number_input(
            "Available study hours per day", min_value=1, max_value=24, value=3
        )
        exam_date = st.date_input("Next exam date")
        submitted = st.form_submit_button("Create Study Plan")

    if submitted:
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
        if not subject_list:
            st.warning("Please enter at least one subject.")
        elif exam_date <= date.today():
            st.warning("Please pick an exam date in the future.")
        else:
            st.session_state.plan_inputs = {
                "subjects": subject_list,
                "hours": int(hours),
                "exam_date": exam_date,
                "weak": [],
            }

    inputs = st.session_state.plan_inputs
    if inputs:
        plan = build_plan(inputs["subjects"], inputs["hours"], inputs["exam_date"], inputs["weak"])
        if plan is None:
            st.warning("The exam date has passed. Create a new plan.")
        else:
            st.markdown(
                f"**Plan:** {len(plan)} days until {inputs['exam_date']}, "
                f"{inputs['hours']} hours/day"
            )
            st.dataframe(plan, hide_index=True)

            st.subheader("Modify Your Plan")
            with st.form("modify_plan_form"):
                new_subjects = st.text_input("Subjects", value=", ".join(inputs["subjects"]))
                new_hours = st.number_input(
                    "Study hours per day", min_value=1, max_value=24, value=inputs["hours"]
                )
                new_date = st.date_input("Exam date", value=inputs["exam_date"])
                weak = st.multiselect(
                    "Subjects needing extra time",
                    options=inputs["subjects"],
                    default=[w for w in inputs["weak"] if w in inputs["subjects"]],
                )
                updated = st.form_submit_button("Update Plan")

            if updated:
                new_list = [s.strip() for s in new_subjects.split(",") if s.strip()]
                if not new_list:
                    st.warning("Keep at least one subject.")
                elif new_date <= date.today():
                    st.warning("Please pick a future exam date.")
                else:
                    st.session_state.plan_inputs = {
                        "subjects": new_list,
                        "hours": int(new_hours),
                        "exam_date": new_date,
                        "weak": [w for w in weak if w in new_list],
                    }
                    st.rerun()