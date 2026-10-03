import streamlit as st

st.set_page_config(
    page_title="College AI Academic Assistant",
    layout="centered"
)

st.title("College AI Academic Assistant")
st.write("Ask academic questions or create a personalized study plan.")

academic_tab, planner_tab = st.tabs(
    ["Academic Assistant", "Study Planner"]
)

# Academic Question Answering
with academic_tab:
    st.subheader("Ask an Academic Question")

    question = st.text_area(
        "Enter your question",
        placeholder="Example: Give me 4th sem syllabus for CSE department."
    )
    st.button("Ask Question")

    # if st.button("Ask Question"):
    #     if question.strip():
    #         st.info(
    #             "Your question is ready. Connect the RAG and LLM modules "
    #             "to generate an answer."
    #         )
    #     else:
    #         st.warning("Please enter a question.")

# Personalized Study Planner
with planner_tab:
    st.subheader("Create Your Study Plan")

    with st.form("study_plan_form"):
        subjects = st.text_input(
            "Subjects",
            placeholder="Example: DAA, DBMS, Computer Networks"
        )

        hours = st.number_input(
            "Available study hours per day",
            min_value=1,
            max_value=24,
            value=3
        )

        exam_date = st.date_input("Next exam date")

        submitted = st.form_submit_button("Create Study Plan")

    if submitted:
        if subjects.strip():
            st.info(
                f"Subjects: {subjects}\n\n"
                f"Available study time: {hours} hours/day\n\n"
                f"Next exam: {exam_date}\n\n"
                "Connect the study planner and LangGraph modules "
                "to generate your personalized plan."
            )
        else:
            st.warning("Please enter at least one subject.")
            