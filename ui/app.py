import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

EXAMPLES = [
    "How many tickets are currently open?",
    "Which agent resolved the most tickets this month?",
    "Show me all Critical tickets not resolved within 12 hours.",
    "What is the average customer rating for Technical category tickets?",
    "Are there any anomalies in resolution times this week?",
]


def ask_question(question: str) -> None:
    if not question:
        st.warning("Enter a question or choose an example.")
        return

    response = requests.post(
        f"{API_URL}/query",
        json={"question": question},
        timeout=45,
    )
    if not response.ok:
        st.error(response.text)
        return

    result = response.json()
    st.subheader(result["answer"])
    if result.get("generated_code"):
        with st.expander("Generated code"):
            st.code(result["generated_code"], language="python")
    st.json(result["raw_result"])


def show_query_panel() -> None:
    question = st.text_input(
        "Question",
        placeholder="How many tickets are currently open?",
    )
    selected = st.selectbox("Examples", [""] + EXAMPLES)
    if st.button("Ask"):
        ask_question(question or selected)


def show_anomaly_panel() -> None:
    st.divider()
    st.subheader("Anomalies")
    days_column, severity_column = st.columns(2)
    days = days_column.number_input(
        "Recent days",
        min_value=1,
        max_value=3650,
        value=7,
    )
    severity = severity_column.selectbox(
        "Severity",
        ["All", "low", "medium", "high", "critical"],
    )

    params = {"days": days}
    if severity != "All":
        params["severity"] = severity

    if st.button("Refresh anomalies"):
        response = requests.get(
            f"{API_URL}/anomalies",
            params=params,
            timeout=15,
        )
        if response.ok:
            st.dataframe(response.json(), use_container_width=True)
        else:
            st.error(response.text)


def main() -> None:
    st.set_page_config(page_title="Ticket Analytics", layout="wide")
    st.title("Ticket Analytics")
    st.caption("Ask a question about the ticket data or check for anomalies.")
    show_query_panel()
    show_anomaly_panel()


if __name__ == "__main__":
    main()
