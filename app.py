"""
streamlit_app.py — interactive UI for the HITL SQL agent.

Run:
    pip install streamlit
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text, inspect
from langchain_core.messages import HumanMessage
from langgraph.types import Command

# ---------------------------------------------------------------------------
# IMPORT YOUR AGENT — needs the compiled graph (sql_agent) and the engine.
# ---------------------------------------------------------------------------
from agents.agent_sql_app import sql_agent, engine

# ---------------------------------------------------------------------------
# PAGE CONFIG + styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SQL Agent", page_icon="🗃️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .step-box { background:#161b22; border:1px solid #30363d; border-radius:10px;
                padding:16px; margin-bottom:12px; }
    .step-label { color:#58a6ff; font-weight:600; font-size:0.85rem;
                  text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }
    .answer-box { background:#0d2818; border:1px solid #238636; border-radius:10px;
                  padding:20px; }
</style>
""", unsafe_allow_html=True)

st.title("🗃️ SQL Analytics Agent")
st.caption("Ask a question → the agent writes SQL → you review it → it runs and answers.")

# ---------------------------------------------------------------------------
# SCHEMA + EXAMPLES — shown in the sidebar so people know what they can ask.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_schema():
    """Return {table: [(column, type), ...]} read from the live database."""
    inspector = inspect(engine)
    schema = {}
    for table in inspector.get_table_names():
        cols = inspector.get_columns(table)
        schema[table] = [(c["name"], str(c["type"])) for c in cols]
    return schema

TABLE_DESCRIPTIONS = {
    "orders":        "One row per order — status and timestamps.",
    "order_items":   "Items in each order — links orders to products, has price.",
    "products":      "Product details and category (categories are in Portuguese).",
    "customers":     "Customer location info.",
    "order_reviews": "Review scores (1–5) and review text per order.",
}

EXAMPLE_QUESTIONS = [
    "Which product category has the highest average review score?",
    "What are the top 5 product categories by number of orders?",
    "How many orders were cancelled?",
    "What percentage of reviews are 5 stars?",
    "Which customer state has the most orders?",
]

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "phase": "idle",              # idle -> awaiting_approval -> revising -> done
        "thread_id": "ui-session-1",
        "question": "",
        "proposed_sql": "",
        "answer": "",
        "history": [],
        "example_clicked": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()

def config():
    return {"configurable": {"thread_id": st.session_state.thread_id},
            "recursion_limit": 35}

def run_sql_preview(sql: str):
    """Run the proposed SQL read-only to PREVIEW results as a dataframe."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df, None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------------------------
# SIDEBAR — database schema, example questions, and history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Database")
    st.caption("This agent queries the Olist Brazilian e-commerce dataset.")

    st.subheader("Tables")
    schema = get_schema()
    for table, columns in schema.items():
        with st.expander(f"🗂️ {table}"):
            desc = TABLE_DESCRIPTIONS.get(table, "")
            if desc:
                st.caption(desc)
            st.dataframe(
                pd.DataFrame(columns, columns=["column", "type"]),
                hide_index=True, use_container_width=True,
            )

    st.divider()

    st.subheader("💡 Try asking")
    st.caption("Click one to fill it in:")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state.example_clicked = q
            st.rerun()

    st.divider()

    st.subheader("📜 History")
    if not st.session_state.history:
        st.caption("No questions yet.")
    for item in reversed(st.session_state.history):
        with st.expander(f"{item['q'][:40]}..."):
            st.code(item["sql"], language="sql")
            st.write(item["a"])

# ---------------------------------------------------------------------------
# INPUT — ask a question (pre-fills if an example was clicked)
# ---------------------------------------------------------------------------
default_question = st.session_state.pop("example_clicked", "") or ""

with st.form("ask_form", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        value=default_question,
        placeholder="e.g. Which product category has the highest average review score?",
    )
    submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

if submitted and question:
    st.session_state.thread_id = f"ui-{abs(hash(question)) % 10_000}-{len(st.session_state.history)}"
    st.session_state.question = question
    st.session_state.phase = "running"

    with st.spinner("Agent is exploring the schema and composing a query..."):
        result = sql_agent.invoke(
            {"input_text": [HumanMessage(content=question)]},
            config(),
        )

    if "__interrupt__" in result:
        st.session_state.proposed_sql = result["__interrupt__"][0].value["proposed_sql"]
        st.session_state.phase = "awaiting_approval"
    else:
        st.session_state.answer = result["input_text"][-1].content
        st.session_state.phase = "done"

# ---------------------------------------------------------------------------
# APPROVAL PHASE
# ---------------------------------------------------------------------------
if st.session_state.phase == "awaiting_approval":
    sql = st.session_state.proposed_sql

    st.markdown('<div class="step-label">Proposed SQL — review before it runs</div>',
                unsafe_allow_html=True)
    st.code(sql, language="sql")

    with st.expander("👁️ Preview results (query not yet approved)", expanded=True):
        df, err = run_sql_preview(sql)
        if err:
            st.error(f"This query errors: {err}")
        elif df is not None:
            st.dataframe(df.head(50), use_container_width=True)
            st.caption(f"{len(df)} rows (showing up to 50)")

    st.markdown("**Your decision:**")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("✅ Approve & Run", type="primary", use_container_width=True):
            with st.spinner("Running approved query..."):
                result = sql_agent.invoke(Command(resume={"action": "approve"}), config())
            while "__interrupt__" in result:
                result = sql_agent.invoke(Command(resume={"action": "approve"}), config())
            st.session_state.answer = result["input_text"][-1].content
            st.session_state.history.append(
                {"q": st.session_state.question, "sql": sql, "a": st.session_state.answer})
            st.session_state.phase = "done"
            st.rerun()

    with c2:
        if st.button("✏️ Revise", use_container_width=True):
            st.session_state.phase = "revising"
            st.rerun()

    with c3:
        if st.button("⏭️ Skip", use_container_width=True):
            result = sql_agent.invoke(Command(resume={"action": "reject"}), config())
            st.session_state.answer = "_Query skipped._"
            st.session_state.phase = "done"
            st.rerun()

# ---------------------------------------------------------------------------
# REVISE PHASE
# ---------------------------------------------------------------------------
if st.session_state.phase == "revising":
    st.markdown('<div class="step-label">Revise the query</div>', unsafe_allow_html=True)
    st.code(st.session_state.proposed_sql, language="sql")

    with st.form("revise_form"):
        notes = st.text_input("What should change?",
                              placeholder="e.g. exclude cancelled orders, only 2018 data...")
        send = st.form_submit_button("Send feedback & re-propose", type="primary")

    if send and notes:
        with st.spinner("Agent is revising the query..."):
            result = sql_agent.invoke(
                Command(resume={"action": "revise", "notes": notes}), config())
        if "__interrupt__" in result:
            st.session_state.proposed_sql = result["__interrupt__"][0].value["proposed_sql"]
            st.session_state.phase = "awaiting_approval"
        else:
            st.session_state.answer = result["input_text"][-1].content
            st.session_state.phase = "done"
        st.rerun()

# ---------------------------------------------------------------------------
# DONE PHASE
# ---------------------------------------------------------------------------
if st.session_state.phase == "done":
    st.markdown('<div class="step-label">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{st.session_state.answer}</div>',
                unsafe_allow_html=True)

    if st.session_state.proposed_sql:
        with st.expander("View the SQL that ran"):
            st.code(st.session_state.proposed_sql, language="sql")

    if st.button("Ask another question"):
        st.session_state.phase = "idle"
        st.session_state.proposed_sql = ""
        st.session_state.answer = ""
        st.rerun()