"""
streamlit_app.py — interactive UI for the HITL SQL agent.

Run:
    pip install streamlit
    streamlit run streamlit_app.py

This imports your compiled `sql_agent` graph. Adjust the import at the top to
match your file. Everything else works with your existing graph unchanged.
"""
from dotenv import load_dotenv
from sqlalchemy import text, create_engine, URL, inspect
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from pydantic import Field, BaseModel
from typing import List, Annotated
from langchain.tools import tool
import os
from langgraph.graph import StateGraph, START, END
import yaml
from pathlib import Path

import streamlit as st
import pandas as pd
from sqlalchemy import text
from langchain_core.messages import HumanMessage
from langgraph.types import Command

# ---------------------------------------------------------------------------
# IMPORT YOUR AGENT
# Change this line to match your file/module. You need: the compiled graph
# (sql_agent) and the engine (to render results as a table).
# ---------------------------------------------------------------------------
from agents.agent_sql_app import sql_agent, engine

# ---------------------------------------------------------------------------
# PAGE CONFIG + a little styling
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
# SESSION STATE — this is what survives Streamlit's reruns.
#   phase: which stage of the flow we're in
#   thread_id: the graph's session id (same one used to resume)
#   proposed_sql / result / answer: what we show at each phase
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "phase": "idle",          # idle -> awaiting_approval -> done
        "thread_id": "ui-session-1",
        "question": "",
        "proposed_sql": "",
        "answer": "",
        "history": [],            # list of past Q&A for display
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()

def config():
    return {"configurable": {"thread_id": st.session_state.thread_id},
            "recursion_limit": 35}

def run_sql_preview(sql: str):
    """Run the proposed SQL just to PREVIEW results as a dataframe in the UI.
    (Read-only; your agent's guard already restricts to SELECT.)"""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df, None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------------------------
# INPUT — ask a question
# ---------------------------------------------------------------------------
with st.form("ask_form", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        placeholder="e.g. Which product category has the highest average review score?",
    )
    submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

if submitted and question:
    # fresh thread per question so sessions don't collide
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
        # no interrupt (agent answered directly, rare)
        st.session_state.answer = result["input_text"][-1].content
        st.session_state.phase = "done"

# ---------------------------------------------------------------------------
# APPROVAL PHASE — show the proposed SQL + preview + approve/revise/skip
# ---------------------------------------------------------------------------
if st.session_state.phase == "awaiting_approval":
    sql = st.session_state.proposed_sql

    st.markdown('<div class="step-label">Proposed SQL — review before it runs</div>',
                unsafe_allow_html=True)
    st.code(sql, language="sql")

    # live preview of what the query WOULD return (so the human can judge it)
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
            # after approve, graph runs execute+answer and finishes
            while "__interrupt__" in result:      # safety: shouldn't loop, but just in case
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
# REVISE PHASE — collect feedback, resume with revise, pause again
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
            st.session_state.phase = "awaiting_approval"     # pause again with new SQL
        else:
            st.session_state.answer = result["input_text"][-1].content
            st.session_state.phase = "done"
        st.rerun()

# ---------------------------------------------------------------------------
# DONE PHASE — show the final answer + the SQL that ran
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

# ---------------------------------------------------------------------------
# SIDEBAR — history of past questions this session
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📜 History")
    if not st.session_state.history:
        st.caption("No questions yet.")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"{item['q'][:40]}..."):
            st.code(item["sql"], language="sql")
            st.write(item["a"])