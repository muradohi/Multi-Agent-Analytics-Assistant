"""
streamlit_app.py — interactive UI for the MULTI-AGENT supervisor.

Ask a question -> the supervisor routes it to the SQL, Pandas, or Direct agent
-> for SQL/Pandas you review the proposed query/code -> it runs and answers.

Run from the PROJECT ROOT:
    pip install streamlit
    streamlit run streamlit_app.py

NOTE: this imports `sup_graph` from agents/supervisor.py. For that import to
work from the project root, supervisor.py's own imports must be package-style
(`from agents.agent_sql_app import ...`, `from agents.agent_pandas_app import ...`),
not bare (`from agent_sql_app import ...`). See the chat notes.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text, inspect
from langchain_core.messages import HumanMessage
from langgraph.types import Command

# ---------------------------------------------------------------------------
# IMPORT THE SUPERVISOR — the compiled multi-agent graph is the entry point now.
# engine is still pulled from the sql app for the schema sidebar + SQL preview.
# ---------------------------------------------------------------------------
from agents import sup_graph, engine

# ---------------------------------------------------------------------------
# PAGE CONFIG + styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Analytics Agent", page_icon="🧭", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .step-box { background:#161b22; border:1px solid #30363d; border-radius:10px;
                padding:16px; margin-bottom:12px; }
    .step-label { color:#58a6ff; font-weight:600; font-size:0.85rem;
                  text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }
    .answer-box { background:#0d2818; border:1px solid #238636; border-radius:10px;
                  padding:20px; }
    .route-box  { background:#161b22; border:1px solid #30363d; border-left:4px solid #58a6ff;
                  border-radius:8px; padding:12px 16px; margin-bottom:14px; }
</style>
""", unsafe_allow_html=True)

st.title("🧭 Multi-Agent Analytics")
st.caption("Ask a question → the supervisor picks the right agent → you review its plan → it runs and answers.")

# How each destination is labelled in the UI.
ROUTE_LABEL = {
    "sql":    "🗃️ SQL agent",
    "pandas": "🐼 Pandas agent",
    "direct": "💬 Direct answer",
}

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
    "Which product category has the highest average review score?",  # sql
    "How many orders were cancelled?",                                # sql
    "Is there a correlation between price and review score?",         # pandas
    "What's the distribution of review scores?",                      # pandas
    "What can you do?",                                               # direct
]

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "phase": "idle",              # idle -> awaiting_approval -> revising -> done
        "thread_id": "ui-session-1",
        "question": "",
        "route": "",                  # sql / pandas / direct
        "reasoning": "",              # why the supervisor chose that route
        "interrupt_payload": {},      # whatever the paused agent sent up
        "artifact": "",               # the proposed SQL or code currently under review
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

def final_answer(result) -> str:
    """The agent's reply is the last message that actually has text
    (a trailing tool-call message can have empty content)."""
    for msg in reversed(result.get("input_text", [])):
        content = getattr(msg, "content", "")
        if content:
            return content
    return "_(no answer returned)_"

def process(result):
    """Take a graph result (from an initial ask OR a resume) and move the UI
    into the right phase. This is the single place that reads the graph output,
    so every path — first ask, approve, revise, skip — funnels through here."""
    # The supervisor writes destination/reasoning; keep them for the badge.
    if result.get("destination"):
        st.session_state.route = result["destination"]
        st.session_state.reasoning = result.get("reasoning", "")

    if "__interrupt__" in result:
        # An agent paused for human review (SQL to approve, or analysis code).
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.phase = "awaiting_approval"
    else:
        st.session_state.answer = final_answer(result)
        st.session_state.phase = "done"

# ---------------------------------------------------------------------------
# SIDEBAR — database schema, example questions, and history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Database")
    st.caption("Questions run against the Olist Brazilian e-commerce dataset.")

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
            st.caption(ROUTE_LABEL.get(item["route"], item["route"]))
            if item["artifact"]:
                lang = "python" if item["route"] == "pandas" else "sql"
                st.code(item["artifact"], language=lang)
            st.write(item["a"])

# ---------------------------------------------------------------------------
# INPUT — ask a question (pre-fills if an example was clicked)
# ---------------------------------------------------------------------------


# If an example was clicked last run, seed the widget's stored value ONCE.
if st.session_state.example_clicked:
    st.session_state.ask_box = st.session_state.example_clicked
    st.session_state.example_clicked = ""

with st.form("ask_form", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        key="ask_box",                 # <- widget value lives in session_state, persists across reruns
        placeholder="e.g. Is there a correlation between price and review score?",
    )
    submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

if submitted and question:
    # Fresh thread per question so each run has its own checkpoint history.
    st.session_state.thread_id = f"ui-{abs(hash(question)) % 10_000}-{len(st.session_state.history)}"
    st.session_state.question = question
    # Reset per-question state.
    st.session_state.route = ""
    st.session_state.reasoning = ""
    st.session_state.interrupt_payload = {}
    st.session_state.artifact = ""
    st.session_state.answer = ""

    with st.spinner("Supervisor is routing your question..."):
        result = sup_graph.invoke(
            {"input_text": [HumanMessage(content=question)]},
            config(),
        )
    st.write("DEBUG destination:", result.get("destination"))
    st.write("DEBUG keys:", list(result.keys()))
    st.write("DEBUG has interrupt:", "__interrupt__" in result)
    st.json({k: str(v)[:300] for k, v in result.items()})
    process(result)

# ---------------------------------------------------------------------------
# ROUTING BADGE — show which agent the supervisor picked, and why.
# ---------------------------------------------------------------------------
if st.session_state.route:
    label = ROUTE_LABEL.get(st.session_state.route, st.session_state.route)
    reason = st.session_state.reasoning
    st.markdown(
        f'<div class="route-box"><b>Routed to {label}</b>'
        + (f'<br><span style="color:#8b949e">{reason}</span>' if reason else "")
        + "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# APPROVAL PHASE — handles BOTH a proposed SQL query and proposed analysis code.
# ---------------------------------------------------------------------------
if st.session_state.phase == "awaiting_approval":
    payload = st.session_state.interrupt_payload

    if "proposed_sql" in payload:
        # ---- SQL agent asked us to approve a query -------------------------
        st.session_state.artifact = payload["proposed_sql"]
        st.markdown('<div class="step-label">Proposed SQL — review before it runs</div>',
                    unsafe_allow_html=True)
        st.code(st.session_state.artifact, language="sql")

        with st.expander("👁️ Preview results (query not yet approved)", expanded=True):
            df, err = run_sql_preview(st.session_state.artifact)
            if err:
                st.error(f"This query errors — fix it via Revise: {err}")
            elif df is not None:
                st.dataframe(df.head(50), use_container_width=True)
                st.caption(f"{len(df)} rows (showing up to 50)")

    elif "proposed_code" in payload:
        # ---- Pandas agent asked us to approve analysis code ----------------
        st.session_state.artifact = payload["proposed_code"]
        if payload.get("qa_report"):
            st.markdown('<div class="step-label">Data-quality report</div>',
                        unsafe_allow_html=True)
            st.text(payload["qa_report"])
        st.markdown('<div class="step-label">Proposed analysis code — review before it runs</div>',
                    unsafe_allow_html=True)
        st.code(st.session_state.artifact, language="python")

    else:
        # ---- Unknown interrupt shape: show it raw so nothing is hidden -----
        st.session_state.artifact = ""
        st.markdown('<div class="step-label">Review</div>', unsafe_allow_html=True)
        st.json(payload)

    st.markdown("**Your decision:**")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("✅ Approve & Run", type="primary", use_container_width=True):
            with st.spinner("Running approved step..."):
                result = sup_graph.invoke(Command(resume={"action": "approve"}), config())
            process(result)   # a follow-up interrupt loops back here automatically
            if st.session_state.phase == "done":
                st.session_state.history.append({
                    "q": st.session_state.question,
                    "route": st.session_state.route,
                    "artifact": st.session_state.artifact,
                    "a": st.session_state.answer,
                })
            st.rerun()

    with c2:
        if st.button("✏️ Revise", use_container_width=True):
            st.session_state.phase = "revising"
            st.rerun()

    with c3:
        if st.button("⏭️ Skip", use_container_width=True):
            result = sup_graph.invoke(Command(resume={"action": "reject"}), config())
            process(result)
            if not st.session_state.answer or st.session_state.answer == "_(no answer returned)_":
                st.session_state.answer = "_Step skipped._"
            st.rerun()

# ---------------------------------------------------------------------------
# REVISE PHASE — send feedback; the agent re-proposes and pauses again.
# ---------------------------------------------------------------------------
if st.session_state.phase == "revising":
    is_code = "proposed_code" in st.session_state.interrupt_payload
    st.markdown('<div class="step-label">Revise the plan</div>', unsafe_allow_html=True)
    st.code(st.session_state.artifact, language="python" if is_code else "sql")

    with st.form("revise_form"):
        notes = st.text_input(
            "What should change?",
            placeholder="e.g. exclude cancelled orders, only 2018 data, use a Spearman correlation...",
        )
        send = st.form_submit_button("Send feedback & re-propose", type="primary")

    if send and notes:
        with st.spinner("Agent is revising..."):
            result = sup_graph.invoke(
                Command(resume={"action": "revise", "notes": notes}), config())
        process(result)
        st.rerun()

# ---------------------------------------------------------------------------
# DONE PHASE
# ---------------------------------------------------------------------------
if st.session_state.phase == "done":
    st.markdown('<div class="step-label">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{st.session_state.answer}</div>',
                unsafe_allow_html=True)

    if st.session_state.artifact:
        lang = "python" if st.session_state.route == "pandas" else "sql"
        label = "the code that ran" if lang == "python" else "the SQL that ran"
        with st.expander(f"View {label}"):
            st.code(st.session_state.artifact, language=lang)

    if st.button("Ask another question"):
        st.session_state.phase = "idle"
        st.session_state.route = ""
        st.session_state.reasoning = ""
        st.session_state.artifact = ""
        st.session_state.answer = ""
        st.rerun()