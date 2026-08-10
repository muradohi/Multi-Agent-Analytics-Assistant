import os
import shutil
import time
from datetime import datetime
import json

import streamlit as st
import pandas as pd
from sqlalchemy import text, inspect
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain_community.callbacks import get_openai_callback

from agents import sup_graph, engine

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Olist Analytics", page_icon="🧭", layout="wide")

AGENT = {
    "sql":    {"label": "SQL",     "icon": "🗃️", "color": "#4C9AFF"},
    "pandas": {"label": "Pandas",  "icon": "🐼", "color": "#F2A93B"},
    "viz":    {"label": "Charts",  "icon": "📊", "color": "#336D62"},
    "direct": {"label": "Direct",  "icon": "💬", "color": "#B392F0"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp { background: #0c0e1a; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* headings a touch tighter */
h1, h2, h3 { letter-spacing: -0.01em; }

/* code + dataframes in mono */
code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

/* the agent pill */
.agent-pill {
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    padding: 3px 10px; border-radius: 999px; margin-bottom: 8px;
    letter-spacing: 0.02em; text-transform: uppercase;
}
.reason { color: #8b90a8; font-size: 0.86rem; margin: 2px 0 10px; }

/* chat bubbles: subtle panel, rounded */
[data-testid="stChatMessage"] {
    background: #12152400; border-radius: 14px; padding: 4px 2px;
}

/* sidebar chat buttons: left-aligned, quiet */
section[data-testid="stSidebar"] .stButton button {
    text-align: left; justify-content: flex-start;
    background: transparent; border: 1px solid transparent;
    color: #c7cbe0; font-weight: 500;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #171a2b; border-color: #262b45;
}

/* empty-state hero */
.hero { text-align: center; padding: 42px 0 10px; }
.hero h2 { font-size: 1.7rem; margin-bottom: 6px; }
.hero p  { color: #8b90a8; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

TABLE_DESCRIPTIONS = {
    "orders":        "One row per order — status and timestamps.",
    "order_items":   "Items in each order — links orders to products, has price.",
    "products":      "Product details and category (categories are Portuguese).",
    "customers":     "Customer location info.",
    "order_reviews": "Review scores (1–5) and review text per order.",
}

EXAMPLES = [
    "Which product category has the highest average review score?",
    "How many orders were cancelled?",
    "Is there a correlation between price and review score?",
    "What's the distribution of review scores?",
    "Plot the top 10 product categories by number of orders",
    "What can you do?",
]

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "phase": "idle",
        "thread_id": "ui-conv-1",
        "question": "",
        "route": "",
        "reasoning": "",
        "interrupt_payload": {},
        "artifact": "",
        "answer": "",
        "conversations": {"conv-1": {"title": "New chat", "messages": [], "thread_id": "ui-conv-1"}},
        "current_conv_id": "conv-1",
        "pending_prompt": None,
        "debug": False,
        "last_debug": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()


def cfg():
    return {"configurable": {"thread_id": st.session_state.thread_id},
            "recursion_limit": 35}

@st.cache_data(show_spinner=False)
def get_schema():
    inspector = inspect(engine)
    return {t: [(c["name"], str(c["type"])) for c in inspector.get_columns(t)]
            for t in inspector.get_table_names()}

def run_sql_preview(sql: str):
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn), None
    except Exception as e:
        return None, str(e)

def final_answer(result) -> str:
    for msg in reversed(result.get("input_text", [])):
        content = getattr(msg, "content", "") or ""
        if not content:
            continue
        s = content.lstrip()
        # skip the model's raw tool-call / code echoes — not real answers
        if s.startswith(("propose_final_query", "propose_analysis_code",
                         "propose_chart_code", "result =", "result=", "{")):
            continue
        return content
    return "_(no answer returned)_"

def pill(route: str) -> str:
    a = AGENT.get(route)
    if not a:
        return ""
    c = a["color"]
    return (f'<span class="agent-pill" '
            f'style="background:{c}1f;color:{c};border:1px solid {c}55">'
            f'{a["icon"]} {a["label"]} agent</span>')

def start_new_chat():
    new_id = f"conv-{int(time.time())}"
    st.session_state.conversations[new_id] = {
        "title": "New chat", "messages": [], "thread_id": f"ui-{new_id}",
    }
    st.session_state.current_conv_id = new_id
    st.session_state.thread_id = f"ui-{new_id}"
    for k in ("phase", "route", "reasoning", "artifact", "answer"):
        st.session_state[k] = "idle" if k == "phase" else ""

def delete_conversation(conv_id):
    st.session_state.conversations.pop(conv_id, None)
    if not st.session_state.conversations:
        start_new_chat()
        return
    if st.session_state.current_conv_id == conv_id:
        fallback = next(iter(st.session_state.conversations))
        st.session_state.current_conv_id = fallback
        st.session_state.thread_id = st.session_state.conversations[fallback]["thread_id"]
        for k in ("phase", "route", "artifact", "answer"):
            st.session_state[k] = "viewing" if k == "phase" else ""

def save_turn():
    conv = st.session_state.conversations[st.session_state.current_conv_id]
    msg = {
        "q": st.session_state.question,
        "a": st.session_state.answer,
        "route": st.session_state.route,
        "artifact": st.session_state.artifact,
        "chart": None,
    }
    # For a viz turn, snapshot the chart to a per-turn file so history survives.
    if st.session_state.route == "viz" and os.path.exists("chart_output.png"):
        os.makedirs("charts", exist_ok=True)
        dest = f"charts/{st.session_state.current_conv_id}-{len(conv['messages'])}.png"
        shutil.copyfile("chart_output.png", dest)
        msg["chart"] = dest
    conv["messages"].append(msg)
    if conv["title"] == "New chat" and st.session_state.question:
        conv["title"] = st.session_state.question[:38]

def process(result, skipped=False):
    if result.get("destination"):
        st.session_state.route = result["destination"]
        st.session_state.reasoning = result.get("reasoning", "")
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.phase = "awaiting_approval"
    else:
        ans = final_answer(result)
        if skipped and (not ans or ans == "_(no answer returned)_"):
            ans = "_Step skipped._"
        st.session_state.answer = ans
        st.session_state.phase = "done"
        save_turn()
        
def show_debug(result, stage: str):
    if not st.session_state.get("debug"):
        return
    interrupted = "__interrupt__" in result
    msgs = result.get("input_text", [])
    lines = []
    for m in msgs:
        kind = type(m).__name__
        tc = getattr(m, "tool_calls", None)
        if tc:
            lines.append(f"`{kind}` → 🔧 {', '.join(c['name'] for c in tc)}")
        else:
            content = (getattr(m, "content", "") or "")[:200]
            lines.append(f"`{kind}` → {content or '_(empty)_'}")
    st.session_state.last_debug = {
        "stage": stage,
        "destination": result.get("destination"),
        "interrupted": interrupted,
        "phase": st.session_state.phase,
        "keys": list(result.keys()),
        "payload": result["__interrupt__"][0].value if interrupted else None,
        "messages": lines,
    }
    
def log_run(question, result, duration, cb):
    row = {
        "ts": datetime.now().isoformat(),
        "thread_id": st.session_state.thread_id,
        "question": question,
        "route": result.get("destination"),
        "interrupted": "__interrupt__" in result,
        "duration_s": round(duration, 2),
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "n_llm_calls": cb.successful_requests,
        "cost_usd": round(cb.total_cost, 4),
    }
    with open("runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

def ask(prompt: str):
    st.session_state.question = prompt
    for k in ("route", "reasoning", "artifact", "answer"):
        st.session_state[k] = ""
    st.session_state.interrupt_payload = {}
    t0 = time.time()
    with st.spinner("Routing your question…"):
        with get_openai_callback() as cb:
            result = sup_graph.invoke({"input_text": [HumanMessage(content=prompt)]}, cfg())
    duration = time.time() - t0
    process(result)
    log_run(prompt, result, duration, cb)
    show_debug(result, "after ask") 

def artifact_lang() -> str:
    return "python" if st.session_state.route in ("pandas", "viz") else "sql"


with st.sidebar:
    st.markdown("### 🧭 Olist Analytics")
    st.markdown(f"- **thread_id:** `{st.session_state.thread_id}`")
    if st.button("＋  New chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.markdown("###### Conversations")
    st.session_state.debug = st.toggle("🔍 Debug mode", value=st.session_state.debug)
    for conv_id, conv in reversed(list(st.session_state.conversations.items())):
        active = conv_id == st.session_state.current_conv_id
        col_open, col_del = st.columns([6, 1])
        with col_open:
            label = ("●  " if active else "　") + conv["title"]
            if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
                st.session_state.current_conv_id = conv_id
                st.session_state.thread_id = conv["thread_id"]
                for k in ("phase", "route", "artifact", "answer"):
                    st.session_state[k] = "viewing" if k == "phase" else ""
                st.rerun()
        with col_del:
            if st.button("🗑", key=f"del_{conv_id}", use_container_width=True):
                delete_conversation(conv_id)
                st.rerun()

    st.divider()
    with st.expander("📊  Database schema"):
        st.caption("Olist Brazilian e-commerce dataset.")
        for table, columns in get_schema().items():
            st.markdown(f"**{table}** — {TABLE_DESCRIPTIONS.get(table, '')}")
            st.dataframe(pd.DataFrame(columns, columns=["column", "type"]),
                         hide_index=True, use_container_width=True)


current = st.session_state.conversations[st.session_state.current_conv_id]


if not current["messages"] and st.session_state.phase in ("idle", "viewing"):
    st.markdown(
        '<div class="hero"><h2>Ask anything about the Olist data</h2>'
        '<p>A supervisor routes your question to the right specialist — '
        'SQL, Pandas, Charts, or a direct answer — and pauses for your review before running anything.</p></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_prompt = ex
            st.rerun()

for msg in current["messages"]:
    with st.chat_message("user"):
        st.markdown(msg["q"])
    with st.chat_message("assistant"):
        st.markdown(pill(msg["route"]), unsafe_allow_html=True)
        st.markdown(msg["a"])
        if msg.get("chart") and os.path.exists(msg["chart"]):
            st.image(msg["chart"], use_container_width=True)
        if msg["artifact"]:
            lang = "python" if msg["route"] in ("pandas", "viz") else "sql"
            with st.expander(f"View {'code' if lang == 'python' else 'SQL'}"):
                st.code(msg["artifact"], language=lang)


if st.session_state.phase in ("awaiting_approval", "revising"):
    with st.chat_message("user"):
        st.markdown(st.session_state.question)

    with st.chat_message("assistant"):
        if st.session_state.route:
            st.markdown(pill(st.session_state.route), unsafe_allow_html=True)
        if st.session_state.reasoning:
            st.markdown(f'<div class="reason">{st.session_state.reasoning}</div>',
                        unsafe_allow_html=True)

        payload = st.session_state.interrupt_payload

       
        if st.session_state.phase == "awaiting_approval":
            if "proposed_sql" in payload:
                st.session_state.artifact = payload["proposed_sql"]
                st.markdown("**Proposed SQL** — review before it runs")
                st.code(payload["proposed_sql"], language="sql")
                with st.expander("👁 Preview results (not yet approved)"):
                    df, err = run_sql_preview(payload["proposed_sql"])
                    if err:
                        st.error(f"This query errors — fix it via Revise: {err}")
                    elif df is not None:
                        st.dataframe(df.head(50), use_container_width=True)
                        st.caption(f"{len(df)} rows (showing up to 50)")

            elif "proposed_code" in payload:
                st.session_state.artifact = payload["proposed_code"]
                if payload.get("qa_report"):
                    with st.expander("Data-quality report"):
                        st.text(payload["qa_report"])
                st.markdown("**Proposed analysis code** — review before it runs")
                st.code(payload["proposed_code"], language="python")

            elif "chart_code" in payload:
                st.session_state.artifact = payload["chart_code"]
                st.markdown("**Proposed chart code** — review before it runs")
                st.code(payload["chart_code"], language="python")

            else:
                st.session_state.artifact = ""
                st.markdown("**Review**")
                st.json(payload)

            c1, c2, c3 = st.columns(3)
            if c1.button("✅ Approve & run", type="primary", use_container_width=True):
                with st.spinner("Running…"):
                    result = sup_graph.invoke(Command(resume={"action": "approve"}), cfg())
                process(result)
                show_debug(result, "after approve") 
                st.rerun()
            if c2.button("✏️ Revise", use_container_width=True):
                
                st.session_state.phase = "revising"
                st.rerun()
            if c3.button("⏭️ Skip", use_container_width=True):
                result = sup_graph.invoke(Command(resume={"action": "reject"}), cfg())
                process(result, skipped=True)
                show_debug(result, "after skip") 
                st.rerun()

        else:
            st.markdown("**Revise the plan** — tell the agent what to change")
            st.code(st.session_state.artifact, language=artifact_lang())
            with st.form("revise_form", clear_on_submit=True):
                notes = st.text_input(
                    "What should change?",
                    placeholder="e.g. exclude cancelled orders, use a Spearman correlation…",
                )
                send = st.form_submit_button("Send feedback & re-propose", type="primary")
            if send and notes:
                with st.spinner("Revising…"):
                    result = sup_graph.invoke(
                        Command(resume={"action": "revise", "notes": notes}), cfg()
                    )
                process(result)
                show_debug(result, "after revise")
                st.rerun()

busy = st.session_state.phase in ("awaiting_approval", "revising")
typed = st.chat_input(
    "Finish the review above first…" if busy else "Ask about the Olist data…",
    disabled=busy,
)

prompt = typed or st.session_state.pending_prompt
st.session_state.pending_prompt = None
if prompt and st.session_state.phase in ("idle", "done", "viewing"):
    ask(prompt)
    st.rerun()
    

if st.session_state.debug and st.session_state.get("last_debug"):
    d = st.session_state.last_debug
    with st.expander(f"🔍 debug — {d['stage']}", expanded=True):
        st.markdown(
            f"- **destination:** `{d['destination']}`  \n"
            f"- **paused for review?** `{d['interrupted']}`  \n"
            f"- **phase now:** `{d['phase']}`  \n"
            f"- **state keys:** `{d['keys']}`"
        )
        if d["payload"]:
            st.caption("interrupt payload:")
            st.json(d["payload"])
        st.caption(f"message history ({len(d['messages'])} messages):")
        for line in d["messages"]:
            st.markdown(line)