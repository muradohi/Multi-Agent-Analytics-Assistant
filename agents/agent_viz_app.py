import os, io, contextlib
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from sqlalchemy import text
import pandas as pd
import scipy
import numpy as np
from langchain.tools import tool
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly
import seaborn as sns

from agents.agent_sql_app import engine, llm, list_tables, schema_tables

load_dotenv()

class VizState(BaseModel):
    input_text: Annotated[list[AnyMessage], add_messages]
    fetch_sql: str = ""
    query_proposed_viz: bool = False
    data_summary: str = ""
    chart_code: str = ""
    chart_path: str = ""
    action: str = ""
    
    
@tool
def propose_fetch_sql(sql: str) -> str:
    """Propose the SELECT that fetches the data to analyze. Call ONCE after
    inspecting the schema. Pull only the needed columns, not SELECT *."""
    return "Fetch query recorded."


tools = [list_tables, schema_tables, propose_fetch_sql]
llm_with_tools = llm.bind_tools(tools)


SYSTEM = SystemMessage(content=(
    "You are a data analyst preparing to analyze e-commerce data with matplotlib. "
    "First inspect the schema (list_tables, schema_tables). Then call "
    "propose_fetch_sql with a SELECT pulling ONLY the needed columns. "
    "Do NOT ask the user questions or explain your plan in prose — "
    "immediately call the tools. "
    "Write SQLite-compatible SQL. Category names are Portuguese."
))


def current_turn_messages(state):
    """Messages from the latest human turn onward — this question's own
    exploration only, not previous questions' tool calls."""
    msgs = state.input_text
    # find the index of the last HumanMessage
    last_human = max(
        (i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)),
        default=0,
    )
    return msgs[last_human:]


def fetch_llm_node(state: VizState):
    
    system_msg = SYSTEM
    user_msg = current_turn_messages(state)
    
    promt = [system_msg , *user_msg]
    llm_respopnse = llm_with_tools.invoke(promt)
    
    return{"input_text": [llm_respopnse], "query_proposed_viz": False, "fetch_sql": ""}

def fetch_tool_node(state: VizState):
    
    tool_name = {t.name: t for t in tools}
    last_msg = state.input_text[-1]
    out = []
    updates = {}
    proposed = False
    
    for call in last_msg.tool_calls:
        if call["name"] == "propose_fetch_sql":
            proposed = True
            updates["fetch_sql"] = call["args"]["sql"]
            out.append(ToolMessage(content= "sql query fetched", tool_call_id = call["id"]))
            
            
        else:
            res = tool_name[call["name"]].invoke(call["args"])
            out.append(ToolMessage(content=str(res), tool_call_id = call["id"]))
            
    updates["input_text"] = out
    updates["query_proposed_viz"] = proposed
    
    return updates


def load_df(fetch_sql: str) -> pd.DataFrame:
    
    
    with engine.connect() as conn:
        res = pd.read_sql(text(fetch_sql),conn)
    
    return res

@tool
def propose_chart_code(code: str):
    
    """Propose only matplotlib code to chart the data. The DataFrame is available as
    `df`. Save the figure to 'chart_output.png'. Pick an appropriate chart type
    (scatter for correlation, bar for categories, line for time series,
    histogram for distributions). Call ONCE; not run yet."""
    
    return "Chart code proposed."

tools_chart = [propose_chart_code]
llm_with_viz_tool = llm.bind_tools(tools_chart)

def propose_chart_node(state: VizState) -> dict:
    system = SystemMessage(content=(
        "Write matplotlib code to chart the data and answer the question. "
        "The DataFrame is in `df`; pd, np, plt available. "
        "Save the figure to 'chart_output.png'. Pick an appropriate chart type.\n\n"
        f"Question: {latest_question(state)}\n"
        f"Data available (use these exact column names):\n{state.data_summary}"
    ))
    return {"input_text": [llm_with_viz_tool.invoke([system])]}

def latest_question(state) -> str:
    for msg in reversed(state.input_text):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def fetch_chart_node(state: VizState):
    
    tool_name = {t.name: t for t in tools_chart}
    last_msg = state.input_text[-1]
    out = []
    updates = {}
    
    for call in last_msg.tool_calls:
        if call["name"] == "propose_chart_code":
            updates["chart_code"] = call["args"]["code"]
            out.append(ToolMessage(content= "chart code fetched", tool_call_id = call["id"]))
            
            
        else:
            res = tool_name[call["name"]].invoke(call["args"])
            out.append(ToolMessage(content=str(res), tool_call_id = call["id"]))
            
    updates["input_text"] = out
    
    return updates




def approval_node(state: VizState) -> dict:
    decision = interrupt({
        "chart_code": state.chart_code,
        "data_summary": state.data_summary,      # <-- now it's in the payload
        "instructions": "action: approve / reject / revise (+notes)",
    })
    action = decision["action"]
    if action == "revise":
        notes = decision.get("notes", "")
        return {
            "input_text": [HumanMessage(content=(
                f"Revise the chart. Feedback: {notes}, and the code you wrote {state.chart_code}, Call propose_chart_code "
                f"again with corrected code. Do NOT answer in prose."))],
            "chart_code": "",
            "action": "revise",
        }
    if action == "reject":
        return {"action": "rejected"}
    return {"action": "approved"}

def approval_router(state: VizState) -> str:
    if state.action == "revise":   return "propose_chart"
    if state.action == "rejected": return "end"
    return "execute"


def execute_chart_node(state: VizState) -> dict:
    df = load_df(state.fetch_sql)
    namespace = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
    try:
        exec(state.chart_code, namespace)
        plt.close("all")                      # free memory
        return {"chart_path": "chart_output.png"}
    except Exception as e:
        return {"chart_path": f"error: {e}"}
    
def data_summary_node(state: VizState) -> dict:
    df = load_df(state.fetch_sql)
    summary = (
    f"Rows: {len(df)}, Columns: {list(df.columns)}\n"
    f"Sample:\n{df.head(3).to_string()}\n"
    f"Numeric summary:\n{df.describe().to_string()}"
)
    return {"data_summary": summary}
    
    
def return_node(state: VizState) -> dict:
    if state.chart_path.startswith("error"):
        msg = f"Chart failed: {state.chart_path}"
    else:
        msg = (
            f"Chart saved to {state.chart_path}\n\n"
            f"**Data behind the chart:**\n{state.data_summary}"
        )
    return {"input_text": [AIMessage(content=msg)]}


def after_fetch_llm(s): return "fetch_tools" if s.input_text[-1].tool_calls else "end"
def after_fetch_tools(s): return "data_summary" if s.query_proposed_viz else "fetch_llm"
def after_propose_chart(s): return "chart_tools" if s.input_text[-1].tool_calls else "end"

graph = StateGraph(VizState)
graph.add_node("fetch_llm", fetch_llm_node)
graph.add_node("fetch_tools", fetch_tool_node)
graph.add_node("data_summary", data_summary_node)
graph.add_node("propose_chart", propose_chart_node)
graph.add_node("chart_tools", fetch_chart_node)
graph.add_node("approval", approval_node)
graph.add_node("execute", execute_chart_node)
graph.add_node("return", return_node)

graph.add_edge(START, "fetch_llm")
graph.add_conditional_edges("fetch_llm", after_fetch_llm, {"fetch_tools":"fetch_tools","end":END})
graph.add_conditional_edges("fetch_tools", after_fetch_tools,
                            {"data_summary": "data_summary", "fetch_llm": "fetch_llm"})
graph.add_edge("data_summary", "propose_chart")
graph.add_conditional_edges("propose_chart", after_propose_chart, {"chart_tools":"chart_tools","end":END})
graph.add_edge("chart_tools", "approval")
graph.add_conditional_edges("approval", approval_router, {"propose_chart":"propose_chart","execute":"execute","end":END})
graph.add_edge("execute", "return")
graph.add_edge("return", END)

viz_agent = graph.compile(checkpointer=InMemorySaver())




def ask(question: str, thread_id: str = "viz-1"):

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}


    result = viz_agent.invoke(
        {"input_text": [HumanMessage(content=question)]}, config
    )


    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n=== DATA-CHART REPORT ===\n", payload["chart_code"])
        print("\n=== DATA-SUMMARY REPORT ===\n", payload["data_summary"])

        choice = input("\n[a]pprove / [r]evise / [s]kip? ").lower().strip()
        if choice.startswith("r"):
            notes = input("What should change? ")
            result = viz_agent.invoke(
                Command(resume={"action": "revise", "notes": notes}), config)
        elif choice.startswith("s"):
            result = viz_agent.invoke(
                Command(resume={"action": "reject"}), config)
        else:
            result = viz_agent.invoke(
                Command(resume={"action": "approve"}), config)


    print("\n=== ANSWER ===\n", result["input_text"][-1].content)


if __name__ == "__main__":
    ask("Top 3 product categories by number of orders")