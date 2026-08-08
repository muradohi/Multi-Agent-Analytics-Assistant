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
import matplotlib.pyplot as plt
import plotly
import seaborn as sns

from agents.agent_sql_app import engine, llm, list_tables, schema_tables

load_dotenv()



class PandasState(BaseModel):

    input_text: Annotated[list[AnyMessage], add_messages]

    fetch_sql: str = "" 
    qa_report: str = ""
    proposed_code: str = ""
    code_result: str = ""
    action: str = ""



@tool
def propose_fetch_sql(sql: str) -> str:
    """Propose the SELECT that fetches the data to analyze. Call ONCE after
    inspecting the schema. Pull only the needed columns, not SELECT *."""
    return "Fetch query recorded."


tools = [list_tables, schema_tables, propose_fetch_sql]

llm_with_tools = llm.bind_tools(tools)


SYSTEM = SystemMessage(content=(
    "You are a data analyst preparing to analyze e-commerce data with pandas. "
    "First inspect the schema (list_tables, schema_tables). Then call "
    "propose_fetch_sql with a SELECT pulling ONLY the needed columns. "
    "Write SQLite-compatible SQL. Category names are Portuguese."
))


def fetch_llm_node(state: PandasState) -> dict:

    response = llm_with_tools.invoke([SYSTEM] + state.input_text)
    return {"input_text": [response]}


def fetch_tool_node(state: PandasState) -> dict:
    tools_by_name = {t.name: t for t in tools}
    last_msg = state.input_text[-1]
    out = []
    updates = {}


    for call in last_msg.tool_calls:
        if call["name"] == "propose_fetch_sql":
 
            updates["fetch_sql"] = call["args"]["sql"]
            out.append(ToolMessage(content="Fetch query recorded.",
                                   tool_call_id=call["id"]))
        else:

            res = tools_by_name[call["name"]].invoke(call["args"])
            out.append(ToolMessage(content=str(res), tool_call_id=call["id"]))

    updates["input_text"] = out
    return updates


def load_dataframe(fetch_sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(fetch_sql), conn)



def data_quality_report(df: pd.DataFrame) -> str:
    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns"]

    # --- CHECK 1: nulls (missing values) per column ---
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) == 0:
        lines.append("Nulls: none")
    else:
        lines.append("Nulls:")
        for col, n in null_cols.items():
            pct = 100 * n / len(df)
            lines.append(f"  - {col}: {n} ({pct:.1f}%)")


    lines.append(f"Duplicate rows: {df.duplicated().sum()}")

    lines.append("Column types:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        flag = ""
        if dtype == "object":
            sample = df[col].dropna().astype(str).head(100)
            # if every sampled value looks like a number, flag it
            if len(sample) > 0 and sample.str.match(r'^-?\d+\.?\d*$').all():
                flag = "  <-- looks numeric but stored as text"
        lines.append(f"  - {col}: {dtype}{flag}")

    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        outs = []
        for col in numeric.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1 
            low  = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            n_out = int(((df[col] < low) | (df[col] > high)).sum())
            if n_out > 0:
                outs.append(f"  - {col}: {n_out} outliers (outside [{low:.1f}, {high:.1f}])")
        lines.append("Outliers (beyond 1.5x IQR):")
        lines.extend(outs if outs else ["  - none"])

    return "\n".join(lines)


def data_quality_node(state: PandasState) -> dict:
    df = load_dataframe(state.fetch_sql)
    report = data_quality_report(df)
    return {"qa_report": report}



@tool
def propose_analysis_code(code: str) -> str:
    """Propose pandas code answering the question. `df` holds the data; `pd`,`np`
    available. Assign the final answer to `result`. Call ONCE; not run yet."""
    return "Analysis code recorded."

analysis_tools = [propose_analysis_code]
llm_with_analysis_tools = llm.bind_tools(analysis_tools)


def propose_analysis_node(state: PandasState) -> dict:
    system = SystemMessage(content=(
        "Write pandas code to answer the question. Data is in `df`; `pd`,`np` available. "
        "Assign the answer to a variable `result`. Account for issues in the quality "
        "report (drop nulls before correlating, etc). Call propose_analysis_code once.\n\n"
        f"Question: {state.input_text[0].content}\n"
        f"Fetch SQL: {state.fetch_sql}\n"
        f"Data-quality report:\n{state.qa_report}"
    ))
    response = llm_with_analysis_tools.invoke([system])
    return {"input_text": [response]}


def analysis_tool_node(state: PandasState) -> dict:
    last = state.input_text[-1]
    out, updates = [], {}
    for call in last.tool_calls:
        if call["name"] == "propose_analysis_code":
            updates["proposed_code"] = call["args"]["code"]
            out.append(ToolMessage(content="Analysis code recorded.",
                                   tool_call_id=call["id"]))
    updates["input_text"] = out
    return updates



def approval_node(state: PandasState) -> dict:
    decision = interrupt({
        "qa_report": state.qa_report,
        "proposed_code": state.proposed_code,
        "instructions": "action: approve / reject / revise (+notes)",
    })

    action = decision["action"]

    if action == "revise":

        notes = decision.get("notes", "")
        return {
            "input_text": [HumanMessage(content=(
                f"Revise the analysis. Feedback: {notes}. "
                f"Call propose_analysis_code again with corrected code. "
                f"Do NOT answer in prose."
            ))],
            "proposed_code": "",
            "action": "revise",
        }
    if action == "reject":
        return {"action": "rejected"}
    return {"action": "approved"}


def approval_router(state: PandasState) -> str:
    if state.action == "revise":   return "propose_analysis"
    if state.action == "rejected": return "end"
    return "execute"



def execute_node(state: PandasState) -> dict:
    df = load_dataframe(state.fetch_sql)     # re-load the data
    # the ONLY names the code can use:
    namespace = {"df": df, "pd": pd, "np": np, "result": None}
    stdout = io.StringIO()                   # a place to capture any print()s

    try:
        # redirect_stdout: anything the code prints goes into our buffer
        with contextlib.redirect_stdout(stdout):
            exec(state.proposed_code, namespace)
        result = namespace.get("result")
        printed = stdout.getvalue()

        out = ""
        if result is not None:
            out += f"result = {result}\n"
        if printed:
            out += f"stdout:\n{printed}"
        code_result = out or "Code ran but produced no `result`."
    except Exception as e:

        code_result = f"Code error: {e}"

    return {"code_result": code_result}



def answer_node(state: PandasState) -> dict:
    prompt = [
        SystemMessage(content=(
            "Answer the question using ONLY the analysis result. Be concise, lead "
            "with the answer, use plain numbers. Note relevant data-quality caveats."
        )),
        HumanMessage(content=(
            f"Question: {state.input_text[0].content}\n"
            f"Data-quality: {state.qa_report}\n"
            f"Analysis code: {state.proposed_code}\n"
            f"Result: {state.code_result}"
        )),
    ]
    answer = llm.invoke(prompt).content
    return {"input_text": [AIMessage(content=answer)]}



def after_fetch_llm(state):
    return "fetch_tools" if state.input_text[-1].tool_calls else "end"


def after_fetch_tools(state):
    return "data_quality" if state.fetch_sql else "fetch_llm"


def after_propose(state):
    return "analysis_tools" if state.input_text[-1].tool_calls else "end"



graph = StateGraph(PandasState)

# add every station (node)
graph.add_node("fetch_llm", fetch_llm_node)
graph.add_node("fetch_tools", fetch_tool_node)
graph.add_node("data_quality", data_quality_node)
graph.add_node("propose_analysis", propose_analysis_node)
graph.add_node("analysis_tools", analysis_tool_node)
graph.add_node("approval", approval_node)
graph.add_node("execute", execute_node)
graph.add_node("answer", answer_node)


graph.add_edge(START, "fetch_llm")


graph.add_conditional_edges("fetch_llm", after_fetch_llm,
                            {"fetch_tools": "fetch_tools", "end": END})


graph.add_conditional_edges("fetch_tools", after_fetch_tools,
                            {"data_quality": "data_quality", "fetch_llm": "fetch_llm"})


graph.add_edge("data_quality", "propose_analysis")


graph.add_conditional_edges("propose_analysis", after_propose,
                            {"analysis_tools": "analysis_tools", "end": END})


graph.add_edge("analysis_tools", "approval")


graph.add_conditional_edges("approval", approval_router,
                            {"propose_analysis": "propose_analysis",
                             "execute": "execute", "end": END})


graph.add_edge("execute", "answer")
graph.add_edge("answer", END)


checkpointer = InMemorySaver()
pandas_agent = graph.compile(checkpointer=checkpointer)



def ask(question: str, thread_id: str = "pandas-1"):

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}


    result = pandas_agent.invoke(
        {"input_text": [HumanMessage(content=question)]}, config
    )


    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n=== DATA-QUALITY REPORT ===\n", payload["qa_report"])
        print("\n=== PROPOSED ANALYSIS CODE ===\n", payload["proposed_code"])

        choice = input("\n[a]pprove / [r]evise / [s]kip? ").lower().strip()
        if choice.startswith("r"):
            notes = input("What should change? ")
            result = pandas_agent.invoke(
                Command(resume={"action": "revise", "notes": notes}), config)
        elif choice.startswith("s"):
            result = pandas_agent.invoke(
                Command(resume={"action": "reject"}), config)
        else:
            result = pandas_agent.invoke(
                Command(resume={"action": "approve"}), config)


    print("\n=== ANSWER ===\n", result["input_text"][-1].content)


if __name__ == "__main__":
    ask("Is there a correlation between product price and review score?")