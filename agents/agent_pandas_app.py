import os, io, contextlib
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import text
from typing import List, Annotated, Literal
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
from functools import lru_cache
from sqlalchemy import inspect

from agents.agent_sql_app import engine, llm, list_tables, schema_tables

load_dotenv()

@lru_cache(maxsize=1)
def get_schema_string() -> str:
    insp = inspect(engine)
    lines = []
    for t in insp.get_table_names():
        cols = ", ".join(f"{c['name']} ({c['type']})" for c in insp.get_columns(t))
        lines.append(f"{t}: {cols}")
    return "\n".join(lines)

#planner
class AnalysisPlan(BaseModel):
    intent: Literal[
    "aggregate", "comparison", "correlation",
    "distribution", "trend", "ranking", "exploratory"
] = Field(..., description="The kind of analysis. Use 'exploratory' ONLY when the "
          "question is too open-ended to fit the others (e.g. 'find unusual patterns').")
    metrics: list[str] = Field(default_factory=list, description="Numeric columns to measure, e.g. ['review_score']")
    grain: str = Field("", description="What one row represents: order, order_item, customer")
    dimensions: list[str] = Field(default_factory=list, description="What to break down / compare by, e.g. ['delivery_time']")
    filters: list[str] = Field(default_factory=list, description="Conditions to apply, e.g. ['order_status = delivered']")
    comparison: str = Field("", description="If intent is comparison, the two groups, e.g. 'on-time vs late'")
    

class PandasState(BaseModel):

    input_text: Annotated[list[AnyMessage], add_messages]
    
    plan:  AnalysisPlan | None = None

    fetch_sql: str = ""
    code_proposed_pandas: bool = False
    query_proposed_sql: bool = False
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


llm_planner = llm.with_structured_output(AnalysisPlan)
def analysis_plan_node(state: PandasState) -> dict:
    system = SystemMessage(content=(
        "You are planning a data analysis. Read the user's question and produce a "
        "structured plan describing WHAT to analyze, not how.\n\n"

        "Choose exactly one intent:\n"
        "- aggregate: a total, count, sum, average, minimum, or maximum.\n"
        "- comparison: compare a metric between two or more explicitly defined groups.\n"
        "- correlation: relationship between two numeric variables.\n"
        "- distribution: distribution or spread of a numeric/categorical variable.\n"
        "- trend: how a metric changes over time or another ordered dimension.\n"
        "- ranking: top/bottom entities according to a metric.\n"
        "- exploratory: the question is open-ended and asks to discover patterns, "
        "anomalies, unusual behavior, or insights without specifying one exact analysis.\n\n"

        "For exploratory questions:\n"
        "- Do NOT leave the plan empty.\n"
        "- Choose a useful grain based on the question.\n"
        "- Identify relevant metrics that would help investigate the question, "
        "even when the user did not explicitly name them.\n"
        "- Identify relevant dimensions that could reveal patterns.\n"
        "- For purchasing behavior, useful metrics may include total spend, "
        "order count, average order value, item count, purchase frequency, "
        "and relevant product/category dimensions.\n\n"

        "- metrics: the numeric column(s) needed for the analysis.\n"
        "- grain: what one row represents (order, order_item, customer, etc.).\n"
        "- dimensions: what to break down or investigate by.\n"
        "- filters: conditions to apply.\n"
        "- comparison: only for comparison intent.\n\n"
        
        f"SCHEMA:\n{get_schema_string()}\n\n"
        "metrics, dimensions, grain must reference REAL columns from this schema. "
        "Do NOT invent derived names like 'total_spend_per_customer' — those are "
        "computed later in analysis, not named here.\n\n"

        "Plan for THIS question only. Keep it minimal, but for exploratory "
        "questions include enough information to investigate the requested pattern."
    ))

    human = HumanMessage(content=latest_question(state))
    plan = llm_planner.invoke([system, human])

    return {"plan": plan}



def latest_question(state) -> str:
    for msg in reversed(state.input_text):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

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

def fetch_llm_node(state: PandasState) -> dict:
    plan = state.plan if state.plan else None

    plan_block = ""
    if plan:
        plan_block = (
            "\n\nANALYSIS PLAN (fetch only what this needs):\n"
            f"- metrics: {plan.metrics}\n"
            f"- grain: {plan.grain}\n"
            f"- dimensions: {plan.dimensions}\n"
            f"- filters: {plan.filters}\n"
        )

    system = SystemMessage(content=(
    "You are a data analyst preparing to fetch data for analysis with pandas.\n\n"

    "First inspect the schema with list_tables and schema_tables. "
    "Then call propose_fetch_sql with a SELECT that retrieves the data "
    "required by the analysis plan.\n\n"

    "GENERAL RULES:\n"
    "- Pull ONLY columns needed for the plan.\n"
    "- Never use SELECT *.\n"
    "- Use SQLite-compatible SQL.\n"
    "- Use the correct grain specified by the plan.\n"
    "- Apply the plan's filters.\n"
    "- Do not invent columns that are not present in the schema.\n"
    "- Category names are Portuguese.\n\n"

    "EXPLORATORY RULE:\n"
    "For exploratory questions, the user is asking you to DISCOVER patterns. "
    "Do not return an empty dataset merely because the user did not specify "
    "a metric.\n"
    "Use the plan to determine the relevant data needed for investigation.\n"
    "For purchasing-pattern questions, customer-level purchasing behavior "
    "may require aggregating order/order-item data into customer-level metrics "
    "such as order count, total spend, average order value, and item count.\n"
    "If the plan requires derived metrics, calculate them in SQL using "
    "GROUP BY rather than expecting pandas to reconstruct missing information.\n\n"

    "Do NOT ask the user questions or answer in prose. Immediately inspect "
    "the schema and call the appropriate tools."
    + plan_block
))

    msgs = current_turn_messages(state)
    response = llm_with_tools.invoke([system, *msgs])
    return {"input_text": [response], "code_proposed_pandas": False, "fetch_sql": ""}


def fetch_tool_node(state: PandasState) -> dict:
    tools_by_name = {t.name: t for t in tools}
    last_msg = state.input_text[-1]
    out = []
    updates = {}
    proposed = False


    for call in last_msg.tool_calls:
        if call["name"] == "propose_fetch_sql":
            proposed = True
 
            updates["fetch_sql"] = call["args"]["sql"]
            out.append(ToolMessage(content="Fetch query recorded.",
                                   tool_call_id=call["id"]))
        else:

            res = tools_by_name[call["name"]].invoke(call["args"])
            out.append(ToolMessage(content=str(res), tool_call_id=call["id"]))

    updates["input_text"] = out
    updates["query_proposed_sql"] = proposed
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

    if df.empty:
        report = (
            "EMPTY DATASET: The fetch query returned 0 rows. "
            "No analysis should be performed."
        )
    else:
        report = data_quality_report(df)

    return {"qa_report": report}



@tool
def propose_analysis_code(code: str) -> str:
    """Propose pandas code to answer the question. `df` holds the data; `pd`, `np`
    are available. Assign the final answer to a variable named `result`.
    Call this ONCE with your complete code."""
    return "Analysis code recorded."

analysis_tools = [propose_analysis_code]
llm_with_analysis_tools = llm.bind_tools(analysis_tools)


def propose_analysis_node(state: PandasState) -> dict:
    plan = state.plan if state.plan else None

    plan_block = ""
    if plan:
        plan_block = (
            f"\n\nANALYSIS PLAN:\n"
            f"- intent: {plan.intent}\n"
            f"- metrics: {plan.metrics}\n"
            f"- dimensions: {plan.dimensions}\n"
            f"- comparison: {plan.comparison}\n"
        )

    system = SystemMessage(content=(
        "Write pandas code to answer the question. `df` holds the fetched data; "
        "`pd`, `np` are available. Assign the final answer to a variable `result`.\n\n"
        "Call the propose_analysis_code TOOL with your code. Do NOT write the code as "
        "text in your reply. After calling the tool, output nothing.\n\n"
        "Compute ONLY what the plan's intent requires — nothing more:\n"
        "- aggregate: the requested totals/means for the metrics.\n"
        "- comparison: compare the metric across the two groups named in 'comparison' "
        "(report each group's value and the difference).\n"
        "- correlation: the correlation coefficient(s) for the metrics (Pearson and "
        "Spearman), nothing else.\n"
        "- distribution: summary statistics and shape of the metric.\n"
        "- trend: the metric over the ordered dimension (e.g. over time).\n"
        "- ranking: the ordered top/bottom results for the metric by dimension.\n\n"
        "- exploratory: the question is open-ended. Do NOT invent columns. Summarize only "
        "the real columns available (basic describe/counts), and note that the question "
        "needs narrowing. Keep result small."
        "Account for issues in the data-quality report (drop nulls before correlating, "
        "handle duplicates/outliers) but do not turn every cleaning choice into a "
        "separate reported number. `result` must be a SMALL dict holding only the few "
        "numbers that answer the question, not a dump of everything you calculated."
        + plan_block
    ))

    # user question + fetch context, anchored to the current turn
    human = HumanMessage(content=(
        f"Question: {latest_question(state)}\n"
        f"Fetch SQL: {state.fetch_sql}\n"
        f"Data-quality report:\n{state.qa_report}"
    ))

    prompt = [system, human]


    return {"input_text": [llm_with_analysis_tools.invoke(prompt)]}


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
    plan = state.plan if state.plan else None
    decision = interrupt({
        "plan": plan.model_dump() if plan else {},
        "qa_report": state.qa_report,
        "proposed_code": state.proposed_code,
        "instructions": "action: approve / reject / revise (+notes)",
    })

    action = decision["action"]

    if action == "revise":

        notes = decision.get("notes", "")
        return {
            "input_text": [HumanMessage(content=(
                f"Revise the analysis. Feedback: {notes}, and the code you wrote {state.proposed_code} "
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
        "You are explaining a data result to a non-technical manager. Answer the "
        "question directly and clearly using ONLY the computed results.\n\n"
        "RULES:\n"
        "- Lead with a direct answer to the exact question (yes / no / it depends), "
        "in one sentence.\n"
        "- Then give 2-3 short supporting points — the numbers that actually answer "
        "the question, not every metric computed.\n"
        "- Translate correlations into plain words: |r| under 0.1 = negligible, "
        "0.1-0.3 = weak, 0.3-0.5 = moderate, above 0.5 = strong. Say 'a moderate "
        "relationship', not 'Pearson = -0.33'.\n"
        "- Prefer concrete comparisons a person feels (e.g. 'on-time orders averaged "
        "4.3 stars vs 2.6 for late ones') over abstract coefficients.\n"
        "- Use plain names ('delivery time', not 'delivery_days'). Round hard "
        "('about 4.3 stars', not '4.2926').\n"
        "- If an effect shrinks after removing outliers, explain what that means in "
        "words (the link is real but modest) — don't just report both numbers.\n"
        "- Keep it under ~120 words. No variable names, no raw dicts, no lists of "
        "every coefficient."
    )),
        HumanMessage(content=(
            f"Question: {latest_question(state)}\n"
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
    return "data_quality" if state.query_proposed_sql else "fetch_llm"


def after_propose(state):
    return "analysis_tools" if state.input_text[-1].tool_calls else "end"



graph = StateGraph(PandasState)

# add every station (node)
graph.add_node("analysis_plan", analysis_plan_node)
graph.add_node("fetch_llm", fetch_llm_node)
graph.add_node("fetch_tools", fetch_tool_node)
graph.add_node("data_quality", data_quality_node)
graph.add_node("propose_analysis", propose_analysis_node)
graph.add_node("analysis_tools", analysis_tool_node)
graph.add_node("approval", approval_node)
graph.add_node("execute", execute_node)
graph.add_node("answer", answer_node)


graph.add_edge(START, "analysis_plan")
graph.add_edge("analysis_plan", "fetch_llm")


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
    breakpoint()
    planner = result.get("plan")
    print(planner)


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