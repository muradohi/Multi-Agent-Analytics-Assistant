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
from functools import lru_cache
from sqlalchemy import inspect


from agents.agent_sql_app import engine, llm, list_tables, schema_tables

load_dotenv()

from typing import Literal
from pydantic import BaseModel, Field

from functools import lru_cache
from sqlalchemy import inspect   # add inspect to your existing sqlalchemy import

@lru_cache(maxsize=1)
def get_schema_string() -> str:
    insp = inspect(engine)
    lines = []
    for t in insp.get_table_names():
        cols = ", ".join(f"{c['name']} ({c['type']})" for c in insp.get_columns(t))
        lines.append(f"{t}: {cols}")
    return "\n".join(lines)

class ChartPlan(BaseModel):
    chart_type: Literal[
        "bar", "line", "scatter", "histogram", "box", "pie", "heatmap"
    ] = Field(..., description="The chart type that best answers the question")
    x: str = Field("", description="Column for the x-axis / categories (real column name)")
    y: str = Field("", description="Column for the y-axis / values (real column; empty for histogram)")
    dimensions: list[str] = Field(default_factory=list, description="Grouping / color-by columns")
    filters: list[str] = Field(default_factory=list, description="Conditions, e.g. ['order_status = delivered']")
    top_n: int = Field(0, description="For rankings, limit to top N (0 = no limit)")
    aggregation: str = Field("", description="How to aggregate y: sum, mean, count (empty if raw)")
    
    
class VizState(BaseModel):
    input_text: Annotated[list[AnyMessage], add_messages]
    chart_plan:  ChartPlan | None = None
    schema_info: str = ""
    fetch_sql: str = ""
    query_proposed_viz: bool = False
    data_summary: str = ""
    chart_code: str = ""
    chart_path: str = ""
    chart_error: str = ""
    chart_attempts: int = 0 
    action: str = ""
    
    
@tool
def propose_fetch_sql(sql: str) -> str:
    """Propose the SELECT that fetches the data to analyze. Call ONCE after
    inspecting the schema. Pull only the needed columns, not SELECT *."""
    return "Fetch query recorded."

def latest_question(state) -> str:
    for msg in reversed(state.input_text):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

schema_tools = [list_tables, schema_tables]
sql_tools = [propose_fetch_sql]

llm_with_schema_tools = llm.bind_tools(schema_tools)
llm_with_sql_tools = llm.bind_tools(sql_tools)


chart_plan_llm = llm.with_structured_output(ChartPlan)


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



def chart_plan_node(state: VizState) -> dict:
    system = SystemMessage(content=(
    "You are planning a chart. Read the user's question and produce a structured "
    "plan describing WHAT to chart.\n"
    f"SCHEMA:\n{get_schema_string()}\n\n"

    "- chart_type: pick the single best type:\n"
    "    bar = category comparison or ranking; line = trend over time; "
    "scatter = relationship between two numerics; histogram = distribution of one "
    "numeric; box = distribution across groups; heatmap = correlation matrix; "
    "pie = part-to-whole (AVOID unless truly part-to-whole with few categories — "
    "prefer bar).\n"

    "- x, y: REAL column names only. Do NOT invent derived names.\n"

    "- dimensions: grouping/color columns.\n"
    "- filters: conditions.\n"
    "- top_n: for rankings, the N to show.\n"
    "- aggregation: sum / mean / count if y needs aggregating.\n"

    "- For 'number of orders', use order_id as y with aggregation='count'.\n"
    "- For unique counts, the SQL stage should use COUNT(DISTINCT ...).\n"

    "Plan for THIS question only. Keep it minimal."
))
    human = HumanMessage(content=latest_question(state))
    plan = chart_plan_llm.invoke([system, human])
    return {"chat_plan": plan}

# def schema_llm_node(state: VizState) -> dict:

#     plan = state.plan

#     plan_block = ""
#     if plan:
#         plan_block = (
#             "\n\nCHART PLAN:\n"
#             f"- chart_type: {plan.chart_type}\n"
#             f"- x: {plan.x}\n"
#             f"- y: {plan.y}\n"
#             f"- dimensions: {plan.dimensions}\n"
#             f"- filters: {plan.filters}\n"
#             f"- aggregation: {plan.aggregation}\n"
#             f"- top_n: {plan.top_n}\n"
#         )

#     system = SystemMessage(content=(
#         "You are inspecting the database schema before generating SQL.\n\n"

#         "Use list_tables first to identify available tables.\n"
#         "Then use schema_tables for the tables relevant to the chart plan.\n\n"
        
#         "Once you have inspected the tables needed for the chart plan, STOP calling "
#         "tools and reply with a single word: 'done'. Do not call any more tools "
#         "after that.\n\n"

#         "Do not generate SQL yet.\n"
#         "Do not answer the user.\n"
#         "Only inspect the schema and call the tools."
#         + plan_block
#     ))

#     response = llm_with_schema_tools.invoke(
#         [system, HumanMessage(content=latest_question(state))]
#     )

#     return {
#         "input_text": [response]
#     }
    
    
def fetch_sql_llm_node(state: VizState) -> dict:
    plan = state.chart_plan if state.chart_plan else None 

    plan_block = ""
    if plan:
        plan_block = (
            "\n\nCHART PLAN:\n"
            f"- chart_type: {plan.chart_type}\n"
            f"- x: {plan.x}\n"
            f"- y: {plan.y}\n"
            f"- dimensions: {plan.dimensions}\n"
            f"- filters: {plan.filters}\n"
            f"- aggregation: {plan.aggregation}\n"
            f"- top_n: {plan.top_n}\n"
        )

    system = SystemMessage(content=(
        "You are a SQL analyst.\n\n"
        "Generate the SQLite SQL required to fetch the data for the chart.\n\n"
        "You are given the database schema below. Use ONLY columns and tables that "
        "actually exist in it.\n\n"
        "RULES:\n"
        "- Never use SELECT *.\n"
        "- Use SQLite-compatible SQL.\n"
        "- Select only columns required for the chart.\n"
        "- Follow the chart plan exactly.\n"
        "- Apply filters, aggregation, and top_n from the plan.\n"
        "- For counts use COUNT; for unique entities use COUNT(DISTINCT ...).\n"
        "- Use correct JOIN conditions based on the schema.\n"
        "- Do not invent columns. Category names are Portuguese.\n\n"
        "Call propose_fetch_sql exactly once.\n"
        + plan_block
    ))

    human = HumanMessage(content=(
        f"User question:\n{latest_question(state)}\n\n"
        f"DATABASE SCHEMA:\n{get_schema_string()}"
    ))

    response = llm_with_sql_tools.invoke([system, human])
    return {"input_text": [response], "query_proposed_viz": False}

def fetch_sql_tool_node(state: VizState) -> dict:

    last_msg = state.input_text[-1]

    out = []
    updates = {}
    proposed = False

    for call in last_msg.tool_calls:

        if call["name"] == "propose_fetch_sql":
            proposed = True

            updates["fetch_sql"] = call["args"]["sql"]

            out.append(
                ToolMessage(
                    content="SQL query recorded.",
                    tool_call_id=call["id"]
                )
            )

    updates["input_text"] = out
    updates["query_proposed_viz"] = proposed

    return updates

# def schema_tool_node(state: VizState) -> dict:

#     tools_by_name = {
#         t.name: t for t in schema_tools
#     }

#     last_msg = state.input_text[-1]

#     out = []
#     schema_parts = []

#     for call in last_msg.tool_calls:

#         result = tools_by_name[call["name"]].invoke(call["args"])

#         schema_parts.append(
#             f"{call['name']}:\n{result}"
#         )

#         out.append(
#             ToolMessage(
#                 content=str(result),
#                 tool_call_id=call["id"]
#             )
#         )

#     schema_info = "\n\n".join(schema_parts)

#     return {
#         "input_text": out,
#         "schema_info": schema_info
#     }

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
    plan = state.chart_plan if state.chart_plan else None

    plan_block = ""
    if plan:
        plan_block = (
            f"\n\nCHART PLAN:\n"
            f"- chart_type: {plan.chart_type}\n"
            f"- x: {plan.x}   y: {plan.y}\n"
            f"- dimensions: {plan.dimensions}   top_n: {plan.top_n}\n"
        )

    system = SystemMessage(content=(
        "Write matplotlib code to build the planned chart. The DataFrame is in `df`; "
        "`pd`, `np`, `plt` are available.\n\n"
        "Build the chart_type the plan specifies:\n"
        "- bar: x on categories, y as bars; if top_n set, show only the top N sorted "
        "descending; use barh for readable long category labels.\n"
        "- line: x on the (time) axis, y as the line; sort by x first.\n"
        "- scatter: x vs y as points; add a light trend line if it aids reading.\n"
        "- histogram: distribution of x. If the data is heavily right-skewed, draw TWO "
        "panels side by side (linear scale and log scale) so the shape is legible; mark "
        "the median.\n"
        "- box: y distribution across the x groups, showing outliers.\n"
        "- heatmap: correlation matrix of the numeric columns (annotated).\n"
        "- pie: only for part-to-whole with few slices.\n\n"
        "STYLE RULES:\n"
        "- Pass color and linestyle SEPARATELY (e.g. color='black', linestyle='--'); "
        "NEVER put a format string like 'k--' inside ls= or linestyle=.\n"
        "- Add a title and axis labels. Use tight_layout.\n"
        "- Save to 'chart_output.png' with bbox_inches='tight', then plt.close().\n\n"
        "Call the propose_chart_code TOOL with your code. Do NOT write code as text. "
        "After calling the tool, output nothing."
        + plan_block
    ))
    human = HumanMessage(content=(
        f"Question: {latest_question(state)}\n"
        f"Data available (use these exact column names):\n{state.data_summary}"
    ))


    return {"input_text": [llm_with_viz_tool.invoke([system, human])]}



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
    plan = state.chart_plan if state.chart_plan else None
    decision = interrupt({
        "plan": plan.model_dump() if plan else {},
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
    if len(df) == 0:
        return {"data_summary": "ERROR: the fetch returned no rows — the planned "
                                 "columns may not exist or the filters excluded everything."}
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
        system = SystemMessage(content=(
            "You are describing a chart to a stakeholder. In ONE sentence, state "
            "what the chart shows and the single most notable takeaway (the top "
            "item, the trend, the outlier). Then present the underlying data as a "
            "short clean list. Round numbers, use plain names. No 'chart saved' "
            "boilerplate, no offers to do more."
        ))
        human = HumanMessage(content=(
            f"Question: {latest_question(state)}\n\n"
            f"Data behind the chart:\n{state.data_summary}"
        ))
        takeaway = llm.invoke([system, human]).content
        msg = takeaway
    return {"input_text": [AIMessage(content=msg)]}


# def after_schema_llm(state):
#     return "schema_tools" if state.input_text[-1].tool_calls else "fetch_sql"
# def after_schema_tools(state):

#     return "schema_llm"
def after_fetch_sql(state):

    if state.input_text[-1].tool_calls:
        return "fetch_sql_tools"

    return "end"
def after_propose_chart(s): return "chart_tools" if s.input_text[-1].tool_calls else "end"

graph = StateGraph(VizState)
graph.add_node("chart_plan", chart_plan_node)


graph.add_node("fetch_sql", fetch_sql_llm_node)
graph.add_node("fetch_sql_tools", fetch_sql_tool_node)
graph.add_node("data_summary", data_summary_node)
graph.add_node("propose_chart", propose_chart_node)
graph.add_node("chart_tools", fetch_chart_node)
graph.add_node("approval", approval_node)
graph.add_node("execute", execute_chart_node)
graph.add_node("return", return_node)

graph.add_edge(START, "chart_plan")

graph.add_edge("chart_plan", "fetch_sql") 

graph.add_conditional_edges(
    "fetch_sql",
    after_fetch_sql,
    {
        "fetch_sql_tools": "fetch_sql_tools",
        "end": END
    }
)

graph.add_edge(
    "fetch_sql_tools",
    "data_summary"
)

graph.add_edge(
    "data_summary",
    "propose_chart"
)

graph.add_conditional_edges(
    "propose_chart",
    after_propose_chart,
    {
        "chart_tools": "chart_tools",
        "end": END
    }
)

graph.add_edge(
    "chart_tools",
    "approval"
)

graph.add_conditional_edges(
    "approval",
    approval_router,
    {
        "propose_chart": "propose_chart",
        "execute": "execute",
        "end": END
    }
)

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