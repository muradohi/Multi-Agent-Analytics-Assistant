from dotenv import load_dotenv
from sqlalchemy import text, create_engine, URL, inspect
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from pydantic import Field, BaseModel
from typing import List, Annotated, Literal
from langchain.tools import tool
import os
from langgraph.graph import StateGraph, START, END
import yaml
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.agent_sql_app import UserInput, llm, ask, sql_agent
from agents.agent_pandas_app import PandasState ,pandas_agent
from agents.agent_viz_app import VizState, viz_agent
import matplotlib.pyplot as plt
import plotly
import sqlite3

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()
DB_URL = os.environ["DATABASE_URL"]

pool = ConnectionPool(
    conninfo=DB_URL,
    max_size=10,
    open=False,
    kwargs={"autocommit": True, "prepare_threshold": 0},
)
pool.open(wait=True, timeout=30)

try:
    config_path = Path(__file__).parent.parent / "config/config.yaml"
    print("Config path is correct")
except:
    print("Config Path is not correct")

with open(config_path, 'r') as f:
    cnf = yaml.safe_load(f)

class Route(BaseModel):
    destination: Literal["sql", "pandas", "viz", "direct"]
    reasoning : str

class SupervisorState(BaseModel):
    input_text: Annotated[list[AnyMessage], add_messages]
    destination: str = ""
    reasoning: str = ""

llm_with_schema = llm.with_structured_output(Route)

SYSTEM = SystemMessage(content="""
        You are a routing agent for a data analytics assistant.

        Your job is ONLY to decide which handler should answer the user's LATEST question.
        Do NOT answer the question yourself.

        --------------------------------------------------
        STEP 1 — Conversation Memory
        --------------------------------------------------

        First check whether the latest question can be answered completely from the current conversation history.

        If the exact answer has already been computed or explicitly stated earlier in this conversation:
        - return route="direct"
        - do NOT query the database again.

        If the previous conversation does not fully answer the question,
        continue to the routing rules below.

        --------------------------------------------------
        STEP 2 — Choose ONE route
        --------------------------------------------------

        ### SQL

        Choose route="sql" when the user is requesting NEW data from the database that can be answered with SQL.

        Typical requests:

        - counts
        - sums
        - averages
        - minimum / maximum
        - percentages
        - rankings
        - top/bottom N
        - filtering
        - joins
        - group-by
        - distinct values
        - distributions
        - time aggregations

        Examples:

        ✓ How many customers are from São Paulo?
        ✓ Total revenue this year.
        ✓ Top 10 selling products.
        ✓ Average delivery time.
        ✓ Number of cancelled orders.
        ✓ Percentage of 5-star reviews.

        Do NOT choose SQL if statistical analysis or visualization is required.

        --------------------------------------------------

        ### PANDAS

        Choose route="pandas" when SQL results require additional computation or when the request is exploratory.

        This includes:

        Statistical analysis

        - correlation
        - covariance
        - regression
        - hypothesis tests
        - ANOVA
        - t-tests
        - variance
        - standard deviation
        - quantiles
        - percentiles

        Exploratory analysis

        - full analysis
        - exploratory data analysis (EDA)
        - summarize the dataset
        - analyze my data
        - business insights
        - trends
        - anomalies
        - patterns
        - key findings
        - recommendations
        - customer segmentation
        - feature relationships
        - outlier detection

        Examples:

        ✓ Is there a correlation between price and review score?
        ✓ Give me a full analysis of the database.
        ✓ What trends do you see?
        ✓ Which variables are most related?
        ✓ Find unusual customers.

        --------------------------------------------------

        ### VIZ

        Choose route="viz" when the primary request is to create or modify a visualization.

        Examples:

        ✓ Plot revenue over time.
        ✓ Draw a histogram.
        ✓ Show a scatter plot.
        ✓ Create a dashboard.
        ✓ Visualize review scores.

        If the request asks for BOTH analysis and visualization,
        choose "viz".

        --------------------------------------------------

        ### DIRECT

        Choose route="direct" ONLY when NO database access is required.

        Examples:

        ✓ What is SQL?
        ✓ Explain correlation.
        ✓ What can you do?
        ✓ Hello.
        ✓ Thanks.

        --------------------------------------------------

        Rules

        - Always choose exactly ONE route.
        - Prefer SQL over Pandas if SQL alone can answer the question.
        - Choose Pandas only when SQL is insufficient.
        - Choose Viz whenever the user's primary goal is a chart or visualization.
        - Never choose Direct if answering requires database access.

        Return ONLY:

        route
        reasoning
    """)
def sup_node(state: SupervisorState):
    system_msg = SYSTEM

    user_msg = state.input_text
    prompt = [system_msg] + user_msg

    
    llm_response = llm_with_schema.invoke(prompt)

    destination = llm_response.destination
    reasoning = llm_response.reasoning

    return {"destination": destination, "reasoning": reasoning}


def direct_node(state: SupervisorState):

    prompt = [
    SystemMessage(
        content= "You answer questions that need no database query. Two cases:\n"
            "1. If the answer was already computed or stated earlier in this "
            "conversation, recall it exactly from the history.\n"
            "2. Otherwise, answer from your own general knowledge "
            "(definitions, explanations, capabilities, greetings).\n"
            "Only say you don't know if it's a live/external fact you genuinely "
            "can't know (e.g. current weather)."
    ),
    *state.input_text
]
    llm_response = llm.invoke(prompt)
    ai_msg = [AIMessage(content=llm_response.content)]

    return {"input_text": ai_msg}

    







def conditional_edge(state: SupervisorState) -> str:
    category = state.destination
    if category == "sql":    return "sql_route"
    elif category == "pandas": return "pandas_route"
    elif category == "viz": return "viz_route"
    elif category == "direct": return "direct_route"
    else: raise ValueError("Invalid category")




graph = StateGraph(SupervisorState)

graph.add_node("sup_node", sup_node)
graph.add_node("sql_node", sql_agent)
graph.add_node("pandas_node", pandas_agent)
graph.add_node("viz_node", viz_agent)
graph.add_node("direct_node", direct_node)

graph.add_edge(START, "sup_node")
graph.add_conditional_edges("sup_node", conditional_edge,{"sql_route":"sql_node", "pandas_route":"pandas_node", "viz_route": "viz_node", "direct_route": "direct_node"})
# graph.add_edge("sql_node", "approval_node")
# graph.add_conditional_edges("sql_node", conditional_edge,{"sql_route":"sql_node", "pandas_route":"pandas_node", "direct_route": "direct_node"})

graph.add_edge("sql_node", END)
graph.add_edge("pandas_node", END)
graph.add_edge("viz_node", END)
graph.add_edge("direct_node", END)


checkpointer = PostgresSaver(pool)
checkpointer.setup()

sup_graph = graph.compile(checkpointer=checkpointer)

def run_supervisor(question: str, thread_id: str=  "sup-1"):

    sup_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 35}
    result = sup_graph.invoke(
        {"input_text": [HumanMessage(content=question)]}, sup_config
    )



    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        if "proposed_sql" in payload:
            print("\n=== PROPOSED SQL ===\n", payload["proposed_sql"])
        elif "proposed_code" in payload:
            print("\n=== DATA-QUALITY REPORT ===\n", payload.get("qa_report", ""))
            print("\n=== PROPOSED CODE ===\n", payload["proposed_code"])
            
        elif "chart_code" in payload:
                    print("\n=== Chart REPORT ===\n", payload.get("chart_code", ""))
                    print("\n=== PROPOSED PATH ===\n", payload["chart_path"])

        choice = input("[a]pprove / [r]evise / [s]kip? ").lower().strip()

        if choice.startswith("r"):
            notes = input("What should change? ")
            result = sup_graph.invoke(Command(resume={"action": "revise", "notes": notes}), sup_config)
        elif choice.startswith("s"):
            result = sup_graph.invoke(Command(resume={"action": "reject"}), sup_config)
        else:
            result = sup_graph.invoke(Command(resume={"action": "approve"}), sup_config)

    print("\n=== ANSWER ===")
    print(result["input_text"][-1].content)

if __name__ == "__main__":
    for q in ["Is there a correlation between product price and review score?",]:
        run_supervisor(q)
