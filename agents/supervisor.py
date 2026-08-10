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

from agents.agent_sql_app import UserInput, llm, ask, sql_agent
from agents.agent_pandas_app import PandasState ,pandas_agent
from agents.agent_viz_app import VizState, viz_agent
import matplotlib.pyplot as plt
import plotly


load_dotenv()

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

class SupervisorState(UserInput, PandasState, VizState):
    destination: str = ""
    reasoning: str = ""

llm_with_schema = llm.with_structured_output(Route)

SYSTEM = SystemMessage(content="""
    You are a routing agent.

    Always base your decision on the user's LATEST question.

    Before choosing a route, first determine whether the latest question can be answered completely from the current conversation history.
    If the answer has already been computed or explicitly stated earlier in the current conversation,
    Do not query the database again.
    If the required information is missing, incomplete, or only partially related, continue to the routing rules below.
    
    routing:
    
    1. SQL
    If answering requires retrieving new data from the SQLite database using SQL
    (counts, sums, averages, joins, filters, rankings, group-bys, simple distributions, etc.),
    return route="sql".

    2. PANDAS
    If SQL results require additional statistical analysis
    (correlation, regression, variance, standard deviation, quantiles, hypothesis tests, etc.),
    return route="pandas".

    3. viz
    If the user requests a chart, plot, graph, draw, or visualization,
    return route="viz".

    4. direct
    If the question requires no database access
    (definitions, explanations, capabilities, greetings, or other general knowledge),
    return route="direct".

    Always return:
    - route
    - a one-sentence reason
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


checkpointer = InMemorySaver()
sup_graph = graph.compile(checkpointer= checkpointer)

def run_supervisor(question: str, thread_id: str=  "sup-1"):

    sup_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 35}
    result = sup_graph.invoke(
        {"input_text": [HumanMessage(content=question)]}, sup_config
    )
    print(result)



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
