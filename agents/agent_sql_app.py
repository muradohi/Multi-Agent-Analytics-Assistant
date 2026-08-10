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
# create the lmm
load_dotenv()

try:
    config_path = Path(__file__).parent.parent / "config/config.yaml"
    print("Config path is correct")
except:
    print("Config Path is not correct")

print(config_path)
with open(config_path, 'r') as f:
    cnf = yaml.safe_load(f)

db = cnf["database"]

# The engine
DB_URL = URL.create(
    db["drivername"],
    username=db["username"],
    password= os.getenv("DB_PASSWORD"),
    host = db["host"],
    port = db["port"],
    database= db["database"]
)

BASE_DIR = Path(__file__).parent.parent
db_path = BASE_DIR / "src" / "olist.db"

engine = create_engine(f"sqlite:///{db_path}")


@tool
def list_tables() -> str:
    """List all table names in the database. Call this first to see what exists."""

    inspector = inspect(engine)
    tables_name_list = inspector.get_table_names()
    tables_name_str = ",".join(tables_name_list)

    return tables_name_str

@tool
def schema_tables(table_name: str) -> str:
    """Show the columns and their types for one table. Call this to learn a
    table's schema before writing a query against it."""

    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    columns_schema = "\n".join(f"{c['name']} ({c['type']})" for c in columns)

    return columns_schema

@tool
def propose_final_query(sql: str) -> str:
    """Propose the final SELECT query to answer the question. Call this ONCE,
    after inspecting the schema, when you have the query you want to run.
    Do NOT call this until you know the exact tables and columns."""
    # This tool does NOT run anything. It only signals 'done exploring'.
    # The actual recording into state happens in tool_node below.
    return "Query proposed."

tools = [list_tables, schema_tables, propose_final_query]






#state and the LLM node

llm = ChatOpenAI(
    model = cnf["llm"]["model"]
)

llm_with_tools = llm.bind_tools(tools)

class UserInput(BaseModel):
    input_text : Annotated[list[AnyMessage], add_messages]
    proposed_sql: str = ""
    query_result: str = ""
    query_proposed: bool = False


SYSTEM = SystemMessage(content=(
    "You are a SQL analyst for a sqlite e-commerce database. "
    "Write sqlite compatible SQL only. Do NOT use PostgreSQL syntax such as "
    "'::type' casts (e.g. ::numeric) or Postgres-only functions. "
    "For rounding use ROUND(expr, 2) directly — no casts. "
    "Workflow: call list_tables, then schema_tables on the relevant tables, "
    "Never assume table or column names."
    "Only use tables and columns returned by list_tables and schema_tables."
    "then call propose_final_query. "
    "Category names are Portuguese."
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

def llm_node(state: UserInput) -> dict:
    turn = current_turn_messages(state)
    prompt = [SYSTEM, *turn]
    return {"input_text": [llm_with_tools.invoke(prompt)]}

def tool_node(state: UserInput) -> str:
    tools_by_name = {t.name: t for t in tools}
    last_msg = state.input_text[-1]
    out_messages = []
    updates = {}
    proposed = False


    for call in last_msg.tool_calls:
        if call["name"] == "propose_final_query":
            proposed = True
            updates["proposed_sql"] = call['args']["sql"]
            out_messages.append(ToolMessage(
                content="Query recorded for approval.",
                tool_call_id=call["id"],
            ))


        else:

            tool_name = call['name']
            tool_id = call['id']
            tool_args = call['args']

            chosen_tool = tools_by_name[tool_name]
            res = chosen_tool.invoke(tool_args)

            out_messages.append(ToolMessage(content=f"{str(res)}", tool_call_id = tool_id))
    updates["input_text"] = out_messages
    updates["query_proposed"] = proposed
    
    return updates


def latest_question(state) -> str:
    for msg in reversed(state.input_text):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def approval_node(state: UserInput):

    decision = interrupt({
        "proposed_sql" : state.proposed_sql,
        "instructions": "action: approve / reject / revise (+notes)",
        
    })

    action = decision['action']
    if action == "revise":
        notes = decision.get("notes", "")

        return {
            "input_text": [HumanMessage(content=(
            f"The reviewer requested changes: {notes} and the query you wrote {state.proposed_sql}. "
        ))],
            "proposed_sql": "",
            "query_result": "revise",
    }
    elif action == "reject":
        return {"query_result": "rejected"}

    
    return {"query_result": "approved"}



def execute_node(state: UserInput) ->str:
    """Run a read-only SQL SELECT query and return the rows. Only SELECT is
    allowed; anything else is rejected."""

    sql = state.proposed_sql

    if not sql:
        return {"query_result": "ERROR: no query was recorded to execute."}
    if not sql.lower().startswith("select"):
        return {"query_result": "Rejected: only SELECT queries are allowed."}

    try:

        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            preview = rows[:50]

        res =  str([tuple(r) for r in preview])

    except Exception as e:
        return f"SQL error: {e}"

    return {"query_result": res}

def answer_node(state: UserInput) -> dict:
    question = latest_question(state)         # the original question
    prompt = [
        SystemMessage(content=(
            "You are a data analyst presenting findings to a non-technical stakeholder. "
            "Answer the question directly and concisely using ONLY the query results.\n\n"
            "STRUCTURE your answer as:\n"
            "1. A one-sentence direct answer to the question.\n"
            "2. 3-5 key findings as short bullets, most important first.\n"
            "3. One sentence of insight or caveat if relevant.\n\n"
            "RULES:\n"
            "- Lead with the answer, not the data.\n"
            "- Do NOT list every row or every metric. Report only what answers the question.\n"
            "- Round numbers and use plain language (say 'about 4.5 stars', not '4.48').\n"
            "- Do NOT offer to do more work or list follow-up options.\n"
            "- Category names are Portuguese; keep them as-is but you may add a short English gloss.\n"
            "- Keep the whole answer under 150 words."
        )),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Query results:\n{state.query_result}"
        )),
    ]

    answer = llm.invoke(prompt).content
    return {"input_text": [AIMessage(content=answer)]}


#routing decision and the graph

def llm_router(state):
    last = state.input_text[-1]

    if getattr(last, "tool_calls", None):
        return "tool_node"

    raise RuntimeError(
        "LLM did not call propose_final_query."
    )

def tools_router(state: UserInput) -> str:
    """If a final query has been proposed, we're done exploring -> stop here
    (for now). Otherwise loop back to keep exploring."""
    return "approval_node" if state.query_proposed else "llm_node"

def approval_router(state: UserInput):
    """Revise your SQL and call propose_final_query again.
        Do NOT answer in prose."""

    qr = state.query_result
    if qr == "revise":   return "llm_node"
    if qr == "rejected": return "end"

    return "execute_node" 




graph = StateGraph(UserInput)


graph.add_node("llm_node", llm_node)
graph.add_node("tool_node", tool_node)
graph.add_node("approval_node", approval_node)
graph.add_node("execute_node", execute_node)
graph.add_node("answer_node", answer_node)

graph.add_edge(START, "llm_node")
graph.add_conditional_edges("llm_node", llm_router,{
    "tool_node": "tool_node", "approval_node": "approval_node", "llm_node": "llm_node"
})
# graph.add_edge("llm_node", "tool_node")
graph.add_conditional_edges( "tool_node", tools_router, {"approval_node": "approval_node", "llm_node": "llm_node"})
graph.add_conditional_edges("approval_node", approval_router, {"execute_node": "execute_node", "llm_node": "llm_node", "end": END})
graph.add_edge("execute_node", "answer_node")
graph.add_edge("answer_node", END)


checkpointer = InMemorySaver()
sql_agent = graph.compile(checkpointer=checkpointer)

def ask(question: str, thread_id: str = "sql-session-1"):

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 35}

    llm_response = sql_agent.invoke(
    {"input_text": [HumanMessage(content=question)]},
    config,
    
    
)
    while "__interrupt__" in llm_response:
        sql = llm_response["__interrupt__"][0].value["proposed_sql"]
        print("\n=== PROPOSED SQL ===")
        print(sql)
        print("\n=====================")

        choice = input("Choose: \n[a]pprove / [r]evise / [s]kip? ").lower().strip()

        if choice.startswith("r"):
            notes = input("What should change? ")
            llm_response = sql_agent.invoke(Command(resume={"action": "revise", "notes": notes}), config)

        elif choice.startswith("s"):
            llm_response = sql_agent.invoke(Command(resume={"action": "reject"}), config)
        else:
            llm_response = sql_agent.invoke(Command(resume={"action": "approve"}), config)





    print("\n=== ANSWER ===")

    print(llm_response["input_text"][-1].content)



# if __name__ == "__main__":
#     q = cnf["sql"]["query"]
#     ask(q)