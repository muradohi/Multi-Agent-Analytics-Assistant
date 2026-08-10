# eval.py
from agents import sup_graph
from langchain_core.messages import HumanMessage

# your test set: question -> which agent SHOULD handle it
CASES = [
    ("How many orders were cancelled?",                          "sql"),
    ("What are the top 5 categories by number of orders?",       "sql"),
    ("Is there a correlation between price and review score?",   "pandas"),
    ("Plot the top 3 categories by number of orders",           "viz"),
    ("What can you do?",                                          "direct"),
]

def run_evals():
    passed = 0
    for question, expected_route in CASES:
        cfg = {"configurable": {"thread_id": f"eval-{hash(question)}"},
               "recursion_limit": 35}
        result = sup_graph.invoke({"input_text": [HumanMessage(content=question)]}, cfg)
        actual = result.get("destination")
        ok = actual == expected_route
        passed += ok
        mark = "✅" if ok else "❌"
        print(f"{mark} [{actual or 'None':>7}] expected [{expected_route:>7}]  {question}")
    print(f"\n{passed}/{len(CASES)} routing cases passed")

if __name__ == "__main__":
    run_evals()