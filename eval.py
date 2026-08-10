# eval.py
from agents import sup_graph, pandas_agent, AnalysisPlan
from langchain_core.messages import HumanMessage
import json

# your test set: question -> which agent SHOULD handle it
CASES = [
    ("Is there a correlation between price and review score?", "correlation"),
    ("What's the distribution of review scores?", "distribution"),
    ("What is the total revenue?", "aggregate"),
    ("Which states generate the most revenue?", "ranking"),
    ("Do late orders have lower review scores?", "comparison"),
    ("How has revenue changed over time?", "trend"),
]

def run_evals():
    passed = 0
    for question, expected_intent in CASES:
        cfg = {"configurable": {"thread_id": f"eval-{hash(question)}"},
               "recursion_limit": 35}
        result = pandas_agent.invoke({"input_text": [HumanMessage(content=question)]}, cfg)
        plan = result.get("plan")
        actual = plan.intent if plan else None
        ok = actual == expected_intent
        passed += ok
        mark = "✅" if ok else "❌"
        print(f"{mark} [{actual or 'None':>7}] expected [{expected_intent:>7}]  {question}")
    print(f"\n{passed}/{len(CASES)} routing cases passed")

if __name__ == "__main__":
    run_evals()